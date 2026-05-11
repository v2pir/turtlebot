import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

from turtlebot4_navigation.turtlebot4_navigator import (
    TurtleBot4Navigator,
    TurtleBot4Directions
)
from nav2_simple_commander.robot_navigator import TaskResult

import random
from math import exp
import numpy as np
from scipy.stats import norm
import time
import copy


"""
Kary Zheng
Delayed Gratification Robot
Test with more Trials
4/15/2026
"""


# ------------------ CONSTANTS ------------------

ACTIONS = 2
LEFT = 0
RIGHT = 1

STATES = 4
EXPM_LR = 0
EXPM_RL = 1
CTRL_LR = 2
CTRL_RL = 3

LIVE_RWD = 5.0
DEAD_RWD = 1.0
UNOBTAINABLE_RWD = 0.5

ALPHA = 0.10
BETA = 1.0
GAMMA = 0.99


# ------------------ MAP COORDINATES ------------------

HOME_X = -0.023635001853108406
HOME_Y = -0.023635001853108406

LEFT_CHAMBER  = (1.9318245649,  0.5751032829)
RIGHT_CHAMBER = (1.9457954168, -0.2844115793)

# ------------------ Q TABLE ------------------

# qTbl = [[0.0 for _ in range(ACTIONS)] for _ in range(STATES)]
qTbl = [
    [3.588, 0.469],  # EXPM_LR
    [0.410, 4.074],  # EXPM_RL
    [0.433, 0.771],  # CTRL_LR
    [0.718, 0.235],  # CTRL_RL
]
#fnish 100 trials of training-->comment out training phase and put back delay phase
# Build command:
# cd ~/turtlebot4_ws
# colcon build
# source install/setup.bash
#desktop terminal and run automation4.py

# ------------------ RL FUNCTIONS ------------------

def prob_wait(tim):
    mean = 70
    std_dev = 20
    beta_weight = 2
    return 1 / np.exp(beta_weight * norm.cdf(tim, loc=mean, scale=std_dev))


def action_select(q, beta):
    softmax_sum = sum(exp(beta * v) for v in q)
    r = random.random()
    cumulative = 0.0

    for i in range(len(q)):
        p = exp(beta * q[i]) / softmax_sum
        cumulative += p
        if cumulative >= r:
            return i

    return RIGHT


# ------------------ ROBOT CLASS ------------------

class DelayedGratificationRobot(Node):

    def __init__(self):
        super().__init__('delayed_gratification_robot')

        self.navigator = TurtleBot4Navigator()

        self.start_pose = self.navigator.getPoseStamped(
            [HOME_X, HOME_Y],
            TurtleBot4Directions.NORTH
        )

        self.navigator.setInitialPose(self.start_pose)

        self.get_logger().info("Waiting for Nav2...")
        self.navigator.waitUntilNav2Active()

        self._wait_for_map_topic(timeout_sec=20.0)

        # Run experiment
        self.run_experiment()
        self.get_logger().info("Experiment finished.")


    # ------------------ MAP WAIT ------------------

    def _wait_for_map_topic(self, timeout_sec=20.0):
        start = time.time()
        while time.time() - start < timeout_sec:
            topics = [name for (name, _) in self.get_topic_names_and_types()]
            if "/map" in topics:
                self.get_logger().info("[Init] /map topic detected.")
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().warn("[Init] Timed out waiting for /map.")
        return False


    # ------------------ NAVIGATION ------------------

    def navigate_to(self, x, y, direction=None, timeout_sec=60.0):

        self.navigator.clearAllCostmaps()

        if direction is None:
            goal = self.navigator.getPoseStamped(
                [x, y],
                TurtleBot4Directions.NORTH
            )
        else:
            goal = self.navigator.getPoseStamped(
                [x, y],
                direction
            )

        self.get_logger().info(f"[Nav] Going to ({x:.3f}, {y:.3f})")

        self.navigator.goToPose(goal)

        start = time.time()

        while not self.navigator.isTaskComplete():
            rclpy.spin_once(self, timeout_sec=0.1)

            if time.time() - start > timeout_sec:
                self.get_logger().warn("[Nav] Timeout — canceling")
                self.navigator.cancelTask()
                return False

        result = self.navigator.getResult()

        if result == TaskResult.SUCCEEDED:
            self.get_logger().info("[Nav] SUCCEEDED")
            return True
        elif result == TaskResult.FAILED:
            self.get_logger().warn("[Nav] FAILED")
            return False
        else:
            self.get_logger().warn("[Nav] CANCELED")
            return False


    def go_home(self):
        self.get_logger().info("Returning home...")
        return self.navigate_to(
            HOME_X,
            HOME_Y,
            TurtleBot4Directions.NORTH
        )


    def go_to_chamber(self, action):
        if action == LEFT:
            x, y = LEFT_CHAMBER
        else:
            x, y = RIGHT_CHAMBER

        # First navigate normally
        success = self.navigate_to(x, y)

        if not success:
            return False

        # Then explicitly rotate to SOUTH
        self.get_logger().info("[Nav] Adjusting orientation to SOUTH")

        return self.navigate_to(
            x,
            y,
            TurtleBot4Directions.SOUTH
        )
        

    # ------------------ EXPERIMENT LOGIC ------------------

    def print_q_table(self):
        self.get_logger().info("Q Table:")
        for s in range(STATES):
            self.get_logger().info(
                f"State {s}: L={qTbl[s][LEFT]:.3f}, R={qTbl[s][RIGHT]:.3f}"
            )


    def run_trial(self, p_wait):

        global qTbl

        state = random.randint(0, STATES - 1)

        # Reset to home
        if not self.go_home():
            self.get_logger().error("Failed to go home — aborting trial")
            return 0.0, False, state, None

        time.sleep(1.0)

        # Thinking phase
        time.sleep(2.0)

        q_thinking = copy.deepcopy(qTbl)

        if state == EXPM_LR:
            q_thinking[state][LEFT] *= p_wait
        elif state == EXPM_RL:
            q_thinking[state][RIGHT] *= p_wait

        act = action_select(q_thinking[state], BETA)

        # Go to chamber
        self.go_to_chamber(act)

        time.sleep(1.0)

        # Reward logic
        if state == EXPM_LR:
            reward = LIVE_RWD if act == LEFT else DEAD_RWD
        elif state == EXPM_RL:
            reward = DEAD_RWD if act == LEFT else LIVE_RWD
        elif state == CTRL_LR:
            reward = UNOBTAINABLE_RWD if act == LEFT else DEAD_RWD
        else:
            reward = DEAD_RWD if act == LEFT else UNOBTAINABLE_RWD

        qTbl[state][act] += ALPHA * (reward - qTbl[state][act])

        return reward, state < CTRL_LR, state, act


    def run_experiment(self):
        experiment_start = time.time()

        # self.get_logger().info("===== TRAINING PHASE START =====")
        # TRIALS = 30
        # for t in range(TRIALS):
        #     self.get_logger().info(f"[Training] Trial {t+1}/{TRIALS}")
        #     self.run_trial(1.0)
        # self.get_logger().info("===== TRAINING PHASE FINISHED =====")

        # self.get_logger().info("===== PRETRAINED Q TABLE LOADED =====")
        # self.print_q_table()

        self.get_logger().info("===== DELAY TEST PHASE START =====")
        delays = np.array([10,20,30,40,50,60])
        DELAY_TRIALS = 5

        for d in delays:
            self.get_logger().info(f"--- Testing Delay = {d} ---")
            for t in range(DELAY_TRIALS):
                self.get_logger().info(f"[Delay {d}] Trial {t+1}/{DELAY_TRIALS}")
                self.run_trial(prob_wait(d))
        self.get_logger().info("===== DELAY TEST PHASE FINISHED =====")

        self.print_q_table()

        self.get_logger().info("===== EXPERIMENT COMPLETE =====")
        self.get_logger().info("Returning robot to home position...")
        self.go_home()
        time.sleep(2.0)

        experiment_end = time.time()
        total_time = experiment_end - experiment_start

        self.get_logger().info("===== EXPERIMENT COMPLETE =====")
        self.get_logger().info(f"Total runtime: {total_time:.2f} seconds")
        self.get_logger().info(f"Total runtime: {total_time/60:.2f} minutes")

# ------------------ MAIN ------------------

def main(args=None):
    rclpy.init(args=args)
    node = DelayedGratificationRobot()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
