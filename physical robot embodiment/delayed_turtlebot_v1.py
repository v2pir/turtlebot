import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator

import random
from math import exp
import numpy as np
from scipy.stats import norm
import time

"""
Kary Zheng
28 January 2026
"""

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
UNOBTAINABLE_RWD = 0.0

ALPHA = 0.10
BETA = 1.0
GAMMA = 0.99

# ------------------ CHAMBER COORDINATES ------------------

LEFT_CHAMBER  = (1.9318245649,  0.5751032829)
RIGHT_CHAMBER = (1.9457954168, -0.2844115793)

# ------------------ Q TABLE ------------------

qTbl = [[0.0 for y in range(ACTIONS)] for x in range(STATES)]

# ------------------ FUNCTIONS (UNCHANGED LOGIC) ------------------

def prob_wait(tim):
    mean = 70
    std_dev = 20
    beta_weight = 2
    pw = 1 / np.exp(beta_weight * norm.cdf(tim, loc=mean, scale=std_dev))
    return pw


def action_select(q, beta):
    sumSoftMax = 0
    for i in range(len(q)):
        sumSoftMax += exp(beta*q[i])

    r = random.random()
    sumP = 0

    for i in range(len(q)):
        p = exp(beta*q[i]) / sumSoftMax
        sumP += p
        if sumP >= r:
            return i
    return RIGHT


# ------------------ ROBOT CLASS ------------------

class DelayedGratificationRobot(Node):
    def __init__(self):
        super().__init__('delayed_gratification_robot')
        self.navigator = BasicNavigator()
        self.navigator.waitUntilNav2Active()

        self.run_experiment()


    def go_to_chamber(self, action):
        if action == LEFT:
            x, y = LEFT_CHAMBER
        else:
            x, y = RIGHT_CHAMBER

        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.w = 1.0

        self.navigator.goToPose(goal)

        while not self.navigator.isTaskComplete():
            rclpy.spin_once(self, timeout_sec=0.1)

        return self.navigator.getResult()


    def run_trial(self, p_wait):
        global qTbl

        # r = random.random()
        # if r < 0.25:
        #     current_state = EXPM_LR
        # elif r < 0.50:
        #     current_state = EXPM_RL
        # elif r < 0.75:
        #     current_state = CTRL_LR
        # else:
        #     current_state = CTRL_RL

        current_state = EXPM_LR

        q = qTbl

        # Apply patience ONLY to live shrimp choices
        if current_state == EXPM_LR:
            q[current_state][LEFT] *= p_wait
        elif current_state == EXPM_RL:
            q[current_state][RIGHT] *= p_wait

        act = action_select(q[current_state], BETA)

        # Robot physically goes to chamber
        self.go_to_chamber(act)

        # Determine reward EXACTLY like your simulation
        if current_state == EXPM_LR:
            rwd = LIVE_RWD if act == LEFT else DEAD_RWD
        elif current_state == EXPM_RL:
            rwd = DEAD_RWD if act == LEFT else LIVE_RWD
        elif current_state == CTRL_LR:
            rwd = UNOBTAINABLE_RWD if act == LEFT else DEAD_RWD
        else:
            rwd = DEAD_RWD if act == LEFT else UNOBTAINABLE_RWD

        # Q update (same)
        qTbl[current_state][act] = qTbl[current_state][act] + ALPHA*(rwd - qTbl[current_state][act])

        experimental_trial = current_state < CTRL_LR
        return rwd, experimental_trial, current_state, act


    def run_experiment(self):

        # ---------- TRAINING PHASE ----------
        #TRIALS = 100
        TRIALS=5
        for t in range(TRIALS):
            self.run_trial(1.0)

        # ---------- DELAY TEST PHASE ----------
        # delays = np.array([10,20,30,40,50,60,70,80,90,100,110,120,130])
        delays = np.array([10,20])
        # DELAY_TRIALS = 100
        DELAY_TRIALS=3
        delay_gratification_experiment_results = np.zeros(len(delays))
        delay_gratification_ctrl_results = np.zeros(len(delays))

        exp_trials = np.zeros(len(delays))
        ctrl_trials = np.zeros(len(delays))

        for d in range(len(delays)):
            for t in range(DELAY_TRIALS):
                rwd, ex, state, act = self.run_trial(prob_wait(delays[d]))

                if ex:
                    exp_trials[d] += 1
                    delay_gratification_experiment_results[d] += int(rwd > DEAD_RWD)
                else:
                    ctrl_trials[d] += 1
                    delay_gratification_ctrl_results[d] += int(rwd > UNOBTAINABLE_RWD)

            self.get_logger().info(
                f"Delay {delays[d]}  EXPM={(delay_gratification_experiment_results[d]*100)/exp_trials[d]:.2f}  CTRL={(delay_gratification_ctrl_results[d]*100)/ctrl_trials[d]:.2f}"
            )


# ------------------ MAIN ------------------

def main(args=None):
    rclpy.init(args=args)
    node = DelayedGratificationRobot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
