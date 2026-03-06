"""
state manager - orchestrator node for the experiment

phase 1: training (100 trials to learn)
phase 2: testing (run 20 trials with physical navigation with TurtleBot4Navigator)

subs to: /carl/trial_result, /carl/zone
publishes to: /carl/trial_cmd
"""

import json
import os
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav2_simple_commander.robot_navigator import TaskResult
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator

# compute repo root from source file location (works with --symlink-install)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.realpath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, 'cuttlefish_sim', 'sim_gazebo'))
from delayed_gratification import (
    prob_wait, STATE_NAMES,
    LEFT, DEAD_RWD, UNOBTAINABLE_RWD, STATES,
)

# coordinates in gazebo
NAV_CENTER = [0.0, -1.5]
NAV_LEFT_CHAMBER = [-2.5, 1.5]
NAV_RIGHT_CHAMBER = [2.5, 1.5]

# spawn position (must match sim.py)
SPAWN_X = 0.0
SPAWN_Y = -1.0
SPAWN_YAW = 90.0  # degrees, facing +y = WEST in TurtleBot4Directions

# experiment parameters
TRAINING_TRIALS = 100
TESTING_TRIALS_PER_DELAY = 20
DELAYS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]

# results file
RESULTS_FILE = os.path.join(REPO_ROOT, 'cuttlefish_sim', 'sim_gazebo', 'results.json')


class StateManager(Node):
    def __init__(self):
        super().__init__('state_manager')

        # pubs and subs
        self.cmd_pub = self.create_publisher(String, '/carl/trial_cmd', 10)

        self.result_sub = self.create_subscription(
            String, '/carl/trial_result', self.result_callback, 10)

        self.zone_sub = self.create_subscription(
            String, '/carl/zone', self.zone_callback, 10)

        # state
        self.last_result = None
        self._result_event = threading.Event()
        self.current_zone = None

        # store results
        self.training_results = []
        self.testing_results = {d: [] for d in DELAYS}

        # nav for physical movement
        self.navigator = TurtleBot4Navigator()

        self.get_logger().info('State manager started. Beginning experiment...')

        # run experiment in a background thread so the executor isn't blocked
        self._experiment_thread = threading.Thread(
            target=self._run_experiment, daemon=True)
        self._experiment_thread.start()

    def result_callback(self, msg: String):
        self.last_result = json.loads(msg.data)
        self._result_event.set()

    def zone_callback(self, msg: String):
        data = json.loads(msg.data)
        self.current_zone = data.get('zone')

    def send_trial_cmd(self, p_wait, forced_state=None):
        """Send a trial command to the brain node and wait for the result."""
        cmd = {'p_wait': p_wait}
        if forced_state is not None:
            cmd['state'] = forced_state

        self.last_result = None
        self._result_event.clear()

        msg = String()
        msg.data = json.dumps(cmd)
        self.cmd_pub.publish(msg)

        # wait for result via threading event (no recursive spinning needed)
        if not self._result_event.wait(timeout=10.0):
            self.get_logger().error('Timed out waiting for brain response!')
            return None

        return self.last_result

    def navigate_to(self, coords, label):
        """Navigate the robot to a position and wait for arrival."""
        self.get_logger().info(f'Navigating to {label} ({coords[0]:.1f}, {coords[1]:.1f})')

        goal = self.navigator.getPoseStamped(coords, TurtleBot4Directions.NORTH)
        self.navigator.startToPose(goal)

        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info(f'Arrived at {label}')
            if self.current_zone:
                self.get_logger().info(f'  Zone: {self.current_zone}')
        else:
            self.get_logger().warn(f'Navigation to {label} failed (status={result})')

        return result == TaskResult.SUCCEEDED

    def _run_experiment(self):
        """Runs in a background thread. Waits for Nav2, then runs full experiment."""
        # brief sleep to let the executor start spinning
        time.sleep(2.0)

        # set initial pose so AMCL knows where the robot is
        self.get_logger().info('Setting initial pose...')
        initial_pose = self.navigator.getPoseStamped(
            [SPAWN_X, SPAWN_Y], SPAWN_YAW)
        self.navigator.setInitialPose(initial_pose)

        self.get_logger().info('Waiting for Nav2 to become active...')
        self.navigator.waitUntilNav2Active()

        # undock before navigating (TurtleBot4 spawns on dock)
        self.get_logger().info('Undocking...')
        self.navigator.undock()

        self.run_training_phase()
        self.run_testing_phase()
        self.save_results()

        self.get_logger().info('Experiment complete!')

    # phase 1: training with no navigation, just learn the state labels
    def run_training_phase(self):
        self.get_logger().info(
            f'=== TRAINING PHASE: {TRAINING_TRIALS} trials ===')

        state_total = np.zeros(STATES)
        state_hits = np.zeros(STATES)

        for t in range(TRAINING_TRIALS):
            result = self.send_trial_cmd(p_wait=1.0)
            if result is None:
                continue

            s = result['state']
            state_total[s] += 1
            state_hits[s] += int(result['correct'])

            self.training_results.append(result)

            if (t + 1) % 25 == 0:
                pcts = []
                for si in range(STATES):
                    if state_total[si] > 0:
                        pcts.append(f'{STATE_NAMES[si]}={100*state_hits[si]/state_total[si]:.0f}%')
                self.get_logger().info(
                    f'Training {t+1}/{TRAINING_TRIALS}: {", ".join(pcts)}')

        self.get_logger().info('Training phase complete.')

    # phase 2: testing with physical navigation and varying delays
    def run_testing_phase(self):
        self.get_logger().info(
            f'=== TESTING PHASE: {len(DELAYS)} delays x {TESTING_TRIALS_PER_DELAY} trials ===')

        # start from center
        self.navigate_to(NAV_CENTER, 'CENTER')

        for delay in DELAYS:
            p_w = prob_wait(delay)
            self.get_logger().info(
                f'--- Delay {delay}s (p_wait={p_w:.3f}) ---')

            exp_total = 0
            exp_correct = 0
            ctrl_total = 0
            ctrl_correct = 0

            for t in range(TESTING_TRIALS_PER_DELAY):
                result = self.send_trial_cmd(p_wait=p_w)
                if result is None:
                    continue

                action = result['action']
                is_exp = result['is_experimental']
                reward = result['reward']

                # nav to chosen chamber
                if action == LEFT:
                    self.navigate_to(NAV_LEFT_CHAMBER, 'LEFT_CHAMBER')
                else:
                    self.navigate_to(NAV_RIGHT_CHAMBER, 'RIGHT_CHAMBER')

                # record result
                if is_exp:
                    exp_total += 1
                    exp_correct += int(reward > DEAD_RWD)
                else:
                    ctrl_total += 1
                    ctrl_correct += int(reward > UNOBTAINABLE_RWD)

                self.testing_results[delay].append(result)

                # return to center for next trial
                self.navigate_to(NAV_CENTER, 'CENTER')

            # log delay summary
            exp_pct = (100 * exp_correct / exp_total) if exp_total > 0 else 0
            ctrl_pct = (100 * ctrl_correct / ctrl_total) if ctrl_total > 0 else 0
            self.get_logger().info(
                f'Delay {delay}s: experimental={exp_pct:.1f}% '
                f'({exp_correct}/{exp_total}), '
                f'control={ctrl_pct:.1f}% ({ctrl_correct}/{ctrl_total})')

        self.get_logger().info('Testing phase complete.')

    def save_results(self):
        """Save all results to a JSON file for later plotting."""
        data = {
            'training': self.training_results,
            'testing': {str(d): trials for d, trials in self.testing_results.items()},
            'delays': DELAYS,
            'params': {
                'training_trials': TRAINING_TRIALS,
                'testing_trials_per_delay': TESTING_TRIALS_PER_DELAY,
            }
        }

        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        with open(RESULTS_FILE, 'w') as f:
            json.dump(data, f, indent=2)

        self.get_logger().info(f'Results saved to {RESULTS_FILE}')


def main():
    rclpy.init()
    node = StateManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
