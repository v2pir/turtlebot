"""
brain node — q-learning decision maker for the delayed gratification experiment.

subs to /carl/trial_cmd (trial comamands)
publishes to /carl/trial_result (action and reward)
"""

import json
import os
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# add sim_gazebo to path to import q-learning lib
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'cuttlefish_sim', 'sim_gazebo'))
from delayed_gratification import (
    QTable, run_trial, is_correct,
    STATE_NAMES, ACTION_NAMES,
    LEFT, RIGHT, EXPM_LR, CTRL_LR,
)

# create brain node class
class BrainNode(Node):
    def __init__(self):
        super().__init__('brain_node')

        self.qtable = QTable()
        self.trial_count = 0

        self.result_pub = self.create_publisher(String, '/carl/trial_result', 10)

        self.cmd_sub = self.create_subscription(
            String, '/carl/trial_cmd', self.trial_callback, 10)

        self.get_logger().info('Brain node started. Waiting for trial commands...')

    # callback for trial commands
    def trial_callback(self, msg: String):
        cmd = json.loads(msg.data)
        p_wait = cmd.get('p_wait', 1.0)
        forced_state = cmd.get('state', None)

        # get reward and action from trial
        rwd, is_exp, state, action = run_trial(
            self.qtable, p_wait, forced_state=forced_state)

        # increase trial count and check if action was correct
        self.trial_count += 1
        correct = is_correct(state, action)

        # log
        self.get_logger().info(
            f'Trial {self.trial_count}: state={STATE_NAMES[state]} '
            f'action={ACTION_NAMES[action]} reward={rwd:.1f} '
            f'correct={correct} experimental={is_exp}')

        # log q-table every 25 trials
        if self.trial_count % 25 == 0:
            self.qtable.print_table()

        # the result
        result = {
            'state': state,
            'action': action,
            'reward': rwd,
            'is_experimental': is_exp,
            'correct': correct,
            'trial': self.trial_count,
        }

        # publish result
        result_msg = String()
        result_msg.data = json.dumps(result)
        self.result_pub.publish(result_msg)


def main():
    rclpy.init()
    node = BrainNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
