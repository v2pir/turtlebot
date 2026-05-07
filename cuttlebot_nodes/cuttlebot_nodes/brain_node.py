"""
brain node — q-learning decision maker for the delayed gratification experiment.

subs to /carl/trial_cmd (trial commands)
publishes to /carl/trial_result (action and reward)
"""

import json
import os
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.realpath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, 'cuttlefish_sim', 'sim_gazebo'))
from delayed_gratification import (
    QTable, run_trial, is_correct,
    STATE_NAMES, ACTION_NAMES, STATES, ACTIONS,
)


class BrainNode(Node):
    def __init__(self):
        super().__init__('brain_node')

        self.qtable = QTable()
        self.trial_count = 0
        self.confirm_count = 0

        self.result_pub = self.create_publisher(String, '/carl/trial_result', 10)

        self.cmd_sub = self.create_subscription(
            String, '/carl/trial_cmd', self.trial_callback, 10)

        self.confirm_sub = self.create_subscription(
            String, '/carl/reward_confirm', self.confirm_callback, 10)

        self._log('init', f'Brain node started. Q-table shape: {STATES}x{ACTIONS}')
        self._log('init', f'Initial Q-table:')
        self._print_qtable('init')
        self._log('init', 'Waiting for trial commands on /carl/trial_cmd...')

    def _log(self, tag, msg):
        self.get_logger().info(f'[brain:{tag}] {msg}')

    def _warn(self, tag, msg):
        self.get_logger().warn(f'[brain:{tag}] {msg}')

    def _print_qtable(self, tag):
        for s in range(STATES):
            row = [self.qtable.table[s][a] for a in range(ACTIONS)]
            self._log(tag, f'  {STATE_NAMES[s]}: L={row[0]:.4f} R={row[1]:.4f}')

    def trial_callback(self, msg: String):
        cmd = json.loads(msg.data)
        p_wait = cmd.get('p_wait', 1.0)
        forced_state = cmd.get('state', None)
        defer = cmd.get('defer_update', False)

        self._log('trial', f'Received cmd: p_wait={p_wait} '
                  f'forced_state={forced_state} defer_update={defer}')

        rwd, is_exp, state, action = run_trial(
            self.qtable, p_wait, forced_state=forced_state,
            update=not defer)

        self.trial_count += 1
        correct = is_correct(state, action)

        self._log('trial', f'Trial #{self.trial_count}: '
                  f'state={STATE_NAMES[state]}({state}) '
                  f'action={ACTION_NAMES[action]}({action}) '
                  f'reward={rwd:.1f} correct={correct} '
                  f'experimental={is_exp} deferred={defer}')

        if not defer:
            self._log('trial', f'Q-table updated immediately (not deferred)')
        else:
            self._log('trial', f'Q-table update DEFERRED — waiting for /carl/reward_confirm')

        if self.trial_count % 25 == 0:
            self._log('qtable', f'Q-table after {self.trial_count} trials:')
            self._print_qtable('qtable')

        result = {
            'state': state,
            'action': action,
            'reward': rwd,
            'is_experimental': is_exp,
            'correct': correct,
            'trial': self.trial_count,
        }

        result_msg = String()
        result_msg.data = json.dumps(result)
        self.result_pub.publish(result_msg)
        self._log('trial', f'Published result to /carl/trial_result')

    def confirm_callback(self, msg: String):
        cmd = json.loads(msg.data)
        state = cmd['state']
        action = cmd['action']
        reward = cmd['reward']

        self.confirm_count += 1
        self._log('confirm', f'Confirm #{self.confirm_count}: '
                  f'state={STATE_NAMES[state]}({state}) '
                  f'action={ACTION_NAMES[action]}({action}) '
                  f'reward={reward:.1f}')

        old_val = self.qtable.table[state][action]
        self.qtable.update(state, action, reward)
        new_val = self.qtable.table[state][action]

        self._log('confirm', f'Q[{STATE_NAMES[state]}][{ACTION_NAMES[action]}]: '
                  f'{old_val:.4f} -> {new_val:.4f} (delta={new_val - old_val:.4f})')


def main():
    rclpy.init()
    node = BrainNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
