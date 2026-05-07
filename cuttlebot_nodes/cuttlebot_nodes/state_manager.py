"""
state manager - orchestrator node for the experiment

phase 1: training (100 trials to learn)
phase 2: testing (run 20 trials with physical navigation with TurtleBot4Navigator)

subs to: /carl/trial_result, /carl/zone, /carl/vision_status
publishes to: /carl/trial_cmd, /carl/vision_cmd
"""

import json
import os
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from nav2_simple_commander.robot_navigator import TaskResult
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator

# compute repo root from source file location (works with --symlink-install)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.realpath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, 'cuttlefish_sim', 'sim_gazebo'))
from delayed_gratification import (
    prob_wait, STATE_NAMES, ACTION_NAMES,
    LEFT, DEAD_RWD, UNOBTAINABLE_RWD, STATES,
)

# coordinates in gazebo
NAV_CENTER = [0.0, -1.5]
NAV_LEFT_CHAMBER = [-2.5, 1.5]
NAV_RIGHT_CHAMBER = [2.5, 1.5]
NAV_GAP_APPROACH = [0.0, 0.3]

# spawn position (must match sim.py)
SPAWN_X = 0.0
SPAWN_Y = -1.0
SPAWN_YAW = 90.0  # degrees, facing +y = NORTH in TurtleBot4Directions

# experiment parameters
TRAINING_TRIALS = 100
TESTING_TRIALS_PER_DELAY = 20
DELAYS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]

# results file
RESULTS_FILE = os.path.join(REPO_ROOT, 'cuttlefish_sim', 'sim_gazebo', 'results.json')

TASK_RESULT_NAMES = {
    TaskResult.SUCCEEDED: 'SUCCEEDED',
    TaskResult.FAILED: 'FAILED',
    TaskResult.CANCELED: 'CANCELED',
    TaskResult.UNKNOWN: 'UNKNOWN',
}


class StateManager(Node):
    def __init__(self):
        super().__init__('state_manager')

        self.declare_parameter('mode', 'test')
        self.mode = self.get_parameter('mode').value

        # pubs and subs
        self.cmd_pub = self.create_publisher(String, '/carl/trial_cmd', 10)
        self.vision_cmd_pub = self.create_publisher(String, '/carl/vision_cmd', 10)
        self.confirm_pub = self.create_publisher(String, '/carl/reward_confirm', 10)

        self.result_sub = self.create_subscription(
            String, '/carl/trial_result', self.result_callback, 10)

        self.zone_sub = self.create_subscription(
            String, '/carl/zone', self.zone_callback, 10)

        self.vision_status_sub = self.create_subscription(
            String, '/carl/vision_status', self.vision_status_callback, 10)

        # state
        self.last_result = None
        self._result_event = threading.Event()
        self.current_zone = None

        # vision state
        self._vision_status = None
        self._vision_event = threading.Event()

        # store results
        self.training_results = []
        self.testing_results = {d: [] for d in DELAYS}

        # nav for physical movement
        self._log('init', 'Creating TurtleBot4Navigator...')
        self.navigator = TurtleBot4Navigator()
        self._log('init', 'TurtleBot4Navigator created.')

        self._log('init', f'State manager ready. mode={self.mode}')

        # run experiment in a background thread so the executor isn't blocked
        self._experiment_thread = threading.Thread(
            target=self._run_experiment, daemon=True)
        self._experiment_thread.start()

    def _log(self, tag, msg):
        self.get_logger().info(f'[{tag}] {msg}')

    def _warn(self, tag, msg):
        self.get_logger().warn(f'[{tag}] {msg}')

    def _err(self, tag, msg):
        self.get_logger().error(f'[{tag}] {msg}')

    # ── callbacks ──

    def result_callback(self, msg: String):
        data = json.loads(msg.data)
        self._log('cb:result', f'Got brain result: {data}')
        self.last_result = data
        self._result_event.set()

    def zone_callback(self, msg: String):
        data = json.loads(msg.data)
        new_zone = data.get('zone')
        if new_zone != self.current_zone:
            self._log('cb:zone', f'Zone changed: {self.current_zone} -> {new_zone}')
        self.current_zone = new_zone

    def vision_status_callback(self, msg: String):
        data = json.loads(msg.data)
        self._log('cb:vision', f'Got vision status: {data}')
        self._vision_status = data
        self._vision_event.set()

    # ── vision helpers ──

    def send_vision_cmd(self, cmd_dict):
        self._log('vision_cmd', f'Sending: {cmd_dict}')
        msg = String()
        msg.data = json.dumps(cmd_dict)
        self.vision_cmd_pub.publish(msg)

    def vision_scan(self, timeout=60.0):
        self._log('vision_scan', f'Starting scan (timeout={timeout}s)')
        self._vision_status = None
        self._vision_event.clear()
        self.send_vision_cmd({'action': 'scan'})

        if not self._vision_event.wait(timeout=timeout):
            self._warn('vision_scan', f'Timed out after {timeout}s')
            self.send_vision_cmd({'action': 'stop'})
            return False

        status = self._vision_status.get('status')
        self._log('vision_scan', f'Completed with status={status}')
        return status == 'both_seen'

    def vision_seek(self, target_color, timeout=90.0):
        self._log('vision_seek', f'Seeking {target_color} (timeout={timeout}s)')
        self._vision_status = None
        self._vision_event.clear()
        self.send_vision_cmd({'action': 'seek', 'color': target_color})

        if not self._vision_event.wait(timeout=timeout):
            self._warn('vision_seek', f'Timed out after {timeout}s for {target_color}')
            self.send_vision_cmd({'action': 'stop'})
            return False

        status = self._vision_status.get('status')
        self._log('vision_seek', f'Completed with status={status}')
        return status == 'reached'

    # ── brain helpers ──

    def send_trial_cmd(self, p_wait, forced_state=None, defer_update=False):
        cmd = {'p_wait': p_wait}
        if forced_state is not None:
            cmd['state'] = forced_state
        if defer_update:
            cmd['defer_update'] = True

        self._log('brain_cmd', f'Sending trial cmd: {cmd}')
        self.last_result = None
        self._result_event.clear()

        msg = String()
        msg.data = json.dumps(cmd)
        self.cmd_pub.publish(msg)

        if not self._result_event.wait(timeout=10.0):
            self._err('brain_cmd', 'Timed out waiting for brain response (10s)!')
            return None

        self._log('brain_cmd', f'Got result: state={STATE_NAMES[self.last_result["state"]]} '
                  f'action={ACTION_NAMES[self.last_result["action"]]} '
                  f'reward={self.last_result["reward"]:.1f} '
                  f'correct={self.last_result["correct"]}')
        return self.last_result

    # ── navigation ──

    def navigate_to(self, coords, label):
        self._log('nav', f'>>> navigate_to {label} ({coords[0]:.2f}, {coords[1]:.2f})')
        self._log('nav', f'    Current zone: {self.current_zone}')

        t0 = time.time()
        goal = self.navigator.getPoseStamped(coords, TurtleBot4Directions.NORTH)
        self._log('nav', f'    Goal pose created, calling startToPose...')
        self.navigator.startToPose(goal)
        elapsed = time.time() - t0

        result = self.navigator.getResult()
        result_name = TASK_RESULT_NAMES.get(result, str(result))
        self._log('nav', f'<<< navigate_to {label} finished: {result_name} '
                  f'(took {elapsed:.1f}s, zone={self.current_zone})')

        return result == TaskResult.SUCCEEDED

    # ── experiment flow ──

    def _run_experiment(self):
        time.sleep(2.0)
        self._log('experiment', '====== EXPERIMENT START ======')
        self._log('experiment', f'Mode: {self.mode}')
        self._log('experiment', f'Spawn: x={SPAWN_X} y={SPAWN_Y} yaw={SPAWN_YAW}')

        self._log('experiment', 'Setting initial pose for AMCL...')
        initial_pose = self.navigator.getPoseStamped(
            [SPAWN_X, SPAWN_Y], SPAWN_YAW)
        self.navigator.setInitialPose(initial_pose)
        self._log('experiment', 'Initial pose set.')

        self._log('experiment', 'Waiting for Nav2 to become active...')
        self.navigator.waitUntilNav2Active()
        self._log('experiment', 'Nav2 is active!')

        self._log('experiment', 'Undocking...')
        self.navigator.undock()
        self._log('experiment', 'Undock complete.')

        if self.mode == 'train':
            self.run_training_phase()
        else:
            self._log('experiment', 'Test mode — skipping training phase.')
        self.run_testing_phase()
        self.save_results()

        self._log('experiment', '====== EXPERIMENT COMPLETE ======')

    # ── phase 1: training ──

    def run_training_phase(self):
        self._log('training', f'=== TRAINING PHASE: {TRAINING_TRIALS} trials ===')

        state_total = np.zeros(STATES)
        state_hits = np.zeros(STATES)

        for t in range(TRAINING_TRIALS):
            self._log('training', f'--- Trial {t+1}/{TRAINING_TRIALS} ---')
            result = self.send_trial_cmd(p_wait=1.0)
            if result is None:
                self._warn('training', f'Trial {t+1} got no result (brain timeout)')
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
                self._log('training', f'Progress {t+1}/{TRAINING_TRIALS}: {", ".join(pcts)}')

        self._log('training', 'Training phase complete.')

    # ── phase 2: testing ──

    def run_testing_phase(self):
        self._log('testing', f'=== TESTING PHASE: {len(DELAYS)} delays x '
                  f'{TESTING_TRIALS_PER_DELAY} trials ===')

        self._log('testing', 'Navigating to CENTER before starting trials...')
        nav_ok = self.navigate_to(NAV_CENTER, 'CENTER')
        self._log('testing', f'Initial nav to CENTER: {"OK" if nav_ok else "FAILED"}')

        for delay_idx, delay in enumerate(DELAYS):
            p_w = prob_wait(delay)
            self._log('testing', f'===== Delay {delay_idx+1}/{len(DELAYS)}: '
                      f'{delay}s (p_wait={p_w:.4f}) =====')

            exp_total = 0
            exp_correct = 0
            ctrl_total = 0
            ctrl_correct = 0

            for t in range(TESTING_TRIALS_PER_DELAY):
                self._log('trial', f'--- Delay={delay}s Trial {t+1}/{TESTING_TRIALS_PER_DELAY} ---')

                # step 1: navigate to gap
                self._log('trial', 'Step 1: Navigate to GAP_APPROACH')
                nav_ok = self.navigate_to(NAV_GAP_APPROACH, 'GAP_APPROACH')
                self._log('trial', f'Nav to GAP_APPROACH: {"OK" if nav_ok else "FAILED"}')

                # step 2: vision scan
                self._log('trial', 'Step 2: Vision scan for both colors')
                scan_ok = self.vision_scan()
                if not scan_ok:
                    self._warn('trial', 'Vision scan failed — skipping trial, returning to CENTER')
                    self.navigate_to(NAV_CENTER, 'CENTER')
                    continue

                # step 3: brain decides
                self._log('trial', f'Step 3: Brain decision (p_wait={p_w:.4f}, defer_update=True)')
                result = self.send_trial_cmd(p_wait=p_w, defer_update=True)
                if result is None:
                    self._warn('trial', 'Brain returned None — skipping trial, returning to CENTER')
                    self.navigate_to(NAV_CENTER, 'CENTER')
                    continue

                action = result['action']
                is_exp = result['is_experimental']
                reward = result['reward']
                state = result['state']
                self._log('trial', f'Brain decided: state={STATE_NAMES[state]} '
                          f'action={ACTION_NAMES[action]} reward={reward:.1f} '
                          f'experimental={is_exp}')

                # step 4: vision seek
                target_color = 'yellow' if action == LEFT else 'red'
                self._log('trial', f'Step 4: Vision seek {target_color} '
                          f'(action={ACTION_NAMES[action]})')
                reached = self.vision_seek(target_color)
                self._log('trial', f'Vision seek {target_color}: '
                          f'{"REACHED" if reached else "NOT REACHED"}')

                # step 5: confirm reward
                actual_reward = reward if reached else 0.0
                self._log('trial', f'Step 5: Confirming reward={actual_reward:.1f} '
                          f'(original={reward:.1f}, reached={reached})')
                confirm = String()
                confirm.data = json.dumps({
                    'state': state,
                    'action': action,
                    'reward': actual_reward,
                })
                self.confirm_pub.publish(confirm)
                result['reward'] = actual_reward
                result['reached'] = reached

                # record
                if is_exp:
                    exp_total += 1
                    exp_correct += int(actual_reward > DEAD_RWD)
                else:
                    ctrl_total += 1
                    ctrl_correct += int(actual_reward > UNOBTAINABLE_RWD)
                self.testing_results[delay].append(result)

                self._log('trial', f'Trial result: reward={actual_reward:.1f} '
                          f'exp_correct={exp_correct}/{exp_total} '
                          f'ctrl_correct={ctrl_correct}/{ctrl_total}')

                # step 6: return to center
                self._log('trial', 'Step 6: Returning to CENTER')
                self.navigate_to(NAV_CENTER, 'CENTER')

            # delay summary
            exp_pct = (100 * exp_correct / exp_total) if exp_total > 0 else 0
            ctrl_pct = (100 * ctrl_correct / ctrl_total) if ctrl_total > 0 else 0
            self._log('testing', f'Delay {delay}s summary: '
                      f'experimental={exp_pct:.1f}% ({exp_correct}/{exp_total}), '
                      f'control={ctrl_pct:.1f}% ({ctrl_correct}/{ctrl_total})')

        self._log('testing', 'Testing phase complete.')

    def save_results(self):
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

        self._log('save', f'Results saved to {RESULTS_FILE}')


def main():
    rclpy.init()
    node = StateManager()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
