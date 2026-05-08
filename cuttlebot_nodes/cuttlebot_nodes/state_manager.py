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
import math
from std_msgs.msg import String
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
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
NAV_CENTER = [0.0, -1.0]
NAV_LEFT_CHAMBER = [-0.8, 1.8]
NAV_RIGHT_CHAMBER = [0.8, 1.8]
NAV_GAP_APPROACH = [0.0, 0.3]

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
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        amcl_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1)
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._amcl_callback, amcl_qos)
        self._current_yaw = None

        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback, odom_qos)
        self._odom_yaw = None

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

    def _amcl_callback(self, msg: PoseWithCovarianceStamped):
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._current_yaw = math.atan2(siny, cosy)

    def _odom_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._odom_yaw = math.atan2(siny, cosy)

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

    def rotate_to_yaw(self, target_yaw_deg, speed=0.5, tolerance_deg=10.0):
        target = math.radians(target_yaw_deg)
        tol = math.radians(tolerance_deg)
        timeout = 30.0
        dt = 0.05

        if self._odom_yaw is None:
            self._warn('rotate', 'No odom data yet — waiting up to 5s...')
            for _ in range(100):
                time.sleep(0.05)
                if self._odom_yaw is not None:
                    break
            if self._odom_yaw is None:
                self._err('rotate', 'No odom data — cannot rotate')
                return

        amcl_yaw = self._current_yaw if self._current_yaw is not None else self._odom_yaw
        needed_delta = math.atan2(math.sin(target - amcl_yaw),
                                  math.cos(target - amcl_yaw))
        target_odom = self._odom_yaw + needed_delta

        self._log('rotate', f'Rotating to yaw={target_yaw_deg:.0f}deg '
                  f'(delta={math.degrees(needed_delta):.1f}deg, '
                  f'amcl={math.degrees(amcl_yaw):.1f}deg)...')

        t0 = time.time()
        while time.time() - t0 < timeout:
            error = math.atan2(math.sin(target_odom - self._odom_yaw),
                               math.cos(target_odom - self._odom_yaw))
            if abs(error) < tol:
                break
            angular_z = speed if error > 0 else -speed
            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.header.frame_id = 'base_link'
            cmd.twist.angular.z = angular_z
            self.cmd_vel_pub.publish(cmd)
            time.sleep(dt)

        stop = TwistStamped()
        stop.header.stamp = self.get_clock().now().to_msg()
        stop.header.frame_id = 'base_link'
        self.cmd_vel_pub.publish(stop)
        time.sleep(0.5)
        elapsed = time.time() - t0
        odom_deg = math.degrees(self._odom_yaw)
        amcl_deg = math.degrees(self._current_yaw) if self._current_yaw else '?'
        self._log('rotate', f'Rotation done in {elapsed:.1f}s. '
                  f'odom_yaw={odom_deg:.1f}deg amcl_yaw={amcl_deg}')

    def navigate_to(self, coords, label, direction=TurtleBot4Directions.WEST):
        self._log('nav', f'>>> navigate_to {label} ({coords[0]:.2f}, {coords[1]:.2f})')
        self._log('nav', f'    Current zone: {self.current_zone}')

        t0 = time.time()
        goal = self.navigator.getPoseStamped(coords, direction)
        self._log('nav', f'    Goal pose created, calling startToPose...')
        self.navigator.startToPose(goal)
        result = self.navigator.getResult()
        elapsed = time.time() - t0
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

        self._log('experiment', 'Post-undock: rotating to face +y (WEST=90deg)...')
        self.rotate_to_yaw(SPAWN_YAW)
        self._log('experiment', 'Post-undock rotation complete.')

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

        self._log('testing', 'Robot is in CENTER zone after undock — starting trials.')

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
                if not nav_ok:
                    self._warn('trial', 'Nav to GAP_APPROACH failed — skipping trial')
                    self.navigate_to(NAV_CENTER, 'CENTER',
                                         direction=TurtleBot4Directions.EAST)
                    continue

                self._log('trial', 'Waiting 4s for Nav2 residual commands to drain...')
                time.sleep(4.0)

                # step 2: vision scan
                self._log('trial', 'Step 2: Vision scan for both colors')
                scan_ok = self.vision_scan()
                if not scan_ok:
                    self._warn('trial', 'Vision scan failed — skipping trial, returning to CENTER')
                    self.navigate_to(NAV_CENTER, 'CENTER',
                                         direction=TurtleBot4Directions.EAST)
                    continue

                # step 3: brain decides
                self._log('trial', f'Step 3: Brain decision (p_wait={p_w:.4f}, defer_update=True)')
                result = self.send_trial_cmd(p_wait=p_w, defer_update=True)
                if result is None:
                    self._warn('trial', 'Brain returned None — skipping trial, returning to CENTER')
                    self.navigate_to(NAV_CENTER, 'CENTER',
                                         direction=TurtleBot4Directions.EAST)
                    continue

                action = result['action']
                is_exp = result['is_experimental']
                reward = result['reward']
                state = result['state']
                self._log('trial', f'Brain decided: state={STATE_NAMES[state]} '
                          f'action={ACTION_NAMES[action]} reward={reward:.1f} '
                          f'experimental={is_exp}')

                # step 4: navigate to chosen chamber
                if action == LEFT:
                    chamber_coords = NAV_LEFT_CHAMBER
                    chamber_label = 'LEFT_CHAMBER'
                else:
                    chamber_coords = NAV_RIGHT_CHAMBER
                    chamber_label = 'RIGHT_CHAMBER'
                self._log('trial', f'Step 4: Navigate to {chamber_label} '
                          f'(action={ACTION_NAMES[action]})')
                reached = self.navigate_to(chamber_coords, chamber_label)
                self._log('trial', f'Nav to {chamber_label}: '
                          f'{"REACHED" if reached else "FAILED"}')
                if reached:
                    time.sleep(2.0)

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
                self.navigate_to(NAV_CENTER, 'CENTER',
                                         direction=TurtleBot4Directions.EAST)

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
