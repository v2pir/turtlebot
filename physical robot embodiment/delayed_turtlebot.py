import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import TwistStamped, PoseWithCovarianceStamped
from irobot_create_msgs.msg import HazardDetectionVector, HazardDetection

from turtlebot4_navigation.turtlebot4_navigator import (
    TurtleBot4Navigator,
    TurtleBot4Directions
)
from nav2_simple_commander.robot_navigator import TaskResult

from cv_bridge import CvBridge

import cv2
import numpy as np
import random
from math import exp
from scipy.stats import norm
import time
import copy


"""
Delayed Gratification Robot
Physical TurtleBot4 version with color-based chamber seeking and a
purple-paper door marker on the live (yellow) chamber.

Experiment assumptions:
- Yellow = live shrimp (better prey)
- Red    = dead shrimp (worse prey)
- Purple = paper marker on the DOOR. The door covers/blocks the yellow
  (live) chamber. Purple is always paired with yellow.
  When the human removes the door, purple disappears from view.
- State 0, EXPM_LR: live/yellow left, dead/red right
- State 1, EXPM_RL: dead/red left, live/yellow right
"""


# ------------------ CONSTANTS ------------------

ACTIONS = 2
LEFT = 0
RIGHT = 1

STATES = 2
EXPM_LR = 0  # Live/yellow left, dead/red right
EXPM_RL = 1  # Dead/red left, live/yellow right

LIVE_RWD = 5.0
DEAD_RWD = 1.0

ALPHA = 0.10
BETA = 1.0
GAMMA = 0.99


# ------------------ STATE/ACTION LOOKUP TABLE ------------------
# Single source of truth: maps (state, action) -> (target_color, reward).
# Replaces several if/elif chains scattered across the file.

STATE_ACTION_TABLE = {
    EXPM_LR: {
        LEFT:  ("yellow", LIVE_RWD),
        RIGHT: ("red",    DEAD_RWD),
    },
    EXPM_RL: {
        LEFT:  ("red",    DEAD_RWD),
        RIGHT: ("yellow", LIVE_RWD),
    },
}


def get_target_color_from_state_action(state, action):
    return STATE_ACTION_TABLE[state][action][0]


def get_reward_from_state_action(state, action):
    return STATE_ACTION_TABLE[state][action][1]


def get_dead_action(state):
    """Return the action that selects the red/dead chamber for `state`."""
    for action, (color, _) in STATE_ACTION_TABLE[state].items():
        if color == "red":
            return action
    return RIGHT  # safety fallback


# ------------------ MAP COORDINATES ------------------

HOME_X = -0.023635001853108406
HOME_Y = -0.023635001853108406


# ------------------ VISION CONTROL PARAMETERS ------------------

MIN_COLOR_AREA = 150

TARGET_CLOSE_ENOUGH_AREA = 800

CENTER_TOLERANCE = 40

APPROACH_FORWARD_SPEED = 0.05
PRE_DECISION_SCAN_SPEED = 0.12
PRE_DECISION_SCAN_SWITCH_SEC = 2.0
PRE_DECISION_SAFETY_TIMEOUT_SEC = 45.0

POST_DETECTION_FORWARD_SEC = 4.0
POST_DETECTION_FORWARD_SPEED = 0.10

SEARCH_TURN_SPEED = 0.25
TURN_GAIN_TRACK = 0.002

FORWARD_SPEED = 0.08
SLOW_FORWARD_SPEED = 0.04

COLOR_TIMEOUT_SEC = 45.0

# All repeated color-related logs are throttled to this interval.
COLOR_LOG_INTERVAL_SEC = 5.0


# ------------------ FINAL BUMP PHASE ------------------

BUMP_FORWARD_SPEED = 0.1
BUMP_PHASE_TIMEOUT_SEC = 20.0

FORCE_BUMP_AFTER_TRACKING_SEC = 18.0
FORCE_BUMP_MIN_AREA = 800
FORCE_BUMP_CENTER_TOLERANCE = 90


# ------------------ PURPLE HSV THRESHOLDS ------------------

PURPLE_HSV_LOWER = (135, 90, 70)
PURPLE_HSV_UPPER = (155, 255, 255)


# ------------------ REST / DELAY / PATIENCE PARAMETERS ------------------

REST_AT_HOME_SEC = 20.0

PATIENCE_CHECK_INTERVAL_SEC = 1.0

DOOR_OPEN_STABLE_SEC = 1.5
DOOR_OPEN_TIMEOUT_SEC = 60.0


# ------------------ SOUTH TURN PARAMETERS ------------------

TURN_SOUTH_SPEED = 0.45
TURN_SOUTH_TIME = 7.0


# ------------------ NORTH ORIENTATION CORRECTION ------------------

NORTH_ALIGN_TOLERANCE_RAD = 0.035  # ~2 degrees
NORTH_ALIGN_TIMEOUT_SEC = 8.0
NORTH_ALIGN_MAX_ANGULAR = 0.40
NORTH_ALIGN_MIN_ANGULAR = 0.12
NORTH_ALIGN_KP = 1.5


# ------------------ Q TABLE ------------------

qTbl = [
    [3.588, 0.469],  # EXPM_LR: LEFT is live/yellow, RIGHT is dead/red
    [0.410, 4.074],  # EXPM_RL: LEFT is dead/red, RIGHT is live/yellow
]


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

    # Floating-point safety net: cumulative should reach 1.0 but may
    # fall ~1e-16 short, in which case we default to the last action.
    return RIGHT


def state_to_string(state):
    if state == EXPM_LR:
        return "STATE 0 (EXPM_LR): yellow/live LEFT, red/dead RIGHT"
    elif state == EXPM_RL:
        return "STATE 1 (EXPM_RL): red/dead LEFT, yellow/live RIGHT"
    return "UNKNOWN_STATE"


def action_to_string(action):
    if action == LEFT:
        return "LEFT"
    elif action == RIGHT:
        return "RIGHT"
    return "UNKNOWN_ACTION"


# ------------------ ROBOT CLASS ------------------

class DelayedGratificationRobot(Node):

    def __init__(self):
        super().__init__('delayed_gratification_robot')

        # Navigation
        self.navigator = TurtleBot4Navigator()

        # Camera / color detection
        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            '/oakd/rgb/preview/image_raw',
            self.image_callback,
            10
        )

        # External debugging topic; not read by decision logic.
        self.color_pub = self.create_publisher(
            String,
            '/detected_color',
            10
        )

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            '/cmd_vel',
            10
        )

        # Map-frame pose, used for fine NORTH orientation correction.
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10
        )
        self.latest_yaw = None

        # Bumper, used as the real stop condition during the final
        # approach to the chosen chamber.
        self.hazard_sub = self.create_subscription(
            HazardDetectionVector,
            '/hazard_detection',
            self.hazard_callback,
            10
        )
        self.bump_detected = False

        # Latest vision data
        self.latest_red_center = None
        self.latest_red_area = 0

        self.latest_yellow_center = None
        self.latest_yellow_area = 0

        # Purple is the marker on the DOOR. While the door is in place,
        # purple is visible and yellow (live shrimp) is hidden behind it.
        self.latest_purple_center = None
        self.latest_purple_area = 0

        self.latest_image_width = None

        # Generic throttled-log timers. Keys are arbitrary strings.
        self._log_timers = {}

        # Nav2 initialization
        start_pose = self.navigator.getPoseStamped(
            [HOME_X, HOME_Y],
            TurtleBot4Directions.NORTH
        )
        self.navigator.setInitialPose(start_pose)

        self.get_logger().info("Waiting for Nav2...")
        self.navigator.waitUntilNav2Active()

        self._wait_for_map_topic(timeout_sec=20.0)

        # Run experiment
        self.run_experiment()
        self.get_logger().info("Experiment finished.")


    # ------------------ HELPERS ------------------

    def _throttled_log(self, key, message, level="info",
                       interval=COLOR_LOG_INTERVAL_SEC):
        """Log `message` only if `interval` seconds have passed since
        the last log under the same `key`."""
        now = time.time()
        last = self._log_timers.get(key, 0.0)

        if now - last < interval:
            return

        self._log_timers[key] = now

        if level == "warn":
            self.get_logger().warn(message)
        else:
            self.get_logger().info(message)


    def _spin_countdown(self, duration_sec, label,
                        should_stop=None, log_extra=None):
        """Spin ROS callbacks for up to `duration_sec` seconds, logging
        a per-second countdown under `[label]`.

        - should_stop: optional callable(elapsed_sec) -> bool; if it
          returns True the countdown ends early. Returns True if it
          stopped early, False if the full duration elapsed.
        - log_extra: optional callable(elapsed_sec) -> str appended to
          the countdown log line.
        """
        start = time.time()
        last_log_sec = -1

        while rclpy.ok():
            elapsed = time.time() - start

            if elapsed >= duration_sec:
                return False

            rclpy.spin_once(self, timeout_sec=0.1)

            if should_stop is not None and should_stop(elapsed):
                return True

            elapsed_int = int(elapsed)
            if elapsed_int != last_log_sec:
                remaining = duration_sec - elapsed
                if remaining > 0:
                    extra = ""
                    if log_extra is not None:
                        extra = " " + log_extra(elapsed)
                    self.get_logger().info(
                        f"[{label}] {remaining:.0f}s remaining...{extra}"
                    )
                last_log_sec = elapsed_int

        return False


    def _spin_sleep(self, duration_sec):
        """Like time.sleep but keeps ROS callbacks alive."""
        start = time.time()
        while rclpy.ok() and time.time() - start < duration_sec:
            rclpy.spin_once(self, timeout_sec=0.05)


    # ------------------ POSE / HAZARD CALLBACKS ------------------

    def pose_callback(self, msg):
        # Convert quaternion to yaw without pulling in tf_transformations.
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.latest_yaw = float(np.arctan2(siny_cosp, cosy_cosp))


    def hazard_callback(self, msg):
        for det in msg.detections:
            self._throttled_log(
                "hazard_type",
                f"[Hazard] detected type={det.type}"
            )

            if det.type == HazardDetection.BUMP:
                self.bump_detected = True
                return


    # ------------------ COLOR DETECTION ------------------

    def get_largest_contour_center(self, mask, min_area=MIN_COLOR_AREA):
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return None, 0

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < min_area:
            return None, area

        M = cv2.moments(largest)

        if M["m00"] == 0:
            return None, area

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        return (cx, cy), area


    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red wraps around HSV hue boundary, so use two masks.
        red1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
        red_mask = cv2.bitwise_or(red1, red2)

        # Yellow mask
        yellow_mask = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))

        # Purple mask (door marker, paired with the yellow chamber).
        # Subtract red so the two are mutually exclusive.
        purple_mask_raw = cv2.inRange(hsv, PURPLE_HSV_LOWER, PURPLE_HSV_UPPER)
        purple_mask = cv2.bitwise_and(purple_mask_raw, cv2.bitwise_not(red_mask))

        red_center, red_area = self.get_largest_contour_center(red_mask)
        yellow_center, yellow_area = self.get_largest_contour_center(yellow_mask)
        purple_center, purple_area = self.get_largest_contour_center(purple_mask)

        self.latest_red_center = red_center
        self.latest_red_area = red_area

        self.latest_yellow_center = yellow_center
        self.latest_yellow_area = yellow_area

        self.latest_purple_center = purple_center
        self.latest_purple_area = purple_area

        self.latest_image_width = frame.shape[1]

        # Publish dominant color for external debugging tools only.
        areas = {
            "red": red_area if red_center is not None else 0,
            "yellow": yellow_area if yellow_center is not None else 0,
            "purple": purple_area if purple_center is not None else 0,
        }
        best = max(areas, key=areas.get)
        detected = best if areas[best] > 0 else "unknown"

        msg_out = String()
        msg_out.data = detected
        self.color_pub.publish(msg_out)


    # ------------------ LOW-LEVEL MOTION ------------------

    def publish_cmd(self, linear_x=0.0, angular_z=0.0):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'

        cmd.twist.linear.x = linear_x
        cmd.twist.linear.y = 0.0
        cmd.twist.linear.z = 0.0

        cmd.twist.angular.x = 0.0
        cmd.twist.angular.y = 0.0
        cmd.twist.angular.z = angular_z

        self.cmd_pub.publish(cmd)


    def stop_robot(self):
        self.publish_cmd(0.0, 0.0)


    # ------------------ REST AT HOME ------------------

    def rest_at_home(self, trial_label, delay_sec, state_label):
        """Stops the robot at home and rests for REST_AT_HOME_SEC seconds.
        Prints trial info AND the current state so the human operator
        knows which trial is up and how to arrange the chambers."""
        self.get_logger().info("======================================")
        self.get_logger().info("[REST AT HOME]")
        self.get_logger().info(f"  >>> {trial_label} <<<")
        self.get_logger().info(f"  >>> STATE: {state_label} <<<")
        self.get_logger().info(f"  >>> Delay this trial: {delay_sec} seconds <<<")
        self.get_logger().info(
            f"  Resting {REST_AT_HOME_SEC:.0f}s. Set up chambers now."
        )
        self.get_logger().info("======================================")

        self.stop_robot()
        self._spin_countdown(REST_AT_HOME_SEC, "REST AT HOME")
        self.get_logger().info("[REST AT HOME] Rest complete. Robot resuming.")


    # ------------------ PATIENCE MODEL ------------------
    def wait_for_door_with_patience(self, delay_sec):
        """Continuous delayed-gratification wait.

        Robot does NOT know `delay_sec`. It just waits, checking two
        things every PATIENCE_CHECK_INTERVAL_SEC:

          1. Has the door been removed? (purple gone for
             DOOR_OPEN_STABLE_SEC continuously) -> return True
          2. Has patience run out? prob_wait(elapsed) drops below a
             random threshold sampled at the start -> return False

        `delay_sec` is used only as a soft hint for the operator log
        and as a hard safety cap (3x the nominal delay) so the trial
        cannot hang forever.
        """
        self.get_logger().info("======================================")
        self.get_logger().info(
            "[Patience] Robot chose delayed yellow/live reward."
        )
        self.get_logger().info(
            "[Patience] Waiting for door removal. Robot does not know the delay."
        )
        self.get_logger().info(
            f"[Operator] Nominal delay this trial: {delay_sec}s. "
            f"Remove the door after that."
        )
        self.get_logger().info("======================================")

        self.stop_robot()

        patience_threshold = random.random()
        self.get_logger().info(
            f"[Patience] Sampled patience threshold={patience_threshold:.3f}"
        )

        # Safety cap so a forgotten door doesn't hang the experiment.
        safety_timeout = max(delay_sec * 3.0, 90.0)

        start = time.time()
        last_check_time = start
        purple_gone_since = None
        last_log_sec = -1

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            now = time.time()
            elapsed = now - start

            # --- Check 1: door removal (purple stably absent) ---
            if not self.purple_visible():
                if purple_gone_since is None:
                    purple_gone_since = now
                    self.get_logger().info(
                        "[Door] Purple no longer detected. Confirming..."
                    )
                elif now - purple_gone_since >= DOOR_OPEN_STABLE_SEC:
                    self.get_logger().info(
                        f"[Door] Door confirmed open after {elapsed:.1f}s "
                        f"of waiting. >>> PROCEEDING TO YELLOW <<<"
                    )
                    return True
            else:
                if purple_gone_since is not None:
                    self.get_logger().info(
                        "[Door] Purple detected again. Resetting door-open timer."
                    )
                purple_gone_since = None

            # --- Check 2: patience (only every PATIENCE_CHECK_INTERVAL_SEC) ---
            if now - last_check_time >= PATIENCE_CHECK_INTERVAL_SEC:
                last_check_time = now
                current_patience = prob_wait(elapsed)
                if patience_threshold > current_patience:
                    self.get_logger().warn(
                        f"[Patience] Robot gave up after {elapsed:.1f}s. "
                        f"patience_probability={current_patience:.3f}, "
                        f"threshold={patience_threshold:.3f}"
                    )
                    self.stop_robot()
                    return False

            # --- Per-second countdown-style log ---
            elapsed_int = int(elapsed)
            if elapsed_int != last_log_sec:
                last_log_sec = elapsed_int
                self.get_logger().info(
                    f"[Patience] elapsed={elapsed:.0f}s, "
                    f"patience_probability={prob_wait(elapsed):.3f}, "
                    f"purple_visible={self.purple_visible()}"
                )

            # --- Safety cap ---
            if elapsed > safety_timeout:
                self.get_logger().warn(
                    f"[Patience] Safety timeout ({safety_timeout:.0f}s) reached "
                    "without door removal. Treating as give-up."
                )
                self.stop_robot()
                return False

        return False


    # ------------------ PRE-DECISION VISION ------------------

    def red_visible(self):
        return (
            self.latest_red_center is not None
            and self.latest_red_area >= MIN_COLOR_AREA
        )


    def yellow_visible(self):
        return (
            self.latest_yellow_center is not None
            and self.latest_yellow_area >= MIN_COLOR_AREA
        )


    def purple_visible(self):
        return (
            self.latest_purple_center is not None
            and self.latest_purple_area >= MIN_COLOR_AREA
        )


    def move_forward_until_options_visible(self):
        """Moves forward while sweeping angular velocity left/right until
        the robot has observed BOTH red (dead chamber) AND purple (the
        door marker on the live/yellow chamber)."""
        self.get_logger().info(
            "[Pre-decision] Moving forward/scanning until both red and "
            "purple have been observed..."
        )

        saw_red = False
        saw_purple = False

        scan_direction = 1
        last_switch_time = time.time()
        start_time = time.time()

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

            if self.red_visible():
                saw_red = True

            if self.purple_visible():
                saw_purple = True

            now = time.time()

            self._throttled_log(
                "pre_decision_vision",
                f"[Pre-decision Vision] "
                f"red_area={self.latest_red_area:.1f}, "
                f"purple_area={self.latest_purple_area:.1f}, "
                f"yellow_area={self.latest_yellow_area:.1f} "
                f"(should be ~0 while door is closed), "
                f"saw_red={saw_red}, saw_purple={saw_purple}"
            )

            if saw_red and saw_purple:
                self.stop_robot()
                self.get_logger().info(
                    "[Pre-decision] Both red and purple have been observed. "
                    "Robot can now wait for the delay."
                )
                self._spin_sleep(0.5)
                return True

            if now - last_switch_time > PRE_DECISION_SCAN_SWITCH_SEC:
                scan_direction *= -1
                last_switch_time = now

            self.publish_cmd(
                linear_x=APPROACH_FORWARD_SPEED,
                angular_z=PRE_DECISION_SCAN_SPEED * scan_direction
            )

            if PRE_DECISION_SAFETY_TIMEOUT_SEC is not None:
                if now - start_time > PRE_DECISION_SAFETY_TIMEOUT_SEC:
                    self.stop_robot()
                    self.get_logger().warn(
                        "[Pre-decision] Safety timeout: could not observe "
                        "both red and purple."
                    )
                    return False

        self.stop_robot()
        return False


    def move_forward_to_decision_point(self):
        """After the robot has seen both red and purple, drive straight
        forward for POST_DETECTION_FORWARD_SEC seconds to bring the
        robot closer to the chambers."""
        self.get_logger().info(
            f"[Pre-decision] Both colors observed. Pushing forward "
            f"{POST_DETECTION_FORWARD_SEC:.1f}s to reach decision point..."
        )

        start = time.time()

        while rclpy.ok() and time.time() - start < POST_DETECTION_FORWARD_SEC:
            rclpy.spin_once(self, timeout_sec=0.1)
            self.publish_cmd(
                linear_x=POST_DETECTION_FORWARD_SPEED,
                angular_z=0.0
            )

        self.stop_robot()
        self._spin_sleep(0.5)
        self.get_logger().info("[Pre-decision] Reached decision point.")


    # ------------------ POST-DECISION COLOR TRACKING ------------------

    def get_target_detection(self, target_color):
        if target_color == "red":
            return self.latest_red_center, self.latest_red_area
        elif target_color == "yellow":
            return self.latest_yellow_center, self.latest_yellow_area
        return None, 0


    def move_toward_color_once(self, target_color):
        center, area = self.get_target_detection(target_color)

        if self.latest_image_width is None:
            self._throttled_log(
                "camera_warning",
                "[Vision] No camera image received yet.",
                level="warn"
            )
            self.stop_robot()
            return False

        if center is None:
            self._throttled_log(
                "search_color",
                f"[Vision] Searching for {target_color}..."
            )
            self.publish_cmd(0.0, SEARCH_TURN_SPEED)
            return False

        target_x = center[0]
        image_center_x = self.latest_image_width // 2
        error = target_x - image_center_x

        self._throttled_log(
            "tracking_color",
            f"[Vision] Tracking {target_color} | center={center} "
            f"area={area:.1f} error={error}"
        )

        if area >= TARGET_CLOSE_ENOUGH_AREA:
            self.get_logger().info(
                f"[Vision] Close to {target_color}. Switching to bump phase. "
                f"area={area:.1f}"
            )
            self.stop_robot()
            return True

        if abs(error) > CENTER_TOLERANCE:
            linear_x = 0.0
            angular_z = -TURN_GAIN_TRACK * error
        else:
            linear_x = FORWARD_SPEED
            angular_z = -TURN_GAIN_TRACK * error
            if area > TARGET_CLOSE_ENOUGH_AREA * 0.6:
                linear_x = SLOW_FORWARD_SPEED

        self.publish_cmd(linear_x, angular_z)
        return False


    # ------------------ NORTH ALIGNMENT ------------------

    def correct_orientation_to_north(self):
        """After Nav2 returns from go_home, the robot can still be a few
        degrees off NORTH. Rotate in place using /amcl_pose feedback
        until the heading is within NORTH_ALIGN_TOLERANCE_RAD of 0."""
        self.get_logger().info("[Orient] Fine-tuning heading to exact NORTH...")

        # Wait briefly for at least one /amcl_pose message.
        wait_start = time.time()
        while self.latest_yaw is None and time.time() - wait_start < 2.0:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.latest_yaw is None:
            self.get_logger().warn(
                "[Orient] No /amcl_pose received. Skipping correction."
            )
            return False

        start = time.time()

        while rclpy.ok() and time.time() - start < NORTH_ALIGN_TIMEOUT_SEC:
            rclpy.spin_once(self, timeout_sec=0.05)

            if self.latest_yaw is None:
                continue

            # Shortest signed angle from current yaw to target (0).
            error = -self.latest_yaw
            while error > np.pi:
                error -= 2.0 * np.pi
            while error < -np.pi:
                error += 2.0 * np.pi

            if abs(error) < NORTH_ALIGN_TOLERANCE_RAD:
                self.stop_robot()
                self.get_logger().info(
                    f"[Orient] Aligned to NORTH. yaw={self.latest_yaw:.3f} rad "
                    f"({np.degrees(self.latest_yaw):.1f} deg)"
                )
                self._spin_sleep(0.3)
                return True

            # Proportional rotation, clamped, with a floor to overcome
            # the motor's static friction.
            angular_z = NORTH_ALIGN_KP * error

            if angular_z > NORTH_ALIGN_MAX_ANGULAR:
                angular_z = NORTH_ALIGN_MAX_ANGULAR
            elif angular_z < -NORTH_ALIGN_MAX_ANGULAR:
                angular_z = -NORTH_ALIGN_MAX_ANGULAR
            elif 0.0 < angular_z < NORTH_ALIGN_MIN_ANGULAR:
                angular_z = NORTH_ALIGN_MIN_ANGULAR
            elif -NORTH_ALIGN_MIN_ANGULAR < angular_z < 0.0:
                angular_z = -NORTH_ALIGN_MIN_ANGULAR

            self.publish_cmd(0.0, angular_z)

        self.stop_robot()
        self.get_logger().warn(
            f"[Orient] Timeout. Final yaw={self.latest_yaw:.3f} rad "
            f"({np.degrees(self.latest_yaw):.1f} deg)"
        )
        return False


    # ------------------ FINAL BUMP PHASE ------------------

    def bump_into_chamber(self, target_color):
        """Keep visually tracking toward target_color while driving slowly
        forward, until the bumper fires. The robot doesn't 'give up' on
        seeing the target — it just keeps adjusting heading until contact."""
        self.get_logger().info(
            f"[Bump] Slow vision-guided approach into {target_color} "
            "until bumper contact..."
        )

        # Clear stale bump events.
        self.bump_detected = False

        start = time.time()

        while rclpy.ok() and time.time() - start < BUMP_PHASE_TIMEOUT_SEC:
            rclpy.spin_once(self, timeout_sec=0.05)

            if self.bump_detected:
                self.stop_robot()
                self.get_logger().info(
                    f"[Bump] Bumper triggered after {time.time() - start:.2f}s. "
                    "Contact made."
                )
                self._spin_sleep(0.3)
                return True

            # Vision-guided forward motion.
            angular_z = 0.0
            center, area = self.get_target_detection(target_color)

            if center is not None and self.latest_image_width is not None:
                image_center_x = self.latest_image_width // 2
                error = center[0] - image_center_x
                # Light steering correction so heading stays on target.
                angular_z = -TURN_GAIN_TRACK * error * 0.5

            self.publish_cmd(linear_x=BUMP_FORWARD_SPEED, angular_z=angular_z)

        self.stop_robot()
        self.get_logger().warn(
            f"[Bump] Timed out after {BUMP_PHASE_TIMEOUT_SEC:.1f}s without "
            "bumper contact. Chamber may be too far, or bumper not firing."
        )
        self._spin_sleep(0.3)
        return False

    # ------------------ TURN SOUTH ------------------

    def turn_south_before_home(self):
        self.get_logger().info("[Reset] Turning toward SOUTH / home direction...")

        start = time.time()

        while rclpy.ok() and time.time() - start < TURN_SOUTH_TIME:
            rclpy.spin_once(self, timeout_sec=0.1)
            self.publish_cmd(0.0, TURN_SOUTH_SPEED)

        self.stop_robot()
        self._spin_sleep(0.5)

        self.get_logger().info("[Reset] Finished rough SOUTH orientation.")
        return True


    def _try_force_bump(self, target_color, tracking_start, reason):
        """Helper: if the target is visible, centered, and big enough,
        return True so the caller can switch to the bump phase."""
        center, area = self.get_target_detection(target_color)

        if center is None or self.latest_image_width is None:
            return False

        image_center_x = self.latest_image_width // 2
        error = center[0] - image_center_x

        if (
            abs(error) <= FORCE_BUMP_CENTER_TOLERANCE
            and area >= FORCE_BUMP_MIN_AREA
        ):
            self.get_logger().warn(
                f"[VisionNav] {reason} for {target_color}: "
                f"tracked_for={time.time() - tracking_start:.1f}s, "
                f"center={center}, area={area:.1f}, error={error}"
            )
            return True

        return False

    
    def commit_turn_toward_target(self, target_color, max_turn_sec=3.0):
        """At the start of the approach, rotate in place until the chosen
        target color is dead-centered. Makes the robot's commitment to
        yellow or red visually obvious from the very beginning."""
        self.get_logger().info(
            f"[Commit] Turning to face {target_color} before approaching..."
        )

        commit_tolerance_px = 15
        start = time.time()

        while rclpy.ok() and time.time() - start < max_turn_sec:
            rclpy.spin_once(self, timeout_sec=0.05)

            center, area = self.get_target_detection(target_color)

            if center is None or self.latest_image_width is None:
                # Briefly lost sight — sweep slowly in case target is just
                # outside FOV. Direction doesn't matter much; pick one.
                self._throttled_log(
                    "commit_search",
                    f"[Commit] Searching for {target_color}..."
                )
                self.publish_cmd(0.0, SEARCH_TURN_SPEED * 0.6)
                continue

            image_center_x = self.latest_image_width // 2
            error = center[0] - image_center_x

            if abs(error) <= commit_tolerance_px:
                self.stop_robot()
                self.get_logger().info(
                    f"[Commit] Facing {target_color} (error={error}px). "
                    "Heading committed."
                )
                # Visible pause so the observer can see the chosen heading.
                self._spin_sleep(1.0)
                return True

            # Rotate in place, with a speed floor so it actually moves.
            angular_z = -TURN_GAIN_TRACK * error * 2.0
            min_speed = 0.12

            if 0 < angular_z < min_speed:
                angular_z = min_speed
            elif -min_speed < angular_z < 0:
                angular_z = -min_speed

            self.publish_cmd(0.0, angular_z)

        self.stop_robot()
        self.get_logger().info(
            f"[Commit] Commit-turn timeout. Proceeding toward {target_color} anyway."
        )
        return False

    def go_to_color(self, target_color, timeout_sec=COLOR_TIMEOUT_SEC):
        self.get_logger().info(f"[VisionNav] Moving toward {target_color}")

        # >>> NEW: visibly commit to the chosen heading BEFORE approaching <
        self.commit_turn_toward_target(target_color)

        start = time.time()

        # Reset color log timers for this tracking phase.
        for key in ("tracking_color", "search_color", "camera_warning"):
            self._log_timers.pop(key, None)

        # Phase 1: vision-track until close enough to the chamber.
        close_enough = False

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

            close_enough = self.move_toward_color_once(target_color)

            if close_enough:
                break

            # Fallback: tracked long enough and target looks reasonable.
            if time.time() - start >= FORCE_BUMP_AFTER_TRACKING_SEC:
                if self._try_force_bump(
                    target_color, start, "Forcing bump phase"
                ):
                    self.stop_robot()
                    close_enough = True
                    break

            if time.time() - start > timeout_sec:
                center, area = self.get_target_detection(target_color)
                self.get_logger().warn(
                    f"[VisionNav] Timeout while approaching {target_color}. "
                    f"Last seen center={center}, area={area:.1f}"
                )

                # Timeout fallback: try bump if target still looks good.
                if self._try_force_bump(
                    target_color, start, "Timeout fallback"
                ):
                    self.stop_robot()
                    close_enough = True
                    break

                self.stop_robot()
                oriented_south = self.turn_south_before_home()
                return False, oriented_south

        # Phase 2: physical bump.
        bumped = self.bump_into_chamber(target_color)

        if bumped:
            self.get_logger().info(
                f"[VisionNav] Robot bumped {target_color} chamber."
            )
        else:
            self.get_logger().info(
                f"[VisionNav] Bump timeout, but already close to "
                f"{target_color}. Counting as caught."
            )

        self.stop_robot()
        self._spin_sleep(0.5)

        oriented_south = self.turn_south_before_home()
        return True, oriented_south


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
            direction = TurtleBot4Directions.NORTH

        goal = self.navigator.getPoseStamped([x, y], direction)

        self.get_logger().info(f"[Nav] Going to ({x:.3f}, {y:.3f})")

        self.navigator.goToPose(goal)

        start = time.time()

        while not self.navigator.isTaskComplete():
            rclpy.spin_once(self, timeout_sec=0.1)

            if time.time() - start > timeout_sec:
                self.get_logger().warn("[Nav] Timeout. Canceling task.")
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

        success = self.navigate_to(
            HOME_X,
            HOME_Y,
            TurtleBot4Directions.NORTH
        )

        # Even on Nav2 success, the heading can be a few degrees off
        # because of goal tolerance. Tighten it so both chambers are
        # guaranteed in the camera FOV.
        self.correct_orientation_to_north()

        return success


    # ------------------ EXPERIMENT LOGIC ------------------

    def print_q_table(self):
        self.get_logger().info("Q Table:")
        for s in range(STATES):
            self.get_logger().info(
                f"State {s}: L={qTbl[s][LEFT]:.3f}, R={qTbl[s][RIGHT]:.3f}"
            )


    def run_trial(self, delay_sec, trial_label):
        global qTbl

        state = random.choice([EXPM_LR, EXPM_RL])

        self.get_logger().info("======================================")
        self.get_logger().info(f"[Trial] {trial_label}")
        self.get_logger().info(f"[Trial] Randomized state: {state_to_string(state)}")

        if not self.go_home():
            self.get_logger().error("Failed to go home. Aborting trial.")
            return 0.0, True, state, None

        # === REST AT HOME ===
        self.rest_at_home(trial_label, delay_sec, state_to_string(state))

        # === MOVE FORWARD AND SCAN UNTIL RED + PURPLE OBSERVED ===
        saw_both = self.move_forward_until_options_visible()

        if not saw_both:
            self.get_logger().warn(
                "[Trial] Robot could not observe both red and purple. "
                "Aborting this trial."
            )
            return 0.0, True, state, None

        # === MOVE TO DECISION POINT ===
        self.move_forward_to_decision_point()

        # === DECISION ===
        # Robot does NOT know this trial's delay. Decide from raw Q-values.
        self.get_logger().info("[Thinking] Robot is at decision point. Deciding...")
        self._spin_sleep(1.0)

        act = action_select(qTbl[state], BETA)

        self.get_logger().info(
            f"[Decision] Robot chose {action_to_string(act)} chamber"
        )

        target_color = get_target_color_from_state_action(state, act)

        self.get_logger().info(
            f"[Decision] Chosen chamber contains target color: {target_color}"
        )

         # === CONTINUOUS PATIENCE: wait for door removal, may give up ===
        if target_color == "yellow":
            door_opened = self.wait_for_door_with_patience(delay_sec)

            if not door_opened:
                # Either ran out of patience, or hit the safety timeout.
                self.get_logger().warn(
                    "[Trial] Robot abandoned delayed yellow/live reward and "
                    "switched to immediate red/dead reward."
                )
                target_color = "red"
                act = get_dead_action(state)
                self.get_logger().warn(
                    f"[Decision] New action after giving up: "
                    f"{action_to_string(act)}, target_color={target_color}"
                )
            else:
                self.get_logger().info(
                    "[Patience] Door removed. Robot proceeding to yellow/live."
                )

        else:
            self.get_logger().info(
                "[Trial] Robot chose immediate red/dead reward. "
                "Going directly to red."
            )
            

        # === APPROACH CHOSEN / FINAL COLOR ===
        reached, oriented_south = self.go_to_color(target_color)

        if reached:
            self.get_logger().info(
                f"[Trial] Robot reached {target_color}. Prey obtained."
            )
        else:
            self.get_logger().warn(
                f"[Trial] Robot did not reach {target_color} target. No prey obtained."
            )

        if oriented_south:
            self.get_logger().info("[Reset] Robot roughly oriented toward SOUTH/home.")
        else:
            self.get_logger().warn("[Reset] Robot may not be oriented toward SOUTH/home.")

        self._spin_sleep(1.0)

        # Only give reward if the robot actually reached the chamber.
        reward = get_reward_from_state_action(state, act) if reached else 0.0

        qTbl[state][act] += ALPHA * (reward - qTbl[state][act])

        self.get_logger().info(
            f"[Reward] State={state}, Action={action_to_string(act)}, "
            f"Target={target_color}, Reward={reward:.2f}"
        )
        self.get_logger().info(
            f"[Q Update] New Q[{state}][{act}] = {qTbl[state][act]:.3f}"
        )

        return reward, True, state, act


    def run_experiment(self):
        experiment_start = time.time()

        self.get_logger().info("===== PRETRAINED Q TABLE LOADED =====")
        self.print_q_table()

        self.get_logger().info("===== DELAY TEST PHASE START =====")

        delays = np.array([10, 30, 60])
        DELAY_TRIALS = 5

        overall_trial = 0
        total_trials = len(delays) * DELAY_TRIALS

        for d_idx, d in enumerate(delays):
            d_int = int(d)

            self.get_logger().info(f"--- Testing Delay = {d_int} seconds ---")

            for t in range(DELAY_TRIALS):
                overall_trial += 1

                trial_label = (
                    f"Trial {overall_trial}/{total_trials} | "
                    f"Delay group {d_idx + 1}/{len(delays)} ({d_int}s) | "
                    f"Sub-trial {t + 1}/{DELAY_TRIALS}"
                )

                self.get_logger().info(f"[Delay {d_int}] {trial_label}")

                self.run_trial(d_int, trial_label)

        self.get_logger().info("===== DELAY TEST PHASE FINISHED =====")

        self.print_q_table()

        self.get_logger().info("Returning robot to home position...")
        self.go_home()
        self._spin_sleep(2.0)

        experiment_end = time.time()
        total_time = experiment_end - experiment_start

        self.get_logger().info("===== EXPERIMENT COMPLETE =====")
        self.get_logger().info(f"Total runtime: {total_time:.2f} seconds")
        self.get_logger().info(f"Total runtime: {total_time / 60:.2f} minutes")


# ------------------ MAIN ------------------

def main(args=None):
    rclpy.init(args=args)

    node = DelayedGratificationRobot()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()