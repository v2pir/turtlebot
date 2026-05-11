import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import TwistStamped, PoseWithCovarianceStamped
from irobot_create_msgs.msg import HazardDetectionVector

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

Per-trial behavior:
- go_home() (face NORTH), then fine-tune heading to exact NORTH using
  /amcl_pose so both chambers are guaranteed inside the camera FOV.
- REST AT HOME for several seconds. Terminal prints which trial this is
  so the operator can swap the chambers.
- Move forward while scanning left/right until BOTH red and purple have
  been observed (yellow is hidden behind the purple door at this stage).
- Drive forward to a decision point closer to the chambers.
- Run the DELAY TIMER countdown (10s, 20s, ..., 60s).
- When the delay elapses, terminal prints
      >>> TAKE OUT THE DOOR! <<<
- WAIT FOR DOOR TO OPEN: robot watches camera; when purple disappears
  (and stays gone briefly) the robot knows the door is open.
- Robot makes its Q-learning decision (with delay-discounted Q values).
- Robot vision-tracks toward the chosen color until close, then runs
  a slow BUMP phase that stops on real bumper contact. This way it
  taps the chamber instead of stopping awkwardly short of it.
- Robot turns toward home, then go_home() for the next trial.
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


# ------------------ MAP COORDINATES ------------------

HOME_X = -0.023635001853108406
HOME_Y = -0.023635001853108406


# ------------------ VISION CONTROL PARAMETERS ------------------

MIN_COLOR_AREA = 150

# Area threshold at which the robot stops vision-tracking and switches
# to the final slow "bump" phase. Lower than a hard "reached" threshold
# because we no longer need the chamber to fill most of the frame —
# the bump phase takes us the rest of the way and gives a clean
# physical contact stop.
TARGET_CLOSE_ENOUGH_AREA = 9000

CENTER_TOLERANCE = 40

APPROACH_FORWARD_SPEED = 0.05
PRE_DECISION_SCAN_SPEED = 0.12
PRE_DECISION_SCAN_SWITCH_SEC = 2.0
PRE_DECISION_SAFETY_TIMEOUT_SEC = 45.0

# After the robot has detected both red and purple, it drives straight
# forward for this many seconds to reach a "decision point" closer to
# the chambers, and only then stops to make its decision and run the
# delay timer.
POST_DETECTION_FORWARD_SEC = 4.0
POST_DETECTION_FORWARD_SPEED = 0.10

SEARCH_TURN_SPEED = 0.25
TURN_GAIN_TRACK = 0.002

FORWARD_SPEED = 0.08
SLOW_FORWARD_SPEED = 0.04

COLOR_TIMEOUT_SEC = 60.0

# All repeated color-related logs are throttled to this interval.
COLOR_LOG_INTERVAL_SEC = 5.0


# ------------------ FINAL BUMP PHASE ------------------
# After vision says we're close enough, drive slowly straight forward
# until the bumper fires, or we hit the timeout. Either way we end up
# touching the chamber, which is what counts as "caught the prey".

BUMP_FORWARD_SPEED = 0.05
BUMP_PHASE_TIMEOUT_SEC = 4.0


# ------------------ PURPLE HSV THRESHOLDS ------------------
# IMPORTANT: tune these for your specific purple paper and lighting.
# OpenCV uses H in [0, 180]. True purple sits roughly H=135-150.
# We deliberately keep the upper bound BELOW 160 so we don't catch
# pink/magenta pixels that often appear on red paper under uneven
# lighting. We also keep S and V floors fairly high so faded/shadowed
# reds don't sneak in.
#
# If the robot misses the purple paper:
#   - widen H range, e.g. (130, 80, 70) to (155, 255, 255)
#   - or lower the S/V floors slightly
# If the robot still confuses red with purple:
#   - narrow H range further, e.g. (138, 100, 80) to (150, 255, 255)
#   - or raise the S/V floors

PURPLE_HSV_LOWER = (135, 90, 70)
PURPLE_HSV_UPPER = (155, 255, 255)


# ------------------ REST / DELAY PARAMETERS ------------------

# How long the robot rests at home between trials so the operator can
# rearrange chambers.
REST_AT_HOME_SEC = 20.0

# After "TAKE OUT THE DOOR!", how long purple has to be continuously
# absent before we decide the door has actually been removed. Stops
# the robot from being fooled by a single bad frame.
DOOR_OPEN_STABLE_SEC = 1.5

# Safety timeout if the operator never removes the door / purple keeps
# being detected. Robot logs a warning and continues.
DOOR_OPEN_TIMEOUT_SEC = 60.0


# ------------------ SOUTH TURN PARAMETERS ------------------

TURN_SOUTH_SPEED = 0.45
TURN_SOUTH_TIME = 3.0


# ------------------ NORTH ORIENTATION CORRECTION ------------------
# After Nav2's go_home reports SUCCEEDED, the robot can still be a few
# degrees off NORTH because of goal-pose tolerance. Those few degrees
# can put one chamber outside the camera FOV. We correct in closed loop
# using /amcl_pose (map frame).

NORTH_ALIGN_TOLERANCE_RAD = 0.035  # ~2 degrees
NORTH_ALIGN_TIMEOUT_SEC = 8.0
NORTH_ALIGN_MAX_ANGULAR = 0.40
NORTH_ALIGN_MIN_ANGULAR = 0.12     # below this, the robot won't actually move
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

    return RIGHT


def get_target_color_from_state_action(state, action):
    if state == EXPM_LR:
        if action == LEFT:
            return "yellow"
        else:
            return "red"

    elif state == EXPM_RL:
        if action == LEFT:
            return "red"
        else:
            return "yellow"

    return None


def get_reward_from_state_action(state, action):
    if state == EXPM_LR:
        if action == LEFT:
            return LIVE_RWD
        else:
            return DEAD_RWD

    elif state == EXPM_RL:
        if action == RIGHT:
            return LIVE_RWD
        else:
            return DEAD_RWD

    return 0.0


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
        self.latest_detected_color = "unknown"

        # Log throttling
        self.last_pre_decision_color_log_time = 0.0
        self.last_tracking_color_log_time = 0.0
        self.last_search_color_log_time = 0.0
        self.last_camera_warning_log_time = 0.0
        self.last_door_wait_log_time = 0.0

        # Nav2 initialization
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


    # ------------------ POSE / HAZARD CALLBACKS ------------------

    def pose_callback(self, msg):
        # Convert quaternion to yaw without pulling in tf_transformations.
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.latest_yaw = float(np.arctan2(siny_cosp, cosy_cosp))


    def hazard_callback(self, msg):
        # type == 1 is BUMP in irobot_create_msgs/HazardDetection.
        # We ignore cliff/stall/wheel-drop/proximity here.
        for det in msg.detections:
            if det.type == 1:
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
        # We then subtract the red mask so any pixel that could be
        # interpreted as red is REMOVED from purple. This makes red
        # and purple mutually exclusive and prevents red paper with
        # magenta-ish pixels from being mis-detected as purple.
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

        # `latest_detected_color` is just for the /detected_color topic
        # (logging / debugging). Decision logic uses the per-color flags.
        areas = {
            "red": red_area if red_center is not None else 0,
            "yellow": yellow_area if yellow_center is not None else 0,
            "purple": purple_area if purple_center is not None else 0,
        }

        best = max(areas, key=areas.get)

        if areas[best] <= 0:
            detected = "unknown"
        else:
            detected = best

        self.latest_detected_color = detected

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
        """
        Stops the robot at home and rests for REST_AT_HOME_SEC seconds.
        Prints trial info AND the current state so the human operator
        knows which trial is up and how to arrange the chambers.
        """
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

        rest_start = time.time()
        last_log_sec = -1

        while rclpy.ok() and time.time() - rest_start < REST_AT_HOME_SEC:
            rclpy.spin_once(self, timeout_sec=0.1)

            elapsed = int(time.time() - rest_start)

            if elapsed != last_log_sec:
                remaining = REST_AT_HOME_SEC - elapsed

                if remaining > 0:
                    self.get_logger().info(
                        f"[REST AT HOME] {remaining:.0f}s remaining..."
                    )

                last_log_sec = elapsed

        self.get_logger().info("[REST AT HOME] Rest complete. Robot resuming.")


    # ------------------ DELAY TIMER ------------------

    def run_delay_timer(self, delay_sec):
        """
        After both options are observed and the robot has stopped,
        wait for `delay_sec` seconds while printing a countdown.
        When the delay elapses, prints the operator instruction to
        remove the chamber door.
        """
        self.get_logger().info("======================================")
        self.get_logger().info(
            f"[DELAY TIMER] Both options detected. "
            f"Starting {delay_sec}s countdown..."
        )
        self.get_logger().info("======================================")

        self.stop_robot()

        delay_start = time.time()
        last_log_sec = -1

        while rclpy.ok() and time.time() - delay_start < delay_sec:
            rclpy.spin_once(self, timeout_sec=0.1)

            elapsed = int(time.time() - delay_start)

            if elapsed != last_log_sec:
                remaining = delay_sec - elapsed

                if remaining > 0:
                    self.get_logger().info(
                        f"[DELAY TIMER] {remaining:.0f}s remaining..."
                    )

                last_log_sec = elapsed

        self.get_logger().info("======================================")
        self.get_logger().info("[DELAY TIMER] >>> TAKE OUT THE DOOR! <<<")
        self.get_logger().info("======================================")


    # ------------------ WAIT FOR DOOR TO OPEN ------------------

    def wait_for_door_open(self, timeout_sec=DOOR_OPEN_TIMEOUT_SEC):
        """
        After the delay timer expires, the operator removes the door.
        The door has purple paper on it, so when the door is gone,
        purple disappears from the camera view.

        We require purple to be continuously absent for
        DOOR_OPEN_STABLE_SEC seconds before declaring the door open,
        so a single noisy frame can't fool us.

        Returns True if the door was confirmed open, False on timeout.
        """
        self.get_logger().info(
            "[Door] Waiting for purple to disappear (door to be removed)..."
        )

        self.stop_robot()

        start = time.time()
        purple_gone_since = None
        self.last_door_wait_log_time = 0.0

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            now = time.time()

            if not self.purple_visible():
                if purple_gone_since is None:
                    purple_gone_since = now
                    self.get_logger().info(
                        "[Door] Purple no longer detected. Confirming..."
                    )

                elif now - purple_gone_since >= DOOR_OPEN_STABLE_SEC:
                    self.get_logger().info(
                        "[Door] Purple has been gone long enough. "
                        ">>> DOOR IS OPEN <<<"
                    )
                    return True

            else:
                # Purple visible again — reset the "gone" timer.
                if purple_gone_since is not None:
                    self.get_logger().info(
                        "[Door] Purple detected again. Resetting door-open timer."
                    )
                purple_gone_since = None

            # Throttled status log so we don't spam.
            if now - self.last_door_wait_log_time >= COLOR_LOG_INTERVAL_SEC:
                self.get_logger().info(
                    f"[Door] Still waiting... "
                    f"purple_area={self.latest_purple_area:.1f}, "
                    f"elapsed={now - start:.1f}s"
                )
                self.last_door_wait_log_time = now

            if now - start > timeout_sec:
                self.get_logger().warn(
                    "[Door] Timeout waiting for door to open. "
                    "Continuing trial anyway."
                )
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


    def both_options_visible(self):
        # Pre-decision: yellow is hidden behind the purple door,
        # so "both options" really means red + purple.
        return self.red_visible() and self.purple_visible()


    def move_forward_until_options_visible(self):
        """
        Moves forward while sweeping angular velocity left/right until
        the robot has observed BOTH red (dead chamber) AND purple (the
        door marker on the live/yellow chamber).

        Yellow itself is hidden by the door at this stage, so seeing
        purple is what tells us the live chamber is over there.
        """
        self.get_logger().info(
            "[Pre-decision] Moving forward/scanning until both red and "
            "purple have been observed..."
        )

        saw_red = False
        saw_purple = False

        scan_direction = 1
        last_switch_time = time.time()
        start_time = time.time()
        self.last_pre_decision_color_log_time = 0.0

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

            if self.red_visible():
                saw_red = True

            if self.purple_visible():
                saw_purple = True

            now = time.time()

            # Color-related log: only every 5 seconds.
            if now - self.last_pre_decision_color_log_time >= COLOR_LOG_INTERVAL_SEC:
                self.get_logger().info(
                    f"[Pre-decision Vision] "
                    f"red_area={self.latest_red_area:.1f}, "
                    f"purple_area={self.latest_purple_area:.1f}, "
                    f"yellow_area={self.latest_yellow_area:.1f} (should be ~0 while door is closed), "
                    f"saw_red={saw_red}, saw_purple={saw_purple}"
                )
                self.last_pre_decision_color_log_time = now

            if saw_red and saw_purple:
                self.stop_robot()

                self.get_logger().info(
                    "[Pre-decision] Both red and purple have been observed. "
                    "Robot can now wait for the delay."
                )

                time.sleep(0.5)
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
        """
        After the robot has seen both red and purple, drive straight
        forward for POST_DETECTION_FORWARD_SEC seconds to bring the
        robot closer to the chambers before stopping. This way the
        "decision point" is near the chambers, not back at home.

        Note: we do NOT scan left/right here. The scan was already
        done in move_forward_until_options_visible. This is just a
        committed forward push.
        """
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
        time.sleep(0.5)

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
        now = time.time()

        if self.latest_image_width is None:
            if now - self.last_camera_warning_log_time >= COLOR_LOG_INTERVAL_SEC:
                self.get_logger().warn("[Vision] No camera image received yet.")
                self.last_camera_warning_log_time = now

            self.stop_robot()
            return False

        if center is None:
            if now - self.last_search_color_log_time >= COLOR_LOG_INTERVAL_SEC:
                self.get_logger().info(f"[Vision] Searching for {target_color}...")
                self.last_search_color_log_time = now

            self.publish_cmd(0.0, SEARCH_TURN_SPEED)
            return False

        target_x = center[0]
        image_center_x = self.latest_image_width // 2
        error = target_x - image_center_x

        # Color-related log: only every 5 seconds.
        if now - self.last_tracking_color_log_time >= COLOR_LOG_INTERVAL_SEC:
            self.get_logger().info(
                f"[Vision] Tracking {target_color} | center={center} "
                f"area={area:.1f} error={error}"
            )
            self.last_tracking_color_log_time = now

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
        """
        After Nav2 returns from go_home, the robot can still be a few
        degrees off NORTH. Rotate in place using /amcl_pose feedback
        until the heading is within NORTH_ALIGN_TOLERANCE_RAD of 0.

        NORTH = 0 rad in the map frame for TurtleBot4Directions.NORTH,
        so target yaw is 0.
        """
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
                time.sleep(0.3)
                return True

            # Proportional rotation, clamped, with a floor so we don't
            # stall under the motor's static friction.
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

    def bump_into_chamber(self):
        """
        Final approach. Drive slowly straight forward until the bumper
        fires or BUMP_PHASE_TIMEOUT_SEC elapses. Gives a clean physical
        tap on the chamber instead of stopping short of it.
        """
        self.get_logger().info("[Bump] Slow approach until bumper contact...")

        # Clear any stale bump events from earlier in the trial (e.g.
        # if the robot brushed something during go_home).
        self.bump_detected = False

        start = time.time()

        while rclpy.ok() and time.time() - start < BUMP_PHASE_TIMEOUT_SEC:
            rclpy.spin_once(self, timeout_sec=0.05)

            if self.bump_detected:
                self.stop_robot()
                self.get_logger().info("[Bump] Bumper triggered. Contact made.")
                time.sleep(0.3)
                return True

            self.publish_cmd(linear_x=BUMP_FORWARD_SPEED, angular_z=0.0)

        self.stop_robot()
        self.get_logger().warn(
            "[Bump] Timed out before bumper triggered. Stopping anyway."
        )
        time.sleep(0.3)
        return False


    # ------------------ TURN SOUTH ------------------

    def turn_south_before_home(self):
        self.get_logger().info("[Reset] Turning toward SOUTH / home direction...")

        start = time.time()

        while rclpy.ok() and time.time() - start < TURN_SOUTH_TIME:
            rclpy.spin_once(self, timeout_sec=0.1)
            self.publish_cmd(0.0, TURN_SOUTH_SPEED)

        self.stop_robot()
        time.sleep(0.5)

        self.get_logger().info("[Reset] Finished rough SOUTH orientation.")
        return True


    def go_to_color(self, target_color, timeout_sec=COLOR_TIMEOUT_SEC):
        self.get_logger().info(f"[VisionNav] Moving toward {target_color}")

        start = time.time()

        # Reset color log timers for this tracking phase.
        self.last_tracking_color_log_time = 0.0
        self.last_search_color_log_time = 0.0
        self.last_camera_warning_log_time = 0.0

        # Phase 1: vision-track until close enough to the chamber.
        close_enough = False

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

            close_enough = self.move_toward_color_once(target_color)

            if close_enough:
                break

            if time.time() - start > timeout_sec:
                self.get_logger().warn(
                    f"[VisionNav] Timeout while approaching {target_color}"
                )
                self.stop_robot()
                oriented_south = self.turn_south_before_home()
                return False, oriented_south

        # Phase 2: physical bump. Stops on bumper contact, falls back
        # to a short timeout. Either way the robot ends up touching
        # the chamber, which counts as "caught the prey".
        bumped = self.bump_into_chamber()

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
        time.sleep(0.5)

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
        # because of goal tolerance. Tighten it before the scan so both
        # chambers are guaranteed to be in the camera FOV.
        # Also run on failure: if Nav2 thought it failed but we're
        # actually near home with bad heading, this still helps.
        self.correct_orientation_to_north()

        return success


    # ------------------ EXPERIMENT LOGIC ------------------

    def print_q_table(self):
        self.get_logger().info("Q Table:")

        for s in range(STATES):
            self.get_logger().info(
                f"State {s}: L={qTbl[s][LEFT]:.3f}, R={qTbl[s][RIGHT]:.3f}"
            )


    def run_trial(self, p_wait, delay_sec, trial_label):
        global qTbl

        state = random.choice([EXPM_LR, EXPM_RL])

        self.get_logger().info("======================================")
        self.get_logger().info(f"[Trial] {trial_label}")
        self.get_logger().info(f"[Trial] Randomized state: {state_to_string(state)}")

        if not self.go_home():
            self.get_logger().error("Failed to go home. Aborting trial.")
            return 0.0, True, state, None

        # === REST AT HOME ===
        # Robot rests at home, prints trial info AND the state, so the
        # operator can arrange the chambers correctly for this trial.
        self.rest_at_home(trial_label, delay_sec, state_to_string(state))

        # === MOVE FORWARD AND SCAN UNTIL RED + PURPLE OBSERVED ===
        # Yellow is behind the purple door, so we look for purple
        # instead of yellow at this stage.
        saw_both = self.move_forward_until_options_visible()

        if not saw_both:
            self.get_logger().warn(
                "[Trial] Robot could not observe both red and purple. "
                "Aborting this trial."
            )
            return 0.0, True, state, None

        # === MOVE TO DECISION POINT ===
        # After seeing both colors, drive forward a bit more so the
        # robot ends up near the chambers, not back at home.
        self.move_forward_to_decision_point()

        # === DECISION ===
        # Robot decides which color to commit to BEFORE the delay
        # starts. The delay-discounted Q-values still depend on
        # delay_sec via p_wait, so the decision math is unchanged.
        self.get_logger().info("[Thinking] Robot is at decision point. Deciding...")
        time.sleep(1.0)

        q_thinking = copy.deepcopy(qTbl)

        # Delay discount only affects the live/yellow option.
        if state == EXPM_LR:
            q_thinking[state][LEFT] *= p_wait

        elif state == EXPM_RL:
            q_thinking[state][RIGHT] *= p_wait

        act = action_select(q_thinking[state], BETA)

        self.get_logger().info(
            f"[Decision] Robot chose {action_to_string(act)} chamber"
        )

        target_color = get_target_color_from_state_action(state, act)

        self.get_logger().info(
            f"[Decision] Chosen chamber contains target color: {target_color}"
        )

        if target_color is None:
            self.get_logger().error("[Decision] Invalid target color. Aborting trial.")
            return 0.0, True, state, act

        # === DELAY TIMER ===
        # Robot has decided. Now wait the delay at the decision point.
        # When the timer hits zero, the terminal prints
        # "TAKE OUT THE DOOR!" so the operator can remove the door.
        self.run_delay_timer(delay_sec)

        # === WAIT FOR DOOR TO OPEN ===
        # Operator removes the door -> purple disappears from the
        # camera view -> robot knows it can move.
        door_opened = self.wait_for_door_open()

        if not door_opened:
            self.get_logger().warn(
                "[Trial] Door-open detection timed out. "
                "Continuing with the trial regardless."
            )

        # === APPROACH CHOSEN COLOR ===
        # Vision-track until close, then a slow bump-into-chamber phase
        # provides the actual stop. After that, robot turns south.
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

        time.sleep(1.0)

        # Reward logic:
        # Only give reward if robot actually got close enough to the selected color.
        if reached:
            reward = get_reward_from_state_action(state, act)
        else:
            reward = 0.0

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

        delays = np.array([10, 20, 30, 40, 50, 60])
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

                self.run_trial(prob_wait(d_int), d_int, trial_label)

        self.get_logger().info("===== DELAY TEST PHASE FINISHED =====")

        self.print_q_table()

        self.get_logger().info("Returning robot to home position...")
        self.go_home()
        time.sleep(2.0)

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
