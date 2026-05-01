"""
Vision node — camera-based color detection and visual servoing.

Subscribes to the robot camera, detects red and yellow targets via HSV
contour analysis, and steers the robot toward a commanded target color.

Topics:
  sub: camera image (parameterized), /carl/vision_cmd
  pub: /carl/color_detection, /carl/vision_status, /cmd_vel
"""

import json
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import TwistStamped
from cv_bridge import CvBridge

MIN_COLOR_AREA = 150
TARGET_REACHED_AREA = 3000
CENTER_TOLERANCE = 40

APPROACH_FORWARD_SPEED = 0.05
SCAN_ANGULAR_SPEED = 0.12
SCAN_SWITCH_SEC = 2.0
SCAN_TIMEOUT_SEC = 45.0

SEARCH_TURN_SPEED = 0.25
TURN_GAIN = 0.002
FORWARD_SPEED = 0.08
SLOW_FORWARD_SPEED = 0.04

SEEK_TIMEOUT_SEC = 60.0
LOG_INTERVAL_SEC = 5.0


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        self.declare_parameter('camera_topic', '/oakd/rgb/preview/image_raw')
        camera_topic = self.get_parameter('camera_topic').value

        self.bridge = CvBridge()

        self.mode = 'idle'
        self.target_color = None
        self.saw_red = False
        self.saw_yellow = False
        self.mode_start_time = 0.0
        self.scan_direction = 1
        self.last_scan_switch = 0.0
        self.last_log_time = 0.0

        self.red_center = None
        self.red_area = 0
        self.yellow_center = None
        self.yellow_area = 0
        self.image_width = None

        self.image_sub = self.create_subscription(
            Image, camera_topic, self.image_callback, 10)
        self.cmd_sub = self.create_subscription(
            String, '/carl/vision_cmd', self.cmd_callback, 10)

        self.detection_pub = self.create_publisher(
            String, '/carl/color_detection', 10)
        self.status_pub = self.create_publisher(
            String, '/carl/vision_status', 10)
        self.cmd_vel_pub = self.create_publisher(
            TwistStamped, '/cmd_vel', 10)

        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info(f'Vision node started. Camera: {camera_topic}')

    def _largest_contour(self, mask):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < MIN_COLOR_AREA:
            return None, area
        M = cv2.moments(largest)
        if M['m00'] == 0:
            return None, area
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy), area

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        red1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
        red_mask = cv2.bitwise_or(red1, red2)

        yellow_mask = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))

        self.red_center, self.red_area = self._largest_contour(red_mask)
        self.yellow_center, self.yellow_area = self._largest_contour(yellow_mask)
        self.image_width = frame.shape[1]

        det = {
            'red_center': list(self.red_center) if self.red_center else None,
            'red_area': self.red_area,
            'yellow_center': list(self.yellow_center) if self.yellow_center else None,
            'yellow_area': self.yellow_area,
        }
        out = String()
        out.data = json.dumps(det)
        self.detection_pub.publish(out)

    def cmd_callback(self, msg):
        cmd = json.loads(msg.data)
        action = cmd.get('action')

        if action == 'scan':
            self.mode = 'scan'
            self.saw_red = False
            self.saw_yellow = False
            self.scan_direction = 1
            self.mode_start_time = time.time()
            self.last_scan_switch = time.time()
            self.last_log_time = 0.0
            self.get_logger().info('Vision: scan mode')

        elif action == 'seek':
            self.target_color = cmd.get('color')
            self.mode = 'seek'
            self.mode_start_time = time.time()
            self.last_log_time = 0.0
            self.get_logger().info(f'Vision: seek {self.target_color}')

        elif action == 'stop':
            self.mode = 'idle'
            self._stop()
            self._publish_status('idle')
            self.get_logger().info('Vision: stopped')

    def control_loop(self):
        if self.mode == 'idle':
            return
        now = time.time()
        if self.mode == 'scan':
            self._do_scan(now)
        elif self.mode == 'seek':
            self._do_seek(now)

    def _do_scan(self, now):
        if self.red_center is not None and self.red_area >= MIN_COLOR_AREA:
            self.saw_red = True
        if self.yellow_center is not None and self.yellow_area >= MIN_COLOR_AREA:
            self.saw_yellow = True

        if now - self.last_log_time >= LOG_INTERVAL_SEC:
            self.get_logger().info(
                f'[Scan] red={self.red_area:.0f} yellow={self.yellow_area:.0f} '
                f'saw_red={self.saw_red} saw_yellow={self.saw_yellow}')
            self.last_log_time = now

        if self.saw_red and self.saw_yellow:
            self._stop()
            self.mode = 'idle'
            self._publish_status('both_seen')
            self.get_logger().info('[Scan] Both colors observed')
            return

        if now - self.mode_start_time > SCAN_TIMEOUT_SEC:
            self._stop()
            self.mode = 'idle'
            self._publish_status('scan_timeout')
            self.get_logger().warn('[Scan] Timeout')
            return

        if now - self.last_scan_switch > SCAN_SWITCH_SEC:
            self.scan_direction *= -1
            self.last_scan_switch = now

        self._publish_cmd(
            APPROACH_FORWARD_SPEED,
            SCAN_ANGULAR_SPEED * self.scan_direction)

    def _do_seek(self, now):
        if self.image_width is None:
            return

        center, area = self._get_target()

        if center is None:
            if now - self.last_log_time >= LOG_INTERVAL_SEC:
                self.get_logger().info(f'[Seek] Searching for {self.target_color}...')
                self.last_log_time = now
            self._publish_cmd(0.0, SEARCH_TURN_SPEED)
            return

        target_x = center[0]
        image_cx = self.image_width // 2
        error = target_x - image_cx

        if now - self.last_log_time >= LOG_INTERVAL_SEC:
            self.get_logger().info(
                f'[Seek] {self.target_color} center={center} '
                f'area={area:.0f} error={error}')
            self.last_log_time = now

        if area >= TARGET_REACHED_AREA:
            self._stop()
            self.mode = 'idle'
            self._publish_status('reached', color=self.target_color)
            self.get_logger().info(
                f'[Seek] Reached {self.target_color} (area={area:.0f})')
            return

        if now - self.mode_start_time > SEEK_TIMEOUT_SEC:
            self._stop()
            self.mode = 'idle'
            self._publish_status('seek_timeout', color=self.target_color)
            self.get_logger().warn(f'[Seek] Timeout for {self.target_color}')
            return

        angular_z = -TURN_GAIN * error
        if abs(error) > CENTER_TOLERANCE:
            linear_x = 0.0
        elif area > TARGET_REACHED_AREA * 0.6:
            linear_x = SLOW_FORWARD_SPEED
        else:
            linear_x = FORWARD_SPEED

        self._publish_cmd(linear_x, angular_z)

    def _get_target(self):
        if self.target_color == 'red':
            return self.red_center, self.red_area
        elif self.target_color == 'yellow':
            return self.yellow_center, self.yellow_area
        return None, 0

    def _publish_cmd(self, linear_x, angular_z):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.twist.linear.x = linear_x
        cmd.twist.angular.z = angular_z
        self.cmd_vel_pub.publish(cmd)

    def _stop(self):
        self._publish_cmd(0.0, 0.0)

    def _publish_status(self, status, **kwargs):
        data = {'status': status}
        data.update(kwargs)
        msg = String()
        msg.data = json.dumps(data)
        self.status_pub.publish(msg)


def main():
    rclpy.init()
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
