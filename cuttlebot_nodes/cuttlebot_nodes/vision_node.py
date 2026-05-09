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
TARGET_REACHED_AREA = 20000
CENTER_TOLERANCE = 40

APPROACH_FORWARD_SPEED = 0.05
SCAN_ANGULAR_SPEED = 0.12
SCAN_SWITCH_SEC = 2.0
SCAN_TIMEOUT_SEC = 10.0

SEARCH_TURN_SPEED = 0.25
TURN_GAIN = 0.002
FORWARD_SPEED = 0.12
SLOW_FORWARD_SPEED = 0.06

SEEK_TIMEOUT_SEC = 60.0
LOG_INTERVAL_SEC = 2.0


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
        self.frame_count = 0
        self.cmd_count = 0

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

        self._log('init', f'Vision node started. Camera: {camera_topic}')

    def _log(self, tag, msg):
        self.get_logger().info(f'[vision:{tag}] {msg}')

    def _warn(self, tag, msg):
        self.get_logger().warn(f'[vision:{tag}] {msg}')

    def _largest_contour(self, mask):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0, None
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < MIN_COLOR_AREA:
            return None, area, None
        M = cv2.moments(largest)
        if M['m00'] == 0:
            return None, area, None
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy), area, largest

    def image_callback(self, msg):
        self.frame_count += 1
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        red1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
        red_mask = cv2.bitwise_or(red1, red2)

        yellow_mask = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))

        self.red_center, self.red_area, red_contour = self._largest_contour(red_mask)
        self.yellow_center, self.yellow_area, yellow_contour = self._largest_contour(yellow_mask)
        self.image_width = frame.shape[1]

        if self.frame_count == 1:
            self._log('camera', f'First frame received: {frame.shape[1]}x{frame.shape[0]}')
        elif self.frame_count % 100 == 0:
            self._log('camera', f'Frame #{self.frame_count} | '
                      f'red_area={self.red_area:.0f} yellow_area={self.yellow_area:.0f}')

        if self.frame_count % 5 == 0:
            debug = frame.copy()
            if red_contour is not None:
                x, y, w, h = cv2.boundingRect(red_contour)
                cv2.rectangle(debug, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(debug, f'RED {self.red_area:.0f}', (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            if yellow_contour is not None:
                x, y, w, h = cv2.boundingRect(yellow_contour)
                cv2.rectangle(debug, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(debug, f'YLW {self.yellow_area:.0f}', (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.imwrite('/tmp/carl_camera.png', debug)

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
        self._log('cmd', f'Received command: {cmd}')

        if action == 'scan':
            self.mode = 'scan'
            self.saw_red = False
            self.saw_yellow = False
            self.scan_direction = 1
            self.mode_start_time = time.time()
            self.last_scan_switch = time.time()
            self.last_log_time = 0.0
            self._log('cmd', f'Entering SCAN mode. frame_count={self.frame_count} '
                      f'image_width={self.image_width}')

        elif action == 'seek':
            self.target_color = cmd.get('color')
            self.mode = 'seek'
            self.mode_start_time = time.time()
            self.last_log_time = 0.0
            self._log('cmd', f'Entering SEEK mode for {self.target_color}')

        elif action == 'stop':
            self._log('cmd', f'STOP received (was in {self.mode} mode)')
            self.mode = 'idle'
            self._stop()
            self._publish_status('idle')

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
            if not self.saw_red:
                self._log('scan', f'RED detected! area={self.red_area:.0f} '
                          f'center={self.red_center}')
            self.saw_red = True
        if self.yellow_center is not None and self.yellow_area >= MIN_COLOR_AREA:
            if not self.saw_yellow:
                self._log('scan', f'YELLOW detected! area={self.yellow_area:.0f} '
                          f'center={self.yellow_center}')
            self.saw_yellow = True

        if now - self.last_log_time >= LOG_INTERVAL_SEC:
            elapsed = now - self.mode_start_time
            self._log('scan', f'[{elapsed:.1f}s] red_area={self.red_area:.0f} '
                      f'yellow_area={self.yellow_area:.0f} '
                      f'saw_red={self.saw_red} saw_yellow={self.saw_yellow} '
                      f'dir={self.scan_direction}')
            self.last_log_time = now

        if self.saw_red and self.saw_yellow:
            self._stop()
            self.mode = 'idle'
            self._publish_status('both_seen',
                red_center=list(self.red_center) if self.red_center else None,
                yellow_center=list(self.yellow_center) if self.yellow_center else None)
            self._log('scan', 'BOTH colors seen! Publishing both_seen.')
            return

        if now - self.mode_start_time > SCAN_TIMEOUT_SEC:
            self._stop()
            self.mode = 'idle'
            self._publish_status('scan_timeout')
            self._warn('scan', f'TIMEOUT after {SCAN_TIMEOUT_SEC}s. '
                       f'saw_red={self.saw_red} saw_yellow={self.saw_yellow}')
            return


    def _do_seek(self, now):
        if self.image_width is None:
            if now - self.last_log_time >= LOG_INTERVAL_SEC:
                self._warn('seek', 'No image data yet (image_width is None)')
                self.last_log_time = now
            return

        center, area = self._get_target()

        if center is None:
            if now - self.last_log_time >= LOG_INTERVAL_SEC:
                elapsed = now - self.mode_start_time
                self._log('seek', f'[{elapsed:.1f}s] Searching for {self.target_color}... '
                          f'(not visible, turning at {SEARCH_TURN_SPEED} rad/s)')
                self.last_log_time = now
            self._publish_cmd(0.0, SEARCH_TURN_SPEED)
            return

        target_x = center[0]
        image_cx = self.image_width // 2
        error = target_x - image_cx

        if now - self.last_log_time >= LOG_INTERVAL_SEC:
            elapsed = now - self.mode_start_time
            self._log('seek', f'[{elapsed:.1f}s] {self.target_color} '
                      f'center={center} area={area:.0f} error={error} '
                      f'threshold={TARGET_REACHED_AREA}')
            self.last_log_time = now

        if area >= TARGET_REACHED_AREA:
            self._stop()
            self.mode = 'idle'
            self._publish_status('reached', color=self.target_color)
            self._log('seek', f'REACHED {self.target_color}! area={area:.0f} >= {TARGET_REACHED_AREA}')
            return

        if now - self.mode_start_time > SEEK_TIMEOUT_SEC:
            self._stop()
            self.mode = 'idle'
            self._publish_status('seek_timeout', color=self.target_color)
            self._warn('seek', f'TIMEOUT after {SEEK_TIMEOUT_SEC}s for {self.target_color} '
                       f'(last area={area:.0f})')
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
        self.cmd_count += 1
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.twist.linear.x = linear_x
        cmd.twist.angular.z = angular_z
        self.cmd_vel_pub.publish(cmd)
        if self.cmd_count <= 3 or self.cmd_count % 50 == 0:
            self._log('vel', f'cmd_vel #{self.cmd_count}: '
                      f'linear={linear_x:.3f} angular={angular_z:.3f}')

    def _stop(self):
        self._log('vel', 'Publishing STOP (0, 0)')
        self._publish_cmd(0.0, 0.0)

    def _publish_status(self, status, **kwargs):
        data = {'status': status}
        data.update(kwargs)
        self._log('status', f'Publishing status: {data}')
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
