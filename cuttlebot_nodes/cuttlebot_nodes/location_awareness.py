"""
Location awareness node — tracks robot position and publishes zone.

Subscribes to AMCL (absolute) and odometry (fallback) pose data.
Publishes current zone to /carl/zone for the experiment.

Zones: CENTER (y<-0.3), GAP (near divider), LEFT_CHAMBER, RIGHT_CHAMBER
"""

import json
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String


class LocationAwareness(Node):
    def __init__(self):
        super().__init__('location_awareness')

        self.x = None
        self.y = None
        self.yaw = None
        self.source = None
        self.current_zone = None

        self.amcl_count = 0
        self.odom_count = 0
        self.zone_change_count = 0

        self.zone_pub = self.create_publisher(String, '/carl/zone', 10)

        amcl_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1)

        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_callback,
            amcl_qos)

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile_sensor_data)

        self.timer = self.create_timer(1.0, self.print_location)

        self._log('init', 'Location awareness node started.')
        self._log('init', 'Subscribed to /amcl_pose (RELIABLE/TRANSIENT_LOCAL) and /odom (sensor_data)')
        self._log('init', 'Publishing zones to /carl/zone')
        self._log('init', 'Zone boundaries: CENTER(y<-0.3), GAP(-0.3<=y<=1.2), '
                  'LEFT_CHAMBER(y>1.2,x<0), RIGHT_CHAMBER(y>1.2,x>=0)')

    def _log(self, tag, msg):
        self.get_logger().info(f'[loc:{tag}] {msg}')

    def _warn(self, tag, msg):
        self.get_logger().warn(f'[loc:{tag}] {msg}')

    def amcl_callback(self, msg: PoseWithCovarianceStamped):
        self.amcl_count += 1
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        prev_source = self.source
        self.source = 'amcl'

        if self.amcl_count == 1:
            self._log('amcl', f'First AMCL pose received: '
                      f'x={self.x:.4f} y={self.y:.4f} yaw={math.degrees(self.yaw):.1f}deg')
        elif prev_source != 'amcl':
            self._log('amcl', f'Switched to AMCL source (was {prev_source})')

        if self.amcl_count <= 5 or self.amcl_count % 50 == 0:
            self._log('amcl', f'AMCL #{self.amcl_count}: '
                      f'x={self.x:.4f} y={self.y:.4f} yaw={math.degrees(self.yaw):.1f}deg')

        self._publish_zone()

    def odom_callback(self, msg: Odometry):
        self.odom_count += 1
        if self.source == 'amcl':
            return
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        self.source = 'odom'

        if self.odom_count == 1:
            self._log('odom', f'First odom pose received: '
                      f'x={self.x:.4f} y={self.y:.4f} yaw={math.degrees(self.yaw):.1f}deg')
        elif self.odom_count % 100 == 0:
            self._log('odom', f'Odom #{self.odom_count}: '
                      f'x={self.x:.4f} y={self.y:.4f} (using odom, no AMCL yet)')

        self._publish_zone()

    def _compute_zone(self):
        if self.x is None:
            return None
        if self.y < -0.3:
            return 'CENTER'
        elif self.y > 1.2 and self.x < 0:
            return 'LEFT_CHAMBER'
        elif self.y > 1.2 and self.x >= 0:
            return 'RIGHT_CHAMBER'
        else:
            return 'GAP'

    def _publish_zone(self):
        zone = self._compute_zone()
        if zone is None:
            return
        if zone != self.current_zone:
            self.zone_change_count += 1
            self._log('zone', f'Zone #{self.zone_change_count}: '
                      f'{self.current_zone} -> {zone} '
                      f'(x={self.x:.4f} y={self.y:.4f})')
            self.current_zone = zone
        msg = String()
        msg.data = json.dumps({'zone': zone})
        self.zone_pub.publish(msg)

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def print_location(self):
        if self.x is None:
            self._warn('pos', f'No position data yet. amcl_msgs={self.amcl_count} odom_msgs={self.odom_count}')
            return

        yaw_deg = math.degrees(self.yaw)
        self._log('pos', f'[{self.source}] x={self.x:.4f} y={self.y:.4f} '
                  f'yaw={yaw_deg:.1f}deg zone={self.current_zone} '
                  f'(amcl={self.amcl_count} odom={self.odom_count})')


def main():
    rclpy.init()
    node = LocationAwareness()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
