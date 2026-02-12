import json
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String


class LocationAwareness(Node):
    def __init__(self):
        super().__init__('location_awareness')

        # latest position
        self.x = None
        self.y = None
        self.yaw = None
        self.source = None
        self.current_zone = None

        # zone publisher for the delayed gratification experiment
        self.zone_pub = self.create_publisher(String, '/carl/zone', 10)

        # amcl absolute coordinates
        amcl_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1)

        # subscribe to amcl topic
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_callback,
            amcl_qos)

        # relative coordinate for fallback, subscribe to odometry topic
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)

        # display position
        self.timer = self.create_timer(1.0, self.print_location)

        self.get_logger().info('Location awareness node started. Waiting for pose data...')

    def amcl_callback(self, msg: PoseWithCovarianceStamped):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        self.source = 'amcl'

    def odom_callback(self, msg: Odometry):
        # use odometry if amcl data doesnt exist
        if self.source == 'amcl':
            return
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        self.source = 'odom'
        self._publish_zone()

    def _compute_zone(self):
        """Determine which zone the robot is in based on coordinates."""
        if self.x is None:
            return None
        if self.y < 0.5:
            return 'CENTER'
        elif self.x < -1.0 and self.y > 0.5:
            return 'LEFT_CHAMBER'
        elif self.x > 1.0 and self.y > 0.5:
            return 'RIGHT_CHAMBER'
        else:
            return 'DIVIDER'

    def _publish_zone(self):
        zone = self._compute_zone()
        if zone is None:
            return
        if zone != self.current_zone:
            self.current_zone = zone
            self.get_logger().info(f'Zone changed: {zone}')
        msg = String()
        msg.data = json.dumps({'zone': zone})
        self.zone_pub.publish(msg)

        # convert quaternion to angles
    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def print_location(self):
        if self.x is None:
            self.get_logger().warn('No position data.')
            return

        yaw_deg = math.degrees(self.yaw)
        self.get_logger().info(
            f'[{self.source}] x={self.x:.4f}, y={self.y:.4f}, yaw={yaw_deg:.1f}°')


def main():
    rclpy.init()
    node = LocationAwareness()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
