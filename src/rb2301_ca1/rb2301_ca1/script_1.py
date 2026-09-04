import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.logging import set_logger_level, LoggingSeverity
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

np.set_printoptions(2, suppress=True)

max_translate_velocity = 0.4
max_turn_velocity = max_translate_velocity * 2

set_logger_level("obstacle_avoidance", level=LoggingSeverity.DEBUG)


class ObstacleAvoidanceNode(Node):

    def __init__(self):
        super().__init__("obstacle_avoidance")
        self.get_logger().info("Starting Obstacle Avoidance")

        self.pub_cmd_vel = self.create_publisher(Twist, "cmd_vel", 10)
        self.sub_scan = self.create_subscription(
            LaserScan, "scan", self.sub_scan_callback, 2
        )

        self.last_scan = None
        self.preferred_direction = 1.0
        self.in_escape_mode = False

        self.timer = self.create_timer(0.05, self.timer_callback)

    def move_2D(self, x=0.0, y=0.0, turn=0.0):
        twist_msg = Twist()

        x = np.clip(x, -max_translate_velocity, max_translate_velocity)
        y = np.clip(y, -max_translate_velocity, max_translate_velocity)
        turn = np.clip(turn, -max_turn_velocity, max_turn_velocity)

        twist_msg.linear.x = float(x)
        twist_msg.linear.y = float(y)
        twist_msg.angular.z = float(turn)

        self.pub_cmd_vel.publish(twist_msg)

    def sub_scan_callback(self, msg):
        self.last_scan = np.array(msg.ranges)[::20]

    def get_avoidance_velocity(self):
        scan = np.nan_to_num(
            self.last_scan,
            nan=10.0,
            posinf=10.0,
            neginf=0.0
        )

        n = len(scan)

        angles = np.linspace(
            0.0,
            2.0 * np.pi,
            n,
            endpoint=False
        )

        safe_distance = 0.8
        avoidance_gain = 0.1

        vx = 0.0
        vy = 0.0

        for distance, angle in zip(scan, angles):

            if distance >= safe_distance:
                continue

            distance = max(distance, 0.05)

            strength = avoidance_gain * (
                1.0 / distance - 1.0 / safe_distance
            )

            vx -= strength * np.cos(angle)
            vy -= strength * np.sin(angle)

        return vx, vy

    def timer_callback(self):
        if self.last_scan is None:
            return

        avoid_x, avoid_y = self.get_avoidance_velocity()

        forward_velocity = 0.3

        if avoid_x < -0.1 and abs(avoid_y) < 0.05:

            if not self.in_escape_mode:
                self.preferred_direction *= -1.0
                self.in_escape_mode = True

            avoid_y = 0.2 * self.preferred_direction

        else:
            self.in_escape_mode = False

        x = forward_velocity + avoid_x
        y = avoid_y

        x = np.clip(x, -0.2, max_translate_velocity)
        y = np.clip(y, -max_translate_velocity, max_translate_velocity)

        self.move_2D(x=x, y=y, turn=0.0)


def main(args=None):
    rclpy.init(args=args)
    obstacle_avoidance_node = ObstacleAvoidanceNode()
    rclpy.spin(obstacle_avoidance_node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()