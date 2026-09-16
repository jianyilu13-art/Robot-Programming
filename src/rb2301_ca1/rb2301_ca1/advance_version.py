import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.logging import set_logger_level, LoggingSeverity
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

np.set_printoptions(
    2, suppress=True
)  # Print numpy arrays to specified d.p. and suppress scientific notation (e.g. 1e-5)

max_translate_velocity = 0.4 # Can be implemented as parameter
max_turn_velocity = max_translate_velocity * 2 # Can be implemented as parameter
set_logger_level("obstacle_avoidance", level=LoggingSeverity.DEBUG) # Configure to either LoggingSeverity.INFO or LoggingSeverity.DEBUG  

class ObstacleAvoidanceNode(Node):
    def __init__(self):
        """Node constructor"""
        super().__init__("obstacle_avoidance")
        self.get_logger().info("Starting Obstacle Avoidance")

        self.pub_cmd_vel = self.create_publisher(Twist, "cmd_vel", 10)  # Publish to cmd_vel node
        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, 2) # The subscriber to the Lidar ranges.
        self.last_scan = None # Copied laser scan message

        self.timer = self.create_timer(0.05, self.timer_callback)  # Runs at 20Hz. Can be changed.

        # for escape mode
        self.preferred_direction = 1.0
        self.in_escape_mode = False
        # for side displacement cehcking
        self.side_displacement = 0.0

    def move_2D(self, x: float = 0.0, y: float = 0.0, turn: float = 0.0):
        """Publishes a twist command to move in 2D space. +ve x is forwards, +ve y is left, and +ve turn is anticlockwise"""
        twist_msg = Twist()
        x = np.clip(x, -max_translate_velocity, max_translate_velocity)
        y = np.clip(y, -max_translate_velocity, max_translate_velocity)
        turn = np.clip(turn, -max_translate_velocity*2, max_translate_velocity*2)
        twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z = float(x), float(y), 0.0
        twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z = 0.0, 0.0, float(turn)
        self.pub_cmd_vel.publish(twist_msg)

    def sub_scan_callback(self, msg):
        """Scan subscriber"""
        self.last_scan = np.array(msg.ranges)[::20] # Slices the 721 scan array to return only 36 scans. Feel free to edit

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

            distance = max(distance, 0.04)

            strength = avoidance_gain * (
                1.0 / distance
                - 1.0 / safe_distance
            )

            vx -= 0.5* strength * np.cos(angle)
            vy -= 0.5* strength * np.sin(angle)

        return vx, vy


    def timer_callback(self):
        """Controller loop"""

        if self.last_scan is None:
            return

        avoid_x, avoid_y = self.get_avoidance_velocity()

        forward_velocity = 0.3

        forward_result = forward_velocity + avoid_x

        correction_gain = 0.5
        
        if abs(self.side_displacement) < 0.03:
            correction_factor = 0.0
        else:
            correction_factor = -correction_gain * self.side_displacement

        #set thresholds
        enter_escape_threshold = 0.05
        exit_escape_threshold = 0.15

        # =====================================================
        # ESCAPE MODE
        # =====================================================

        if self.in_escape_mode:

            #leave after there is clearly enough
            if forward_result > exit_escape_threshold:

                self.in_escape_mode = False

                x = forward_result
                y = avoid_y + correction_factor

            else:

                # Keep the SAME escape direction
                x = 0.0
                y = 0.25 * self.preferred_direction

        # =====================================================
        # NORMAL MODE
        # =====================================================

        else:

            # Potential-field local minimum / trap
            if forward_result < enter_escape_threshold:

                self.in_escape_mode = True

                # Choose escape direction once
                if avoid_y > 0.03:
                    self.preferred_direction = 1.0

                elif avoid_y < -0.03:
                    self.preferred_direction = -1.0

                else:
                    # No obvious direction
                    self.preferred_direction *= -1.0

                x = 0.0
                y = 0.25 * self.preferred_direction

            else:

                # Normal potential-field control
                x = forward_result
                y = avoid_y + correction_factor

        # =====================================================
        # VELOCITY LIMITS
        # =====================================================

        x = np.clip(
            x,
            0.0,
            max_translate_velocity
        )

        y = np.clip(
            y,
            -max_translate_velocity,
            max_translate_velocity
        )

        self.side_displacement += y * 0.05

        self.move_2D(
            x=x,
            y=y,
            turn=0.0
    )

def main(args=None):
    rclpy.init(args=args)
    obstacle_avoidance_node = ObstacleAvoidanceNode()
    rclpy.spin(obstacle_avoidance_node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()