import argparse
import configparser
import math
import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.logging import set_logger_level, LoggingSeverity
from rclpy.qos import ReliabilityPolicy, QoSProfile
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

np.set_printoptions(2, suppress=True, threshold=np.inf) # Print numpy arrays to specified d.p., suppress scientific notation and do not truncate
set_logger_level("obstaclecourse", level=LoggingSeverity.INFO) # Configure to either LoggingSeverity.INFO or LoggingSeverity.DEBUG  


_PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))

def wrap_deg(angle:float) -> float:
    '''Wrap an angle in degrees into [-180, 180).'''
    return (angle + 180.0) % 360.0 - 180.0

def optitrack_to_maze(x:float, y:float, heading_deg:float, frame:dict) -> tuple:
    '''Optitrack-frame pose (x, y, heading in degrees) -> CA2 maze-frame pose (see optitrack_variables.config).'''
    dx, dy = x - frame["origin_x"], y - frame["origin_y"]
    r = math.radians(frame["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (c * dx + s * dy, -s * dx + c * dy,
            wrap_deg(heading_deg - frame["rotation_deg"] - frame["heading_offset_deg"]))

def maze_to_course(x:float, y:float, heading_deg:float, course:dict) -> tuple:
    '''CA2 maze-frame pose -> final-project course frame (origin/rotation of the course in the maze frame, [course] in the config).'''
    dx, dy = x - course["origin_x"], y - course["origin_y"]
    r = math.radians(course["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (c * dx + s * dy, -s * dx + c * dy, wrap_deg(heading_deg - course["rotation_deg"]))

def load_frame(robot_number:int) -> tuple:
    '''Read optitrack_variables.config: returns (frame, course) for this robot.'''
    parser = configparser.ConfigParser()
    path = os.path.join(_PACKAGE_DIR, "optitrack_variables.config")
    if not parser.read(path):
        raise FileNotFoundError(f"Could not read {path}")
    frame = {k: parser.getfloat("frame", k) for k in ("origin_x", "origin_y", "rotation_deg")}
    raw = parser.get("frame", "heading_offset_deg", fallback="auto").strip().lower()
    # Rigid bodies are created facing +y of the maze frame, so Motive's zero heading is maze +y (90 deg)
    frame["heading_offset_deg"] = wrap_deg(-(frame["rotation_deg"] + 90.0)) if raw == "auto" else float(raw)
    if parser.has_section("heading_offsets") and parser.has_option("heading_offsets", str(robot_number)):
        frame["heading_offset_deg"] = parser.getfloat("heading_offsets", str(robot_number))
    course = {k: parser.getfloat("course", k, fallback=0.0) for k in ("origin_x", "origin_y", "rotation_deg")}
    return frame, course

class LidarNode(Node):
    def __init__(self):
        """Node constructor"""
        super().__init__("LidarNode")
        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, 2) # Subscribe to lidar
        self.is_simulation = None

    def sub_scan_callback(self, msg):
        """Scan subscriber"""
        if len(msg.ranges) <= 360: 
            self.is_simulation = True
        else:
            self.is_simulation = False

class ObstacleCourseNode(Node):
    '''Node to navigate obstacle course, using pose from either gazebo odometer or optitrack and lidar scan data'''
    def __init__(self, is_simulation:bool=True, calibrate:bool=False):
        super().__init__('obstaclecourse')
        self.get_logger().info("Starting ObstacleCourseNode")

        self.is_simulation = is_simulation
        self.calibrate = calibrate # Real robot: only print poses, never drive
        self.raw_pose = None # Pose exactly as Optitrack reported it (real robot only)
        self.frame = self.course = None
        self._ticks = 0

        if is_simulation:
            self.max_translate_velocity = 1.4
        else:
            self.max_translate_velocity = 0.3 # Please keep this in place; 0.3m/s is more than fast enough 

        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, 2) # Subscribe to LiDAR scan data

        # Subscribe to the dynamic_pose topic from Gazebo that publishes ground-truth pose data
        if self.is_simulation:
            self.subscription = self.create_subscription(Odometry, 'odom', self.odometer_callback, 2)
        else:
            qos_profile = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
            # On the Bingda robots ROS_DOMAIN_ID is the robot's number, so the Optitrack topic is picked automatically
            domain = os.environ.get("ROS_DOMAIN_ID", "")
            self.robot_number = int(domain) if domain.isdigit() and 1 <= int(domain) <= 99 else 3
            self.frame, self.course = load_frame(self.robot_number)
            self.get_logger().info(f"Reading pose from /vrpn_mocap/bingda_{self.robot_number:03d}/pose")

            self.map_sub = self.create_subscription( 
                PoseStamped,
                f'/vrpn_mocap/bingda_{self.robot_number:03d}/pose',
                self.optitrack_callback, 
                qos_profile
                )
            
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10) # Publish to cmd_vel node       
        self.timer = self.create_timer(0.05, self.timer_callback)  # Runs at 20Hz. Can be changed.

        self.pose = None

        # CA1 obstacle avoidance parameters
        self.last_scan = None

        self.safe_distance = 1.0
        self.avoidance_gain = 0.1

        self.forward_velocity = 0.3

        # Escape mode
        self.preferred_direction = -1.0
        self.in_escape_mode = False

        # Lateral displacement correction
        self.side_displacement = 0.0

        self.stage = 0

        self.stage_names = [
            "FORWARD_X_1",
            "MOVE_Y",
            "FORWARD_X_2",
            "FINISHED"
        ]

        self.escape_direction_x = -1.0
        self.escape_direction_y = -1.0

        # Separate displacement correction for each axis
        self.x_displacement = 0.0
        self.y_displacement = 0.0

        self.state1_end_x = 1.7
        self.state2_end_x = 3.0
        self.state3_end_x = 5.7

        self.state2_end_y = -2.5

    def sub_scan_callback(self, msg):
        """Scan subscriber"""
        if len(msg.ranges) <= 360:
            self.last_scan = np.array(msg.ranges)
        else:
            self.last_scan = np.array(msg.ranges)[::2] 

    def yaw_from_quaternion(self, q):
        '''Returns yaw angle (in rad) for orientation based on given quaternion input q'''
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def optitrack_callback(self, msg:PoseStamped):
        '''Callback to calculate 2D pose info from Optitrack node. Pose info includes x and y coordinates, as well as heading in degrees.
        This callback will run everytime the rclpy executor spins'''
        x, y = msg.pose.position.x, msg.pose.position.y
        heading = np.rad2deg(self.yaw_from_quaternion(msg.pose.orientation))
        self.raw_pose = np.array((x,y,heading))
        x, y, heading = optitrack_to_maze(x, y, heading, self.frame) # CA2 maze frame, see optitrack_variables.config
        self.pose = np.array((x,y,heading))
        return self.pose

    def to_course(self, pose=None):
        '''Convert a pose [x, y, heading] (default: self.pose) from the CA2 maze frame to the final-project course frame. Optional; see [course] in optitrack_variables.config'''
        pose = self.pose if pose is None else pose
        return np.array(maze_to_course(pose[0], pose[1], pose[2], self.course)) if self.course else np.array(pose)

    def odometer_callback(self, msg):
        '''Callback to calculate 2D pose info from Gazebo odomoter. Pose info includes x and y coordinates, as well as heading in degrees.
        This callback will run everytime the rclpy executor spins'''
        latest_pose_msg = msg.pose.pose 
        heading = np.rad2deg(self.yaw_from_quaternion(latest_pose_msg.orientation))
        self.pose = np.array((latest_pose_msg.position.x, latest_pose_msg.position.y, heading))
        return self.pose

    def move_2D(self, x:float=0.0, y:float=0.0, turn:float=0.0):
        '''Publishes a Twist message to ROS to move a robot. Inputs are x and y linear velocities, as well as turn (z-axis yaw) angular velocity.'''
        twist_msg = Twist()
        x = np.clip(x, -self.max_translate_velocity, self.max_translate_velocity)
        y = np.clip(y, -self.max_translate_velocity, self.max_translate_velocity)
        turn = np.clip(turn, -self.max_translate_velocity*2, self.max_translate_velocity*2)
        twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z = float(x), float(y), 0.0
        twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z = 0.0, 0.0, float(turn)
        self.publisher_.publish(twist_msg)

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

        vx = 0.0
        vy = 0.0

        for distance, angle in zip(scan, angles):

            if distance >= self.safe_distance:
                continue

            distance = max(distance, 0.04)

            angle_deg = np.degrees(angle)

            # Original CA1 directional weights
            if angle_deg < 1.0 or angle_deg >= 359.0:
                direction_weight = 1.0

            elif 89.0 <= angle_deg <= 91.0:
                direction_weight = 1.0

            elif 269.0 <= angle_deg <= 271.0:
                direction_weight = 1.0

            elif 179.0 <= angle_deg <= 181.0:
                direction_weight = 1.0

            else:
                direction_weight = 0.01

            strength = (
                self.avoidance_gain
                * direction_weight
                * (1.0 / distance - 1.0 / self.safe_distance) ** 2
            )

            vx -= 0.5 * strength * np.cos(angle)
            vy -= 0.1 * strength * np.sin(angle)

            # Ignore lateral movement when left/right repulsion is balanced
            if abs(vy) < 0.05:
                vy = 0.0

        return vx, vy

    def ca1_x(self, avoid_x, avoid_y):

        forward_velocity = 0.3
        forward_result = forward_velocity + avoid_x

        correction_gain = 0.5

        if abs(self.y_displacement) < 0.03:
            correction_factor = 0.0
        else:
            correction_factor = -correction_gain * self.y_displacement

        enter_escape_threshold = 0.05
        exit_escape_threshold = 0.15

        # ==========================================
        # ESCAPE MODE
        # ==========================================

        if self.in_escape_mode:

            if forward_result > exit_escape_threshold:

                self.in_escape_mode = False

                x = forward_result
                y = avoid_y + correction_factor

            else:

                if abs(avoid_y) < 0.20:

                    x = 0.0
                    y = 0.15 * self.escape_direction_x

                else:

                    x = 0.0
                    y = avoid_y
                    self.in_escape_mode = False

        # ==========================================
        # NORMAL MODE
        # ==========================================

        else:

            if forward_result < enter_escape_threshold:

                self.in_escape_mode = True

                if avoid_y > 0.1:
                    self.escape_direction_x = 1.0

                elif avoid_y < -0.1:
                    self.escape_direction_x = -1.0

                else:
                    self.escape_direction_x *= -1.0

                x = 0.0
                y = 0.15 * self.escape_direction_x

            else:

                x = forward_result

                if abs(avoid_y) > 0.02:
                    y = avoid_y
                else:
                    y = avoid_y + correction_factor

        # ==========================================
        # VELOCITY LIMITS
        # ==========================================

        x = np.clip(
            x,
            0.0,
            self.max_translate_velocity
        )

        y = np.clip(
            y,
            -0.15,
            0.15
        )

        self.y_displacement += y * 0.05

        return x, y

    def ca1_y(self, avoid_x, avoid_y):

        forward_velocity = 0.3

        # Primary direction is negative Y
        forward_result = forward_velocity - avoid_y

        correction_gain = 0.5

        if abs(self.x_displacement) < 0.03:
            correction_factor = 0.0
        else:
            correction_factor = -correction_gain * self.x_displacement

        enter_escape_threshold = 0.05
        exit_escape_threshold = 0.15

        # ==========================================
        # ESCAPE MODE
        # ==========================================

        if self.in_escape_mode:

            if forward_result > exit_escape_threshold:

                self.in_escape_mode = False

                y = -forward_result
                x = avoid_x + correction_factor

            else:

                if abs(avoid_x) < 0.20:

                    y = 0.0
                    x = 0.15 * self.escape_direction_y

                else:

                    y = 0.0
                    x = avoid_x
                    self.in_escape_mode = False

        # ==========================================
        # NORMAL MODE
        # ==========================================

        else:

            if forward_result < enter_escape_threshold:

                self.in_escape_mode = True

                if avoid_x > 0.1:
                    self.escape_direction_y = 1.0

                elif avoid_x < -0.1:
                    self.escape_direction_y = -1.0

                else:
                    self.escape_direction_y *= -1.0

                y = 0.0
                x = 0.15 * self.escape_direction_y

            else:

                y = -forward_result

                if abs(avoid_x) > 0.02:
                    x = avoid_x
                else:
                    x = avoid_x + correction_factor

        # ==========================================
        # VELOCITY LIMITS
        # ==========================================

        x = np.clip(
            x,
            -0.15,
            0.15
        )

        y = np.clip(
            y,
            -self.max_translate_velocity,
            0.0
        )

        self.x_displacement += x * 0.05

        return x, y

    def timer_callback(self):
        """Three-stage navigation: +X -> -Y -> +X"""

        if self.pose is None:
            print("No pose detected")
            return

        if self.calibrate:
            if self.raw_pose is not None:
                self._ticks += 1
                if self._ticks % 10 == 0:
                    ox, oy, oh = self.raw_pose
                    self.get_logger().info(
                        f"optitrack ({ox:+.3f}, {oy:+.3f}, {oh:+7.1f} deg)"
                    )
            return

        if self.last_scan is None:
            self.move_2D(0.0, 0.0, 0.0)
            return

        # ==========================================
        # 1. CURRENT POSE
        # ==========================================

        px, py, heading = self.pose

        # ==========================================
        # 2. STATE MACHINE
        # ==========================================

        previous_stage = self.stage

        # Stage 0: +X
        if self.stage == 0 and px >= self.state1_end_x:
            self.stage = 1

        # Stage 1: -Y
        elif self.stage == 1 and py <= self.state2_end_y:
            self.stage = 2

        # Stage 2: +X
        elif self.stage == 2 and px >= self.state3_end_x:
            self.stage = 3

        # ==========================================
        # 3. STATE TRANSITION
        # ==========================================

        if self.stage != previous_stage:

            self.in_escape_mode = False

            self.x_displacement = 0.0
            self.y_displacement = 0.0

            self.get_logger().info(
                f"STATE TRANSITION: "
                f"{self.stage_names[previous_stage]} "
                f"-> {self.stage_names[self.stage]}"
            )

        # ==========================================
        # 4. FINISHED
        # ==========================================

        if self.stage == 3:
            self.move_2D(0.0, 0.0, 0.0)
            return

        # ==========================================
        # 5. OBSTACLE AVOIDANCE
        # ==========================================

        avoid_x, avoid_y = self.get_avoidance_velocity()

        # ==========================================
        # 6. SELECT CONTROLLER
        # ==========================================

        if self.stage == 0:

            # Stage 1: Move +X
            x, y = self.ca1_x(avoid_x, avoid_y)

        elif self.stage == 1:

            # Stage 2: Move -Y
            x, y = self.ca1_y(avoid_x, avoid_y)

        elif self.stage == 2:

            # Stage 3: Move +X
            x, y = self.ca1_x(avoid_x, avoid_y)

        # ==========================================
        # 7. DEBUG
        # ==========================================

        self._ticks += 1

        if self._ticks % 10 == 0:
            self.get_logger().info(
                f"State: {self.stage_names[self.stage]} | "
                f"Mode: {'ESCAPE' if self.in_escape_mode else 'NORMAL'} | "
                f"Pose: ({px:.2f}, {py:.2f}, {heading:.1f}) | "
                f"Avoid: ({avoid_x:.3f}, {avoid_y:.3f}) | "
                f"Velocity: ({x:.3f}, {y:.3f})"
            )

        # ==========================================
        # 8. MOVE
        # ==========================================

        self.move_2D(
            x=x,
            y=y,
            turn=0.0
        )

                

def main(args=None):
    arg_parser = argparse.ArgumentParser(description="RB2301 final project")
    arg_parser.add_argument('--calibrate', action='store_true', help="Real robot: only print the Optitrack pose and the maze-frame pose twice a second, never drive")
    cli_args, ros_args = arg_parser.parse_known_args(args=args)
    print("Starting obstacle course")
    rclpy.init(args=ros_args)

    # Use lidar scan length to determine if in simulation or not
    lidar = LidarNode()
    while lidar.is_simulation is None:
        rclpy.spin_once(lidar)
    is_simulation = lidar.is_simulation
    lidar.destroy_node()

    # Start spinning the obstacle node and only stop once SystemExit error is raised within the node callback
    obstacle = ObstacleCourseNode(is_simulation, cli_args.calibrate)
    try:
        rclpy.spin(obstacle)
    except SystemExit:
        print("Shutting down")
    obstacle.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()