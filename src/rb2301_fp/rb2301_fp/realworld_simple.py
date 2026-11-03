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

        # ==========================================
        # ARTIFICIAL POTENTIAL FIELD PARAMETERS
        # ==========================================
        self.navigation_stage = 0
        self.state1_end_y = 1.74
        self.state2_end_x = 1.30
        self.state3_end_y = 0.0

        self.k_att = 0.7
        self.k_rep = 0.65

        self.obstacle_influence_distance = 0.7
        self.minimum_distance = 0.1

        self.lateral_deadband = 0

        self.forward_velocity_limit = 0.22
        self.lateral_velocity_limit = 0.12

        self.enter_escape_threshold = 0.13
        self.exit_escape_threshold = 0.16
        self.preferred_direction = -1.0
        self.in_escape_mode = False


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

    # ==========================================
    # ARTIFICIAL POTENTIAL FIELD HELPERS
    # ==========================================

    def update_navigation_stage(self, x, y):
        """Advance navigation stages using real-world coordinates."""

        next_stage = self.navigation_stage

        # Stage 0: Move along global -Y
        if self.navigation_stage == 0 and y <= self.state1_end_y:
            next_stage = 1

        # Stage 1: Move along global -X
        elif self.navigation_stage == 1 and x <= self.state2_end_x:
            next_stage = 2

        # Stage 2: Move along global -Y
        elif self.navigation_stage == 2 and y <= self.state3_end_y:
            next_stage = 3

        if next_stage != self.navigation_stage:
            previous_stage = self.navigation_stage
            self.navigation_stage = next_stage
            self.in_escape_mode = False

            self.get_logger().info(
                f"APF stage transition: {previous_stage} -> {next_stage} "
                f"at ({x:.2f}, {y:.2f})"
            )



    def get_avoidance_velocity(self, heading_deg):

        if not hasattr(self, "last_scan"):
            return 0.0, 0.0

        scan = np.asarray(self.last_scan, dtype=float)

        n = len(scan)

        # Official scan convention:
        # 0 = front, 90 = left, 180 = back, 270 = right
        angles = np.linspace(
            0.0,
            2.0 * np.pi,
            n,
            endpoint=False
        )

        valid = (
            np.isfinite(scan)
            & (scan > 0.0)
            & (scan < self.obstacle_influence_distance)
        )

        if not np.any(valid):
            return 0.0, 0.0

        distances = np.maximum(
            scan[valid],
            self.minimum_distance
        )

        # Keep only the angles corresponding to valid obstacles
        angles = angles[valid]

        magnitudes = self.k_rep * (
            1.0 / distances
            - 1.0 / self.obstacle_influence_distance
        ) / (distances * distances)

        angular_weight = 2.0 * np.pi / n

        # Sum repulsive forces along robot X
        force_robot_x = float(np.sum(
            -magnitudes * np.cos(angles) * angular_weight
        ))

        # Sum repulsive forces along robot Y
        force_robot_y = float(np.sum(
            -magnitudes * np.sin(angles) * angular_weight
        ))

        # Convert robot heading from degrees to radians
        yaw = np.deg2rad(heading_deg)

        force_global_x = (
            np.cos(yaw) * force_robot_x
            - np.sin(yaw) * force_robot_y
        )

        force_global_y = (
            np.sin(yaw) * force_robot_x
            + np.cos(yaw) * force_robot_y
        )

        return force_global_x, force_global_y


    def ca1_x(self, repulsive_x, repulsive_y):
        """APF for stages moving along global -Y."""

        attractive = (0.0, -self.k_att)

        lateral = (
            0.0 if abs(repulsive_x) < self.lateral_deadband
            else repulsive_x
        )

        total = (
            attractive[0] + lateral,
            attractive[1] + repulsive_y
        )

        return attractive, total
    

    def ca1_y(self, repulsive_x, repulsive_y):
        """APF for stage moving along global -X."""

        attractive = (-self.k_att, 0.0)

        lateral = (
            0.0 if abs(repulsive_y) < self.lateral_deadband
            else repulsive_y
        )

        total = (
            attractive[0] + repulsive_x,
            attractive[1] + lateral
        )

        return attractive, total



    def limit_apf_velocity(self, force_x, force_y):

        if self.navigation_stage in (0, 2):

            # Primary -Y, lateral X
            velocity_x = np.clip(
                force_x, -self.lateral_velocity_limit, self.lateral_velocity_limit
            )

            velocity_y = np.clip(
                force_y, -self.forward_velocity_limit, self.forward_velocity_limit
            )

        else:

            # Primary -X, lateral Y
            velocity_x = np.clip(
                force_x, -self.forward_velocity_limit, self.forward_velocity_limit
            )

            velocity_y = np.clip(
                force_y, -self.lateral_velocity_limit, self.lateral_velocity_limit
            )

        return float(velocity_x), float(velocity_y)



    def timer_callback(self):
        """Controller loop. Insert path planning and PID control logic here"""
        if self.pose is None:
            print("No pose detected")
            return # Does not run if no pose received from Odom or Optitrack

        if self.calibrate and self.raw_pose is not None:
            self._ticks += 1
            if self._ticks % 10 == 0: # twice a second
                ox, oy, oh = self.raw_pose
                self.get_logger().info(f"optitrack ({ox:+.3f}, {oy:+.3f}, {oh:+7.1f} deg)  ->  maze ({self.pose[0]:+.3f}, {self.pose[1]:+.3f}, {self.pose[2]:+7.1f} deg)")
            return # Calibration mode: look, don't drive

        ###### INSERT CODE HERE ######

        px, py, heading_deg = self.pose

        self.update_navigation_stage(px, py)

        if self.navigation_stage == 3:
            self.move_2D(0.0, 0.0, 0.0)
            return

        repulsive_x, repulsive_y = self.get_avoidance_velocity(heading_deg)

        if self.navigation_stage in (0, 2):
            attractive, total = self.ca1_x(repulsive_x, repulsive_y)
        else:
            attractive, total = self.ca1_y(repulsive_x, repulsive_y)

        velocity_global_x, velocity_global_y = self.limit_apf_velocity(
            total[0], total[1]
        )

        # ==========================================
        # LOCAL MINIMUM RECOVERY
        # ==========================================

        # Select primary and escape axes.
        if self.navigation_stage in (0, 2):

            # Primary -Y, escape along X
            primary_velocity = -velocity_global_y
            lateral_avoidance = repulsive_x

        else:

            # Primary -X, escape along Y
            primary_velocity = -velocity_global_x
            lateral_avoidance = repulsive_y

        escape_velocity = None

        if self.in_escape_mode:

            # Resume normal APF when the primary velocity recovers.
            if primary_velocity > self.exit_escape_threshold:
                self.in_escape_mode = False

            else:
                if abs(lateral_avoidance) < 0.20:
                    # Keep the preferred direction.
                    escape_velocity = 0.15 * self.preferred_direction

                else:
                    # Follow lateral avoidance when strong.
                    escape_velocity = lateral_avoidance
                    self.in_escape_mode = False

        else:

            # Enter escape mode at an APF local minimum.
            if primary_velocity < self.enter_escape_threshold:
                self.in_escape_mode = True

                if lateral_avoidance > 0.1:
                    self.preferred_direction = 1.0
                elif lateral_avoidance < -0.1:
                    self.preferred_direction = -1.0
                else:
                    self.preferred_direction *= -1.0

                escape_velocity = 0.15 * self.preferred_direction


        if escape_velocity is not None:

            if self.navigation_stage in (0, 2):

                # Primary -Y, escape along +/-X
                velocity_global_y = 0.0
                velocity_global_x = escape_velocity

            else:

                # Primary -X, escape along +/-Y
                velocity_global_x = 0.0
                velocity_global_y = escape_velocity


        yaw = math.radians(heading_deg)

        velocity_robot_x = (
            math.cos(yaw) * velocity_global_x
            + math.sin(yaw) * velocity_global_y
        )

        velocity_robot_y = (
            -math.sin(yaw) * velocity_global_x
            + math.cos(yaw) * velocity_global_y
        )

        self.move_2D(
            x=velocity_robot_x,
            y=velocity_robot_y,
            turn=0.0
        )

        self.get_logger().info(
            f"APF stage={self.navigation_stage} | "
            f"pose=({px:.2f}, {py:.2f}, {heading_deg:.1f}) | "
            f"repulsive=({repulsive_x:.3f}, {repulsive_y:.3f}) | "
            f"escape_mode={self.in_escape_mode} | "
            f"preferred_direction={self.preferred_direction:+.1f} | "
            f"command=({velocity_robot_x:.3f}, {velocity_robot_y:.3f})"
        )
        ###### INSERT CODE HERE ######
                

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
