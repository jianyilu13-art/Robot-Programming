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
        self.state1_end_x = 1.8
        self.state2_end_y = -2.3
        self.state3_end_x = 5.0

        self.k_att = 3.2
        self.k_rep = 0.15

        self.obstacle_influence_distance = 0.65
        self.minimum_distance = 0.1

        self.lateral_deadband = 0

        self.forward_velocity_limit = 0.30
        self.lateral_velocity_limit = 0.20

        self.push_force = 0.13
        self.enter_escape_threshold = 0.05
        self.exit_escape_threshold = 0.1
        self.push_direction = 0
        self.push_active = False
        self.recovery_waiting = False

        # Conservative rectangular footprint from the NanoCar URDF wheel-centre
        # extents (y = +/-0.08725 m, x = -0.00015 to 0.14398 m).  The mesh has
        # no primitive dimensions, so the margin covers its unreported overhang.

        # Robot dimensions used for collision checking
        self.robot_width = 0.175
        self.robot_length = 0.288

        # Additional clearance around the robot
        self.clearance_safety_margin = 0.01

        # Controller update period (20 Hz)
        self.control_period = 0.05

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
        """Advance the three-stage route using global course-frame pose."""
        next_stage = self.navigation_stage

        # Stage 0: Move along global +X
        if self.navigation_stage == 0 and x >= self.state1_end_x:
            next_stage = 1

        # Stage 1: Move along global -Y
        elif self.navigation_stage == 1 and y <= self.state2_end_y:
            next_stage = 2

         # Stage 2: Move along global +X
        elif self.navigation_stage == 2 and x >= self.state3_end_x:
            next_stage = 3

        if next_stage != self.navigation_stage:
            previous_stage = self.navigation_stage
            self.navigation_stage = next_stage
            self.stage_entry_pose = np.array((x, y), dtype=float)
            self.apf_debug_ticks = 0
            self.push_active = False
            self.push_direction = 0
            self.recovery_waiting = False
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
        """APF force for stages whose primary direction is global +X."""
        # Constant attraction toward global +X
        attractive = (self.k_att, 0.0)

        # Remove small lateral repulsion
        lateral = 0.0 if abs(repulsive_y) < self.lateral_deadband else repulsive_y

        # Total force = attraction + repulsion
        total = (attractive[0] + repulsive_x, attractive[1] + lateral)
        return attractive, total

    def ca1_y(self, repulsive_x, repulsive_y):
        """APF force for the stage whose primary direction is global -Y."""
        # Constant attraction toward global -Y
        attractive = (0.0, -self.k_att)

        # Remove small lateral repulsion
        lateral = 0.0 if abs(repulsive_x) < self.lateral_deadband else repulsive_x

        # Total force = attraction + repulsion
        total = (attractive[0] + lateral, attractive[1] + repulsive_y)
        return attractive, total



    def limit_apf_velocity(self, force_x, force_y):
        """Apply primary/lateral limits without forcing a forward direction."""
        if self.navigation_stage in (0, 2):
            velocity_x = np.clip(
                force_x, -self.forward_velocity_limit, self.forward_velocity_limit
            )
            velocity_y = np.clip(
                force_y, -self.lateral_velocity_limit, self.lateral_velocity_limit
            )
        else:
            velocity_x = np.clip(
                force_x, -self.lateral_velocity_limit, self.lateral_velocity_limit
            )
            velocity_y = np.clip(
                force_y, -self.forward_velocity_limit, self.forward_velocity_limit
            )
        return float(velocity_x), float(velocity_y)



    def directional_clearance(self, direction_global, heading_deg, speed=None):
        """Return free travel in a footprint-wide corridor, or zero if unknown."""

        if not hasattr(self, "last_scan"):
            return 0.0

        scan = np.asarray(self.last_scan, dtype=float)
        if scan.size == 0:
            return 0.0
        angles = np.arange(scan.size) * 2.0 * np.pi / scan.size
        # An infinite range is known-free only up to a conservative finite cap;
        # NaN, non-positive, and missing readings never establish clearance.
        valid = ((np.isfinite(scan) & (scan > 0.0)) | np.isposinf(scan))
        if not np.any(valid):
            return 0.0

        # Keep the official scan convention: angle zero is robot forward.
        distances = np.minimum(scan[valid], self.obstacle_influence_distance)
        robot_angles = angles[valid]
        x = distances * np.cos(robot_angles)
        y = distances * np.sin(robot_angles)

         # Express the desired global direction in the robot frame
        direction_robot = direction_global - math.radians(heading_deg)
        c, s = math.cos(direction_robot), math.sin(direction_robot)
        along = x * c + y * s
        across = -x * s + y * c

        # Inflated robot dimensions
        half_width = self.robot_width / 2.0 + self.clearance_safety_margin
        half_length = self.robot_length / 2.0 + self.clearance_safety_margin

         # Expected movement during one controller cycle
        travel = (self.push_force if speed is None else abs(speed)) * self.control_period
        in_corridor = (
            (np.abs(across) <= half_width)
            & (along >= -half_length)
        )
        if not np.any(in_corridor):
            return float("inf")

        # Clearance is the distance the footprint may translate before contact.
        clearance = max(0.0, float(np.min(along[in_corridor]) - half_length))
        return 0.0 if clearance <= travel else clearance



    def direction_is_traversable(self, direction_global, heading_deg):
        """Check that one recovery step fits in the inflated footprint corridor."""
        return self.directional_clearance(direction_global, heading_deg) > (
            self.push_force * self.control_period
        )

    @staticmethod
    def recovery_direction_name(axis, direction):
        return f"{('+' if direction > 0 else '-')}{axis}" if direction else "NONE"



    def choose_y_push(self, heading_deg):
        """Choose global +/-Y using footprint-wide, not single-beam, clearance."""
        positive = self.directional_clearance(math.pi / 2.0, heading_deg)
        negative = self.directional_clearance(-math.pi / 2.0, heading_deg)
        return 1.0 if positive >= negative else -1.0

    def choose_x_push(self, heading_deg):
        """Choose global +/-X using footprint-wide, not single-beam, clearance."""
        positive = self.directional_clearance(0.0, heading_deg)
        negative = self.directional_clearance(math.pi, heading_deg)
        return 1.0 if positive >= negative else -1.0



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

        if self.navigation_stage in (0, 2):
            primary_velocity = velocity_global_x
            primary_angle = 0.0
            recovery_axis = "Y"
            recovery_angles = (math.pi / 2.0, -math.pi / 2.0)
        else:
            primary_velocity = -velocity_global_y
            primary_angle = -math.pi / 2.0
            recovery_axis = "X"
            recovery_angles = (0.0, math.pi)

        primary_clearance = self.directional_clearance(
            primary_angle, heading_deg, abs(primary_velocity)
        )
        primary_clear = primary_clearance > abs(primary_velocity) * self.control_period
        stuck = primary_velocity <= self.enter_escape_threshold

        if not self.push_active:
            if stuck:
                self.push_active = True
                preferred = (self.choose_y_push(heading_deg)
                             if self.navigation_stage in (0, 2)
                             else self.choose_x_push(heading_deg))
                preferred_angle = recovery_angles[0 if preferred > 0 else 1]
                opposite = -preferred
                opposite_angle = recovery_angles[0 if opposite > 0 else 1]
                if self.direction_is_traversable(preferred_angle, heading_deg):
                    self.push_direction = preferred
                elif self.direction_is_traversable(opposite_angle, heading_deg):
                    self.push_direction = opposite
                else:
                    self.push_direction = 0
                self.recovery_waiting = self.push_direction == 0
                self.get_logger().info(
                    f"RECOVERY START: stage={self.navigation_stage}, "
                    f"direction={self.recovery_direction_name(recovery_axis, self.push_direction)}"
                )
                if self.recovery_waiting:
                    self.get_logger().info("RECOVERY STOP: both directions blocked")
        else:
            # Hysteresis: do not leave recovery until APF forward speed has
            # recovered and the primary footprint corridor is clear.
            if primary_velocity >= self.exit_escape_threshold and primary_clear:
                self.push_active = False
                self.push_direction = 0
                self.recovery_waiting = False
                self.get_logger().info(
                    "RECOVERY SUCCESS: primary direction clear"
                )
            else:
                # No direction is latched only after both were blocked.  Choose
                # again once either fo# Total force = attraction + repulsionotprint corridor becomes traversable.
                if not self.push_direction:
                    preferred = (self.choose_y_push(heading_deg)
                                 if self.navigation_stage in (0, 2)
                                 else self.choose_x_push(heading_deg))
                    preferred_angle = recovery_angles[0 if preferred > 0 else 1]
                    opposite = -preferred
                    opposite_angle = recovery_angles[0 if opposite > 0 else 1]
                    if self.direction_is_traversable(preferred_angle, heading_deg):
                        self.push_direction = preferred
                    elif self.direction_is_traversable(opposite_angle, heading_deg):
                        self.push_direction = opposite
                current_angle = recovery_angles[0 if self.push_direction > 0 else 1]
                opposite_direction = -self.push_direction if self.push_direction else 1.0
                opposite_angle = recovery_angles[0 if opposite_direction > 0 else 1]
                current_clearance = (self.directional_clearance(current_angle, heading_deg)
                                     if self.push_direction else 0.0)
                current_clear = (self.push_direction != 0
                                 and self.direction_is_traversable(current_angle, heading_deg))
                opposite_clear = self.direction_is_traversable(opposite_angle, heading_deg)

                if current_clear:
                    if self.recovery_waiting:
                        self.recovery_waiting = False
                        self.get_logger().info(
                            f"RECOVERY CONTINUE: direction="
                            f"{self.recovery_direction_name(recovery_axis, self.push_direction)}, "
                            f"clearance={current_clearance:.2f}"
                        )
                elif opposite_clear:
                    if self.push_direction:
                        self.get_logger().info(
                            f"RECOVERY BLOCKED: direction="
                            f"{self.recovery_direction_name(recovery_axis, self.push_direction)}"
                        )
                    old_direction = self.push_direction
                    self.push_direction = opposite_direction
                    self.recovery_waiting = False
                    self.get_logger().info(
                        f"RECOVERY SWITCH: "
                        f"{self.recovery_direction_name(recovery_axis, old_direction)} -> "
                        f"{self.recovery_direction_name(recovery_axis, self.push_direction)}"
                    )
                elif not self.recovery_waiting:
                    if self.push_direction:
                        self.get_logger().info(
                            f"RECOVERY BLOCKED: direction="
                            f"{self.recovery_direction_name(recovery_axis, self.push_direction)}"
                        )
                    self.recovery_waiting = True
                    self.get_logger().info("RECOVERY STOP: both directions blocked")

        # Apply recovery velocity
        if self.push_active and not self.recovery_waiting:

            if self.navigation_stage in (0, 2):

                velocity_global_x = max(0.0, velocity_global_x)
                velocity_global_y = self.push_force * self.push_direction

            else:

                velocity_global_y = min(0.0, velocity_global_y)
                velocity_global_x = self.push_force * self.push_direction


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

        push_x = 0.0
        push_y = 0.0

        if self.push_active and not self.recovery_waiting:
            if self.navigation_stage in (0, 2):
                push_y = self.push_force * self.push_direction
            elif self.navigation_stage == 1:
                push_x = self.push_force * self.push_direction

        self.get_logger().info(
            f"APF stage={self.navigation_stage} | "
            f"pose=({px:.2f}, {py:.2f}, {heading_deg:.1f}) | "
            f"repulsive=({repulsive_x:.3f}, {repulsive_y:.3f}) | "
            f"push_active={self.push_active} | "
            f"push=({push_x:.3f}, {push_y:.3f}) | "
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
