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
        self.state1_end_x = 1.6
        self.state2_end_y = -2.3

        # Stage 2 -> Stage 3
        self.dynamic_stage_start_x = 4.0  # Adjust to dynamic-obstacle entry

        # Stage 3 -> Stage 4 (finish)
        self.state3_end_x = 5.0


        self.k_att = 6
        self.k_rep = 1

        self.obstacle_influence_distance = 0.7
        self.minimum_distance = 0.1

        self.lateral_deadband = 0

        self.forward_velocity_limit = 0.45
        self.lateral_velocity_limit = 0.45


        # ==========================================
        # UPGRADED ESCAPE MODE PARAMETERS
        # STAGES 0, 1, 2 ONLY
        # ==========================================

        # Robot physical dimensions (metres)
        # Replace with measured dimensions
        self.robot_width = 0.1
        self.robot_length = 0.2
        self.safety_margin = 0.03

        # Actual forward progress monitoring
        self.progress_window = 3.0
        self.minimum_forward_progress = 0.3

        # Escape movement
        self.escape_speed = 0.2
        self.backtrack_speed = 0.2
        self.preferred_direction = -1.0
        self.escape_switch_count = 0
        self.max_escape_switches = 4

        # Clearance checking
        self.escape_lookahead = 0.50
        self.escape_side_lookahead = 0.1
        self.backtrack_lookahead = 0.10
        self.side_clearance_threshold = 0.30
        self.side_difference_threshold = 0.10
        self.wall_stop_distance = 0.10

        # Recovery stability
        self.forward_clear_duration = 0.30
        self.recovery_duration = 0.50
        # RECOVER movement
        self.recover_forward_speed = 0.20
        self.recover_lateral_gain = 0.10

        # Backtracking limits
        self.max_backtrack_distance = 1.50
        self.min_backtrack_distance = 0.20

        self.backtrack_lateral_multiplier = 1.4

        # Recovery state
        self.recovery_state = "NORMAL"
        self.recovery_state_start = None

        # Progress history
        self.progress_history = []

        # Direction exploration
        self.left_attempted = False
        self.right_attempted = False

        # Backtrack reference
        self.backtrack_start_pose = None
        self.backtrack_forward_lookahead = None

        # Forward-clear timer
        self.forward_clear_start = None

        # Existing detailed logging fields
        self.last_forward_progress = 0.0
        self.last_left_clear = False
        self.last_right_clear = False

        # ==========================================
        # STAGE 3: DYNAMIC OBSTACLE PARAMETERS
        # ==========================================

        self.dynamic_k_att = 7.0
        self.dynamic_k_rep = 4.0
        self.dynamic_influence_distance = 0.3

        self.dynamic_forward_limit = 0.8
        self.dynamic_lateral_limit = 0.6


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

        # Stage 0: Move along global +X
        if self.navigation_stage == 0 and x >= self.state1_end_x:
            next_stage = 1

        # Stage 1: Move along global -Y (slightly +X)
        elif self.navigation_stage == 1 and y <= self.state2_end_y:
            next_stage = 2

        # Stage 2 -> Stage 3: Continue along +X
        elif self.navigation_stage == 2 and x >= self.dynamic_stage_start_x:
            next_stage = 3

        # Stage 3 -> Stage 4: Finish along +X
        elif self.navigation_stage == 3 and x >= self.state3_end_x:
            next_stage = 4

        if next_stage != self.navigation_stage:
            previous_stage = self.navigation_stage
            self.navigation_stage = next_stage
            self.in_escape_mode = False

            # Reset upgraded recovery state
            self.recovery_state = "NORMAL"
            self.recovery_state_start = None

            self.progress_history.clear()

            self.left_attempted = False
            self.right_attempted = False

            self.backtrack_start_pose = None
            self.forward_clear_start = None
            self.escape_switch_count = 0
            self.backtrack_forward_lookahead = None

            self.get_logger().info(
                f"APF stage transition: {previous_stage} -> {next_stage} "
                f"at ({x:.2f}, {y:.2f})"
            )

    def get_stage_directions(self):
        """
        Return unit vectors for the stage's forward
        and lateral escape directions in global coordinates.
        """

        if self.navigation_stage in (0, 2, 3):
            # Forward +X; positive lateral +Y
            forward = np.array([1.0, 0.0])
            lateral = np.array([0.0, 1.0])
        else:
            # Stage 1: primarily -Y, slightly +X
            forward = np.array([0.1, -1.0])
            forward /= np.linalg.norm(forward)
            # Positive lateral is approximately +X
            lateral = np.array([-forward[1], forward[0]])

        return forward, lateral

    
    def check_forward_progress(self, px, py):
        """
        Check actual displacement along the stage's
        forward direction over a rolling time window.
        """

        now = self.get_clock().now().nanoseconds * 1e-9

        current_position = np.array([px, py], dtype=float)

        self.progress_history.append(
            (now, current_position.copy())
        )

        # Remove old samples while retaining a reference
        # sample at or just before the window boundary.
        cutoff = now - self.progress_window

        while (
            len(self.progress_history) > 2
            and self.progress_history[1][0] <= cutoff
        ):
            self.progress_history.pop(0)

        if len(self.progress_history) < 2:
            return False

        old_time, old_position = self.progress_history[0]

        # Wait until we have a complete monitoring window
        if now - old_time < self.progress_window:
            return False

        forward, _ = self.get_stage_directions()

        displacement = current_position - old_position

        forward_progress = float(
            np.dot(displacement, forward)
        )

        self.last_forward_progress = forward_progress

        return forward_progress < self.minimum_forward_progress


    def get_clearance(self, direction_global, travel_distance):
        """
        Check whether the robot can translate safely
        in a specified global direction.

        Uses a rectangular swept footprint.
        """

        if not hasattr(self, "last_scan") or self.pose is None:
            return False

        scan = np.asarray(self.last_scan, dtype=float)
        n = len(scan)

        if n == 0:
            return False

        direction = np.asarray(direction_global, dtype=float)
        norm = np.linalg.norm(direction)

        if norm < 1e-9:
            return False

        direction /= norm

        yaw = math.radians(float(self.pose[2]))

        # Global direction -> robot frame
        dx = math.cos(yaw) * direction[0] + math.sin(yaw) * direction[1]
        dy = -math.sin(yaw) * direction[0] + math.cos(yaw) * direction[1]

        # Robot footprint dimensions
        half_length = self.robot_length / 2.0 + self.safety_margin
        half_width = self.robot_width / 2.0 + self.safety_margin

        angles = np.linspace(
            0.0, 2.0 * math.pi, n, endpoint=False
        )

        # Unknown readings must be handled conservatively.
        # A sensor-specific scan-validity check is needed
        # before this is used on the real robot.
        valid = np.isfinite(scan) & (scan > 0.0)

        if not np.any(valid):
            return False

        # LiDAR points in robot coordinates
        points_x = scan[valid] * np.cos(angles[valid])
        points_y = scan[valid] * np.sin(angles[valid])

        # Sample the robot footprint along the intended movement
        steps = max(2, int(np.ceil(travel_distance / 0.02)) + 1)

        for distance in np.linspace(0.0, travel_distance, steps):

            center_x = dx * distance
            center_y = dy * distance

            collision = (
                (np.abs(points_x - center_x) <= half_length)
                &
                (np.abs(points_y - center_y) <= half_width)
            )

            if np.any(collision):
                return False

        return True
        
    def get_forward_distance(self):
        scan = np.asarray(self.last_scan, dtype=float)

        forward, _ = self.get_stage_directions()

        yaw = math.radians(float(self.pose[2]))

        angle = (
            math.atan2(forward[1], forward[0]) - yaw
        ) % (2.0 * math.pi)

        index = int(round(angle / (2.0 * math.pi) * len(scan))) % len(scan)

        return float(scan[index])

    def select_escape_direction(self):
        """
        Select lateral escape direction.

        +1: positive lateral direction
        -1: negative lateral direction
        0: neither direction available
        """

        forward, lateral = self.get_stage_directions()

        left_clear = self.get_clearance(
            lateral,
            self.escape_side_lookahead
        )

        right_clear = self.get_clearance(
            -lateral,
            self.escape_side_lookahead
        )

        self.last_left_clear = left_clear
        self.last_right_clear = right_clear

        if left_clear and not self.left_attempted:
            if not right_clear or self.right_attempted:
                return 1.0

        if right_clear and not self.right_attempted:
            if not left_clear or self.left_attempted:
                return -1.0

        if left_clear and right_clear:
            if not self.left_attempted and not self.right_attempted:
                return self.preferred_direction

        return 0.0



    def recovery_controller(self, px, py, velocity_global_x, velocity_global_y):

        now = self.get_clock().now().nanoseconds * 1e-9

        forward, lateral = self.get_stage_directions()

        position = np.array([px, py], dtype=float)

        # ==========================================
        # NORMAL
        # ==========================================

        if self.recovery_state == "NORMAL":

            stuck = self.check_forward_progress(px, py)

            if stuck:
                self.recovery_state = "SELECT"
                self.recovery_state_start = now
                self.progress_history.clear()
            
        # ==========================================
        # SELECT
        # ==========================================

        if self.recovery_state == "SELECT":

            direction = self.select_escape_direction()

            if direction != 0.0:

                if direction != self.preferred_direction:
                    self.escape_switch_count += 1

                self.preferred_direction = direction

                if self.escape_switch_count >= self.max_escape_switches:
                    self.recovery_state = "BACKTRACK"
                    self.backtrack_start_pose = position.copy()
                else:
                    self.recovery_state = "ESCAPE"

                self.recovery_state_start = now
                self.forward_clear_start = None

            else:

                self.recovery_state = "BACKTRACK"
                self.recovery_state_start = now
                self.backtrack_start_pose = position.copy()

        # ==========================================
        # ESCAPE
        # ==========================================

        if self.recovery_state == "ESCAPE":

            escape_direction = (
                lateral * self.preferred_direction
            )

            # Check whether forward passage is available
            forward_clear = self.get_clearance(
                forward,
                self.backtrack_forward_lookahead
                if self.backtrack_forward_lookahead is not None
                else self.escape_lookahead
            )

            if forward_clear:

                if self.forward_clear_start is None:
                    self.forward_clear_start = now

                elif (
                    now - self.forward_clear_start
                    >= self.forward_clear_duration
                ):

                    self.recovery_state = "RECOVER"
                    self.recovery_state_start = now
                    self.progress_history.clear()

            else:
                self.forward_clear_start = None

            if self.recovery_state == "ESCAPE":

                side_clear = self.get_clearance(
                    escape_direction,
                    self.escape_side_lookahead
                )

                if not side_clear:

                    if self.preferred_direction > 0:
                        self.left_attempted = True
                    else:
                        self.right_attempted = True

                    self.recovery_state = "SELECT"

                    return 0.0, 0.0

                velocity = escape_direction * self.escape_speed

                return float(velocity[0]), float(velocity[1])

        # ==========================================
        # BACKTRACK
        # ==========================================

        if self.recovery_state == "BACKTRACK":

            if self.backtrack_start_pose is None:
                self.backtrack_start_pose = position.copy()

            displacement = position - self.backtrack_start_pose

            retreat_distance = float(
                np.dot(displacement, -forward)
            )

            if retreat_distance >= self.max_backtrack_distance:
                self.recovery_state = "FAILED"
                return 0.0, 0.0

            # Check lateral space only after minimum backtracking
            if retreat_distance >= self.min_backtrack_distance:

                left_clear = self.get_clearance(
                    lateral,
                    self.escape_side_lookahead
                )

                right_clear = self.get_clearance(
                    -lateral,
                    self.escape_side_lookahead
                )

                if left_clear or right_clear:
                    forward_distance = self.get_forward_distance()

                    if np.isfinite(forward_distance) and forward_distance > 0:
                        self.backtrack_forward_lookahead = (
                            self.backtrack_lateral_multiplier * forward_distance
                        )

                        self.left_attempted = False
                        self.right_attempted = False
                        self.escape_switch_count = 0

                        self.recovery_state = "SELECT"
                        return 0.0, 0.0

            # Check backward clearance
            backward_clear = self.get_clearance(
                -forward,
                self.backtrack_lookahead
            )

            if not backward_clear:
                self.recovery_state = "FAILED"
                return 0.0, 0.0

            velocity = -forward * self.backtrack_speed

            return float(velocity[0]), float(velocity[1])

        # ==========================================
        # RECOVER
        # ==========================================

        if self.recovery_state == "RECOVER":

            # Resume existing APF command
            # while checking for sustained progress.

            if self.check_forward_progress(px, py):

                self.recovery_state = "SELECT"
                self.progress_history.clear()

                return 0.0, 0.0

            if now - self.recovery_state_start >= max(
                self.recovery_duration,
                self.progress_window
            ):

                self.recovery_state = "NORMAL"

                self.left_attempted = False
                self.right_attempted = False
                self.escape_switch_count = 0
                self.backtrack_forward_lookahead = None

                self.progress_history.clear()

            # Forward movement with small lateral APF correction
            forward, lateral = self.get_stage_directions()

            velocity = np.array([velocity_global_x, velocity_global_y])

            recover_velocity = (
                self.recover_forward_speed * forward
                + self.recover_lateral_gain * np.dot(velocity, lateral) * lateral
            )

            return float(recover_velocity[0]), float(recover_velocity[1])

        # ==========================================
        # FAILED
        # ==========================================

        if self.recovery_state == "FAILED":
            return 0.0, 0.0

        return velocity_global_x, velocity_global_y
    

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

        # ==========================================
        # DIRECTIONAL OBSTACLE WEIGHTING
        # ==========================================

        angle_deg = np.degrees(angles)

        def angular_difference(a, b):
            return (a - b + 180.0) % 360.0 - 180.0

        # Base weight: weak diagonal and rear repulsion
        direction_weight = np.full_like(angles, 0.2)

        # Stage 3 uses wider angular sensitivity
        if self.navigation_stage == 3:
            sigma = 25.0
        else:
            sigma = 3.0

        # Front: strongest
        direction_weight += 2.3 * np.exp(
            -0.5 * (angular_difference(angle_deg, 0.0) / sigma) ** 2
        )

        # Left: strong
        direction_weight += 1.8 * np.exp(
            -0.5 * (angular_difference(angle_deg, 90.0) / sigma) ** 2
        )

        # Right: strong
        direction_weight += 1.8 * np.exp(
            -0.5 * (angular_difference(angle_deg, 270.0) / sigma) ** 2
        )

        # ==========================================
        # REPULSIVE FORCE
        # ==========================================

        magnitudes = self.k_rep * direction_weight * (
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
        """APF for stages moving along global +X."""
        attractive = (self.k_att, 0.0)
        lateral = (
            0.0 if abs(repulsive_y) < self.lateral_deadband
            else repulsive_y
        )
        total = (attractive[0] + repulsive_x, attractive[1] + lateral)
        return attractive, total

    def ca1_y(self, repulsive_x, repulsive_y):
        """APF for stage moving along global -Y, slightly +X."""
        attractive = (self.k_att * 0.1, -self.k_att)
        lateral = (
            0.0 if abs(repulsive_x) < self.lateral_deadband
            else repulsive_x
        )
        total = (attractive[0] + lateral, attractive[1] + repulsive_y)
        return attractive, total

    def stage3_controller(self, heading_deg):
        """
        Stage 3: Dynamic obstacle avoidance.
        Stronger repulsion, longer detection range,
        and independent velocity limits.
        """

        if not hasattr(self, "last_scan"):
            return 0.0, 0.0

        # Save original APF parameters
        original_k_att = self.k_att
        original_k_rep = self.k_rep
        original_influence = self.obstacle_influence_distance

        try:
            # Apply Stage 3 parameters temporarily
            self.k_att = self.dynamic_k_att
            self.k_rep = self.dynamic_k_rep
            self.obstacle_influence_distance = (
                self.dynamic_influence_distance
            )

            # Calculate repulsive force using existing
            # Gaussian directional weighting
            repulsive_x, repulsive_y = (
                self.get_avoidance_velocity(heading_deg)
            )

            # Attraction toward global +X
            _, total = self.ca1_x(
                repulsive_x,
                repulsive_y
            )

        finally:
            # Restore original Stage 0-2 parameters
            self.k_att = original_k_att
            self.k_rep = original_k_rep
            self.obstacle_influence_distance = original_influence

        # ==========================================
        # GLOBAL VELOCITY LIMITS
        # ==========================================

        # Forward along global +X
        velocity_global_x = float(np.clip(
            total[0], 0.0, self.dynamic_forward_limit
        ))

        # Lateral along global +/-Y
        velocity_global_y = float(np.clip(
            total[1], -self.dynamic_lateral_limit,
            self.dynamic_lateral_limit
        ))

        # ==========================================
        # GLOBAL -> ROBOT FRAME
        # ==========================================

        yaw = math.radians(heading_deg)

        velocity_robot_x = (
            math.cos(yaw) * velocity_global_x
            + math.sin(yaw) * velocity_global_y
        )

        velocity_robot_y = (
            -math.sin(yaw) * velocity_global_x
            + math.cos(yaw) * velocity_global_y
        )

        return velocity_robot_x, velocity_robot_y



    def limit_apf_velocity(self, force_x, force_y):

        if self.navigation_stage in (0, 2):
            # Primary +X, lateral Y
            velocity_x = np.clip(
                force_x, -self.forward_velocity_limit, self.forward_velocity_limit
            )
            velocity_y = np.clip(
                force_y, -self.lateral_velocity_limit, self.lateral_velocity_limit
            )
        else:
            # Primary -Y, lateral X
            velocity_x = np.clip(
                force_x, -self.lateral_velocity_limit, self.lateral_velocity_limit
            )
            velocity_y = np.clip(
                force_y, -self.forward_velocity_limit, self.forward_velocity_limit
            )

        return float(velocity_x), float(velocity_y)



    def timer_callback(self):
        """Controller loop. Insert path planning and PID control logic here"""
        self.get_logger().debug("Timer callback")
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

        # ==========================================
        # STAGE 4: FINAL STOP
        # ==========================================

        if self.navigation_stage == 4:
            self.move_2D(0.0, 0.0, 0.0)
            return

        # ==========================================
        # STAGE 3: DYNAMIC OBSTACLE AVOIDANCE
        # ==========================================

        if self.navigation_stage == 3:

            vx, vy = self.stage3_controller(heading_deg)

            self.move_2D(
                x=vx,
                y=vy,
                turn=0.0
            )

            self.get_logger().info(
                f"STAGE 3 | "
                f"pose=({px:.2f}, {py:.2f}) | "
                f"command=({vx:.3f}, {vy:.3f})"
            )

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
        # UPGRADED LOCAL MINIMUM RECOVERY
        # ==========================================

        velocity_global_x, velocity_global_y = (
            self.recovery_controller(
                px,
                py,
                velocity_global_x,
                velocity_global_y
            )
        )

        self.in_escape_mode = (
            self.recovery_state != "NORMAL"
        )

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
            f"state={self.recovery_state} | "
            f"pose=({px:.2f}, {py:.2f}, {heading_deg:.1f}) | "
            f"forward_progress={self.last_forward_progress:.3f}m | "
            f"repulsive=({repulsive_x:.3f}, {repulsive_y:.3f}) | "
            f"left_clear={self.last_left_clear} | "
            f"right_clear={self.last_right_clear} | "
            f"preferred_direction={self.preferred_direction:+.1f} | "
            f"left_attempted={self.left_attempted} | "
            f"right_attempted={self.right_attempted} | "
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