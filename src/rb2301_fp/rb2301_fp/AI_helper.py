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
        # HARMONIC POTENTIAL FIELD PARAMETERS
        # ==========================================

        # Goal
        self.goal_x = 5.0
        self.goal_y = -2.6
        self.goal_tolerance = 0.18

        # Motion
        self.speed = 0.12

        # Occupancy grid
        self.resolution = 0.08

        self.map_x_min = -0.5
        self.map_x_max = 6.5
        self.map_y_min = -5.0
        self.map_y_max = 1.0

        self.nx = int(np.ceil(
            (self.map_x_max - self.map_x_min) / self.resolution
        )) + 1

        self.ny = int(np.ceil(
            (self.map_y_max - self.map_y_min) / self.resolution
        )) + 1

        self.occupancy = np.zeros(
            (self.ny, self.nx), dtype=bool
        )

        # Robot clearance
        self.robot_radius = 0.17
        self.safety_margin = 0.035

        # Harmonic field
        self.potential = None
        self.plan_period = 0.35
        self.last_plan_time = None
        self.harmonic_iterations = 500

        # LiDAR
        self.scan_angles = None
        self.scan_min = 0.05
        self.scan_max = 10.0

        # Debug
        self._ticks = 0

        self.xmin, self.xmax = self.map_x_min, self.map_x_max
        self.ymin, self.ymax = self.map_y_min, self.map_y_max

        self.log_odds = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.occupied_inflated = None

        self.hpf_ranges = None
        self.scan_stamp = None
        self.last_integrated_stamp = None

        self.hpf_scan_sub = self.create_subscription(
            LaserScan,
            "scan",
            self.hpf_scan_callback,
            2
        )

        # HPF mapping/planning state. Occupancy evidence distinguishes mapped
        # obstacles from the still-unobserved portion of the Laplace domain.
        self.known_evidence = 0.12
        self.max_mapping_range = 7.5
        self.plan_ray_stride = 4
        self.harmonic_tolerance = 1.0e-5
        self.last_solver_iterations = 0
        self.last_solver_delta = float("inf")
        self.last_goal_reachable = None
        self.last_plan_status = "waiting for map"
        self.last_stop_reason = "waiting for scan"

        # Smooth-field sampling and command filtering.  Commands are retained
        # in the world frame so robot heading does not alter filter history.
        self.sor_omega = 1.75
        self.minimum_gradient = 2.0e-4
        self.full_speed_gradient = 3.0e-3
        self.velocity_filter_alpha = 0.24
        self.reversal_filter_alpha = 0.10
        self.max_velocity_step = 0.018
        self.previous_world_velocity = np.zeros(2, dtype=float)
        self.initial_goal_distance = None
        self.best_goal_distance = float("inf")

        # Pose-source diagnostics and a QoS-tolerant /odom observer.  The
        # official subscription above is left untouched.  BEST_EFFORT can
        # receive from both reliable and best-effort odometry publishers.
        self.detected_scan_length = None
        self.original_simulation_detection = bool(is_simulation)
        self.selected_pose_topic = (
            "/odom" if self.is_simulation
            else f"/vrpn_mocap/bingda_{self.robot_number:03d}/pose"
        )
        self.odom_callback_count = 0
        self.last_pose_update_ns = None
        self.pose_diagnostic_ticks = 0
        self.pose_bootstrap_started_ns = None
        self.pose_bootstrap_active = False
        self.pose_bootstrap_done = False
        odom_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pose_diagnostic_odom_sub = self.create_subscription(
            Odometry, "odom", self.diagnostic_odom_callback, odom_qos
        )
        self.pose_diagnostic_timer = self.create_timer(0.1, self.pose_diagnostic_callback)
        self.get_logger().info(
            f"Pose diagnostics: detected simulation={self.is_simulation}, "
            f"selected pose topic={self.selected_pose_topic}"
        )

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
    # HARMONIC POTENTIAL FIELD HELPERS
    # ==========================================

    def hpf_scan_callback(self, msg):
        """Store LiDAR data and angles in the robot frame."""

        scan_length = len(msg.ranges)
        if self.detected_scan_length != scan_length:
            self.detected_scan_length = scan_length
            self.get_logger().info(
                f"LiDAR scan length={scan_length}; original automatic "
                f"simulation detection={self.original_simulation_detection}"
            )

        step = 1 if len(msg.ranges) <= 360 else 2

        self.hpf_ranges = np.asarray(
            msg.ranges, dtype=float
        )[::step]

        # laser_joint has rpy="0 0 3.1416" in nanocar_description.urdf.
        # Scan angles are therefore laser-frame angles and need this one (and
        # only this one) conversion into base_link/robot-frame angles.
        self.scan_angles = (
            msg.angle_min
            + np.arange(len(self.hpf_ranges))
            * msg.angle_increment * step
            + np.pi
        )

        self.scan_min = float(msg.range_min)
        self.scan_max = float(msg.range_max)

        self.scan_stamp = (
            msg.header.stamp.sec,
            msg.header.stamp.nanosec
        )

    def diagnostic_odom_callback(self, msg):
        """Observe /odom without changing the protected official callback."""
        self.odom_callback_count += 1
        self.last_pose_update_ns = self.get_clock().now().nanoseconds
        # Reuse the official pose conversion instead of duplicating it.
        self.odometer_callback(msg)
        if self.odom_callback_count == 1:
            self.get_logger().info(
                "Odometry callback received its first /odom message "
                "through the BEST_EFFORT diagnostic subscription"
            )

    def gazebo_odom_evidence(self):
        """Return True only when the ROS graph provides Gazebo evidence."""
        publishers = self.get_publishers_info_by_topic("/odom")
        bridge_publisher = any(
            "bridge" in getattr(info, "node_name", "").lower()
            for info in publishers
        )
        gazebo_scan = (
            self.detected_scan_length is not None
            and self.detected_scan_length <= 360
        )
        return bool(publishers) and (bridge_publisher or gazebo_scan)

    def pose_diagnostic_callback(self):
        """Report pose health and bootstrap stationary Gazebo odometry."""
        now_ns = self.get_clock().now().nanoseconds
        self.pose_diagnostic_ticks += 1

        gazebo_evidence = self.gazebo_odom_evidence()
        if gazebo_evidence and not self.is_simulation:
            # Runtime-only override: real hardware is not forced into Gazebo
            # mode unless an /odom publisher plus Gazebo evidence is present.
            self.is_simulation = True
            self.selected_pose_topic = "/odom (safe Gazebo override)"
            self.get_logger().warning(
                "Automatic mode selected hardware, but Gazebo /odom evidence "
                "was found; enabling the runtime simulation pose override"
            )

        if self.pose is not None and self.last_pose_update_ns is None:
            # The protected official subscriber received a pose first.
            self.last_pose_update_ns = now_ns
            self.get_logger().info("Official pose callback has executed")

        if self.pose_bootstrap_active:
            # End the one-cycle, 1 mm/s nudge even if /odom is still absent.
            self.move_2D(0.0, 0.0, 0.0)
            self.pose_bootstrap_active = False

        if self.pose is None and gazebo_evidence and not self.pose_bootstrap_done:
            if self.pose_bootstrap_started_ns is None:
                self.pose_bootstrap_started_ns = now_ns
            elif now_ns - self.pose_bootstrap_started_ns >= 1_000_000_000:
                # Gazebo's odometry publisher goes silent while stationary.
                # A single 0.1 s command at 1 mm/s moves at most 0.1 mm and
                # makes it publish a fresh pose for late subscribers.
                self.get_logger().warning(
                    "No stationary /odom sample received; issuing a one-cycle "
                    "0.001 m/s Gazebo-only pose bootstrap"
                )
                self.move_2D(0.001, 0.0, 0.0)
                self.pose_bootstrap_active = True
                self.pose_bootstrap_done = True

        if self.pose_diagnostic_ticks % 20 == 0:
            if self.last_pose_update_ns is None:
                pose_age = "never"
            else:
                pose_age = f"{max(0.0, (now_ns - self.last_pose_update_ns) / 1e9):.2f}s"
            self.get_logger().info(
                f"Pose status: scan_length={self.detected_scan_length}, "
                f"simulation={self.is_simulation}, topic={self.selected_pose_topic}, "
                f"odom_callbacks={self.odom_callback_count}, "
                f"time_since_pose={pose_age}"
            )


    def grid_cell(self, x, y):
        """Convert world coordinates into grid indices."""

        c = int(round((x - self.xmin) / self.resolution))
        r = int(round((y - self.ymin) / self.resolution))

        if 0 <= r < self.ny and 0 <= c < self.nx:
            return r, c

        return None


    def update_map(self, pose):
        """Update occupancy probabilities using LiDAR."""

        if self.scan_angles is None or self.hpf_ranges is None:
            return

        if self.scan_stamp == self.last_integrated_stamp:
            return

        self.last_integrated_stamp = self.scan_stamp

        px, py, heading_deg = pose
        heading = np.deg2rad(heading_deg)

        for distance, angle in zip(
            self.hpf_ranges[::self.plan_ray_stride],
            self.scan_angles[::self.plan_ray_stride]
        ):

            valid_hit = (
                np.isfinite(distance)
                and self.scan_min <= distance
                < self.scan_max - 0.05
            )

            ray_length = (
                float(distance)
                if valid_hit
                else min(self.scan_max, self.max_mapping_range)
            )

            if not np.isfinite(ray_length) or ray_length <= 0:
                continue

            ray_length = min(ray_length, self.max_mapping_range)

            world_angle = heading + angle

            # Mark free cells along each ray
            # Do not clear the endpoint of a hit: otherwise repeated rays
            # erase exactly the obstacle they are meant to reinforce.
            free_length = max(0.0, ray_length - (self.resolution if valid_hit else 0.0))
            for t in np.arange(
                0.0, free_length, self.resolution * 0.8
            ):

                cell = self.grid_cell(
                    px + t * np.cos(world_angle),
                    py + t * np.sin(world_angle)
                )

                if cell is not None:
                    r, c = cell

                    self.log_odds[r, c] = max(
                        -3.0,
                        self.log_odds[r, c] - 0.15
                    )

            # Mark detected obstacle
            if valid_hit and distance <= self.max_mapping_range:

                cell = self.grid_cell(
                    px + distance * np.cos(world_angle),
                    py + distance * np.sin(world_angle)
                )

                if cell is not None:
                    r, c = cell

                    self.log_odds[r, c] = min(
                        4.0,
                        self.log_odds[r, c] + 1.0
                    )


    def inflated_obstacles(self):
        """Inflate occupied cells by the robot clearance radius."""

        occupied = self.log_odds > 0.8

        radius_cells = int(np.ceil(
            (self.robot_radius + self.safety_margin)
            / self.resolution
        ))

        inflated = np.zeros_like(occupied)

        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):

                if dr*dr + dc*dc > radius_cells*radius_cells:
                    continue

                src_r0 = max(0, -dr)
                src_r1 = min(self.ny, self.ny-dr)

                src_c0 = max(0, -dc)
                src_c1 = min(self.nx, self.nx-dc)

                inflated[
                    src_r0+dr:src_r1+dr,
                    src_c0+dc:src_c1+dc
                ] |= occupied[
                    src_r0:src_r1,
                    src_c0:src_c1
                ]

        return inflated

    def goal_is_reachable(self, blocked, pose, goal):
        """Check 4-connected free-space reachability before trusting a field."""
        start = self.grid_cell(pose[0], pose[1])
        if start is None or blocked[start] or blocked[goal]:
            return False
        pending = [start]
        visited = np.zeros_like(blocked, dtype=bool)
        visited[start] = True
        while pending:
            r, c = pending.pop()
            if (r, c) == goal:
                return True
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (0 <= nr < self.ny and 0 <= nc < self.nx and
                        not visited[nr, nc] and not blocked[nr, nc]):
                    visited[nr, nc] = True
                    pending.append((nr, nc))
        return False


    def solve_harmonic(self, pose):
        """Solve Laplace's equation on the occupancy grid."""

        blocked = self.inflated_obstacles()

        self.occupied_inflated = blocked

        goal = self.grid_cell(
            self.goal_x,
            self.goal_y
        )

        if goal is None:
            self.last_goal_reachable = False
            self.last_plan_status = "goal is outside occupancy map"
            return None

        gr, gc = goal

        if blocked[gr, gc]:
            self.last_goal_reachable = False
            self.last_plan_status = "goal cell is occupied"
            return None

        self.last_goal_reachable = self.goal_is_reachable(blocked, pose, goal)
        if not self.last_goal_reachable:
            self.last_plan_status = "no collision-free grid path to goal"
            return None
        self.last_plan_status = "ok"

        # Initialize potential
        u = np.full(
            (self.ny, self.nx),
            0.5,
            dtype=np.float32
        )

        # Obstacle potential = 1
        u[blocked] = 1.0

        # Outer boundaries = 1
        u[0, :] = 1.0
        u[-1, :] = 1.0
        u[:, 0] = 1.0
        u[:, -1] = 1.0

        # Goal potential = 0
        goal_mask = np.zeros_like(blocked)

        goal_mask[
            max(1, gr-1):min(self.ny-1, gr+2),
            max(1, gc-1):min(self.nx-1, gc+2)
        ] = True

        goal_mask &= ~blocked

        u[goal_mask] = 0.0

        # Unknown cells are part of the Laplace domain, not fixed internal
        # boundaries.  Collision safety is still enforced from live LiDAR and
        # occupied cells.  Only obstacles, the goal, and the map perimeter are
        # Dirichlet boundaries, allowing goal influence to reach the robot.
        fixed = blocked | goal_mask
        fixed[0, :] = True
        fixed[-1, :] = True
        fixed[:, 0] = True
        fixed[:, -1] = True
        free_inner = ~fixed[1:-1, 1:-1]
        rr, cc = np.indices((self.ny - 2, self.nx - 2))

        # Red-black SOR solves the same discrete Laplace equation as Jacobi,
        # but converges far faster and provides an actual maximum update size.
        self.last_solver_delta = float("inf")
        self.last_solver_iterations = 0
        for iteration in range(self.harmonic_iterations):
            max_delta = 0.0
            for colour in (0, 1):
                avg = (
                    u[:-2, 1:-1]
                    + u[2:, 1:-1]
                    + u[1:-1, :-2]
                    + u[1:-1, 2:]
                ) * 0.25
                inner = u[1:-1, 1:-1]
                mask = free_inner & (((rr + cc) & 1) == colour)
                if np.any(mask):
                    change = self.sor_omega * (avg[mask] - inner[mask])
                    inner[mask] += change
                    max_delta = max(max_delta, float(np.max(np.abs(change))))

            self.last_solver_delta = max_delta
            self.last_solver_iterations = iteration + 1
            if max_delta < self.harmonic_tolerance:
                break

        return u

    def interpolated_harmonic_gradient(self, potential, x, y):
        """Bilinearly interpolate the world-frame gradient at a pose."""
        grad_y, grad_x = np.gradient(potential, self.resolution, self.resolution)
        cf = np.clip((x - self.xmin) / self.resolution, 0.0, self.nx - 1.000001)
        rf = np.clip((y - self.ymin) / self.resolution, 0.0, self.ny - 1.000001)
        c0, r0 = int(math.floor(cf)), int(math.floor(rf))
        c1, r1 = min(c0 + 1, self.nx - 1), min(r0 + 1, self.ny - 1)
        tx, ty = cf - c0, rf - r0

        def sample(field):
            top = (1.0 - tx) * field[r0, c0] + tx * field[r0, c1]
            bottom = (1.0 - tx) * field[r1, c0] + tx * field[r1, c1]
            return float((1.0 - ty) * top + ty * bottom)

        return sample(grad_x), sample(grad_y)

    def smooth_world_velocity(self, desired_x, desired_y):
        """Low-pass and rate-limit velocity, braking before a reversal."""
        desired = np.array((desired_x, desired_y), dtype=float)
        previous = self.previous_world_velocity
        reversing = np.dot(previous, desired) < 0.0
        alpha = self.reversal_filter_alpha if reversing else self.velocity_filter_alpha
        candidate = previous + alpha * (desired - previous)
        step = candidate - previous
        step_norm = float(np.hypot(step[0], step[1]))
        if step_norm > self.max_velocity_step:
            step *= self.max_velocity_step / step_norm
        smoothed = previous + step
        return float(smoothed[0]), float(smoothed[1])


    def safe_velocity(self, vx, vy, pose):
        """Reject movement when LiDAR detects an obstacle in its swept corridor."""

        if self.scan_angles is None or self.hpf_ranges is None:
            self.last_stop_reason = "no LiDAR data"
            return 0.0, 0.0

        speed = np.hypot(vx, vy)

        if speed < 1e-6:
            self.last_stop_reason = "zero desired velocity"
            return 0.0, 0.0

        ranges = self.hpf_ranges

        valid = (
            np.isfinite(ranges)
            & (ranges >= self.scan_min)
            & (ranges < self.scan_max)
        )

        xx = ranges[valid] * np.cos(
            self.scan_angles[valid]
        )

        yy = ranges[valid] * np.sin(
            self.scan_angles[valid]
        )

        ux = vx / speed
        uy = vy / speed

        def clear(candidate_x, candidate_y):
            candidate_speed = np.hypot(candidate_x, candidate_y)
            ux, uy = candidate_x / candidate_speed, candidate_y / candidate_speed
            candidate_along = xx * ux + yy * uy
            candidate_across = np.abs(-xx * uy + yy * ux)
            return not np.any((candidate_along > -0.03) &
                              (candidate_along < max(0.14, candidate_speed * 1.2)) &
                              (candidate_across < self.robot_radius + self.safety_margin))

        # A single return of (0, 0) used to latch the robot at every grazing
        # hit.  Preserve footprint protection but try small holonomic steering
        # deviations before declaring the corridor blocked.
        candidates = [(vx, vy)]
        for offset_deg in (-55, 55, -95, 95):
            a = math.radians(offset_deg)
            candidates.append((speed * (ux * math.cos(a) - uy * math.sin(a)),
                               speed * (ux * math.sin(a) + uy * math.cos(a))))
        for candidate_x, candidate_y in candidates:
            if clear(candidate_x, candidate_y):
                self.last_stop_reason = ""
                return candidate_x, candidate_y

        self.last_stop_reason = "LiDAR footprint corridor blocked"
        return 0.0, 0.0

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

        # ==========================================
        # 1. CHECK SENSOR DATA
        # ==========================================

        if not hasattr(self, "last_scan") or self.last_scan is None or self.scan_angles is None:
            self.last_stop_reason = "waiting for LiDAR scan"
            self.move_2D(0.0, 0.0, 0.0)
            return

        # ==========================================
        # 2. CURRENT POSE
        # ==========================================

        pose = (
            self.to_course()
            if not self.is_simulation
            else self.pose
        )

        px, py, heading_deg = pose
        goal_distance = float(np.hypot(px - self.goal_x, py - self.goal_y))
        if self.initial_goal_distance is None:
            self.initial_goal_distance = goal_distance
        self.best_goal_distance = min(self.best_goal_distance, goal_distance)

        # ==========================================
        # 3. GOAL CHECK
        # ==========================================

        if goal_distance <= self.goal_tolerance:

            self.last_stop_reason = "goal reached"
            self.previous_world_velocity[:] = 0.0
            self.move_2D(0.0, 0.0, 0.0)

            return

        # ==========================================
        # 4. UPDATE OCCUPANCY GRID
        # ==========================================

        self.update_map(pose)

        # ==========================================
        # 5. SOLVE HARMONIC FIELD
        # ==========================================

        now = self.get_clock().now().nanoseconds / 1e9

        if (
            self.last_plan_time is None
            or now - self.last_plan_time >= self.plan_period
        ):

            self.potential = self.solve_harmonic(pose)

            self.last_plan_time = now

        if self.potential is None:
            self.last_stop_reason = self.last_plan_status
            self.move_2D(0.0, 0.0, 0.0)
            return

        # ==========================================
        # 6. CURRENT GRID POSITION
        # ==========================================

        cell = self.grid_cell(px, py)

        if cell is None:
            self.last_stop_reason = "robot outside occupancy map"
            self.move_2D(0.0, 0.0, 0.0)
            return

        r, c = cell

        if not (
            1 <= r < self.ny-1
            and 1 <= c < self.nx-1
        ):
            self.last_stop_reason = "robot on map boundary"
            self.move_2D(0.0, 0.0, 0.0)
            return

        if self.occupied_inflated is not None and self.occupied_inflated[r, c]:
            # Do not blindly clear this cell: it may be a real wall inside the
            # footprint.  Logging it distinguishes a map/angle issue from a
            # legitimate collision-protection stop.
            self.last_stop_reason = "current cell is inside inflated obstacle"
            self.move_2D(0.0, 0.0, 0.0)
            return

        # ==========================================
        # 7. HARMONIC GRADIENT
        # ==========================================

        gx, gy = self.interpolated_harmonic_gradient(
            self.potential, px, py
        )

        norm = np.hypot(gx, gy)

        if norm < self.minimum_gradient:
            self.last_stop_reason = "harmonic gradient below numerical confidence"
            self.previous_world_velocity[:] = 0.0
            self.move_2D(0.0, 0.0, 0.0)
            return

        # ==========================================
        # 8. GLOBAL VELOCITY
        # ==========================================

        gradient_speed_scale = min(1.0, norm / self.full_speed_gradient)
        desired_wx = -self.speed * gradient_speed_scale * gx / norm
        desired_wy = -self.speed * gradient_speed_scale * gy / norm
        smoothed_wx, smoothed_wy = self.smooth_world_velocity(
            desired_wx, desired_wy
        )

        # ==========================================
        # 9. GLOBAL -> ROBOT FRAME
        # ==========================================

        yaw = np.deg2rad(heading_deg)

        vx = (
            np.cos(yaw) * smoothed_wx
            + np.sin(yaw) * smoothed_wy
        )

        vy = (
            -np.sin(yaw) * smoothed_wx
            + np.cos(yaw) * smoothed_wy
        )

        # ==========================================
        # 10. COLLISION CHECK
        # ==========================================

        vx, vy = self.safe_velocity(vx, vy, pose)

        # Feed the collision-checked command back into the world-frame filter;
        # a safety stop therefore resets momentum instead of fighting it.
        self.previous_world_velocity[:] = (
            np.cos(yaw) * vx - np.sin(yaw) * vy,
            np.sin(yaw) * vx + np.cos(yaw) * vy,
        )

        # ==========================================
        # 11. EXECUTE MOVEMENT
        # ==========================================

        self.move_2D(
            x=vx,
            y=vy,
            turn=0.0
        )

        # ==========================================
        # 12. DEBUG
        # ==========================================

        self._ticks += 1

        if self._ticks % 10 == 0:

            goal_cell = self.grid_cell(self.goal_x, self.goal_y)
            current_occupied = bool(self.occupied_inflated[r, c]) if self.occupied_inflated is not None else False
            goal_occupied = (bool(self.occupied_inflated[goal_cell])
                             if self.occupied_inflated is not None and goal_cell is not None else True)

            self.get_logger().info(
                f"HPF Pose: ({px:.2f}, {py:.2f}, {heading_deg:.1f}) | "
                f"cell=({r},{c}) occupied={current_occupied} | "
                f"goal_cell={goal_cell} occupied={goal_occupied} | "
                f"interp_gradient=({gx:.5f}, {gy:.5f}) mag={norm:.5f} | "
                f"desired_world=({desired_wx:.3f}, {desired_wy:.3f}) | "
                f"smoothed_world=({smoothed_wx:.3f}, {smoothed_wy:.3f}) | "
                f"final_robot=({vx:.3f}, {vy:.3f}) | "
                f"goal_distance={goal_distance:.2f}m "
                f"progress={self.initial_goal_distance - goal_distance:+.2f}m "
                f"best={self.initial_goal_distance - self.best_goal_distance:+.2f}m | "
                f"stop={self.last_stop_reason or 'none'} | "
                f"Laplace={self.last_solver_iterations} iter, "
                f"residual={self.last_solver_delta:.2e}"
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
