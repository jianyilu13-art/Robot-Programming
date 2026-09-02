from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch.actions import SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution, Command

import sys, os
sys.path.insert(2, os.path.dirname(os.path.realpath(__file__))[:-17]+'/rb2301_ca1/rb2301_ca1')
print(os.path.dirname(os.path.realpath(__file__))[:-17])
import obstacle_generator 

def generate_launch_description():
    obstacle_generator.generate_sdf_file()
    ld = LaunchDescription()
    pkg_rb2301_gz = FindPackageShare('rb2301_gz') 
    pkg_ros_gz_sim = FindPackageShare('ros_gz_sim')

    arg_model = DeclareLaunchArgument(
        'model', 
        default_value='nanocar_description.urdf',
        description='Name of the URDF description to load'
    )

    path_model = PathJoinSubstitution([
        pkg_rb2301_gz, 'urdf', LaunchConfiguration('model')
    ])

    ld.add_action(arg_model)

    # for loading the model
    env_gz = SetEnvironmentVariable("GAZEBO_MODEL_PATH", pkg_rb2301_gz)
    ld.add_action(env_gz)
    env_lib_gl = SetEnvironmentVariable("LIBGL_ALWAYS_SOFTWARE", "0") # 1 for VBox Users, 0 for non-Vbox users
    ld.add_action(env_lib_gl) # FOR VBOX USERS

    # world
    arg_world = DeclareLaunchArgument(
        'world', 
        # default_value='empty.sdf',
        default_value='obstacle_world_ca1.sdf', # RB2301 CA1 Obstacles
        description='Name of the Gazebo world file to load'
    )
    path_world = PathJoinSubstitution([
        pkg_rb2301_gz, 'worlds', LaunchConfiguration('world')
    ])
    ld.add_action(arg_world)

    # publishes the robot states into robot_description topic, along with transforms.
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'robot_description': ParameterValue(Command(['xacro ', path_model]), value_type=str), # Parameter Value required to wrap around xacro (if file accidentally contains colons).
             'use_sim_time': True},
        ],
    )
    ld.add_action(robot_state_publisher_node)

    # launch Gz Harmonic
    launch_gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py'
            ])
        ]),
        launch_arguments={
            'gz_args': [
                path_world,
                TextSubstitution(text=' -r -v -v1'), # for non-VBox users
                # TextSubstitution(text=' -r -v -v1 --render-engine ogre'), # DO NOT USE: this may cause the last reading for VBox users to become 0.05. -r for autorun, -v for verbose, v1 for level 1 verbose.
            ], 
        }.items()
    )
    ld.add_action(launch_gz_sim)
    
    x_arg = DeclareLaunchArgument('x', default_value='0.0', description='Initial x-coordinates within maze')
    y_arg = DeclareLaunchArgument('y', default_value='0.0', description='Initial y-coordinates within maze')
    yaw_arg = DeclareLaunchArgument('yaw', default_value='0.0', description='Initial yaw rotation (in radians) within maze' )
    ld.add_action(x_arg)
    ld.add_action(y_arg)
    ld.add_action(yaw_arg)
    x, y, yaw = LaunchConfiguration('x'), LaunchConfiguration('y'), LaunchConfiguration('yaw') 
                
    # Spawn the URDF model using the `/world/<world_name>/create` service
    spawn_urdf_node = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "nanocar",
            "-topic", "robot_description",
            "-x", x, 
            "-y", y, 
            "-z", "0.03", 
            "-Y", yaw, 
        ],
        output="screen",
        parameters=[
            {'use_sim_time': True},
        ]
    )
    ld.add_action(spawn_urdf_node)

    # Node to bridge messages like /cmd_vel and /odom
    gz_bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clockgz.msgs.Clock",
            "/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry",
            "/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model",
            "/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V",
            "/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan",
            "/imu@sensor_msgs/msg/Imu@gz.msgs.IMU",
            "/world/empty/dynamic_pose/info@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V", # This topic publishes ground-truth pose of the gz sim actors
        ],
        output="screen",
        parameters=[
            {'use_sim_time': True},
        ]
    )
    ld.add_action(gz_bridge_node)

    return ld


