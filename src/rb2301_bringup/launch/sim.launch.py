import os
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.descriptions import ParameterValue
from launch.actions import SetEnvironmentVariable
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution, Command

def generate_launch_description():

    ld = LaunchDescription()
    pkg_rb2301_description = FindPackageShare('rb2301_description') 
    pkg_rb2301_bringup = FindPackageShare('rb2301_bringup') 
    pkg_ros_gz_sim = FindPackageShare('ros_gz_sim')

    arg_model = DeclareLaunchArgument(
        'model', 
        default_value='nanocar_description.urdf',
        description='Name of the URDF description to load'
    )
    path_model = PathJoinSubstitution([
        pkg_rb2301_description, 'urdf', LaunchConfiguration('model')
    ])
    ld.add_action(arg_model)

    # for loading the model
    env_gz = SetEnvironmentVariable("GAZEBO_MODEL_PATH", pkg_rb2301_description)
    ld.add_action(env_gz)
    env_lib_gl = SetEnvironmentVariable("LIBGL_ALWAYS_SOFTWARE", "0") # 1 for VBox Users, 0 for non-Vbox users
    ld.add_action(env_lib_gl) # FOR VBOX USERS

    # world
    arg_world = DeclareLaunchArgument(
        'world', 
        default_value='test.sdf',
        description='Name of the Gazebo world file to load. Must be in rb2301_bringup/worlds/'
    )
    path_world = PathJoinSubstitution([
        pkg_rb2301_bringup, 'worlds', LaunchConfiguration('world')
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

    # publishes the static tf betw. map and odom (bcos no localization)
    node_static_tf_publisher = Node(
        package = 'tf2_ros', 
        executable = 'static_transform_publisher',
        name = 'static_transform_publisher',
        arguments = ['0', '0', '0', '0', '0', '0', '/map', '/odom'])
    ld.add_action(node_static_tf_publisher)

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

    # Spawn the URDF model using the `/world/<world_name>/create` service
    spawn_urdf_node = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "nanocar",
            "-topic", "robot_description",
            "-x", "0.0", 
            "-y", "0.0", 
            "-z", "0.1", 
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
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry",
            "/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model",
            "/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V",
            "/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan",
            "/imu@sensor_msgs/msg/Imu@gz.msgs.IMU",
        ],
        output="screen",
        parameters=[
            {'use_sim_time': True},
        ]
    )
    ld.add_action(gz_bridge_node)

    # rviz
    # node_rviz = Node(
    #     package='rviz2',
    #     executable='rviz2',
    #     arguments=['-d', PathJoinSubstitution([pkg_rb2301_bringup, 'rviz', 'sim.rviz']),],
    #     output='screen'
    # )
    # ld.add_action(node_rviz)

    return ld
