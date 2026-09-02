#!/bin/bash
source install/setup.bash
ros2 launch rb2301_gz ca1_gazebo.launch.py
./kill.sh # kills any lingering Gazebo