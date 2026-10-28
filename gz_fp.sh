#!/bin/bash
source install/setup.bash
ros2 launch rb2301_gz fp_gazebo.launch.py x:=0.0 y:=0.0
# ros2 launch rb2301_gz fp_gazebo.launch.py x:=3.8 y:=-2.6 # Start past the gate

pkill -9 ruby # kills any lingering Gazebo