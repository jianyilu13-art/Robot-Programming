# RB2301 ROS2 Tutorial

This workspace contains a simple ROS 2 Python package for the RB2301 tutorial exercises. It is organized as a standard `ament_python` package and includes a basic tutorial node plus a second minimal ROS 2 executable.

## Workspace / package structure

```text
~/rb2301/
├── src/
│   └── rb2301_tutorial/
│       ├── package.xml
│       ├── setup.py
│       ├── setup.cfg
│       ├── resource/
│       ├── test/
│       └── rb2301_tutorial/
│           ├── __init__.py
│           ├── tutorial.py
│           └── fake.py
├── build/
├── install/
├── log/
├── .gitignore
└── README.md
```

## Software requirements

- ROS 2 Jazzy
- Python 3

## Build the workspace

```bash
cd ~/rb2301
colcon build
source install/setup.bash
```

## Run the executables

```bash
ros2 run rb2301_tutorial tut
ros2 run rb2301_tutorial fake
```

## Included ROS 2 nodes

- `tutorial.py` implements a ROS 2 node with a 0.5 s timer, a `/turtle1/pose` subscriber, a `/turtle1/cmd_vel` publisher, and a `/spawn` service client.
- `fake.py` is a minimal ROS 2 node created to satisfy the second executable requirement.

## Ignored files

The workspace ignores generated and editor files, including `build/`, `install/`, `log/`, `.vscode/`, and Python cache files (`__pycache__`).
