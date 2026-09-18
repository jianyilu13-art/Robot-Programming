import numpy as np
import xml.etree.ElementTree as ET
import os


height = 20
size_div = 5
width = 22
randomise = True

workspace_directory = os.path.dirname(os.path.realpath(__file__))[:-22]
overwrite_file = workspace_directory + '/rb2301_gz/worlds/obstacle_world_ca1.sdf'
obstacle_model = f'file:///{workspace_directory}/rb2301_gz/meshes/coke/6'


def generate_maze():
    maze_arr = np.zeros((height, width))

    centre_y = int(width / 2)

    # =========================================================
    # 1. FIXED L-SHAPE SURROUNDING THE ROBOT
    # =========================================================

    # Horizontal wall in front of the robot
    front_x = 4

    # Left wall beside the robot
    left_y = centre_y - 4

    # Long horizontal wall in FRONT of robot
    #
    # ● ● ● ● ● ● ● ● ● ●
    # ●
    # ●
    # ● ROBOT
    #
    for y in range(left_y, centre_y + 6):
        maze_arr[front_x, y] = 1

    # Left wall extending toward the robot
    for x in range(1, front_x + 1):
        maze_arr[x, left_y] = 1

    # =========================================================
    # 2. RANDOM OBSTACLES
    # =========================================================

    for x in range(2, height - 2):
        for y in range(1, width - 1):

            # Don't overwrite the fixed L-shape
            if maze_arr[x, y] == 1:
                continue

            # Keep the robot's immediate starting area clear
            if x <= 2 and centre_y - 2 <= y <= centre_y + 2:
                continue

            # Count how many obstacles are nearby
            chance = np.sum(
                maze_arr[x - 1:x + 2, y - 1:y + 2]
            )

            roll = np.random.random()

            # Randomly generate additional Coke obstacles
            if roll < 0.5 - 0.4 * 2**chance + x / (3 * height):
                maze_arr[x, y] = 1

    # =========================================================
    # 3. CLEAR ROBOT STARTING AREA
    # =========================================================

    maze_arr[:2, centre_y - 1:centre_y + 2] = 0

    return maze_arr


def add_coke_element(x, y, n):
    obstacle = ET.Element("include")

    uri = ET.Element("uri")
    uri.text = obstacle_model
    obstacle.append(uri)

    name = ET.Element("name")
    name.text = f'coke{n}'
    obstacle.append(name)

    pose = ET.Element("pose")
    pose.text = f'{x} {y} 0 0 0 0'
    obstacle.append(pose)

    return obstacle


def generate_sdf_file():
    if randomise:
        maze_arr = generate_maze()

    n = 1

    print("Generating L-shaped obstacle world with random obstacles...")

    tree = ET.parse(overwrite_file)
    root = tree.getroot()
    world = root[0]

    # =====================================================
    # REMOVE OLD COKE OBSTACLES
    # =====================================================

    for element in reversed(world):
        if element.tag == 'include':
            world.remove(element)

    tree.write(overwrite_file)

    # =====================================================
    # ADD NEW OBSTACLES
    # =====================================================

    for x in range(height):
        for y in range(width):

            if maze_arr[x, y] == 1:

                x_pos = x * 2 / size_div
                y_pos = (y - width / 2) / size_div

                world.append(
                    add_coke_element(
                        x_pos,
                        y_pos,
                        n
                    )
                )

                n += 1

    tree.write(overwrite_file)

    print(f"Generated {n - 1} obstacles.")


if __name__ == '__main__':
    generate_sdf_file()