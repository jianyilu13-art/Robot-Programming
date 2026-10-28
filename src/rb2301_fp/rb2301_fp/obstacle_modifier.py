import numpy as np
import xml.etree.ElementTree as ET
import os
from shutil import copy2

randomise_gate_position = True
randomise_zigzag_size = True
randomise_obstacle_speed = True

spawn_static_obstacles = True
randomise_static_obstacles = True

workspace_directory = os.path.dirname(os.path.realpath(__file__))[:-20]
overwrite_file = workspace_directory + '/rb2301_gz/worlds/obstacle_course_world_fp.sdf'
original_file = workspace_directory + '/rb2301_gz/worlds/obstacle_course_world_fp_original.sdf'

coke_model = '://rb2301_gz/meshes/coke/6'
drawer_model = '://rb2301_gz/meshes/drawer/2'
litterbin_model = '://rb2301_gz/meshes/litterbin/2'


def add_element(x, y, n, element_model):
    obstacle = ET.Element("include")
    uri = ET.Element("uri")
    uri.text = element_model
    obstacle.append(uri)
    name = ET.Element("name")
    name.text = f'static_obstacle{n}'
    obstacle.append(name)
    pose = ET.Element("pose")
    pose.text = f'{x} {y} 0 0 0 0'
    obstacle.append(pose)
    return obstacle

def modify_sdf_file():
    if os.path.exists(overwrite_file):
        os.remove(overwrite_file)
    copy2(original_file, overwrite_file)

    print(f"Randomising: Gate position={randomise_gate_position}, zigzag size:{randomise_zigzag_size}, obstacle speed:{randomise_obstacle_speed}")
    print(f"Spawn coke obstacles:{spawn_static_obstacles} and randomise coke positions:{randomise_static_obstacles}") 

    tree = ET.parse(overwrite_file)
    root = tree.getroot()
    world = root[0]

    if randomise_zigzag_size:
        x_offset = -np.random.random()*0.8 
        y_offset = -np.random.random()*0.6
    else:
        x_offset = 0
        y_offset = 0           

    gate_y_start = -np.random.random() * 0.5 - 1.4
    gate_y_end = gate_y_start - 1.6

    for element in reversed(world): # Check through all walls and edit offsets as needed
        if element.tag == 'include':
            pose = element[2].text.split(' ')

            # Edit gate positions
            if element[1].text == 'nist_maze_wall_120_configurable_left': 
                if randomise_gate_position:
                    element[2].text = f'{4.2+x_offset} {gate_y_start+y_offset} -0.7 0 0 1.57079633'
            elif element[1].text == 'nist_maze_wall_120_configurable_right':
                if randomise_gate_position:
                    element[2].text = f'{4.2+x_offset} {gate_y_end+y_offset} -0.7 0 0 1.57079633'

            # Edit wall sizes for zigzag
            elif randomise_zigzag_size and 'nist_maze_wall' in element[1].text:
                if float(pose[0]) > 1.0:
                    pose[0] = str(float(pose[0]) + x_offset)
                if float(pose[1]) < -2.6:
                    pose[1] = str(float(pose[1]) + y_offset) 
                element[2].text = ' '.join(pose)    

        # Edit obstacles
        elif element.tag == 'actor':
            trajectory = element.find('script').find('trajectory') # waypoints of this obstacle
            if randomise_obstacle_speed:
                random_time = np.random.random()*4.0 + 2.0 # Random half-period between [2, 5] seconds
                trajectory[1][0].text = str(random_time) # Edit first waypoint time
                trajectory[2][0].text = str(random_time*2) # 2nd waypoint time
                
            if randomise_zigzag_size:
                for waypoint_idx in range(3):
                    waypoint_pose = trajectory[waypoint_idx][1].text.split(' ')
                    waypoint_pose[0] = str(float(waypoint_pose[0]) + x_offset)
                    waypoint_pose[1] = str(float(waypoint_pose[1]) + y_offset)
                    trajectory[waypoint_idx][1].text = ' '.join(waypoint_pose)  

    if spawn_static_obstacles:
        static_obstacle_positions = np.array([
            [0.6, 0.0, 0],
            [2.4+x_offset, -0.6+y_offset, 0],
            [2.4+x_offset, -1.4+y_offset, 1],
            [2.4+x_offset, -2.4+y_offset, 2],
        ])

        if randomise_static_obstacles:
            for idx in range(len(static_obstacle_positions)):
                static_obstacle_positions[idx][0] += (np.random.random()-0.5)*0.8
                static_obstacle_positions[idx][1] += (np.random.random()-0.5)*0.6
                static_obstacle_positions[idx][2] = np.random.randint(3)
            
        for idx in range(len(static_obstacle_positions)):
            if static_obstacle_positions[idx][2] == 0:
                obstacle_model = coke_model
            elif static_obstacle_positions[idx][2] == 1:
                obstacle_model = drawer_model
            else:
                obstacle_model = litterbin_model
            print(f"Adding obstacle {obstacle_model[20:-2]} to position {static_obstacle_positions[idx][:2]}")
            world.append(add_element(static_obstacle_positions[idx][0], static_obstacle_positions[idx][1], idx, obstacle_model))
                    
    tree.write(overwrite_file)

if __name__ == '__main__':
    modify_sdf_file()