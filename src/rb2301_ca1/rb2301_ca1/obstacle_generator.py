import numpy as np
import xml.etree.ElementTree as ET
import os

height = 20
size_div = 5
width = 22
randomise = True

workspace_directory = os.path.dirname(os.path.realpath(__file__))[:-22]
overwrite_file =  workspace_directory + '/rb2301_gz/worlds/obstacle_world_ca1.sdf'
obstacle_model = f'file:///{workspace_directory}/rb2301_gz/meshes/coke/6'

def generate_maze():
    maze_arr = np.zeros((height, width))
    maze_arr[2, int(width/2)] = 1
    for x in range(2, height-2):
        for y in range(1, width-1):
            chance = np.sum(maze_arr[x-1:x+2, y-1:y+2])
            roll = np.random.random()
            if roll < 0.5 - 0.4*2**chance + x/(3*height): # Tweak the chance of cans spawning here
                maze_arr[x,y] = 1

    maze_arr[:2, int(width/2)-1:int(width/2)+2] = 0 # Clear the starting area
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
        n=1

        print("Generating new obstacle world...")
        tree = ET.parse(overwrite_file)
        root = tree.getroot()
        world = root[0]
        for element in reversed(world): # Remove all coke obstacles
            if element.tag == 'include':
                world.remove(element)
        tree.write(overwrite_file)

        for x in range(height): # Add in new obstacles
            for y in range(width):
                if maze_arr[x,y] == 1:
                    x_pos = x*2/(size_div)
                    y_pos = (y-width/2)/size_div
                    world.append(add_coke_element(x_pos, y_pos, n))
                    n+=1

        tree.write(overwrite_file)

if __name__ == '__main__':
    generate_sdf_file()
