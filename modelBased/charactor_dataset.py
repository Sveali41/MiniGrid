import numpy as np
import random
from minigrid_custom_env import CustomEnvFromFile
from path import Paths
import os
from minigrid.manual_control import ManualControl
import json


def generate_maps(w, h, num_keys=1, num_doors=1):
    # amount of wall is random
    num_walls = random.randint(5, w + 8)
    # Initialize object and color grids
    objects = np.full((h, w), 'E')  # Default: Empty
    colors = np.full((h, w), 'N')  # Default: No color

    # Set boundaries with walls
    objects[0, :] = objects[-1, :] = objects[:, 0] = objects[:, -1] = 'W'
    colors[objects == 'W'] = 'N'  # Walls have no color

    # Function to place items randomly ensuring they don't overlap
    def place_item(item, has_color=False):
        while True:
            x, y = random.randint(1, w - 2), random.randint(1, h - 2)
            if objects[y, x] == 'E':  # Only place in empty spaces
                objects[y, x] = item
                if has_color:
                    colors[y, x] = random.choice(['R', 'G', 'B', 'Y'])  # Assign a random color
                return x, y

    # Randomly place the goal
    place_item('G')

    # Place keys and doors with colors
    for _ in range(num_keys):
        place_item('K', has_color=True)
    for _ in range(num_doors):
        place_item('D', has_color=True)

    # Randomly place interior walls, which have no color
    for _ in range(num_walls):
        place_item('W')

    return objects, colors


def format_and_concatenate_maps(obj_map, color_map):
    # Convert numpy arrays to string format
    obj_map_str = '\n'.join(''.join(row) for row in obj_map)
    color_map_str = '\n'.join(''.join(row) for row in color_map)

    # Concatenate maps with two newlines as separator
    combined_map_str = obj_map_str + '\n\n' + color_map_str
    return combined_map_str


def save_map_to_file(map_string, filename):
    with open(filename, 'w') as file:
        file.write(map_string)


# Saving to a JSON file
def save_json(file_name, env_list):
    with open(file_name, 'w') as json_file:
        json.dump(env_list, json_file)


# Example usage
width, height = 10, 10
obj_map, color_map = generate_maps(width, height, num_keys=1, num_doors=1)
map_str = format_and_concatenate_maps(obj_map, color_map)
# visualize the map
path = Paths()
name_string = 'test.txt'
file_save = os.path.join(path.CHARACTOR_DATA, name_string)
save_map_to_file(map_str, file_save)
env = CustomEnvFromFile(txt_file_path=file_save, custom_mission="Find the key and open the door.",
                        render_mode="human")
env.reset()
manual_control = ManualControl(env)  # Allows manual control for testing and visualization
# manual_control.start()
# create the dataset for generator
env_list = list()

for i in range(200):
    width, height = 10, 10
    obj_map, color_map = generate_maps(width, height, num_keys=1, num_doors=1)
    map_str = format_and_concatenate_maps(obj_map, color_map)
    env_list.append(map_str)
name_string = 'training_data.json'
file_save = os.path.join(path.CHARACTOR_DATA, name_string)
save_json(file_save, env_list)
