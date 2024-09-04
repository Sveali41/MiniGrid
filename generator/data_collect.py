import hydra
from omegaconf import DictConfig, OmegaConf
import random
from collections import deque
import sys
sys.path.append('/home/siyao/project/rlPractice/MiniGrid')
import json
from modelBased.common.utils import GENERATOR_PATH
import os

@hydra.main(version_base=None, config_path=str(GENERATOR_PATH / "conf"), config_name="config")
def generate_map(cfg: DictConfig):
    params = cfg.data_generator
    width = params.map_width
    height = params.map_height
    num_keys = params.num_keys  # Default to 0 keys if not specified
    num_episodes = params.num_episodes  # Number of episodes to generate
    output_file = params.output_file  # File to save the generated data
    # List to hold all generated maps
    episodes_list = []
    for episode in range(num_episodes):
        object_map, color_map = generate_obj_map(width, height, num_keys)
        combined_map = format_maps(object_map, color_map)
        episodes_list.append(combined_map)
    # Save the list to the output file as a JSON array
    with open(output_file, 'w') as f:
        json.dump(episodes_list, f, indent=4)
    print(f"Data for {num_episodes} episodes saved to {output_file}")

def generate_obj_map(width, height, num_keys=0):
    # Initialize the object map randomly with walls (W) and empty spaces (E)
    object_map = [
        [
            'W' if random.choice([True, False]) else 'E' 
            for _ in range(width)
        ] 
        for _ in range(height)
    ]
    
    # Ensure the borders are walls (W)
    for x in range(width):
        object_map[0][x] = 'W'
        object_map[height-1][x] = 'W'
    for y in range(height):
        object_map[y][0] = 'W'
        object_map[y][width-1] = 'W'
    
    # Initialize the color map (same size as object map)
    color_map = [['' for _ in range(width)] for _ in range(height)]
    
    # Set W and E to themselves in the color map
    for y in range(height):
        for x in range(width):
            if object_map[y][x] in ['W', 'E']:
                color_map[y][x] = object_map[y][x]
    
    # Place the goal (G) on a random empty space
    goal_placed = False
    while not goal_placed:
        goal_x, goal_y = random.randint(1, width-2), random.randint(1, height-2)
        if object_map[goal_y][goal_x] == 'E':  # Place the goal only on empty spaces
            object_map[goal_y][goal_x] = 'G'
            color_map[goal_y][goal_x] = 'G'
            goal_placed = True

    if num_keys > 0:
        # Set K and D to 'Y' in the color map
        key_door_color = 'Y'
        
        for _ in range(num_keys):
            key_pos = place_random(object_map, 'K', width, height)
            color_map[key_pos[1]][key_pos[0]] = key_door_color
            
            door_pos = place_random(object_map, 'D', width, height)
            color_map[door_pos[1]][door_pos[0]] = key_door_color

    return object_map, color_map
    
def place_random(map_data, obj_type, width, height):
    while True:
        x, y = random.randint(1, width-2), random.randint(1, height-2)
        if map_data[y][x] == 'E':  # Place the object only on empty spaces
            map_data[y][x] = obj_type
            return (x, y)
        
def format_maps(object_map, color_map):
    # Convert object_map and color_map into string format
    object_map_str = '\n'.join([''.join(row) for row in object_map])
    color_map_str = '\n'.join([''.join(row) for row in color_map])
    
    # Combine the two maps with a double newline separating them
    return f"{object_map_str}\n\n{color_map_str}"

def load_existing_data(file_path):
    # Check if the file exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = []
    return existing_data

@hydra.main(version_base=None, config_path=str(GENERATOR_PATH / "conf"), config_name="config")
def add_new_maps(cfg: DictConfig):
    # Load the existing data from the file
    output_file = cfg.data_generator.output_file
    existing_data = load_existing_data(output_file)
    
    # Generate new data without keys and doors
    width = cfg.data_generator.map_width
    height = cfg.data_generator.map_height
    num_episodes = cfg.data_generator.num_episodes  # Number of new episodes to generate

    new_episodes_list = []
    for _ in range(num_episodes):
        object_map, color_map = generate_obj_map(width, height, num_keys=0)  # No keys or doors
        combined_map = format_maps(object_map, color_map)
        new_episodes_list.append(combined_map)
    
    # Merge existing data with new data
    updated_data = existing_data + new_episodes_list

    # Save the merged data back to the file (or another file if preferred)
    with open(output_file, 'w') as f:
        json.dump(updated_data, f, indent=4)
    
    print(f"Added {num_episodes} new episodes without keys to {output_file}")


if __name__ == "__main__":
    # Example of usage
    # Generate a basic valid map
    add_new_maps()
    # generate_map()
    pass
