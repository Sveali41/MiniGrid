import hydra
from omegaconf import DictConfig, OmegaConf
import random
from collections import deque
import sys
sys.path.append('/home/siyao/project/rlPractice/MiniGrid')
from modelBased.common.utils import GENERATOR_PATH

# @hydra.main(version_base=None, config_path=GENERATOR_PATH / "conf", config_name="config")
@hydra.main(version_base=None, config_path=str(GENERATOR_PATH / "conf"), config_name="config")
def generate_random_map(cfg: DictConfig):
    params = cfg.data_generator
    width = params.map_width
    height = params.map_height
    num_keys = params.num_keys  # Default to 0 keys if not specified


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
    
    # Set W, E, and G to themselves in the color map
    for y in range(height):
        for x in range(width):
            if object_map[y][x] in ['W', 'E', 'G']:
                color_map[y][x] = object_map[y][x]
    
    # Place the goal (G) on a random empty space
    goal_x, goal_y = random.randint(1, width-2), random.randint(1, height-2)
    if object_map[goal_y][goal_x] == 'E':
        object_map[goal_y][goal_x] = 'G'
        color_map[goal_y][goal_x] = 'G'

    if num_keys > 0:
        # Set K and D to 'Y' in the color map
        key_door_color = 'Y'
        
        for _ in range(num_keys):
            key_pos = place_random(object_map, 'K', width, height)
            color_map[key_pos[1]][key_pos[0]] = key_door_color
            
            door_pos = place_random(object_map, 'D', width, height)
            color_map[door_pos[1]][door_pos[0]] = key_door_color

            print(object_map)
            print(color_map)
            return object_map, color_map
        
def place_random(map_data, obj_type, width, height):
    while True:
        x, y = random.randint(1, width-2), random.randint(1, height-2)
        if map_data[y][x] == 'E':  # Place the object only on empty spaces
            map_data[y][x] = obj_type
            return (x, y)

if __name__ == "__main__":
    # Example of usage
    # Generate a basic valid map
    object_map, color_map = generate_random_map()