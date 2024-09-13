import random
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
sys.path.append('/home/siyao/project/rlPractice/MiniGrid')
import json
from modelBased.common.utils import GENERATOR_PATH

def create_base_map(width, height):
    map = [['E' for _ in range(width)] for _ in range(height)]

    for row in range(height):
        for col in range(width):
            if row == 0 or row == height-1 or col == 0 or col == width-1:
                map[row][col] = 'W'
            elif col == round(1/3*(width)) or col == round(2/3*(width)):
                map[row][col] = 'W'
    return map


def empty_map(width, height):
    map = create_base_map(width, height)
    # room3（col9-10，row1-4）
    third_room_positions = [(row, col) for row in range(1,height-1) for col in range(round(2/3*(width))+1,width-1)]
    b_row, b_col = random.choice(third_room_positions)
    map[b_row][b_col] = 'B'
    # assert E on both wall as a exit
    possible_rows = [row for row in range(1,height-1)]
    row_4 = random.choice(possible_rows)
    map[row_4][round(1/3*(width))] = 'E'
    row_8 = random.choice(possible_rows)
    map[row_8][round(2/3*(width))] = 'E'
    return map

def map_with_door(width, height):
    map = create_base_map(width, height)
    # put B in room3
    third_room_positions = [(row, col) for row in range(1,height-1) for col in range(round(2/3*(width))+1,width-1)]
    b_row, b_col = random.choice(third_room_positions)
    map[b_row][b_col] = 'B'
    # put O（col4，row1-4）
    vertical_wall_positions = [(row, round(1/3*(width))) for row in range(1,height-1)]
    o_row, o_col = random.choice(vertical_wall_positions)
    map[o_row][o_col] = 'O'
    # assert E on both wall as a exit
    possible_rows = [row for row in range(1,height-1)]
    row_8 = random.choice(possible_rows)
    map[row_8][round(2/3*(width))] = 'E'
    return map

def map_with_key_door(width, height):
    map = create_base_map(width, height)
    # put B in room3
    third_room_positions = [(row, col) for row in range(1,height-1) for col in range(round(2/3*(width))+1,width-1)]
    b_row, b_col = random.choice(third_room_positions)
    map[b_row][b_col] = 'B'
    # put K in room1
    first_room_positions = [(row, col) for row in range(1,height-1) for col in range(1,round(1/3*(width))-1)]
    k_row, k_col = random.choice(first_room_positions)
    map[k_row][k_col] = 'K'
    # put D on vertical wall（col8，row1-4）
    vertical_wall_positions = [(row, round(2/3*(width))) for row in range(1,height-1)]
    d_row, d_col = random.choice(vertical_wall_positions)
    map[d_row][d_col] = 'D'
    # assert E on both wall as a exit
    possible_rows = [row for row in range(1,height-1)]
    row_4 = random.choice(possible_rows)
    map[row_4][round(1/3*(width))] = 'E'
    return map

def final_map(width, height):
    map = create_base_map(width, height)
    # put B in room3
    third_room_positions = [(row, col) for row in range(1,height-1) for col in range(round(2/3*(width))+1,width-1)]
    b_row, b_col = random.choice(third_room_positions)
    map[b_row][b_col] = 'B'
    # put K in room1
    first_room_positions = [(row, col) for row in range(1,height-1) for col in range(1,round(1/3*(width))-1)]
    k_row, k_col = random.choice(first_room_positions)
    map[k_row][k_col] = 'K'
    # put O（col4，row1-4）
    vertical_wall_positions = [(row, round(1/3*(width))) for row in range(1,height-1)]
    o_row, o_col = random.choice(vertical_wall_positions)
    map[o_row][o_col] = 'O'
    # put D on vertical wall（col8，row1-4）
    vertical_wall_positions = [(row, round(2/3*(width))) for row in range(1,height-1)]
    d_row, d_col = random.choice(vertical_wall_positions)
    map[d_row][d_col] = 'D'
    return map

def print_map(map):
    for row in map:
        print(''.join(row))

def format_maps(object_map):
    # Convert object_map and color_map into string format
    object_map_str = '\n'.join([''.join(row) for row in object_map])
    # Combine the two maps with a double newline separating them
    return object_map_str

@hydra.main(version_base=None, config_path=str(GENERATOR_PATH / "conf"), config_name="config")
def generate_map(cfg: DictConfig):
    params = cfg.data_generator
    width = params.map_width
    height = params.map_height
    num_episodes = params.num_episodes  # Number of episodes to generate
    output_file = params.output_file  # File to save the generated data
    # List to hold all generated maps
    episodes_list = []
    for _ in range(round(num_episodes/4)):
        object_map = empty_map(width, height)
        combined_map = format_maps(object_map)
        episodes_list.append(combined_map)
    for _ in range(round(num_episodes/4)):
        object_map = map_with_door(width, height)
        combined_map = format_maps(object_map)
        episodes_list.append(combined_map)
    for _ in range(round(num_episodes/4)):
        object_map = map_with_key_door(width, height)
        combined_map = format_maps(object_map)
        episodes_list.append(combined_map)
    for _ in range(round(num_episodes/4)):
        object_map = final_map(width, height)
        combined_map = format_maps(object_map)
        episodes_list.append(combined_map)
    # Save the list to the output file as a JSON array
    with open(output_file, 'w') as f:
        json.dump(episodes_list, f, indent=4)
    print(f"Data for {num_episodes} episodes saved to {output_file}")

if __name__ == "__main__":
    generate_map()