from typing import List
import numpy as np
import os 
import sys
ROOTPATH = os.path.abspath(os.path.join(__file__, '..'))
sys.path.append(ROOTPATH)
from data.env_dataset_support import *
import random

# 生成可达的地图，保证每个生成的地图都可到达终点
def generate_envs_dataset(rows, cols, num_maps, wall_p_range=(0.1, 0.5), door_p_range=(0, 0), key_p_range=(0,0), max_len = 1e7,random_gen_max=2e4, save_flag=False, save_path=None):
    all_map = True  #  True: Feature can be repearted, False: no repeat
    index = 0
    archive = {}
    while max_len > 0:  # 如果最大尝试次数超过1500000还没有积累够num_maps 就停止
        print(f'gen minigrid remain attempt: {max_len}')
        wall_p = random.uniform(wall_p_range[0], wall_p_range[1])
        door_p = random.uniform(door_p_range[0], door_p_range[1])
        key_p = random.uniform(door_p, key_p_range[1])
        try_again = True
        gen_times = 0 
        while try_again:
            gen_times += 1
            max_len -= 1
            if gen_times > random_gen_max:  # 如果随机生成超过15000次还是不可达，就用使用路径生成的方法
                grid = generate_single_env(rows, cols, 'path', wall_prob=wall_p, key_prob=key_p, door_prob=key_p)
                try_again = False
            else:                  # 默认使用随机生成迷宫策略
                grid = generate_single_env(rows, cols, 'random', wall_prob=wall_p, key_prob=key_p, door_prob=key_p)
                if is_reachable(grid):
                    try_again = False
        if all_map:  
            archive[index] = grid
            if save_flag:
                visualize_grid(grid, save_flag=True,save_path=save_path, idx=f'map_{index}')  # 这里的路径要主动修改
            index += 1
        else:
            # 特征描述
            desc = compute_descriptors(grid)
            bd1 = min(num_maps - 1, int(desc[0] * num_maps))
            bd2 = min(num_maps - 1, int(desc[1] * num_maps))
            key = (bd1, bd2)
            if key not in archive:
                archive[key] = grid
                if save_flag:
                    visualize_grid(grid, save_flag=True,save_path=save_path, idx=f'{bd1}_{bd2}')  # 这里的路径要主动修改
        if len(archive) >= num_maps:  # 如果积累够， 提前跳出
            break

    return archive
        

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    rows, cols = 12, 12         # grid 尺寸
    num_maps = 5000                  # grid数量（可达的数量，内部会自动校验）
    wall_p_range = (0.1, 0.5)   # walls的概率最小值最大值
    door_p_range = (0, 0)       # doors的概率最小值最大值
    key_p_range = (0, 0)        # keys的概率最小值最大值
    archive = generate_envs_dataset(rows, cols, num_maps,wall_p_range=wall_p_range, door_p_range=door_p_range, key_p_range=key_p_range)
    save_dic(archive, '/home/siyao/project/rlPractice/MiniGrid/generator/data/grid5000.pkl')  # 这里的路径要主动修改
    pass    
    
    

