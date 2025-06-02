import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
from typing import List
import pickle
import torch
def save_dic(dict, save_path='dict.pkl'):
    with open(save_path, 'wb') as f:
        pickle.dump(dict, f)
        
def load_dic(save_path):
    with open(save_path, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

def replace_vector_value(grid, src_type='chart'):
    if src_type == 'value':
        mapping = {
            2: 'W',
            1: 'E',
            5: 'K',
            4: 'D',
            8: 'G'
        }
    else:
        mapping = {
            'W': 2,
            'E': 1,
            'K': 5,
            'D': 4,
            'G': 8
        }
    vectorized_replace = np.vectorize(lambda x: mapping.get(x, x)) 
    return vectorized_replace(grid)
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

def is_reachable(grid: List[List[str]]) -> bool:
    h, w = grid.shape
    # 查找目标点（'G'）的位置
    starts = np.argwhere(grid != 2)

    # BFS 遍历从所有起始点及目标可达性
    visited = set()
    queue = deque()

    # 将所有起始点加入队列
    queue.append(starts[0])

    # 标记目标是否可达
    reached_goal = False

    while queue:
        y, x = queue.popleft()

        # 如果当前位置是目标，标记为到达目标
        if grid[y, x] == 8:
            reached_goal = True

        # 如果已经访问过这个节点，跳过
        if (y, x) in visited:
            continue

        # 标记为已访问
        visited.add((y, x))

        # 4个方向扩展
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 1 <= ny < h-1 and 1 <= nx < w-1 and (ny, nx) not in visited:
                ntile = grid[ny, nx]
                if ntile != 2:  # 如果不是墙壁，继续扩展
                    queue.append((ny, nx))

    if not reached_goal:
        return False  # 如果目标不可达，返回 False

    # 检查所有非墙壁区域是否都被访问，确保没有封闭区域
    if len(visited) == len(starts):
        return True  # 没有封闭区域并且目标可达
    else:
        return False  # 存在封闭区域

def visualize_grid(grid, count=10, save_flag=False, save_path='', idx =''):

    if isinstance(grid, torch.Tensor):
        grid = grid.detach().cpu().numpy()

    singlet = True
    if save_flag:
        plt.ioff()
    if isinstance(grid, list) or isinstance(grid, np.ndarray):
        if isinstance(grid[0], list) or isinstance(grid[0], np.ndarray):
            if isinstance(grid[0][0], list) or isinstance(grid[0][0], np.ndarray):
                singlet = False
                if count > len(grid):
                    count = len(grid)
    if not np.issubdtype(grid.dtype, np.integer):
        grid = replace_vector_value(grid, 'chart')

    def show_grid(map, index=''):
        if save_flag:
            save_file_name = os.path.join(save_path, f'gen_{index}.png')
            plt.imshow(map, cmap='viridis')  # 显示图像
            plt.colorbar()
            plt.savefig(save_file_name)  # 保存图像
            plt.close()  # 关闭图像，避免显示
        else:
            plt.imshow(map, cmap='viridis')  # 显示图像
            plt.colorbar()
            plt.show()  # 显示图像
            
    if singlet:
        show_grid(grid, idx)
    else:
        
        for index in range(min(grid.shape[0],count)):
            plt.close()
            show_grid(grid[index,:,:], index)

def generate_path_branch(height, width, path_length=50, branch_num=5, branch_length=7):
    grid = 2 * np.ones((height, width), dtype=int)
    
    def valid(y, x):
        return 1 <= y < height - 1 and 1 <= x < width - 1
    path = []
    y, x = random.randint(1, height-2), random.randint(1, width-2)
    goal = y, x
    grid[y, x] = 1
    path.append((y, x))
    directions = [(-1,0), (1,0), (0,-1), (0,1)]

    # 主路径
    for _ in range(path_length):
        random.shuffle(directions)
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if valid(ny, nx) and grid[ny, nx] == 2:
                y, x = ny, nx
                grid[y, x] = 1
                path.append((y, x))
                break

    # 分支
    for _ in range(branch_num):
        if not path: break
        sy, sx = random.choice(path)
        for _ in range(branch_length):
            random.shuffle(directions)
            for dy, dx in directions:
                ny, nx = sy + dy, sx + dx
                if valid(ny, nx) and grid[ny, nx] == 2:
                    grid[ny, nx] = 1
                    sy, sx = ny, nx
                    path.append((sy, sx))
                    break
    grid[goal] = 8
    return grid

#region
# def generate_main_path(height, width, path_len):
#     grid = 2 * np.ones((height, width), np.uint8)
#     start = (random.randint(1, height - 2), random.randint(1, width - 2))
#     path = [start]
#     grid[start] = 1

#     directions = [(-1,0), (1,0), (0,-1), (0,1)]
#     for _ in range(path_len):
#         y, x = path[-1]
#         random.shuffle(directions)
#         for dy, dx in directions:
#             ny, nx = y + dy, x + dx
#             if 1 <= ny < height-1 and 1 <= nx < width-1 and grid[ny, nx] == 2:
#                 grid[ny, nx] = 1
#                 path.append((ny, nx))
#                 break  # 添加一格就 break，控制路径连贯性

#     # 随机在主路径周围扩张空地（支路）
#     for _ in range(path_len * 2):
#         y, x = random.choice(path)
#         random.shuffle(directions)
#         for dy, dx in directions:
#             ny, nx = y + dy, x + dx
#             if 1 <= ny < height -1  and 1 <= nx < width - 1 and grid[ny, nx] == 2:
#                 grid[ny, nx] = 1
#                 break
#     return grid
#endregion

def random_generate(height, width, wall_ratio=0.1, key_ratio=0.05, door_ratio=0.05):
    grid = 2 * np.ones((height, width), np.uint8)
    grid[1:-1, 1:-1] = 1
    goal_yx = (random.randint(1, height - 2), random.randint(1, width - 2))  # random G

    grid[goal_yx[0],goal_yx[1]] = 8 # add G
    empty_pos = np.argwhere(grid == 1)
    empty_count = len(empty_pos)
    
    # Number of each object to place
    num_keys = int(empty_count * key_ratio)
    num_doors = int(empty_count * door_ratio)
    num_walls = int(empty_count * wall_ratio)

    # Shuffle to randomize placement
    empty_pos = empty_pos[np.random.permutation(len(empty_pos))]

    # Place walls
    wall_positions = empty_pos[num_keys + num_doors : num_keys + num_doors + num_walls]
    for y, x in wall_positions:
        grid[y, x] = 2

    return grid
    
def generate_single_env(height, width, mode, wall_prob=0.3, key_prob=0.05, door_prob=0.05):
    if mode == 'random':
        grid = random_generate(height, width, wall_prob, key_prob, door_prob)
    else:
        path_len = int((height-2) * (width-2) * (1-wall_prob))
        branch_num = random.randint(height//2,width)
        branch_length = random.randint(height//3,height-2)
        grid = generate_path_branch(height, width, path_len,branch_num=branch_num, branch_length=branch_length) 
    return grid

# =========================
# 2. 行为描述子提取
# =========================
def compute_descriptors(grid):
    empty_ratio = np.mean(grid == 1)
    path_length = estimate_path_length(grid)
    norm_path = path_length / (grid.shape[0] + grid.shape[1])
    return np.array([empty_ratio, norm_path])

def estimate_path_length(grid):
    h, w = grid.shape
    empties = np.argwhere(grid == 1)
    if len(empties) < 2:
        return 0

    max_dist = 0
    for i in range(len(empties)):
        visited = np.zeros_like(grid, dtype=bool)
        q = deque([(tuple(empties[i]), 0)])
        visited[tuple(empties[i])] = True
        while q:
            (x, y), d = q.popleft()
            max_dist = max(max_dist, d)
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and grid[nx, ny] == 1 and not visited[nx, ny]:
                    visited[nx, ny] = True
                    q.append(((nx, ny), d + 1))
    return max_dist