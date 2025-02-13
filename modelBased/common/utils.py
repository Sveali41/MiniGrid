import os
from pathlib import Path
from typing import Optional
import dotenv
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os

def get_env(env_name: str, default: Optional[str] = None) -> str:
    if env_name not in os.environ:
        if default is None:
            raise KeyError(f"{env_name} not defined and no default value is present!")
        return default

    env_value: str = os.environ[env_name]
    if not env_value:
        if default is None:
            raise ValueError(
                f"{env_name} has yet to be configured and no default value is present!"
            )
        return default

    return env_value

def load_envs(env_file: Optional[str] = '.env') -> None:
    dotenv.load_dotenv(dotenv_path=env_file, override=True)

load_envs()

PROJECT_ROOT : Path = Path(get_env("PROJECT_ROOT"))
GENERATOR_PATH : Path = Path(get_env("GENERATOR_PATH"))
TRAINER_PATH : Path = Path(get_env("TRAINER_PATH"))


def replace_values(arr, old_values, new_values):
    assert arr.ndim >= 2 and len(old_values) == len(new_values)
    mapping = np.arange(256, dtype=arr.dtype)
    mapping[np.array(old_values, dtype=arr.dtype)] = np.array(new_values, dtype=arr.dtype)
    arr[:, :] = np.take(mapping, arr[:, :])
    return arr

def create_mask(state_shape, agent_position, mask_size):
    mask = np.zeros((state_shape[0], state_shape[1]), dtype=bool)
    y, x = agent_position
    half_size = mask_size // 2
    y_start, y_end = max(0, y - half_size), min(state_shape[0], y + half_size + 1)
    x_start, x_end = max(0, x - half_size), min(state_shape[1], x + half_size + 1)
    mask[y_start:y_end, x_start:x_end] = True
    return mask

def ColRowCanl_to_CanlRowCol(state):
    state = state.transpose(0, 3, 2, 1)
    return state

def extract_masked_state(state, agent_position_yx, mask_size):
    channels, rows, cols = state.shape
    y, x = agent_position_yx
    half = mask_size // 2
    margin_data = state[:, 0, 0]
    region = np.tile(margin_data.reshape(channels, 1, 1),
                     (1, mask_size, mask_size))

    src_slice_y = slice(max(y - half, 0), min(y + half + 1, rows))
    src_slice_x = slice(max(x - half, 0), min(x + half + 1, cols))

    dest_slice_y = slice(max(0, half - y), max(0, half - y) + (min(y + half + 1, rows) - max(y - half, 0)))
    dest_slice_x = slice(max(0, half - x), max(0, half - x) + (min(x + half + 1, cols) - max(x - half, 0)))

    # 将 state 中的有效区域复制到预填充区域中
    region[:, dest_slice_y, dest_slice_x] = state[:, src_slice_y, src_slice_x]
    return region

    
def visualize_attention_2d(attn_weights, sample_idx=0, query_idx=0):
    """
    可视化指定样本中某个 query token 的注意力分布，
    将 25 维注意力向量 reshape 成 5×5 的网格。

    参数：
      attn_weights: Tensor，形状 (B, 25, 25)，例如来自模型输出的注意力权重
      sample_idx: 选择 batch 中的样本索引
      query_idx: 选择某个 query token 的索引（0~24），
                 该索引可映射为 5×5 区域中的 (row, col) = (query_idx//5, query_idx%5)
    """
    # 取出指定样本和 query token 对应的注意力向量，形状 (25,)
    attn_vector = attn_weights[sample_idx, query_idx]  # 例如 shape: (25,)
    
    # reshape 为 5×5 矩阵
    attn_grid = attn_vector.reshape(5, 5).detach().cpu().numpy()
    
    plt.figure(figsize=(6, 5))
    plt.imshow(attn_grid, cmap='viridis')
    plt.colorbar()
    plt.title(f'Attention (Sample {sample_idx}, Query token {query_idx}\nGrid loc: ({query_idx//5}, {query_idx%5}))')
    plt.xlabel('Key Token Grid Column')
    plt.ylabel('Key Token Grid Row')
    plt.show()

class Visualization:
    def __init__(self, config=''):
        self.cfg = config

    def visualize_attention(self, obs, act, attentionWeight, obs_next, obs_pred, step_counter, size=(14, 10), shrink=1):
        
        mask_size = self.cfg.attention_mask_size
        channel, row, col = 3, mask_size, mask_size
        # Preprocess data
        obs_next_temp = obs_next.view(obs_next.shape[0], channel, row, col) 

        if obs.max() <= 1:
            dir_ratio, obj_ratio, act_ratio = 3, 10, 6
        else:
            dir_ratio, obj_ratio, act_ratio = 1, 1, 1

        state_image = obs[-1, 0, :, :].detach().cpu().numpy() * obj_ratio  # Convert tensor to numpy
        direction = self.cfg.direction_map[round(obs[-1, 2, :, :].detach().cpu().numpy().max() * dir_ratio)]
        action = self.cfg.action_map[round(act[-1].item() * act_ratio)]
        next_direction = self.cfg.direction_map[round(obs_next_temp[-1, 2, :, :].detach().cpu().numpy().max() * dir_ratio)]
        obs_next = obs_next_temp[-1, 0, :, :].detach().cpu().numpy() * obj_ratio  # Convert tensor to numpy
        pred_direction_idx = round(obs_pred[-1, :].reshape(channel, row, col)[2, :, :].detach().cpu().numpy().max() * dir_ratio)
        obs_pred = np.round(obs_pred[-1, :].reshape(channel, row, col)[0, :, :].detach().cpu().numpy() * obj_ratio)  # Convert tensor to nump
        if len(attentionWeight.shape) == 2:
            heat_map = attentionWeight[-1, :].reshape(row, col).detach().cpu().numpy()  # Convert tensor to numpy
        elif len(attentionWeight.shape) == 3:
            heat_map = attentionWeight[-1, (mask_size**2)//2, :].reshape(row, col).detach().cpu().numpy()
        else:
            raise ValueError("Attention weight shape is not supported.")
        pre_direction = self.cfg.direction_map.get(pred_direction_idx, "Unknown")

        
        # Set up colormap for states
        num_colors = 13
        custom_cmap = plt.cm.get_cmap('gray', num_colors)
        plt.figure(figsize=size)
        self._plot_subplot(2, 2, 1, state_image, custom_cmap, 'State', f"State  Dir: {direction}  Action: {action}", shrink)
        self._plot_subplot(2, 2, 3, heat_map, 'viridis', 'Attention', "Attention Heatmap", shrink)
        self._plot_subplot(2, 2, 2, obs_next, custom_cmap, 'Next State', f"Next State  Dir:{next_direction}", shrink)
        self._plot_subplot(2, 2, 4, obs_pred, custom_cmap, 'Predicted', f"Pre State  Dir: {pre_direction}", shrink)
        plt.tight_layout()

        # Save plot
        if not os.path.exists(self.cfg.save_path):
            os.mkdir(self.cfg.save_path)
        save_file = os.path.join(self.cfg.save_path, f"Attention_{step_counter}.png")
        plt.savefig(save_file)
        plt.close()

    def _plot_subplot(self, row, col, position, data, cmap, colorbar_label, title, shrink):
        plt.subplot(row, col, position)
        im = plt.imshow(data, cmap=cmap, interpolation='nearest')
        plt.colorbar(im, shrink=shrink, label=colorbar_label)
        plt.title(title)





