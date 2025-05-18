import os
import sys
from pathlib import Path
from typing import Optional
import dotenv
from matplotlib import pyplot as plt
import numpy as np
import torch
from . import utilis_support
from typing import Dict

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
    if len(state.shape) == 3:
        dims = (2, 1 ,0)
    elif len(state.shape) == 4:
        dims = (0, 3, 2, 1)
    else:
        raise ValueError("Input must be a 3D or 4D array.")
    
    transpose_func = getattr(state, "permute", None) or getattr(state, "transpose", None)  # 如果有permute方法就用permute，否则用transpose
    if transpose_func:
        return transpose_func(*dims)
    else:
        raise TypeError("Input must be a PyTorch tensor or a NumPy array.")

def get_agent_position(state):
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    if len(state.shape) == 3:
        channel, row, col = state.shape
        agent_position_index = np.argmax(state[0, :, :])
        agent_position_yx = np.unravel_index(agent_position_index, (row, col))
        return agent_position_yx

    elif len(state.shape) == 4:
        B, channel, row, col = state.shape
        agent_position_index = np.argmax(state[:, 0, :, :].reshape(B, -1), axis=1)
        agent_position_yx_batch = np.stack(np.unravel_index(agent_position_index, (row, col)), axis=1)
        return agent_position_yx_batch
    else:
        raise ValueError("Input must be a 3D or 4D array.")

def extract_masked_state(state, mask_size, agent_position_yx):
    tensor_flag = False
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
        tensor_flag = True

    #区分带batch 和不带batch的情况
    if len(state.shape) == 3:
        state_masked = utilis_support.extract_masked_state_support(state, agent_position_yx , mask_size)

    if len(state.shape) == 4:
        B, channel, row, col = state.shape
        
        state_masked = np.zeros((B, channel, mask_size, mask_size), dtype=state.dtype) 
        for i in range(B):
            state_masked[i, :, :, :] = utilis_support.extract_masked_state_support(state[i], agent_position_yx[i], mask_size)

    if tensor_flag:
        state_masked = torch.from_numpy(state_masked).cuda()
    return state_masked

def put_back_masked_state(state_masked, orginal_state, mask_size, agent_position_yx):
    tensor_flag = False
    if isinstance(state_masked, torch.Tensor):
        state_masked = state_masked.detach().cpu().numpy()
        tensor_flag = True
    
    if isinstance(orginal_state, torch.Tensor):
        orginal_state = orginal_state.detach().cpu().numpy()
        tensor_flag = True

    if len(state_masked.shape) == 3:
        channels, rows, cols = orginal_state.shape
        y, x = agent_position_yx
        half = mask_size // 2

        src_slice_y = slice(max(y - half, 0), min(y + half + 1, rows))
        src_slice_x = slice(max(x - half, 0), min(x + half + 1, cols))

        dest_slice_y = slice(max(0, half - y), max(0, half - y) + (min(y + half + 1, rows) - max(y - half, 0)))
        dest_slice_x = slice(max(0, half - x), max(0, half - x) + (min(x + half + 1, cols) - max(x - half, 0)))
        orginal_state[:, src_slice_y, src_slice_x] = state_masked[:, dest_slice_y, dest_slice_x]



    if tensor_flag:
        orginal_state = torch.from_numpy(orginal_state).cuda()
    return orginal_state

def load_model_weight(model, weight_path, freeze=True):
    try:
        # 加载 checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(weight_path)
        state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()  # 切换到评估模式

        # 冻结参数
        if freeze:
            for param in model.parameters():
                param.requires_grad = False

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Weight file not found at {weight_path}")
    except KeyError as e:
        raise KeyError(f"Error: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error occurred: {e}")

def normalize_obs(x, obs_norm_values):
    if isinstance(x, np.ndarray):
        # NumPy array case
        if not np.issubdtype(x.dtype, np.floating):
            x = x.astype(np.float32)
    elif isinstance(x, torch.Tensor):
        # PyTorch tensor case
        if not torch.is_floating_point(x):
            x = x.to(torch.float32)
    else:
        raise TypeError("Input must be a NumPy array or PyTorch tensor.")
    
    # Normalize based on the dimensionality of x
    if x.ndim == 3:
        if obs_norm_values is None or len(obs_norm_values) != x.shape[0]:
            raise ValueError("Normalization values must be provided and must match the number of channels in the data.")
        # x is of shape (H, W, C)
        channel, row, col = x.shape
        for i in range(channel):
            max_val = obs_norm_values[i]
            if max_val != 0:  # Avoid division by zero
                x[i, :, :] /= max_val
        # Flatten the data into a 1D vector and convert to a torch tensor
        x = x.reshape(channel, row, col)
        
    elif x.ndim == 4:
        if obs_norm_values is None or len(obs_norm_values) != x.shape[1]:
            raise ValueError("Normalization values must be provided and must match the number of channels in the data.")
        # x is of shape (B, H, W, C)
        B, channel, row, col = x.shape
        for i in range(channel):
            max_val = obs_norm_values[i]
            if max_val != 0:
                x[:, i, :, :] /= max_val
        # Flatten each observation in the batch
        x = x.reshape(B, channel, row, col)
    else:
        raise ValueError("Input must be a 3D or 4D array.")
    
    return x

def denormalize_obj(x, obs_norm_values):    
    # Normalize based on the dimensionality of x
    if x.ndim == 3:
        if obs_norm_values is None or len(obs_norm_values) != x.shape[0]:
            raise ValueError("Normalization values must be provided and must match the number of channels in the data.")
        # x is of shape (H, W, C)
        channel, row, col = x.shape
        for i in range(channel):
            max_val = obs_norm_values[i]
            if max_val != 0:  # Avoid division by zero
                x[i, :, :] *= max_val
        # Flatten the data into a 1D vector and convert to a torch tensor
        x = x.reshape(channel, row, col)
        
    elif x.ndim == 4:
        if obs_norm_values is None or len(obs_norm_values) != x.shape[1]:
            raise ValueError("Normalization values must be provided and must match the number of channels in the data.")
        # x is of shape (B, H, W, C)
        B, channel, row, col = x.shape
        for i in range(channel):
            max_val = obs_norm_values[i]
            if max_val != 0:
                x[:, i, :, :] *= max_val
        # Flatten each observation in the batch
        x = x.reshape(B, channel, row, col)
    else:
        raise ValueError("Input must be a 3D or 4D array.")
    return x
    
def map_obs_to_nearest_value(obs_denorm, obj_values, color_values, state_values):
    obs_denorm[0, :, :] = utilis_support.map_to_nearest_value_support(obs_denorm[0, :, :], obj_values)
    obs_denorm[1, :, :] = utilis_support.map_to_nearest_value_support(obs_denorm[1, :, :], color_values)
    obs_denorm[2, :, :] = utilis_support.map_to_nearest_value_support(obs_denorm[2, :, :], state_values)
    return obs_denorm

class Visualization:
    def __init__(self, config=''):
        self.cfg = config
        if not os.path.exists(self.cfg.save_path):
            os.mkdir(self.cfg.save_path)
    

    def compare_states(self, obs, obs_next, act, step_counter=0, saveImage=False, size=(10, 4), shrink=0.5):
        if isinstance(obs, np.ndarray): 
            obs = torch.from_numpy(obs).cuda()
        if isinstance(obs_next, np.ndarray): 
            obs_next = torch.from_numpy(obs_next).cuda()
        plt.close()
        if obs.max() <= 1:
            dir_ratio, obj_ratio, act_ratio = 3, 10, 6
        else:
            dir_ratio, obj_ratio, act_ratio = 1, 1, 1

        state_image = obs[0, :, :].detach().cpu().numpy() * obj_ratio
        direction = self.cfg.direction_map[round(obs[2, :, :].detach().cpu().numpy().max() * dir_ratio)]

        state_image_next = obs_next[0, :, :].detach().cpu().numpy() * obj_ratio
        direction_next = self.cfg.direction_map[round(obs_next[2, :, :].detach().cpu().numpy().max() * dir_ratio)]
        if act is None:
            action = "None"
        else:
            action = self.cfg.action_map[round(act * act_ratio)]
    
        num_colors = 11
        custom_cmap = plt.cm.get_cmap('jet', num_colors)
        self._plot_subplot(1, 2, 1, state_image, custom_cmap, 'State', f"Dir: {direction}  Action: {action}", shrink)
        self._plot_subplot(1, 2, 2, state_image_next, custom_cmap, 'State Pre', f"Dir: {direction_next}", shrink)
        plt.tight_layout()
        if saveImage:
            save_file = os.path.join(self.cfg.save_path, f"Compare_{step_counter}.png")
            plt.savefig(save_file)
            plt.close()
        else:
            plt.show()

    def visualize_single_state(self, obs, act=None, shrink=1):
        if isinstance(obs, np.ndarray): 
            obs = torch.from_numpy(obs).cuda()

        plt.close()
        if obs.max() <= 1:
            dir_ratio, obj_ratio, act_ratio = 3, 10, 6
        else:
            dir_ratio, obj_ratio, act_ratio = 1, 1, 1

        state_image = obs[0, :, :].detach().cpu().numpy() * obj_ratio
        direction = self.cfg.direction_map[round(obs[2, :, :].detach().cpu().numpy().max() * dir_ratio)]
        if act is None:
            action = "None"
        else:
            action = self.cfg.action_map[round(act * act_ratio)]
    
        num_colors = 13
        custom_cmap = plt.cm.get_cmap('jet', num_colors)
        self._plot(state_image, custom_cmap, f"Dir: {direction}  Act: {action}", shrink)

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
        custom_cmap = plt.cm.get_cmap('jet', num_colors)
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
    
    def _plot(self, data, cmap, title, shrink):
        plt.imshow(data, cmap=cmap, interpolation='nearest')
        plt.colorbar(shrink=shrink, label=title)
        plt.title(title)
        plt.show()

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


def merge_data_dicts(d1: Dict[str, np.ndarray], d2: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    merged = {}
    for key in d1.keys():
        if key in d2:
            merged[key] = np.concatenate([d1[key], d2[key]], axis=0)
        else:
            merged[key] = d1[key]
    return merged


load_envs()
PROJECT_ROOT : Path = Path(get_env("PROJECT_ROOT"))
GENERATOR_PATH : Path = Path(get_env("GENERATOR_PATH"))
TRAINER_PATH : Path = Path(get_env("TRAINER_PATH"))
WORLD_MODEL_PATH = Path(get_env("WORLD_MODEL_PATH"))
sys.path.append(str(PROJECT_ROOT.resolve()))	




