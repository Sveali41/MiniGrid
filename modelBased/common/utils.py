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

class Visualization:
    def __init__(self, config=''):
        self.cfg = config

    def visualize_attention(self, obs, act, attentionWeight, obs_next, obs_pred, step_counter, size=(14, 10), shrink=1):
        channel, row, col = self.cfg.grid_shape
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
        obs_next = obs_next_temp[-1, 0, :, :].detach().cpu().numpy() * 10  # Convert tensor to numpy
        pred_direction_idx = round(obs_pred[-1, :].reshape(channel, row, col)[2, :, :].detach().cpu().numpy().max() * dir_ratio)
        obs_pred = np.round(obs_pred[-1, :].reshape(channel, row, col)[0, :, :].detach().cpu().numpy() * obj_ratio)  # Convert tensor to nump
        heat_map = attentionWeight[-1, :].reshape(row, col).detach().cpu().numpy()  # Convert tensor to numpy
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





