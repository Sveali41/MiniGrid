import hydra
import sys
sys.path.append('/home/siyao/project/rlPractice/MiniGrid')
from modelBased.common.utils import GENERATOR_PATH
from omegaconf import DictConfig
from generator.common.utils import load_gen, generate_color_map, generate_obj_map, layout_to_string, combine_maps, clean_and_place_goal
from generator.gen import GAN
from minigrid_custom_env import *
import textwrap
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper
import torch
import os
from modelBased.data_collect import *
from modelBased.data.datamodule import *
# from generator.data.env_dataset_support import generate_valid_minigrid_with_key_door
from matplotlib import pyplot as plt
class Support:
    def __init__(self, cfg):
        self.cfg = cfg
        pass
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

    def visualize_dataset(self, file_name, customize_name='',total_count=20, saveImage=False, size=(10, 4), shrink=0.5):
        data = np.load(file_name)
        obs = data['a']
        next_obs = data['b']
        act = data['c']
        
        for idx in range(total_count):
            plt.close()
            direction = self.cfg.attention_model.direction_map[round(obs[idx, 2, :, :].max())]
            state_image = obs[idx, 0, :, :]
            state_image_next = next_obs[idx, 0, :, :]
            direction_next = self.cfg.attention_model.direction_map[round(next_obs[idx, 2, :, :].max())]
            if act is None:
                action = "None"
            else:
                action = self.cfg.attention_model.action_map[round(act[idx])]
        
            num_colors = 11
            custom_cmap = plt.cm.get_cmap('jet', num_colors)
            self._plot_subplot(1, 2, 1, state_image, custom_cmap, 'State', f"Dir: {direction}  Action: {action}", shrink)
            self._plot_subplot(1, 2, 2, state_image_next, custom_cmap, 'State Pre', f"Dir: {direction_next}", shrink)
            plt.tight_layout()
            if saveImage:
                save_file = os.path.join(self.cfg.attention_model.save_path, f"Compare_{customize_name}_{idx}.png")
                plt.savefig(save_file)
                plt.close()
            else:
                plt.show()



    def load_gen_func(self):
        model = load_gen(self.cfg)
        return model

    def generate_env(self, model):
        z = torch.randn(1, self.cfg.training_generator.z_shape)
        env_layout = torch.argmax(model(z), dim=1)
        return env_layout
    
    def wrap_env(self, env):
        if self.cfg.env.visualize:
            render_mode = "human"
        else:
            render_mode = None
        layout_string = generate_obj_map(env, self.cfg.training_generator.map_element)
        layout_string = clean_and_place_goal(layout_string)
        color_string = generate_color_map(layout_string)
        print("layout_string: ", layout_string)
        env = FullyObsWrapper(CustomMiniGridEnv(
            layout_str=layout_string,
            color_str=color_string,
            custom_mission="Navigate to the start position.",
            render_mode = render_mode
        ))
        # env.reset()
        # manual_control = ManualControl(env)  # Allows manual control for testing and visualization
        # manual_control.start()  # Start the manual control interface
        return env
    
    def wrap_env_from_text(self, file_path):
        if self.cfg.env.visualize:
            render_mode = "human"
        else:
            render_mode = None
        env = FullyObsWrapper(CustomMiniGridEnv(
            txt_file_path=file_path,
            custom_mission="Navigate to the start position.",
            max_steps=4000,
            render_mode=render_mode
        ))
        return env
    
    def collect_data_trainer(self, env):
        if not os.path.exists(self.cfg.env.collect.data_save_path):
            data_collect_api(self.cfg, env)
    
    # def save_data_to_buffer(self, data):        
    #     pass

    # def generate_final_task(self):
    #     # Generate the final task
    #     layout_list = generate_valid_minigrid_with_key_door(15, 15, start=None, goal=None, wall_prob=0.5, max_attempts=1000)
    #     layout_string = layout_to_string(layout_list)
    #     color_string = generate_color_map(layout_string)
    #     combine_maps(layout_string, color_string, self.cfg.PPO.env_path)

    def del_env_data_file(self):
        # delete the env data file
        if os.path.exists(self.cfg.env.collect.data_save_path):
            os.remove(self.cfg.env.collect.data_save_path)




