import hydra
import sys
sys.path.append('/home/siyao/project/rlPractice/MiniGrid')
from modelBased.common.utils import GENERATOR_PATH
from omegaconf import DictConfig
from generator.common.utils import load_gen, generate_color_map, generate_obj_map
from generator.gen import GAN
from minigrid_custom_env import *
import textwrap
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper
import torch
import os
from modelBased.data_collect import *
from modelBased.data.datamodule import *

class Support:
    def __init__(self, cfg):
        self.cfg = cfg
        pass

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
        color_string = generate_color_map(layout_string)
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
    
    def collect_data(self, env):
        if not os.path.exists(self.cfg.env.collect.data_train):
            data_collect(self.cfg, env)
        data_loader = WMRLDataModule(hparams = self.cfg.world_model)
        return data_loader
    
    # def save_data_to_buffer(self, data):        
    #     pass

    def train_WM(self, data_loader):
        # model = load_model(self.cfg)
        # trainer = Trainer(model=model, dataloader=data_loader)
        # trainer.train()
        pass

    

