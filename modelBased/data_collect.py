import sys
sys.path.append('/home/siyao/project/rlPractice/MiniGrid')
from minigrid_custom_env import *
from minigrid.wrappers import FullyObsWrapper,  ImgObsWrapper
from path import *
import pandas as pd
import json
import hydra
from modelBased.common.utils import PROJECT_ROOT
from omegaconf import DictConfig, OmegaConf
import time
from tqdm import tqdm

def run_env(env, cfg: DictConfig):
    obs_list, obs_next_list, act_list, rew_list, done_list = [], [], [], [], []
    episodes = 0
    obs = env.reset()[0]

    # Use tqdm to provide a progress bar
    with tqdm(total=cfg.collect.episodes, desc="Collecting Episodes") as pbar:
        while episodes < cfg.collect.episodes:
            obs_list.append([obs['image']])
            act = env.action_space.sample()  # Restrict the number of actions to 2
            obs, reward, done, _, _ = env.step(act)
            act_list.append([act])
            obs_next_list.append([obs['image']])
            rew_list.append([reward])
            done_list.append([done])

            if cfg.env.visualize:  # Set to false to hide the GUI
                env.render()
                time.sleep(0.1)

            if done:
                episodes += 1
                pbar.update(1)  # Update the progress bar

                if episodes % 100 == 0:
                    print("Episode", episodes)
                env.reset()

    obs_np = np.concatenate(obs_list)
    obs_next_np = np.concatenate(obs_next_list)
    act_np = np.concatenate(act_list)
    rew_np = np.concatenate(rew_list)
    done_np = np.concatenate(done_list)

    print(obs_np.shape)
    print(obs_next_np.shape)
    print(rew_np.shape)
    print(done_np.shape)
    print("Num episodes started: ", episodes)

    return obs_np, obs_next_np, act_np, rew_np, done_np

def save_experiments(cfg: DictConfig, obs, obs_next, act, rew, done):
    np.savez_compressed(cfg.collect.data_train, a=obs, b=obs_next, c=act, d=rew, e=done)

@hydra.main(version_base=None, config_path = PROJECT_ROOT / "conf/env", config_name="config")
def main(cfg: DictConfig):
    path = Paths()
    env = FullyObsWrapper(CustomEnvFromFile(txt_file_path=path.LEVEL_FILE, custom_mission="Find the key "
                                                                                      "and open the "
                                                                                      "door.",
                                        max_steps=2000, render_mode=None))
    obs, obs_next, act,rew, done = run_env(env, cfg)
    save_experiments(cfg,obs,obs_next, act, rew, done)

if __name__ == "__main__": 
    main()