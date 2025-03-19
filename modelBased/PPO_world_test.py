import os
import glob
import time
from datetime import datetime
import torch
import numpy as np
import pandas as pd
import gym
import hydra
from common.utils import PROJECT_ROOT, normalize_obs, map_obs_to_nearest_value
from PPO import PPO
from omegaconf import DictConfig, OmegaConf
from path import Paths
from minigrid.wrappers import FullyObsWrapper
from minigrid_custom_env import *
from common import utils

# set device to cpu or cuda
device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
#################################### Testing ###################################
@hydra.main(version_base=None, config_path= str(PROJECT_ROOT / "modelBased/config"), config_name="config")
def test(cfg: DictConfig):
    print("============================================================================================")
    ################## hyperparameters ##################
    has_continuous_action_space = cfg.PPO.has_continuous_action_space
    action_std = cfg.PPO.action_std
    lr_actor = cfg.PPO.lr_actor
    lr_critic = cfg.PPO.lr_critic
    gamma = cfg.PPO.gamma
    K_epochs = cfg.PPO.K_epochs
    eps_clip = cfg.PPO.eps_clip
    total_test_episodes = 100
    max_ep_len = cfg.PPO.max_ep_len
    render = cfg.PPO.render
    checkpoint_path = cfg.PPO.checkpoint_path
    env_type =  cfg.PPO.env_type
    obs_norm_values = cfg.attention_model.obs_norm_values
    visualize_obs = utils.Visualization(cfg.attention_model)
    visualize_flag = cfg.PPO.visualize
    csv_output = True
    data = []
    #####################################################
    # load environment
    path = Paths()
    env = FullyObsWrapper(CustomEnvFromFile(txt_file_path=path.LEVEL_FILE, custom_mission="Find the key "
                                                                                      "and open the "
                                                                                      "door.",
                                        max_steps=2000, render_mode=None))

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space
    else:
        action_dim = env.action_space.n
    state_dim = np.prod(env.observation_space['image'].shape)

    # initialize a PPO agent
    if has_continuous_action_space:
        if env_type == 'empty':
            action_dim = 3
        else:
            action_dim = env.action_space
    else:
        if env_type == 'empty':
            action_dim = 3
        else:
            action_dim = env.action_space.n
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory
    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state_init = env.reset()[0]['image']
        state = utils.ColRowCanl_to_CanlRowCol(state_init)
        for step in range(1, max_ep_len+1):
            state_norm = normalize_obs(state, obs_norm_values)
            if isinstance(state_norm, np.ndarray):
                state_norm = torch.tensor(state_norm, dtype=torch.float32).to(device)
            action = ppo_agent.select_action(state_norm.flatten())
            state_next, reward, done, _, _ = env.step(action)
            ep_reward += reward
            state_next = utils.ColRowCanl_to_CanlRowCol(state_next['image'])
            if visualize_flag:
                print(f'Episode: {ep} \t Step: {step} \t Reward: {round(ep_reward, 2)}')
                visualize_obs.compare_states(state, state_next, action, f'ep:{ep} step:{step}', True)
            
            state = state_next
                
            if render:
                env.render()

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print(f'Episode: {ep} \t\t Reward: {round(ep_reward, 2)}')
        
        if csv_output:
            data.append(ep_reward)
        ep_reward = 0
    if csv_output:
        df = pd.DataFrame(data, columns=["reward"])
        df.to_csv("test_PPO_reward.csv", index=False, header=True)

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':
    test()