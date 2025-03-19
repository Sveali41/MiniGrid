import os
import glob
import time
from datetime import datetime
import torch
import numpy as np
import gym
from PPO import PPO
import hydra
from modelBased.common.utils import PROJECT_ROOT, normalize, map_obs_to_nearest_value
from omegaconf import DictConfig, OmegaConf
from path import Paths
from minigrid.wrappers import FullyObsWrapper
from minigrid_custom_env import *

# set device to cpu or cuda
device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
#################################### Testing ###################################
@hydra.main(version_base=None, config_path= str(PROJECT_ROOT / "conf/model"), config_name="config")
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
        state = env.reset()[0]

        for _ in range(1, max_ep_len+1):
            state = normalize(state['image']).to(device)
            action = ppo_agent.select_action(state)
            state, reward, done, _, _ = env.step(action)
            ep_reward += reward

            if render:
                env.render()

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':

    test()