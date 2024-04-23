import time
import gym
from PPO import PPO
from path import Paths
from minigrid_custom_env import CustomEnvFromFile
from minigrid.wrappers import FullyObsWrapper
import torch
import numpy as np
import os

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


def test(env, path):
    has_continuous_action_space = False
    max_ep_len = 1000  # max timesteps in one episode
    action_std = 0.1  # set same std for action distribution which was used while saving

    render = True  # render environment on screen
    frame_delay = 0  # if required; add delay b/w frames

    total_test_episodes = 10  # total num of testing episodes

    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor
    lr_critic = 0.001  # learning rate for critic
    checkpoint_path = os.path.join(path, 'PPO_model.pth')

    # state space dimension
    state_dim = np.prod(env_0.observation_space['image'].shape)

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)

    # preTrained weights directory
    print("loading network from : " + checkpoint_path)
    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes + 1):
        ep_reward = 0
        state = env.reset()
        state = preprocess_observation(state[0]['image']).to(device)

        for t in range(1, max_ep_len + 1):
            action = ppo_agent.select_action(state)
            state, reward, done, _, _ = env.step(action)
            state = preprocess_observation(state['image']).to(device)
            ep_reward += reward

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


def preprocess_observation(obs):
    obs = obs / np.array([10, 5, 2])
    return torch.from_numpy(obs.flatten()).float()


if __name__ == '__main__':
    path = Paths()
    env_0 = FullyObsWrapper(
        CustomEnvFromFile(txt_file_path=path.LEVEL_FILE, custom_mission="Find the key and open the door.",
                          max_steps=1000,
                          render_mode="human"))
    test(env_0, path.TRAINED_MODEL)
