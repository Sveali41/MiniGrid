from path import Paths
from minigrid_custom_env import CustomEnvFromFile
from minigrid.wrappers import FullyObsWrapper
import torch
import torch.nn as nn
import numpy as np
import torch.distributions as td
from PPO import PPO
import os
from datetime import datetime
from model_based import SimpleNN, get_destination
from test import denorm_and_round, norm

# set device to cpu or cuda
device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


def preprocess_observation(obs):
    obs = obs / np.array([10, 5, 2])
    return torch.from_numpy(obs.flatten()).float().to(device)


def training_agent(env, model, path):
    has_continuous_action_space = False
    max_ep_len = 400  # max timesteps in one episode
    action_std = 0.1  # set same std for action distribution which was used while saving
    i_episode = 0
    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor
    lr_critic = 0.001  # learning rate for critic
    checkpoint_path = os.path.join(path.TRAINED_MODEL, 'env_model.pth')
    time_step = 0
    max_training_timesteps = int(3e5)
    update_timestep = max_ep_len * 4  # update policy every n timesteps
    action_std_decay_freq = int(2.5e5)
    action_std_decay_rate = 0.05
    print_freq = max_ep_len * 4
    min_action_std = 0.1
    save_model_freq = int(2e4)
    print_running_reward = 0
    print_running_episodes = 0

    # state space dimension
    state_dim = np.prod(env_0.observation_space['image'].shape)

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space
    else:
        action_dim = env.action_space.n

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)
    # training loop
    while time_step <= max_training_timesteps:
        state = env.reset()
        current_ep_reward = 0
        state = preprocess_observation(state[0]['image']).to(device).unsqueeze(0)
        action = ppo_agent.select_action(state)
        for t in range(1, max_ep_len + 1):
            # select action with policy
            if t > 1:
                action = ppo_agent.select_action(state_denorm.view(state_denorm.size(0), -1))
            state = model(state, torch.tensor(action).to(device).unsqueeze(0).unsqueeze(0))
            state = state.to(dtype=torch.float32)
            # denorm the state
            state_denorm = denorm_and_round(state.reshape(-1, 6, 3, 3), (10, 5, 2))

            # obtain reward from the state representation & done
            done, reward = get_destination(state_denorm, t, max_ep_len, device)
            state = norm(state_denorm, (10, 5, 2)).view(state_denorm.size(0), -1)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                if use_wandb:
                    wandb.log({"average_reward": print_avg_reward})

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        i_episode += 1

    env.close()


if __name__ == "__main__":
    use_wandb = False
    if use_wandb:
        import wandb

        wandb.login(key="ae0b0db53ae05bebce869b5ccc77b9efd0d62c73")
        wandb.init(project='world_test', entity='svea41')
    # test the model
    loaded_model = SimpleNN(54, 54, 1, 50).to(device)
    start_time = datetime.now().replace(microsecond=0)
    path = Paths()
    env_0 = FullyObsWrapper(
        CustomEnvFromFile(txt_file_path=path.LEVEL_FILE, custom_mission="Find the key and open the door.",
                          max_steps=2000,
                          render_mode="rgb"))
    # Please note that the default observation format is a partially observable view of the environment using a compact
    # and efficient encoding, with 3 input values per visible grid cell, 7x7x3 values total.
    loaded_model.load_state_dict(torch.load('env_model.pth'))

    training_agent(env_0, loaded_model, path)
    if use_wandb:
        wandb.finish()
