from path import Paths
from minigrid_custom_env import CustomEnvFromFile
from minigrid.wrappers import FullyObsWrapper
import torch
import torch.nn as nn
import numpy as np
import torch.distributions as td
from PPO import PPO
import os
import hydra
from modelBased.common.utils import PROJECT_ROOT
from datetime import datetime
from modelBased.world_model import SimpleNN
from modelBased.world_model_training import get_destination, denormalize, normalize
from omegaconf import DictConfig, OmegaConf

# set device to cpu or cuda
device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

def get_destination(obs, episode, maxstep, device):
    """
    from the obs state, check if the agent has reached the destination
    and return done and reward

    1.object:("unseen": 0,  "empty": 1, "wall": 2, "door": 4, "key": 5, "goal": 8, "agent": 10)
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10

    2. color:
    "red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5

    3. status
    State, 0: open, 1: closed, 2: locked

    check from wrappers.py full_obs-->encode
    """
    
    destination = torch.tensor(np.array(
        [[[2, 5, 0],
          [2, 5, 0],
          [2, 5, 0]],

         [[2, 5, 0],
          [1, 0, 0],
          [2, 5, 0]],

         [[2, 5, 0],
          [1, 0, 0],
          [2, 5, 0]],

         [[2, 5, 0],
          [4, 0, 0],
          [2, 5, 0]],

         [[2, 5, 0],
          [10, 0, 0],
          [2, 5, 0]],

         [[2, 5, 0],
          [2, 5, 0],
          [2, 5, 0]]])).unsqueeze(0).to(device).float()

    # when next_obs = destination-> done = True, otherwise = False
    if torch.isclose(destination, obs, rtol=1, atol=1).all():
        if episode >= maxstep:
            done = True
            reward = 0
        else:
            reward = 1 - 0.9 * (episode / maxstep)
            done = True
    else:
        done = False
        reward = 0

    return done, reward

@hydra.main(version_base=None, config_path=PROJECT_ROOT / "conf/model", config_name="config")
def training_agent(cfg: DictConfig):
    hparams = cfg
    
    # 1. world Model
    model = SimpleNN(hparams=hparams.world_model).to(device)
    checkpoint = torch.load(hparams.world_model.pth_folder)
    model.load_state_dict(checkpoint['state_dict'])
    
    # 2. PPO
    # hyperparameters
    lr_actor = hparams.PPO.lr_actor
    lr_critic = hparams.PPO.lr_critic
    gamma = hparams.PPO.gamma
    K_epochs = hparams.PPO.K_epochs
    eps_clip = hparams.PPO.eps_clip
    action_std = hparams.PPO.action_std
    action_std_decay_rate = hparams.PPO.action_std_decay_rate
    min_action_std = hparams.PPO.min_action_std
    action_std_decay_freq = hparams.PPO.action_std_decay_freq
    max_training_timesteps = hparams.PPO.max_training_timesteps
    save_model_freq = hparams.PPO.save_model_freq
    max_ep_len = hparams.PPO.max_ep_len
    has_continuous_action_space = hparams.PPO.has_continuous_action_space
    
    # 3. real env
    path = Paths()
    env = FullyObsWrapper(
        CustomEnvFromFile(txt_file_path=path.LEVEL_FILE, custom_mission="Find the key and open the door.",
                          max_steps=2000,
                          render_mode="rgb"))
    
    # 4. initialize training
    i_episode = 0
    update_timestep = max_ep_len * 4  # update policy every n timesteps
    print_freq = max_ep_len * 4
    print_running_reward = 0
    print_running_episodes = 0
    
    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space
    else:
        action_dim = env.action_space.n
    state_dim = np.prod(env.observation_space['image'].shape)
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)
    # training loop
    while time_step <= max_training_timesteps:
        state = env.reset()
        current_ep_reward = 0
        state = preprocess_observation(state[0]['image']).to(device).unsqueeze(0)
        action = ppo_agent.select_action(state)
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
    loaded_model = SimpleNN(54, 54, 1, 50).to(device)

    # Please note that the default observation format is a partially observable view of the environment using a compact
    # and efficient encoding, with 3 input values per visible grid cell, 7x7x3 values total.
    loaded_model.load_state_dict(torch.load('env_model.pth'))

    training_agent(env_0, loaded_model, path)
    if use_wandb:
        wandb.finish()
