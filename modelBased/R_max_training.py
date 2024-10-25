import sys
sys.path.append('/home/siyao/project/rlPractice/MiniGrid')
from minigrid_custom_env import *
from minigrid.wrappers import FullyObsWrapper,  ImgObsWrapper
from path import *
import pandas as pd
from modelBased.common.utils import PROJECT_ROOT
from PPO import PPO
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import hydra
from modelBased.world_model_training import normalize, map_obs_to_nearest_value
import torch
from Rmax import RMaxExploration


# set device to cpu or cuda
device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "conf/Rmax"), config_name="config")
def training_agent_with_rmax(cfg: DictConfig):
    """
    This function trains the agent using PPO algorithm based on R_max concept
    1. collect env data from real env
    2. train world model
    3. collect data from world model
    4. train PPO agent
    """
    hparams = cfg

    # 1. hyperparameters
    # PPO hyperparameters
    start_time = datetime.now().replace(microsecond=0)
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
    checkpoint_path = hparams.PPO.checkpoint_path
    # world model hyperparameters
    grid_size = [hparams.world_model.map_height, hparams.world_model.map_width]
    # R_max hyperparameters
    wm_data_ep = hparams.data_collect.n_rollouts


    # 1. Real env initialization 
    path = Paths()
    env = FullyObsWrapper(CustomEnvFromFile(txt_file_path=path.LEVEL_FILE_Rmax, custom_mission="pick up the yellow ball",
                        max_steps=2000, render_mode="human"))
    
    # 2. Initialize training PPO
    i_episode = 0
    update_timestep = max_ep_len * 4  # update policy every n timesteps
    print_freq = max_ep_len * 4
    print_running_reward = 0
    print_running_episodes = 0
    time_step = 0
    
    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space
    else:
        action_dim = env.action_space.n
    state_dim = np.prod(env.observation_space['image'].shape)
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)
    
    # 3. Initialize R_max exploration
    rmax_exploration = RMaxExploration(state_dim, action_dim, R_max=1.0, exploration_threshold=10)
    for time_step in range(1, hparams.R_max.max_training_timesteps + 1):
        if time_step <= hparams.R_max.exploration_timesteps:
            # when timestep < Rmax exploration threshold, collect data from real env
            state = env.reset()[0]['image']
            done = False
            while not done:
            # collect data from real env under the ppo policy to train World Model
                state = normalize(state).to(device)
                action = ppo_agent.select_action(state)
                next_state, reward, done, _, _ =  env.step(action)
                next_state = normalize(next_state['image']).to(device)
                # save this to data buffer to train world model
                # Count and state and action pair
                rmax_exploration.update_visit_count(state.cpu().numpy(), action, reward, next_state.cpu().numpy())
                state = next_state

        else:
            # collect data from world model and update PPO agent
            state = state.squeeze()
            action = ppo_agent.select_action(state)
            state = model(state, torch.tensor(action/hparams.world_model.action_norm_values).to(device).unsqueeze(0))
            state = state.to(dtype=torch.float32)
            # denorm the state
            state_denorm = map_obs_to_nearest_value(cfg, state)
            # obtain reward from the state representation & done
            done, reward = get_destination(state_denorm, t, max_ep_len, device)
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
            # if done:
            #     break

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            i_episode += 1

    env.close()

if __name__ == "__main__":
    training_agent_with_rmax()