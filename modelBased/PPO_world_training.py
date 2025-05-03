import sys
from .common.utils import PROJECT_ROOT
from minigrid_custom_env import CustomMiniGridEnv
from minigrid.wrappers import FullyObsWrapper
import torch
import numpy as np
from .PPO import PPO
import hydra
from datetime import datetime
from .common import utils

from omegaconf import DictConfig, OmegaConf 
from . import AttentionWM_support
from . import Embedding_support
from . import MLP_support#
import wandb
from PPO import preprocess_observation 


# set device to cpu or cuda
device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

def get_destination(obs, episode, maxstep, destination):
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
    if obs[0, destination[0], destination[1]] == 10:
        # agent has reached the destination
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


def find_position(array, target):
    """
    Find the position of a target value in a 3D numpy array.
    
    Args:
        array (np.ndarray): The 3D array to search.
        target (tuple): The target value to locate (e.g., (8, 1, 0)).

    Returns:
        tuple: The position (x, y) of the target in the array if found, otherwise None.
    """
    # Find all indices where the value matches the target
    target = np.array(target).reshape(-1, 1, 1)
    result = np.argwhere((array == target).all(axis=0))

    # Check if any matches were found
    if result.size > 0:
        return tuple(result[0])  # Return the first match as a tuple (x, y)
    else:
        return None

def process_data(state, action, maks_size):
    action = action
    agent_postion_yx = utils.get_agent_position(state)
    state_masked = utils.extract_masked_state(state, maks_size, agent_postion_yx) 
    return state_masked, action

def evaluate_policy(policy, env, episodes, obs_norm_values):
    """
    Evaluate the policy by running it in the environment for a number of episodes.
    """
    total_reward = 0
    for _ in range(episodes):
        obs = env.reset()[0]['image']
        state = utils.ColRowCanl_to_CanlRowCol(obs)
        done = False
        ep_reward = 0
        for _ in range(env.max_steps):
            state_tensor = torch.tensor(utils.normalize_obs(state, obs_norm_values)).to(device)
            action = policy.select_action(state_tensor.flatten())
            obs, reward, done, _, _ = env.step(action)
            state = utils.ColRowCanl_to_CanlRowCol(obs['image'])
            ep_reward += reward
            if done:
                break
        total_reward += ep_reward
    return total_reward / episodes


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "modelBased/config"), config_name="config")
def training_agent(cfg: DictConfig):
    run_training_wm(cfg)

def run_training_wm(cfg):
    hparams = cfg
    
    # 1. World Model
    hparams_world_model = hparams.attention_model

    MODEL_MAPPING = {
            'attention': AttentionWM_support.AttentionModule,
            'embedding': Embedding_support.EmbeddingModule,
            'mlp': MLP_support.SimpleNNModule
        }
    # 初始化模型
    module_class = MODEL_MAPPING.get(hparams_world_model.model_type.lower())
    if module_class is not None:
        model = module_class(
            hparams_world_model.data_type,  
            hparams_world_model.grid_shape, 
            hparams_world_model.attention_mask_size, 
            hparams_world_model.embed_dim, 
            hparams_world_model.num_heads
        )
    else:
        print(f"Model type: {hparams_world_model.model_type} not supported")
        exit()
    utils.load_model_weight(model, hparams_world_model.model_save_path)
    


    # 2. PPO
    # hyperparameters
    # compute regret
    compute_regret = hparams_PPO.compute_regret
    if compute_regret:
        regret_eval_freq = hparams_PPO.get("regret_eval_freq", 5000)
        regret_eval_episodes = hparams_PPO.get("regret_eval_episodes", 5)
        real_policy_path = hparams_PPO.get("real_policy_path")


    hparams_PPO = hparams.PPO
    start_time = datetime.now().replace(microsecond=0)
    lr_actor = hparams_PPO.lr_actor
    lr_critic = hparams_PPO.lr_critic
    gamma = hparams_PPO.gamma
    K_epochs = hparams_PPO.K_epochs
    eps_clip = hparams_PPO.eps_clip
    action_std = hparams_PPO.action_std
    action_std_decay_rate = hparams_PPO.action_std_decay_rate
    min_action_std = hparams_PPO.min_action_std
    action_std_decay_freq = hparams_PPO.action_std_decay_freq
    max_training_timesteps = hparams_PPO.max_training_timesteps
    save_model_freq = hparams_PPO.save_model_freq
    max_ep_len = hparams_PPO.max_ep_len
    has_continuous_action_space = hparams_PPO.has_continuous_action_space
    checkpoint_path = hparams_PPO.checkpoint_path
    env_path = hparams_PPO.env_path
    visualize_flag = hparams_PPO.visualize
    env_type =  hparams_PPO.env_type
    use_wandb = hparams_PPO.use_wandb

    if use_wandb:
        wandb.login(key="eeecc8f761c161927a5713203b0362dfcb3181c4")
        wandb.init(project='Trainer_policy', entity='18920011663-king-s-college-london', group="experiment-EWC", reinit=True)

    # training_agent()

    if visualize_flag:
        visualize = utils.Visualization(hparams_world_model)
    # 3. Real environment
    env = FullyObsWrapper(
        CustomMiniGridEnv(txt_file_path=env_path, custom_mission="Find the key and open the door.",
                        max_steps=4000, render_mode=None))
    
    # 4. Initialize training
    i_episode = 0
    update_timestep = max_ep_len * 4  # update policy every n timesteps
    print_freq = max_ep_len * 2
    print_running_reward = 0
    print_running_episodes = 0
    time_step = 0
    
    # action space dimension
    if has_continuous_action_space:
        if env_type == 'empty':
            action_dim = 3
        else:
            action_dim = 6
    else:
        if env_type == 'empty':
            action_dim = 3
        else:
            action_dim = 6
    state_dim = np.prod(env.observation_space['image'].shape)
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)
    if compute_regret:
        real_policy_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                        has_continuous_action_space, action_std)
        real_policy_agent.load(real_policy_path)

    

    # training loop
    while time_step <= max_training_timesteps:
        print(f'time step: {time_step}')
        state_init = env.reset()[0]['image']
        state_0 = utils.ColRowCanl_to_CanlRowCol(state_init)
        goal_position_yx = find_position(state_0, (8, 1, 0)) # find the goal position
        current_ep_reward = 0
        for t in range(1, max_ep_len + 1):
            # self.buffer.states = [state.squeeze(0) if state.dim() > 1 else state for state in self.buffer.states]
            if t==1:
                # state = utils.normalize_obs(state_0, hparams_world_model.obs_norm_values)
                state_0 = torch.tensor(state_0).to(device)
            
            state_norm = utils.normalize_obs(state_0, hparams_world_model.obs_norm_values)
            action = ppo_agent.select_action(state_norm.flatten()) # state is the dimension of flatten
            state_masked, action = process_data(state_0.clone(), action, 
                                         hparams_world_model.attention_mask_size)
            
            delta_masked, _ = model(state_masked, action)
            
            state_pre_masked = state_masked + delta_masked
            if visualize_flag:
                visualize.compare_states(state_masked, state_pre_masked, action, t, True)
            # delta_state_pre = delta_state_pre.to(dtype=torch.float32)
            # denorm the state
            # state_pre_denorm = utils.denormalize_obj(state_pre, hparams_world_model.obs_norm_values)

            state_pre_masked = utils.map_obs_to_nearest_value(state_pre_masked, 
                                                              hparams_world_model.valid_values_obj,
                                                              hparams_world_model.valid_values_color,
                                                              hparams_world_model.valid_values_state)


            agent_postion_yx = utils.get_agent_position(state_0)
            state_pre = utils.put_back_masked_state(state_pre_masked, state_0, hparams_world_model.attention_mask_size, agent_postion_yx)
            

                
            state_0 = state_pre
            # obtain reward from the state representation & done
            done, reward = get_destination(state_0, t, max_ep_len, goal_position_yx)
            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward
            if use_wandb:
                wandb.log({"episode_reward": current_ep_reward})
            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            if time_step % print_freq == 0 and print_running_episodes > 0:
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


                if compute_regret and time_step % regret_eval_freq == 0:
                    # compute the regret
                    print(f"Evaluating regret at timestep {time_step}...")
                    R_wm = evaluate_policy(ppo_agent, env, regret_eval_episodes, hparams_world_model.obs_norm_values)
                    R_real = evaluate_policy(real_policy_agent, env, regret_eval_episodes, hparams_world_model.obs_norm_values)
                    regret = R_real - R_wm
                    norm_regret = regret / max(R_real, 1e-8)

                    print(f"[Regret @ timestep {time_step}] Real: {R_real:.2f}, WM: {R_wm:.2f}, Regret: {regret:.2f}, Norm: {norm_regret:.2%}")
                    
                    if use_wandb:
                        wandb.log({
                            "regret": regret,
                            "normalized_regret": norm_regret,
                            "real_policy_reward": R_real,
                            "wm_policy_reward": R_wm,
                            "timestep": time_step
                        })
                        
            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        i_episode += 1
    env.close()
    if use_wandb:
        wandb.finish() 

@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "modelBased/config"), config_name="config")
def training_agent_real_env(cfg: DictConfig):
    # parameters
    hparams = cfg
    hparams_PPO = hparams.PPO
    has_continuous_action_space = hparams_PPO.has_continuous_action_space
    max_ep_len =  hparams_PPO.max_ep_len
    max_training_timesteps = hparams_PPO.max_training_timesteps
    print_freq = hparams_PPO.print_freq
    save_model_freq = hparams_PPO.save_model_freq
    update_timestep = max_ep_len * 4  # update policy every n timesteps
    print_running_reward = 0
    print_running_episodes = 0

    time_step = 0
    i_episode = 0
    action_std_decay_freq = hparams_PPO.action_std_decay_freq
    action_std_decay_rate = hparams_PPO.action_std_decay_rate
    min_action_std = hparams_PPO.min_action_std
    checkpoint_path = hparams_PPO.get("real_policy_path")

    # param for agent
    K_epochs = hparams_PPO.K_epochs
    eps_clip = hparams_PPO.eps_clip
    gamma = hparams_PPO.gamma
    lr_actor = hparams_PPO.lr_actor  # learning rate for actor network
    lr_critic = hparams_PPO.lr_critic  # learning rate for critic network
    action_std = hparams_PPO.action_std  # default std for action distribution (can be overwritten by action_std_decay_rate)
    has_continuous_action_space = hparams_PPO.has_continuous_action_space
    env_path = hparams_PPO.env_path

    # state space dimension
    env = FullyObsWrapper(
        CustomMiniGridEnv(txt_file_path=env_path, custom_mission="Find the key and open the door.",
                        max_steps=4000, render_mode=None))
    
    state_dim = np.prod(env.observation_space['image'].shape)

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
        state = preprocess_observation(state[0]['image']).to(device)

        for t in range(1, max_ep_len + 1):

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _, _ = env.step(action)
            state = preprocess_observation(state['image']).to(device)
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
                print_avg_reward = round(print_avg_reward, 2)
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
    use_wandb = True
    if use_wandb:
        import wandb

        wandb.login(key="eeecc8f761c161927a5713203b0362dfcb3181c4")
        wandb.init(project='WM Attention PPO', entity='svea41')

    training_agent()

    if use_wandb:
        wandb.finish()
