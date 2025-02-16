import sys
sys.path.append('/home/siyao/project/rlPractice/MiniGrid')
from minigrid_custom_env import *
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from path import *
import hydra
from omegaconf import DictConfig, OmegaConf
import time
from tqdm import tqdm
import torch
from common.utils import normalize_obs, ColRowCanl_to_CanlRowCol, WORLD_MODEL_PATH

# set device to cpu or cuda
device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

# def run_env(env, cfg: DictConfig, policy=None, rmax_exploration=None):
#     obs_list, obs_next_list, act_list, rew_list, done_list = [], [], [], [], []
#     episodes = 0
#     obs = env.reset()[0]

#     # Use tqdm to provide a progress bar
#     with tqdm(total=cfg.episodes, desc="Collecting Episodes") as pbar:
#         while episodes < cfg.episodes:
#             obs_list.append([obs['image']])
#             if policy is None:
#                 act = env.action_space.sample()  # Restrict the number of actions to 2
#             else:
#                 state_norm = normalize(obs['image']).to(device)
#                 act = policy.select_action(state_norm)
#             obs, reward, done, _, _ = env.step(act)
#             act_list.append([act])
#             obs_next_list.append([obs['image']])
#             rew_list.append([reward])
#             done_list.append([done])

#             # Update RMax visit count and store interaction
#             # if apply rmax exploration
#             if rmax_exploration is not None:
#                 rmax_exploration.update_visit_count(obs['image'], act)


#             if cfg.visualize:  # Set to false to hide the GUI
#                 env.render()
#                 time.sleep(0.1)

#             if done:
#                 episodes += 1
#                 pbar.update(1)  # Update the progress bar

#                 if episodes % 100 == 0:
#                     print("Episode", episodes)
#                 env.reset()

#     obs_np = np.concatenate(obs_list)
#     obs_next_np = np.concatenate(obs_next_list)
#     act_np = np.concatenate(act_list)
#     rew_np = np.concatenate(rew_list)
#     done_np = np.concatenate(done_list)

#     print(obs_np.shape)
#     print(obs_next_np.shape)
#     print(rew_np.shape)
#     print(done_np.shape)
#     print("Num episodes started: ", episodes)

#     return obs_np, obs_next_np, act_np, rew_np, done_np

def run_env(env, cfg: DictConfig, policy=None, rmax_exploration=None):
    obs_list, obs_next_list, act_list, rew_list, done_list = [], [], [], [], []
    episodes = 0
    obs = env.reset()[0]

    # Visit count for RMax or exploration tracking
    visit_count = {}

    # Define meaningful actions (forward, turn_left, turn_right)
    meaningful_actions = [env.unwrapped.actions.forward, env.unwrapped.actions.left, env.unwrapped.actions.right]

    # Use tqdm for progress tracking
    with tqdm(total=cfg.collect.episodes, desc="Collecting Episodes") as pbar:
        while episodes < cfg.collect.episodes:
            obs_list.append([obs['image']])

            # Select an action
            if policy is None:
                act = np.random.choice(meaningful_actions, p=[0.8, 0.1, 0.1])  # Weighted random sampling
            else:
                state_norm = normalize_obs(obs['image']).to(device)
                act = policy.select_action(state_norm)

            # Step in the environment
            obs_next, reward, done, trunc, _ = env.step(act)

            # Collect data
            act_list.append([act])
            obs_next_list.append([obs_next['image']])
            rew_list.append([reward])
            done_list.append([done])

            # Update visit count and RMax exploration
            state_action_key = (tuple(obs['image'].flatten()), act)
            visit_count[state_action_key] = visit_count.get(state_action_key, 0) + 1
            if rmax_exploration is not None:
                rmax_exploration.update_visit_count(obs['image'], act)

            # Visualize if needed
            if cfg.visualize:
                env.render()
                time.sleep(0.1)

            # Reset environment on episode end
            if done or trunc:
                episodes += 1
                pbar.update(1)
                if episodes % 100 == 0:
                    print(f"Episode {episodes}")
                obs = env.reset()[0]
            else:
                obs = obs_next

    # Convert collected data to numpy arrays
    obs_np = np.concatenate(obs_list)
    obs_next_np = np.concatenate(obs_next_list)
    act_np = np.concatenate(act_list)
    rew_np = np.concatenate(rew_list)
    done_np = np.concatenate(done_list)

    # Log statistics
    print(f"Observation shape: {obs_np.shape}")
    print(f"Next observation shape: {obs_next_np.shape}")
    print(f"Actions shape: {act_np.shape}")
    print(f"Rewards shape: {rew_np.shape}")
    print(f"Dones shape: {done_np.shape}")
    print(f"Number of episodes started: {episodes}")
    print(f"Unique state-action pairs visited: {len(visit_count)}")

    return obs_np, obs_next_np, act_np, rew_np, done_np


def save_experiments(cfg: DictConfig, obs, obs_next, act, rew, done):
    obs = ColRowCanl_to_CanlRowCol(obs)
    obs_next = ColRowCanl_to_CanlRowCol(obs_next)
    np.savez_compressed(cfg.collect.data_save_path, a=obs, b=obs_next, c=act, d=rew, e=done)

def data_augmentation(cfg: DictConfig, obs, obs_next, act, rew, done):
    """
    Adding more forward and turning data for empty env
    """
    obs_aug = np.concatenate([obs, obs_next])
    obs_next_aug = np.concatenate([obs_next, obs])
    act_aug = np.concatenate([act, act])
    rew_aug = np.concatenate([rew, rew])
    done_aug = np.concatenate([done, done])

    # Shuffle the data
    idx = np.random.permutation(len(obs_aug))
    obs_aug = obs_aug[idx]
    obs_next_aug = obs_next_aug[idx]
    act_aug = act_aug[idx]
    rew_aug = rew_aug[idx]
    done_aug = done_aug[idx]

    print("Data Augmented")
    print(obs_aug.shape)
    print(obs_next_aug.shape)
    print(rew_aug.shape)
    print(done_aug.shape)

    return obs_aug, obs_next_aug, act_aug, rew_aug, done_aug

@hydra.main(version_base=None, config_path = str(WORLD_MODEL_PATH / "config"), config_name="config")
def data_collect(cfg: DictConfig):
    hparam = cfg.env
    mode =None
    if hparam.visualize:
        mode = 'human'
    env = FullyObsWrapper(CustomEnvFromFile(txt_file_path=hparam.env_path, 
                                        custom_mission="Find the key and open the door.",
                                        max_steps=4000, render_mode=mode))
    obs, obs_next, act,rew, done = run_env(env, hparam)
    save_experiments(cfg.env,obs,obs_next, act, rew, done)

if __name__ == "__main__": 
    data_collect()