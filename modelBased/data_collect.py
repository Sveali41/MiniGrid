import sys
sys.path.append('/home/siyao/project/rlPractice/MiniGrid/modelBased')
from .common.utils import normalize_obs, ColRowCanl_to_CanlRowCol, WORLD_MODEL_PATH, PROJECT_ROOT
from minigrid_custom_env import *
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from path import *
import hydra
from omegaconf import DictConfig, OmegaConf
import time
from tqdm import tqdm
import torch
import wandb


import os
import hydra
from omegaconf import DictConfig
import numpy as np
from multiprocessing import Pool, get_context


def run_env_single(env, cfg: DictConfig, seed, policy=None, rmax_exploration=None, save_img=False):
    """
    Run a specified number of episodes in one environment instance,
    collect (obs, obs_next, act, rew, done, info) tuples, and return them.
    
    Args:
        env:            A Gym‐style environment with .reset(), .step(), etc.
        cfg:            A DictConfig containing `cfg.collect.episodes` and other flags.
        seed (int):     Random seed for this worker to ensure each process
                        collects different data.
        save_img (bool): If True, you could log the first frame to W&B (not shown here).

    Returns:
        obs_np:      numpy array of all observations (shape = [total_steps, ...])
        obs_next_np: numpy array of next observations
        act_np:      numpy array of actions taken
        rew_np:      numpy array of rewards received
        done_np:     numpy array of done flags (True/False)
        info_np:     numpy array of info dictionaries (or structured arrays)
    """
    import random, torch

    # 1) Set seeds for reproducibility in this process
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 2) Reset environment and prepare containers
    obs_list, obs_next_list = [], []
    act_list, rew_list, done_list, info_list = [], [], [], []
    episodes_collected = 0

    # If you want to log a first frame to W&B, you could do it here:
    # if save_img and wandb_run is not None:
    #     img = env.get_frame()
    #     wandb_run.log({"Example Frame": wandb.Image(img)})

    # 3) Configure env’s random seed if supported
    try:
        env.seed(seed)
    except AttributeError:
        pass  # Some wrappers don’t support .seed(); ignore if not available

    # 4) Reset environment to start collecting
    obs = env.reset()[0]  # assuming env.reset() returns a tuple where [0] has the image

    # 5) Track unique state‐action visits (optional)
    visit_count = {}

    # Define meaningful actions (forward, turn_left, turn_right)
    meaningful_actions = [env.unwrapped.actions.forward, env.unwrapped.actions.left, env.unwrapped.actions.right, env.unwrapped.actions.pickup, env.unwrapped.actions.toggle]


    # 6) Decide how many episodes this worker should run
    target_episodes = cfg.collect.episodes

    while episodes_collected < target_episodes:
        # Record the current raw observation (image)
        obs_list.append([obs['image']])
        # Select an action
        if policy is None:
            act = np.random.choice(meaningful_actions, p=[0.3, 0.15, 0.15, 0.2, 0.2])  # Weighted random sampling
        else:
            state_norm = normalize_obs(obs['image']).to(device)
            act = policy.select_action(state_norm)

        # Step the environment
        obs_next, reward, done, trunc, info = env.step(act)

        # Optionally add custom info fields, for example whether agent is carrying a key
        if hasattr(env.unwrapped, "carrying") and env.unwrapped.carrying:
            info["carrying_key"] = (env.unwrapped.carrying.type == 'key')
        else:
            info["carrying_key"] = False

        # Append to lists
        act_list.append([act])
        obs_next_list.append([obs_next['image']])
        rew_list.append([reward])
        done_list.append([done])
        info_list.append([info])

        # Update visit count for (state, action) pairs (optional)
        key = (tuple(obs['image'].flatten()), act)
        visit_count[key] = visit_count.get(key, 0) + 1

        # If visualization is enabled, render frames
        if getattr(cfg.collect, "visualize", False):
            env.render()
            time.sleep(0.1)

        # Check if episode ended
        if done or trunc:
            episodes_collected += 1
            obs = env.reset()[0]
            if episodes_collected % 100 == 0:
                print(f"[Worker {seed}] Completed {episodes_collected} episodes")
        else:
            obs = obs_next

    # Convert lists to numpy arrays
    obs_np      = np.concatenate(obs_list,      axis=0)
    obs_next_np = np.concatenate(obs_next_list, axis=0)
    act_np      = np.concatenate(act_list,      axis=0)
    rew_np      = np.concatenate(rew_list,      axis=0)
    done_np     = np.concatenate(done_list,     axis=0)
    info_np     = np.concatenate(info_list,     axis=0)

    # Print summary of this worker’s collection
    print(f"[Worker {seed}] Final counts: obs {obs_np.shape}, actions {act_np.shape}, rewards {rew_np.shape}, dones {done_np.shape}")
    print(f"[Worker {seed}] Unique state-action pairs visited: {len(visit_count)}")

    return obs_np, obs_next_np, act_np, rew_np, done_np, info_np

def visualize_env(env, cfg: DictConfig, save_img=False):
    env.reset()[0]
    img = env.get_frame()
    return img 

# -----------------------------------
# Entry point for each worker process
# -----------------------------------
def worker_entry(args):
    """
    Each worker:
      1. Receives (worker_id, cfg) as arguments.
      2. Builds its own environment instance.
      3. Calls run_env_single to collect episodes.
      4. Returns the collected numpy arrays.

    Args:
        args: Tuple (worker_id, cfg)
    Returns:
        Same tuple of numpy arrays as run_env_single.
    """
    worker_id, cfg, env, save_img= args
    visualize_env(env, cfg, save_img)
    return run_env_single(env, cfg, worker_id, policy=None, rmax_exploration=None, save_img=False)


def run_env_multiprocess(env, hparam, wandb_run, save_img):
    """
    Parallel data collection using multiple processes.

    Steps:
      1. Read cfg.collect.episodes (total number of episodes to collect)
         and cfg.collect.num_workers (number of parallel worker processes).
      2. Split total episodes among workers, so each worker runs a subset.
      3. Spawn a Pool of workers, each calling worker_entry((worker_id, cfg_copy)).
      4. Collect results (numpy arrays) from all workers, concatenate them,
         and finally save into one .npz file.
    """
    # 1) Extract configuration parameters
    total_episodes = hparam.collect.episodes       # e.g. 1000
    num_workers    = hparam.collect.num_workers    # e.g. 4

    # 2) Divide episodes as evenly as possible
    base  = total_episodes // num_workers
    extra = total_episodes % num_workers
    episodes_per_worker = [base + (1 if i < extra else 0) for i in range(num_workers)]
    print(f"Spawning {num_workers} workers, each will collect these episode counts: {episodes_per_worker}")

    # 3) Create a separate cfg copy for each worker, adjusting only .collect.episodes
    import copy
    worker_args = []
    for i in range(num_workers):
        single_cfg = copy.deepcopy(hparam)
        single_cfg.collect.episodes = episodes_per_worker[i]
        worker_args.append((i, single_cfg))  # (worker_id, config)

    # 4) Use multiprocessing Pool with "spawn" context for better compatibility
    ctx = get_context("spawn")
    with Pool(processes=num_workers, context=ctx) as pool:
        # map blocks until all workers finish, and gathers their return values
        results = pool.map(worker_entry, worker_args)

    # 5) Merge data from all workers
    all_obs, all_obs_next = [], []
    all_act, all_rew, all_done, all_info = [], [], [], []

    for (obs_np, obs_next_np, act_np, rew_np, done_np, info_np) in results:
        all_obs.append(obs_np)
        all_obs_next.append(obs_next_np)
        all_act.append(act_np)
        all_rew.append(rew_np)
        all_done.append(done_np)
        all_info.append(info_np)

    obs_all      = np.concatenate(all_obs,      axis=0)
    obs_next_all = np.concatenate(all_obs_next, axis=0)
    act_all      = np.concatenate(all_act,      axis=0)
    rew_all      = np.concatenate(all_rew,      axis=0)
    done_all     = np.concatenate(all_done,     axis=0)
    info_all     = np.concatenate(all_info,     axis=0)

    # Log statistics
    print(f"Observation shape: {obs_np.shape}")
    print(f"Next observation shape: {obs_next_np.shape}")
    print(f"Actions shape: {act_np.shape}")
    print(f"Rewards shape: {rew_np.shape}")
    print(f"Dones shape: {done_np.shape}")

    return obs_all, obs_next_all, act_all, rew_all, done_all, info_all

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


def augment_interactions(obs, obs_next, act, rew, done, actions_to_oversample, N=10):
    # 如果 N==0 或者关键样本太少，不做增强
    if N <= 1:
        return obs, obs_next, act, rew, done
    """
    对指定的关键动作做过采样 N 倍，其余样本保留一次。
    actions_to_oversample: list of action indices, e.g. [pickup, toggle]
    """
    # Rest of augment_interactions ...(obs, obs_next, act, rew, done, actions_to_oversample, N=10):
    """
    对指定的关键动作做过采样 N 倍，其余样本保留一次。
    actions_to_oversample: list of action indices, e.g. [pickup, toggle]
    """
    # obs shape e.g. (num_samples, 1, H, W, C)
    # act shape e.g. (num_samples, 1)
    num_samples = obs.shape[0]
    flat_act = act.reshape(num_samples)  # (num_samples,)

    # 检测哪些 transition 实际改变了环境
    changed = np.any(obs != obs_next, axis=tuple(range(1, obs.ndim)))  # (num_samples,)

    # 标记要过采样的关键动作
    mask_key = np.zeros(num_samples, dtype=bool)
    for a in actions_to_oversample:
        mask_key |= (flat_act == a)

    # 最终 mask：既是关键动作，又发生了环境变化
    mask = mask_key & changed

    # 分离关键交互与普通样本
    obs_key    = obs[mask]
    obsn_key   = obs_next[mask]
    act_key    = act[mask]
    rew_key    = rew[mask]
    done_key   = done[mask]

    obs_norm    = obs[~mask]
    obsn_norm   = obs_next[~mask]
    act_norm    = act[~mask]
    rew_norm    = rew[~mask]
    done_norm   = done[~mask]

    # 过采样关键交互
    obs_aug      = np.concatenate([obs_norm] + [obs_key] * N, axis=0)
    obsn_aug     = np.concatenate([obsn_norm] + [obsn_key] * N, axis=0)
    act_aug      = np.concatenate([act_norm] + [act_key] * N, axis=0)
    rew_aug      = np.concatenate([rew_norm] + [rew_key] * N, axis=0)
    done_aug     = np.concatenate([done_norm] + [done_key] * N, axis=0)

    # 打乱数据
    idx = np.random.permutation(len(obs_aug))
    return obs_aug[idx], obsn_aug[idx], act_aug[idx], rew_aug[idx], done_aug[idx]


def run_env(env, cfg: DictConfig, wandb_run, policy=None, rmax_exploration=None, save_img=False):
    obs_list, obs_next_list, act_list, rew_list, done_list, info_list = [], [], [], [], [], []
    episodes = 0
    obs = env.reset()[0]
    if save_img and wandb_run is not None:
        img = env.get_frame()
        wandb_run.log({"Mini-tasks": wandb.Image(img)})
    # Visit count for RMax or exploration tracking
    visit_count = {}

    # Define meaningful actions (forward, turn_left, turn_right)
    meaningful_actions = [env.unwrapped.actions.forward, env.unwrapped.actions.left, env.unwrapped.actions.right, env.unwrapped.actions.pickup, env.unwrapped.actions.toggle]

    # Use tqdm for progress tracking
    with tqdm(total=cfg.collect.episodes, desc="Collecting Episodes") as pbar:

        while episodes < cfg.collect.episodes:
            obs_list.append([obs['image']])

            # Select an action
            if policy is None:
                act = np.random.choice(meaningful_actions, p=[0.3, 0.15, 0.15, 0.2, 0.2])  # Weighted random sampling
            else:
                state_norm = normalize_obs(obs['image']).to(device)
                act = policy.select_action(state_norm)

            # Step in the environment
            obs_next, reward, done, trunc, info = env.step(act)
            if env.env.carrying and env.env.carrying.type == 'key':
                info['carrying_key'] = True
            else:
                info['carrying_key'] = False


            # Collect data
            act_list.append([act])
            obs_next_list.append([obs_next['image']])
            rew_list.append([reward])
            done_list.append([done])
            info_list.append([info])

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
    info_np = np.concatenate(info_list)

    # Log statistics
    print(f"Observation shape: {obs_np.shape}")
    print(f"Next observation shape: {obs_next_np.shape}")
    print(f"Actions shape: {act_np.shape}")
    print(f"Rewards shape: {rew_np.shape}")
    print(f"Dones shape: {done_np.shape}")
    print(f"Number of episodes started: {episodes}")
    print(f"Unique state-action pairs visited: {len(visit_count)}")
    env.close()

    return obs_np, obs_next_np, act_np, rew_np, done_np, info_np


def save_experiments(cfg: DictConfig, obs, obs_next, act, rew, done, info=None):
    obs = ColRowCanl_to_CanlRowCol(obs)
    obs_next = ColRowCanl_to_CanlRowCol(obs_next)
    np.savez_compressed(cfg.collect.data_save_path, a=obs, b=obs_next, c=act, d=rew, e=done, f=info)

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
    env = FullyObsWrapper(CustomMiniGridEnv(txt_file_path=hparam.env_path, 
                                        custom_mission="Find the key and open the door.",
                                        max_steps=5000, render_mode=mode))
    obs, obs_next, act,rew, done = run_env(env, hparam, wandb_run=None, save_img=False)
    save_experiments(cfg.env,obs,obs_next, act, rew, done)



def data_collect_api(cfg: DictConfig, env, wandb_run, save_img=False):
    hparam = cfg.env
    obs, obs_next, act,rew, done, info = run_env(env, hparam, wandb_run, save_img=save_img)
        # 指定要过采样的动作：pickup 和 toggle
    # actions_to_oversample = [env.unwrapped.actions.toggle]
    # obs, obs_next, act, rew, done = augment_interactions(
    #     obs, obs_next, act, rew, done,
    #     actions_to_oversample,
    #     N=10  # 过采样倍数
    # )
    save_experiments(cfg.env,obs,obs_next, act, rew, done, info)

def data_collect_api_multiprocess(cfg: DictConfig, env, wandb_run, save_img=False):
    hparam = cfg.env
    obs, obs_next, act, rew, done, info = run_env_multiprocess(env, hparam, wandb_run, save_img=save_img)
    save_experiments(cfg.env,obs,obs_next, act, rew, done, info)

if __name__ == "__main__": 
    data_collect() 