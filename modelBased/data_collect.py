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

def visualize_env(env, cfg: DictConfig, save_img=False):
    env.reset()[0]
    img = env.get_frame()
    return img 


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

def run_env_vectorized(env, cfg: DictConfig, wandb_run, policy=None, rmax_exploration=None, save_img=False):
    import copy
    from gym.vector import AsyncVectorEnv
    """
    Vectorized data collection with AsyncVectorEnv, handling variable episode lengths.
    Collects cfg.collect.episodes full episodes per parallel env.
    """
        # set device to cpu or cuda
    device = torch.device('cpu')
    if save_img and wandb_run is not None:
        _ = env.reset()[0]
        img = env.get_frame()
        wandb_run.log({"Mini-tasks": wandb.Image(img)})

    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    num_envs = cfg.collect.num_workers

        # Build factory that deep-copies the single env into independent instances
    def make_env():
        base = copy.deepcopy(env)
        # convert dict observation to image-only Box space
        return ImgObsWrapper(base)

    envs = AsyncVectorEnv([make_env for _ in range(num_envs)])
    # Reset all envs and prepare trackers
    obs_batch = envs.reset()  # shape: (num_envs, H, W, C)
    episodes_done = [0] * num_envs
    obs_list, obs_next_list, act_list, rew_list, done_list, info_list = [], [], [], [], [], []
    meaningful_actions = [env.unwrapped.actions.forward, env.unwrapped.actions.left, env.unwrapped.actions.right, env.unwrapped.actions.pickup, env.unwrapped.actions.toggle]

    # Continue until each env has completed desired episodes
    while any(ed < cfg.collect.episodes for ed in episodes_done):
        # Sample actions
        if policy is None:
            acts = np.random.choice(
                meaningful_actions,
                size=num_envs,
                p=[0.3,0.15,0.15,0.2,0.2]
            )
        else:
            # Batch forward
            state_norm = normalize_obs(obs_batch['image']).to(device)
            act = policy.select_action(state_norm)

        # Step all envs
        next_obs, rewards, dones, _, infos = envs.step(acts)

        # Optionally add custom info fields, for example whether agent is carrying a key
        if hasattr(env.unwrapped, "carrying") and env.unwrapped.carrying:
            infos["carrying_key"] = (env.unwrapped.carrying.type == 'key')
        else:
            infos["carrying_key"] = False

                # Record transitions for envs still collecting
        for i in range(num_envs):
            if episodes_done[i] < cfg.collect.episodes:
                obs_list.append(obs_batch[i])
                obs_next_list.append(next_obs[i])
                act_list.append([acts[i]])
                rew_list.append([rewards[i]])
                done_list.append([dones[i]])
                info_list.append(infos[i])
                if dones[i]:
                    episodes_done[i] += 1

        obs_batch = next_obs

    # Convert lists to arrays
    obs_buf      = np.stack(obs_list,      axis=0)
    obs_next_buf = np.stack(obs_next_list, axis=0)
    act_buf      = np.array(act_list,      dtype=np.int32)
    rew_buf      = np.array(rew_list,      dtype=np.float32)
    done_buf     = np.array(done_list,     dtype=bool)
    # infos may be a list of dicts; keep as list or convert to object array
    info_buf     = np.array(info_list,     dtype=object)
    envs.close()
    print(f"Collected: {obs_buf.shape[0]} steps from {num_envs} envs")
    return obs_buf, obs_next_buf, act_buf, rew_buf, done_buf, info_buf

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


def run_env_worker(args):
    env_fn, cfg, wandb_run, policy, rmax_exploration, save_img = args
    env = env_fn()  # 每个子进程单独创建自己的环境
    return run_env(env, cfg, wandb_run, policy, rmax_exploration, save_img)

def run_env_multiprocess(cfg, wandb_run, policy=None, rmax_exploration=None, save_img=False, num_workers=4):
    import multiprocessing as mp
    from modelBased.common.utils import get_env

    # 设置多进程启动方式（只需设置一次）
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # 如果已经设置过就忽略

    # 环境构造函数：每个子进程用它创建独立环境
    env_fn = lambda: get_env(cfg.env.name)

    # 多个子进程的参数列表
    args_list = [(env_fn, cfg, wandb_run, policy, rmax_exploration, save_img) for _ in range(num_workers)]

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(run_env_worker, args_list)

    # 合并结果
    obs_np, obs_next_np, act_np, rew_np, done_np, info_np = zip(*results)

    return (
        np.concatenate(obs_np),
        np.concatenate(obs_next_np),
        np.concatenate(act_np),
        np.concatenate(rew_np),
        np.concatenate(done_np),
        np.concatenate(info_np),
    )
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



def data_collect_api(cfg: DictConfig, env, wandb_run, save_img):
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

def data_collect_api_vectorized(cfg: DictConfig, env, wandb_run, save_img=False):
    hparam = cfg.env
    obs, obs_next, act, rew, done, info = run_env_vectorized(env, hparam, wandb_run, save_img=save_img)
    save_experiments(cfg.env,obs,obs_next, act, rew, done, info)

if __name__ == "__main__": 
    data_collect() 