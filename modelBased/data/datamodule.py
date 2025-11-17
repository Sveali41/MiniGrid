import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.append('/home/siyao/project/rlPractice/MiniGrid')
from modelBased.common.utils import get_env, normalize_obs
from typing import Tuple, List, Any, Dict, Optional
# import src.env.run_env_save as env_run_save
import torch
from modelBased.common import utils
from func_timeout import func_set_timeout
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional
from func_timeout import func_set_timeout  # Ensure you have this package
from modelBased.common.utils import merge_data_dicts


def extract_agent_cross_mask(state):
        """
        Extract a cross-shaped mask centered on the agent's position.
        
        Parameters:
            state (np.ndarray): The 3D array representing the gridworld state.
                                
        Returns:
            np.ndarray: A 3D array of extracted content for the cross-shaped area
                        around the agent, with the layout of 3*3 square, padding with 0.
                        or None if agent is not found.
        """
        # Find agent's position in the grid
        # For the agent position, the object value is 10

                
        agent_position = np.argwhere(state[:, :, 0] == 10)

        # Check if the agent position is found
        if len(agent_position) == 0:
            # Could't find the agent position where =10,  take the position where closest to 10 as agent position
            index = np.argmax(state[:, :, 0])
            print(f"Warning! Agent position not found, assume max value: {state[:, :, 0].max()} as agent")
            y, x = index // state.shape[1], index % state.shape[1]
            # return None
        else:
            # Extract y, x coordinates of the agent's position
            y, x = agent_position[0]
            

        cross_structure = np.full((3, 3, state.shape[2]), 0)  # Create a 3x3 structure with None values

        # Extract the content for each valid neighbor position
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < state.shape[0] and 0 <= nx < state.shape[1]:
                cross_structure[dy + 1, dx + 1] = state[ny, nx]  # Place content in the cross structure

        return cross_structure

class WMRLDataset(Dataset):
    @func_set_timeout(100)
    def __init__(self, loaded, hparams, replay_data=None):
        self.hparams = hparams
        self.obs_norm_values = hparams.obs_norm_values
        self.act_norm_values = hparams.action_norm_values
        self.data = self.make_data(loaded, replay_data)

    def state_batch_preprocess(self, state):
        obs = np.zeros((state.shape[0], 3, 3, state.shape[-1])) # The mask will extract a 3x3 square around the agent
        for i in range(state.shape[0]):  # Loop over the last dimension (channels)
            obs[i] = extract_agent_cross_mask(state[i])
        return obs

    @func_set_timeout(1000)
    def make_data(self, loaded, replay_data=None):
        import numpy as np
        rng = np.random.default_rng()  # 统一随机源

        # ===== 基础取数 =====
        mask_size = self.hparams.attention_mask_size
        env_type  = self.hparams.env_type
        obs, obs_next, act = loaded['a'], loaded['b'], loaded['c']
        info = loaded.get('f', None) if env_type == 'with_obj' else None

        current_n = len(obs)
        assert current_n == len(obs_next) == len(act), "[BUG] Current lengths inconsistent!"

        # ===== (1) 控制 replay 占比：replay ≤ new =====
        # 从 hparams 读取可选的比例配置；默认 0.5
        replay_frac = float(getattr(self.hparams, "replay_frac", 0.5))
        replay_frac = max(0.0, min(1.0, replay_frac))  # clamp 到 [0,1]
        max_replay = int(current_n * replay_frac)

        if replay_data is not None and 'obs' in replay_data and replay_data['obs'] is not None:
            R = len(replay_data['obs'])
            # 只抽取不超过 max_replay 的样本，避免“replay > new”
            if R > max_replay and max_replay > 0:
                idx = rng.choice(R, size=max_replay, replace=False)
                r_obs      = replay_data['obs'][idx]
                r_obs_next = replay_data['obs_next'][idx]
                r_act      = replay_data['act'][idx]
                r_info     = (replay_data['info'][idx]
                            if (env_type == 'with_obj' and 'info' in replay_data and replay_data['info'] is not None)
                            else None)
            else:
                r_obs, r_obs_next, r_act = replay_data['obs'], replay_data['obs_next'], replay_data['act']
                r_info = (replay_data['info']
                        if (env_type == 'with_obj' and 'info' in replay_data and replay_data['info'] is not None)
                        else None)

            # 拼接
            obs      = np.concatenate([obs,      r_obs     ], axis=0)
            obs_next = np.concatenate([obs_next, r_obs_next], axis=0)
            act      = np.concatenate([act,      r_act     ], axis=0)
            if env_type == 'with_obj' and info is not None and r_info is not None:
                info = np.concatenate([info, r_info], axis=0)

            # 统一洗牌（很重要：避免“当前在前、replay 在后”的顺序偏置）
            N = len(obs)
            perm = rng.permutation(N)
            obs, obs_next, act = obs[perm], obs_next[perm], act[perm]
            if env_type == 'with_obj' and info is not None and len(info) == N:
                info = info[perm]

            print(f"Adding replay buffer with {min(R, max_replay)} samples (capped by ratio {replay_frac:.2f}).")

        # ===== (2) 生成 Δ 并做数值稳定 =====
        if self.hparams.data_type == 'norm':
            obs_f      = normalize_obs(obs,      self.obs_norm_values).astype(np.float32)
            obs_next_f = normalize_obs(obs_next, self.obs_norm_values).astype(np.float32)
            act_f      = act.astype(np.float32) / self.act_norm_values
            obs_delta  = (obs_next_f - obs_f).astype(np.float32)

        elif self.hparams.data_type == 'discrete':
            # 用 int16 做差，再转回 float32 做 MSE，防止梯度爆/精度损失
            obs_delta = (obs_next.astype(np.int16) - obs.astype(np.int16)).astype(np.float32)
            # 若你的离散网格相邻变化为主，建议裁剪到 [-1, 1]（可配开关）
            if getattr(self.hparams, "clip_discrete_delta", True):
                np.clip(obs_delta, -1, 1, out=obs_delta)
            obs_f = obs  # 保留原离散值以便可视化/调试
            act_f = act.astype(np.int64)

        else:
            raise ValueError(f"Invalid data type: {self.hparams.data_type}")

        # ===== (3) 打包 =====
        data = {'obs': obs_f, 'obs_next': obs_delta, 'act': act_f}
        if env_type == 'with_obj' and info is not None:
            data['info'] = info
        return data

            



    def __len__(self):
        lengths = [len(self.data[k]) for k in self.data]
        if not all(l == lengths[0] for l in lengths):
            print(f"[BUG] Inconsistent lengths! { {k: len(self.data[k]) for k in self.data} }")
        return lengths[0]  # 以第一个 key 的长度为准

    def __getitem__(self, idx):
        try:
            return {key: self.data[key][idx] for key in self.data}
        except IndexError as e:
            print(f"[ERROR] idx={idx}, dataset length={len(self)}")
            raise e


class WMRLDataModule(pl.LightningDataModule):
    def __init__(self, hparams=None, data: Optional[Dict[str, np.ndarray]] = None, replay_data: Optional[Dict[str, np.ndarray]] = None):
        """
        Initialize with hyperparameters and optionally directly with data.

        Parameters:
            hparams: Hyperparameters for data processing and dataloaders
            data: Optional data dictionary, e.g., {'a': np.array(...), 'b': np.array(...), 'c': np.array(...)}
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_dir = self.hparams.data_dir
        self.direct_data = data  # Store the data passed directly
        self.replay_data = replay_data
        
    def setup(self, stage: Optional[str] = None):
        if self.direct_data is not None:
            loaded = self.direct_data  # Use directly passed data
        else:
            # Load data from a file if `self.data_dir` is set and data is not provided directly
            loaded = np.load(self.data_dir, allow_pickle=True) # Allow pickle for safety with complex data structures
        data = WMRLDataset(loaded, self.hparams, self.replay_data)
        split_size = int(len(data) * 9 / 10)
        # self.data_train, self.data_test = torch.utils.data.random_split(
        #     data, [split_size, len(data) - split_size]
        # )
        self.data_train = torch.utils.data.Subset(data, range(0, split_size))
        self.data_test = torch.utils.data.Subset(data, range(split_size, len(data)))
        if len(self.data_test) < 256:
            raise ValueError("The test set is too small. Please ensure the dataset is large enough for splitting.")

    def train_dataloader(self):
        return DataLoader(
            self.data_train, 
            batch_size=self.hparams.batch_size, 
            shuffle=True,
            drop_last=True,
            num_workers=self.hparams.n_cpu,
            pin_memory=True,
            persistent_workers=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_test, 
            batch_size=self.hparams.batch_size, 
            shuffle=True,
            drop_last=True,
            num_workers=self.hparams.n_cpu,
            pin_memory=True,
            persistent_workers=False
        )
