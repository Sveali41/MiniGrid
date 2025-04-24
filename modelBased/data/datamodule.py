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
    def __init__(self, loaded, hparams):
        self.hparams = hparams
        self.obs_norm_values = hparams.obs_norm_values
        self.act_norm_values = hparams.action_norm_values
        self.data = self.make_data(loaded)

    def state_batch_preprocess(self, state):
        obs = np.zeros((state.shape[0], 3, 3, state.shape[-1])) # The mask will extract a 3x3 square around the agent
        for i in range(state.shape[0]):  # Loop over the last dimension (channels)
            obs[i] = extract_agent_cross_mask(state[i])
        return obs

    @func_set_timeout(1000)
    def make_data(self, loaded):
        mask_size = self.hparams.attention_mask_size
        B, row, col, channel = loaded['a'].shape
        obs = loaded['a']
        obs_next = loaded['b']
        act = loaded['c']
        
        if self.hparams.data_type == 'norm':
            obs = normalize_obs(loaded['a'], self.obs_norm_values)
            obs_next = normalize_obs(loaded['b'], self.obs_norm_values)
            act = act.astype(np.float32) / self.act_norm_values 
            obs_delta = obs_next.astype(np.float32)-obs.astype(np.float32)

        elif self.hparams.data_type == 'discrete':
            obs = loaded['a']
            obs_next = loaded['b']
            act = loaded['c']
            # obs[:,0,:,:] = utils.replace_values(obs[:,0,:,:], np.array([1,2,8,10]), np.array([0, 1, 2, 3]))
            # obs[:,1,:,:] = utils.replace_values(obs[:,1,:,:], np.array([5]), np.array([2]))

            # obs_next[:,0,:,:] = utils.replace_values(obs_next[:,0,:,:], np.array([1,2,8,10]), np.array([0, 1, 2, 3]))
            # obs_next[:,1,:,:] = utils.replace_values(obs_next[:,1,:,:], np.array([5]), np.array([2]))
            obs_delta = obs_next.astype(np.int16)- obs.astype(np.int16)
        else:
            raise ValueError(f"Invalid data type: {self.hparams.data_type}")

        if mask_size > 0:
            agent_position_yx = utils.get_agent_position(obs)
            obs_delta = utils.extract_masked_state(obs_delta, mask_size, agent_position_yx)

        data = {
            'obs': obs,
            'obs_next': obs_delta, # obs_next is the delta between the current state and the next state
            'act': act,
        }
        return data
        



    def __len__(self):
        return len(self.data['obs'])  # Assuming 'obs' is the main reference for dataset length

    def __getitem__(self, idx):
        return {key: self.data[key][idx] for key in self.data}  # Return a dictionary of all data items

class WMRLDataModule(pl.LightningDataModule):
    def __init__(self, hparams=None, data: Optional[Dict[str, np.ndarray]] = None):
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
        
    def setup(self, stage: Optional[str] = None):
        if self.direct_data is not None:
            loaded = self.direct_data  # Use directly passed data
        else:
            # Load data from a file if `self.data_dir` is set and data is not provided directly
            loaded = np.load(self.data_dir, allow_pickle=True) # Allow pickle for safety with complex data structures
        data = WMRLDataset(loaded, self.hparams)
        split_size = int(len(data) * 9 / 10)
        self.data_train, self.data_test = torch.utils.data.random_split(
            data, [split_size, len(data) - split_size]
        )
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
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_test, 
            batch_size=self.hparams.batch_size, 
            shuffle=False,
            drop_last=True,
            num_workers=self.hparams.n_cpu,
            pin_memory=True,
            persistent_workers=True
        )
