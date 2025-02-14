import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.append('/home/siyao/project/rlPractice/MiniGrid')
from modelBased.common.utils import get_env
from typing import Tuple, List, Any, Dict, Optional
# import src.env.run_env_save as env_run_save
import torch
from common import utils
from func_timeout import func_set_timeout

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional
from func_timeout import func_set_timeout  # Ensure you have this package
import hydra
from common.utils import PROJECT_ROOT, get_env

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
        if self.hparams.model == 'Rmax':
            self.data = self.make_data_Rmax(loaded)
        if self.hparams.model == 'Attention':
            self.data = self.make_data_attention(loaded)
        else:
            self.data = self.make_data(loaded)


    @func_set_timeout(100)
    def make_data(self, loaded):
        if not self.hparams.feature_extration_mode == 'discrete':
            obs = self.normalize(loaded['a'])
            obs_next = self.normalize(loaded['b'])
            act = loaded['c'].astype(np.float32) / self.act_norm_values 
        else:
            obs = loaded['a']
            obs_next = loaded['b']
            act = loaded['c']
            #  因为要做one hot 所以数据要连续
            from common import utils
            obs[:,:,:,0] = utils.replace_values(obs[:,:,:,0], np.array([1,2,8,10]), np.array([0, 1, 2, 3]))
            obs[:,:,:,1] = utils.replace_values(obs[:,:,:,1], np.array([5]), np.array([2]))

            obs_next[:,:,:,0] = utils.replace_values(obs_next[:,:,:,0], np.array([1,2,8,10]), np.array([0, 1, 2, 3]))
            obs_next[:,:,:,1] = utils.replace_values(obs_next[:,:,:,1], np.array([5]), np.array([2]))

        data = {
            'obs': obs,
            'obs_next': obs_next,
            'act': act,
            # 'done': done
        }
        return data
    

    
    def state_batch_preprocess(self, state):
        obs = np.zeros((state.shape[0], 3, 3, state.shape[-1])) # The mask will extract a 3x3 square around the agent
        for i in range(state.shape[0]):  # Loop over the last dimension (channels)
            obs[i] = extract_agent_cross_mask(state[i])
        return obs

    def make_data_Rmax(self, loaded):
        obs = self.state_batch_preprocess(loaded['a'])
        obs = self.normalize(obs)
        # output is the changes between the current state and the next state
        obs_delta = self.delta_batch_preprocess(loaded['a'], loaded['b'])
        act = loaded['c'].astype(np.float32) / self.act_norm_values # Normalize the action with the max 6
        # done = loaded['d'].astype(int)  # Convert boolean values to binary 0-1

        # Create a dictionary to store processed data
        data = {
            'obs': obs,
            'obs_next': obs_delta, # obs_next is the delta between the current state and the next state
            'act': act,
            # 'done': done
        }
        return data
    
    def make_data_attention(self, loaded):
        mask_size = self.hparams.attention_mask_size
        _, col, row, channel = loaded['a'].shape


        if not self.hparams.feature_extration_mode == 'discrete':
            obs = self.normalize(loaded['a']).reshape(-1, col, row, channel)
            obs_next = self.normalize(loaded['b']).reshape(-1, col, row, channel)
            act = loaded['c'].astype(np.float32) / self.act_norm_values 
        else:
            obs = loaded['a']
            obs_next = loaded['b']
            act = loaded['c']
            obs[:,:,:,0] = utils.replace_values(obs[:,:,:,0], np.array([1,2,8,10]), np.array([0, 1, 2, 3]))
            obs[:,:,:,1] = utils.replace_values(obs[:,:,:,1], np.array([5]), np.array([2]))

            obs_next[:,:,:,0] = utils.replace_values(obs_next[:,:,:,0], np.array([1,2,8,10]), np.array([0, 1, 2, 3]))
            obs_next[:,:,:,1] = utils.replace_values(obs_next[:,:,:,1], np.array([5]), np.array([2]))

        obs = utils.ColRowCanl_to_CanlRowCol(obs)
        obs_next = utils.ColRowCanl_to_CanlRowCol(obs_next)
        obs_delta = self.delta_batch_preprocess(obs, obs_next, mask_size, channel) 

  
        # Create a dictionary to store processed data
        data = {
            'obs': obs,
            'obs_next': obs_delta, # obs_next is the delta between the current state and the next state
            'act': act,
            # 'done': done
        }
        return data
        

    def normalize(self, x):
        """Normalize the obs data and flatten it."""
        if not np.issubdtype(x.dtype, np.floating):
            x = x.astype(np.float32) 
        
        # Ensure that the norm_values is not None and has the correct length
        if self.obs_norm_values is None or len(self.obs_norm_values) != x.shape[-1]:
            raise ValueError("Normalization values must be provided and must match the number of channels in the data.")
        
        # Normalize each channel using the provided norm values
        for i in range(x.shape[-1]):  # Loop over the last dimension (channels)
            max_value = self.obs_norm_values[i]
            if max_value != 0:  # Avoid division by zero
                x[:, :, :, i] = x[:, :, :, i] / max_value  
        
        # Flatten the data
        x = x.reshape(x.shape[0], -1)
        return x
    

    
    def delta_batch_preprocess(self, state, next_state, mask_size, channel):
        delta_state = np.zeros((state.shape[0], channel * mask_size * mask_size)) 
        for i in range(state.shape[0]):
            delta_state[i] = self.delta_state(state[i], next_state[i], mask_size).reshape(-1)
        return delta_state

    def delta_state(self, state, next_state, mask_size):
        if self.hparams.feature_extration_mode == 'discrete':
            agent_position_yx = np.argwhere(state[0, :, :] == 3)[0]
            delta_state = next_state.astype(np.int16)-state.astype(np.int16)
        else:
            agent_position_yx = np.argwhere(state[0, :, :] == 1)[0]
            delta_state = next_state.astype(np.float32)-state.astype(np.float32)
        masked_delta_state = utils.extract_masked_state(delta_state, agent_position_yx, mask_size)
        return masked_delta_state


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
