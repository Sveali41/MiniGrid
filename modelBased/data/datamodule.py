import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import sys
sys.path.append('/home/siyao/project/rlPractice/MiniGrid')
from modelBased.common.utils import get_env
from typing import Tuple, List, Any, Dict, Optional
import os.path
# import src.env.run_env_save as env_run_save
import numpy as np
import matplotlib.pyplot as plt
import torch
import multiprocessing
import time
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
    # @func_set_timeout(100)
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

    @func_set_timeout(1000)
    def make_data(self, loaded):
        obs = self.normalize(loaded['a'])
        obs_next = self.normalize(loaded['b'])

        act = loaded['c'].astype(np.float32) / self.act_norm_values # Normalize the action with the max 6
        # done = loaded['d'].astype(int)  # Convert boolean values to binary 0-1

        # Create a dictionary to store processed data
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
        """
        obs: same as normal
        obs_next: the 3x3 square around the agent with delta change
        """
        obs = self.normalize(loaded['a']) # (batch_size, width*height*channels)
        # output is the changes between the current state and the next state
        obs_delta = self.delta_batch_preprocess(loaded['a'], loaded['b']) # (batch_size, 3*3*3)
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
    
    def create_3x3_mask(self, state_shape, agent_position):
        """
        Creates a mask that selects a 3x3 square around the agent's position.

        Parameters:
            state_shape (tuple): The shape of the gridworld state array, e.g., (height, width, channels).
            agent_position (tuple): The (y, x) coordinates of the agent's position.

        Returns:
            np.ndarray: A boolean mask array of the same shape as the state, with True in the 3x3 square around the agent.
        """
        # Initialize a mask of the same shape as the state but only for height and width dimensions
        mask = np.zeros((state_shape[0], state_shape[1]), dtype=bool)
        
        # Calculate the boundaries of the 3x3 square around the agent, ensuring they stay within bounds
        y, x = agent_position
        y_start, y_end = max(0, y - 1), min(state_shape[0], y + 2)
        x_start, x_end = max(0, x - 1), min(state_shape[1], x + 2)
        
        # Set the 3x3 square area around the agent to True
        mask[y_start:y_end, x_start:x_end] = True
        
        return mask
    
    def delta_batch_preprocess(self, state, next_state):
        delta_state = np.zeros((state.shape[0], 3 * 3 * state.shape[-1])) 
        for i in range(state.shape[0]):
            delta_state[i] = self.delta_state(state[i], next_state[i]).reshape(-1)
        return delta_state

    def delta_state(self, state, next_state):
        agent_position = np.argwhere(state[:, :, 0] == 10)[0]
        delta_state = next_state.astype(np.int16)-state.astype(np.int16)
        mask = self.create_3x3_mask(state.shape, agent_position)
        delta_state = delta_state[mask].reshape((3, 3, 3))
        return delta_state


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
            num_workers=self.hparams.n_cpu,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_test, 
            batch_size=self.hparams.batch_size, 
            shuffle=False,
            num_workers=self.hparams.n_cpu,
            pin_memory=True,
            persistent_workers=True
        )
