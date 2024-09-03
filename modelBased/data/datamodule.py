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

class WMRLDataset(Dataset):
    @func_set_timeout(100)
    def __init__(self, loaded, hparams):
        self.hparams = hparams
        self.obs_norm_values = hparams.obs_norm_values
        self.act_norm_values = hparams.action_norm_values
        self.data = self.make_data(loaded)

    @func_set_timeout(100)
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

    def __len__(self):
        return len(self.data['obs'])  # Assuming 'obs' is the main reference for dataset length

    def __getitem__(self, idx):
        return {key: self.data[key][idx] for key in self.data}  # Return a dictionary of all data items

class WMRLDataModule(pl.LightningDataModule):
    def __init__(self, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_dir = self.hparams.data_dir
        print(self.hparams)
        
    def setup(self, stage: Optional[str] = None):
        loaded = np.load(self.data_dir, allow_pickle=True)  # Allow pickle for safety with complex data structures
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
