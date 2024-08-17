import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import sys
sys.path.append('/home/siyao/project/rlPractice/MiniGrid')
from path import *
import os
import json
import numpy as np
import pytorch_lightning as pl
from typing import Sequence, List, Dict, Tuple, Optional, Any, Set, Union, Callable, Mapping
import wandb

class SimpleNN(pl.LightningModule):
    def __init__(self, hparams):
        super(SimpleNN, self).__init__()
        hparams = hparams.world_model
        self.save_hyperparameters(hparams)
        self.obs_size = hparams.obs_size 
        self.n_hidden = hparams.hidden_size
        self.action_size = hparams.action_size
        self.total_input_size = self.obs_size + self.action_size
        # # Define the first dense layer to process the combined input
        # self.shared_layers = nn.Sequential(
        #     nn.Linear(self.total_input_size, self.n_hidden),
        #     nn.ReLU()
        # )
        # self.state_head = nn.Sequential(
        #     nn.Linear(self.n_hidden, self.obs_size),
        #     nn.ReLU()
        # )
        # # self.reward_head = nn.Linear(hidden_size, 1)
        # # self.done_head = nn.Linear(hidden_size, 1)

        self.shared_layers = nn.Sequential(
            nn.Linear(self.total_input_size, self.n_hidden),
            nn.BatchNorm1d(self.n_hidden),  
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),  # Added hidden layer
            nn.ReLU()
        )

        self.state_head = nn.Sequential(
            nn.Linear(self.n_hidden, self.obs_size),  # Added hidden layer
            nn.ReLU()
            # nn.Linear(self.n_hidden),
            # nn.ReLU()
        )


    def forward(self, input_obs, input_action):
        input_action = input_action.unsqueeze(1)
        combined_input = torch.cat((input_obs, input_action), dim=1)
        out = self.shared_layers(combined_input)
        obs_out = self.state_head(out)
        # reward_out = torch.sigmoid(self.reward_head(out))
        # done_out = self.done_head(out)
        #
        # done_out = td.independent.Independent(
        #     td.Bernoulli(logits=done_out), 1
        # )

        return obs_out
    

    
    # def done_loss(self, predict, original):
    #     predict = predict.view(-1, predict.shape[-1])
    #     original = original.view(-1, original.shape[-1])
    #     return F.mse_loss(predict, original, reduction='mean')

    def loss_function(self, next_observation_predict, next_observations_true):
        loss = nn.MSELoss()
        loss_obs = loss(next_observation_predict, next_observations_true)
        loss = {'loss_obs':loss_obs}
        return loss
    
    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=self.hparams.wd)
        reduce_lr_on_plateau = ReduceLROnPlateau(optimizer, mode='min',verbose=True, min_lr=1e-8)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr_on_plateau,
                "monitor": 'avg_val_loss_wm',
                "frequency": 1
            },
        }


    def training_step(self, batch, batch_idx):
        obs = batch['obs']
        act = batch['act']
        obs_pred = self(obs, act)
        obs_next = batch['obs_next']
        loss = self.loss_function(obs_pred, obs_next)
        self.log_dict(loss)
        return loss['loss_obs']

    def validation_step(self, batch, batch_idx):
        obs = batch['obs']
        act = batch['act']
        obs_pred = self(obs, act)
        obs_next = batch['obs_next']
        loss = self.loss_function(obs_pred, obs_next)
        self.log_dict(loss)
        return {"loss_wm_val": loss['loss_obs']}

    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Union[torch.Tensor, Dict[str, Union[torch.Tensor, Sequence[wandb.Image]]]]]:
        avg_loss = torch.stack([x["loss_wm_val"] for x in outputs]).mean()
        self.log("avg_val_loss_wm", avg_loss)
        return {"avg_val_loss_wm": avg_loss}

    def on_save_checkpoint(self, checkpoint):
        # Example checkpoint customization: removing specific keys if needed
        t = checkpoint['state_dict']
        # You can filter out certain keys if required, for example:
        # checkpoint['state_dict'] = {key: t[key] for key in t if not key.startswith('unwanted_prefix')}
        # But since there's no VAE or specific filtering needed, this method can be left simple.
        pass  # No specific filtering needed for a simple NN