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
from modelBased.AttentionWM import *
import transformer_support

class SimpleNN(pl.LightningModule):
    def __init__(self, hparams):
        super(SimpleNN, self).__init__()
        hparams = hparams.world_model
        self.save_hyperparameters(hparams)
        self.obs_size = hparams.obs_size 
        self.n_hidden = hparams.hidden_size
        self.action_size = hparams.action_size
        self.total_input_size = self.obs_size + self.action_size
        self.algo = hparams.model
        # self.visualizationFlag = hparams.visualization
        # self.visualize_every = hparams.visualize_every
        # self.save_path = hparams.save_path
        # self.step_counter = 0  # init step counter 
        # self.action_map = hparams.action_map 
        # self.direction_map = hparams.direction_map

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
        if self.algo == 'Attention':
            self.topk = hparams.attention_model.topk
            self.total_input_size = 4 * self.topk + self.action_size
            self.extract_layer = self.load_attention_model(hparams.attention_model)
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
            
            self.state_head_Att = nn.Linear(self.n_hidden, 27) # predict the change centric of the agent

        else:
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
            
            self.state_head_Rmax = nn.Linear(self.n_hidden, self.obs_size)


    def forward(self, input_obs, input_action):
        if self.algo.lower() != 'Attention'.lower():
            if input_obs.dim() == 1 and input_action.dim() == 1:
                # Add a batch dimension to both inputs
                input_obs = input_obs.unsqueeze(0)  # Shape becomes [1, 54]
                input_action = input_action.unsqueeze(0)  # Shape becomes [1, 1]
                combined_input = torch.cat((input_obs, input_action), dim=1)
            else:
                input_action = input_action.unsqueeze(1)
                combined_input = torch.cat((input_obs, input_action), dim=1)
            # Convert combined_input to Float if necessary
            if combined_input.dtype != torch.float32:
                combined_input = combined_input.float()
            out = self.shared_layers(combined_input)
            if self.algo=='Rmax':
                obs_out = self.state_head_Rmax(out) 
            else:
                obs_out = self.state_head(out)
        else:
            _, attention_weight = self.extract_layer(input_obs, input_action)
            extracted_regions = self.extract_topk_regions_without_padding(input_obs, attention_weight, self.topk) #(batch_size, 3 channel, 9topk)
            extracted_regions = extracted_regions.view(extracted_regions.size(0), -1)
            input_action = input_action.unsqueeze(1)
            combined_input = torch.cat((extracted_regions, input_action), dim=1)  # Shape becomes [1, 1]
            out = self.shared_layers(combined_input)
            obs_out = self.state_head_Att(out)
            # input the extracted feature to the model


        # reward_out = torch.sigmoid(self.reward_head(out))
        # done_out = self.done_head(out)
        #
        # done_out = td.independent.Independent(
        #     td.Bernoulli(logits=done_out), 1
        # )

        return obs_out
    

    def load_attention_model(self, cfg):
        """
        Load the attention model.

        Parameters:
            cfg: The configuration object.

        Returns:
            The loaded attention model.
        """
        hparams = cfg
        model = transformer_support.ExtractionModule(hparams.action_size, hparams.embed_dim, hparams.num_heads)
        # Load the checkpoint
        checkpoint = torch.load(hparams.pth_folder)
        # Load state_dict into the model
        model.load_state_dict(checkpoint['state_dict'])
        # Set the model to evaluation mode (optional, depends on use case)
        extraction_module = model
        extraction_module.eval()
        for param in extraction_module.parameters():
            param.requires_grad = False # Freeze the model
        return extraction_module
    
    def extract_topk_regions_padding(self, state, attention_weights, topk=9):
        """
        according to attention weights, extract top-k regions from state
        :param state: state features, shape = (batch_size, seq_len, state_dim)
        :param attention_weights: attention weights, shape = (batch_size, seq_len)
        :param topk: number of regions to extract
        :return: 
            extracted_regions: extracted regions, shape = (batch_size, topk, state_dim)
            topk_indices: selected indices, shape = (batch_size, topk)
        """
        # acquire top-k indices
        batch_size = state.size(0)
        _, topk_indices = torch.topk(attention_weights, topk, dim=1)  # (batch_size, topk)
        state = state.permute(0, 2, 3, 1)
        attention_weights = attention_weights.view(batch_size, state.size(1), state.size(2))
        state = state.reshape(batch_size, -1, 3)
        output_data = torch.zeros_like(state)
        for i in range(batch_size):
            for idx in topk_indices[i]:
                output_data[i, idx] = state[i, idx]
        return output_data


    def generate_positional_encoding(self, position, embed_dim):
        position_encoding = torch.zeros(position.size(0), embed_dim, device=position.device)
        for i in range(embed_dim // 2):
            position_encoding[:, 2 * i] = torch.sin(position[:, 0] / (10000 ** (2 * i / embed_dim)))
            position_encoding[:, 2 * i + 1] = torch.cos(position[:, 0] / (10000 ** (2 * i / embed_dim)))
        return position_encoding


    def extract_topk_regions_without_padding(self, state, attention_weights, topk=5):
        """
        according to attention weights, extract top-k regions from state
        :param state: state features, shape = (batch_size, seq_len, state_dim)
        :param attention_weights: attention weights, shape = (batch_size, seq_len)
        :param topk: number of regions to extract
        :return: 
            extracted_regions: extracted regions, shape = (batch_size, topk, state_dim)
            topk_indices: selected indices, shape = (batch_size, topk)
        """
        # acquire top-k indices
        topk_values, topk_indices = torch.topk(attention_weights, topk, dim=1)  # (batch_size, topk)

        # acquire relative posistion
        agent_position = torch.argwhere(state[:, 0, :, :] == 1)[:,1:]
        agent_position_1d = agent_position[:, 0] * 12 + agent_position[:, 1]
        relative_position = topk_indices- agent_position_1d.unsqueeze(1)
        batch_size, channel, row, col = state.shape
        relative_position = relative_position / (row * col - 1)  # normlization 

        # topK regions
        extracted_regions = torch.gather(state.flatten(2), dim=2, index=topk_indices.unsqueeze(1).expand(-1, channel, -1))

        # topK regions + relative posistion
        extracted_regions_position = torch.cat((extracted_regions, relative_position.unsqueeze(1)), dim=1)
        return extracted_regions_position



    def loss_function(self, next_observations_predict, next_observations_true):
        loss = nn.MSELoss()
        loss_obs = loss(next_observations_predict, next_observations_true)
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
        if self.algo == 'Attention':
            obs_temp = obs.view(obs.size(0), 12, 6, 3)  # convert to (batch_size, width, height, channels)
            obs = obs_temp.permute(0, 3, 2, 1)  # convert to (batch_size, channels, height, width)
        obs_pred = self(obs, act)
        obs_next = batch['obs_next']
        if obs_next.dtype != obs_pred.dtype:
            obs_next = obs_next.float()
        loss = self.loss_function(obs_pred, obs_next)
        self.log_dict(loss)

        # ## visualization
        # self.step_counter += 1
        # if self.visualizationFlag and (self.step_counter % self.visualize_every == 0):
        #     """
        #     obs: curent state
        #     action = action taken
        #     attentionWeight = attention weight 
        #     obs_next = next state
        #     obs_pred = predicted next state
        #     """
        #     self.visualization(obs, act, attention_weight)
        return loss['loss_obs']

    def validation_step(self, batch, batch_idx):
        obs = batch['obs']
        act = batch['act']
        if self.algo == 'Attention':
            obs_temp = obs.view(obs.size(0), 12, 6, 3)  # convert to (batch_size, width, height, channels)
            obs = obs_temp.permute(0, 3, 2, 1)  # convert to (batch_size, channels, height, width)
        obs_pred = self(obs, act)
        obs_next = batch['obs_next']
        if obs_next.dtype != obs_pred.dtype:
            obs_next = obs_next.float()
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
        pass  # No specific filtering needed for a simple NN

    


