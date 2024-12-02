import torch
from torch import nn
import pytorch_lightning as pl
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Sequence, List, Dict, Tuple, Optional, Any, Set, Union, Callable, Mapping
import numpy as np
import matplotlib.pyplot as plt
import os
"""
在copy5的基础上, 增加了direction embedding的维度

"""
print("start training-> tansformer6_best.py")
class ExtractionModule(nn.Module):
    def __init__(self, action_dim, embed_dim, num_heads):
        super(ExtractionModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, embed_dim -2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim - 2),
            nn.LeakyReLU(negative_slope=0.01),
            # nn.Conv2d(embed_dim -2, embed_dim -2, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(negative_slope=0.01)
            # nn.Dropout(0.3)
        )
        self.action_embedding = nn.Linear(action_dim, embed_dim // 2)
        self.dir_embedding = nn.Linear(1, embed_dim // 2 - 2)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)

    def generate_positional_encoding(self, position, embed_dim):
        position_encoding = torch.zeros(position.size(0), embed_dim, device=position.device)
        for i in range(embed_dim // 2):
            position_encoding[:, 2 * i] = torch.sin(position[:, 0] / (10000 ** (2 * i / embed_dim)))
            position_encoding[:, 2 * i + 1] = torch.cos(position[:, 0] / (10000 ** (2 * i / embed_dim)))
        return position_encoding

    def forward(self, state, action):
        # state embedding
        state_embed = self.conv(state)  # (batch_size, embed_dim, grid_height, grid_width)
        state_embed = state_embed.flatten(2).permute(0, 2, 1)  # (128, 72, 62)

        # agent position embedding
        agent_position = torch.argwhere(state[:, 0, :, :] == 1)[:,1:]
        agent_position_embed = agent_position.unsqueeze(1)  # (128, 1, 2)

        # direction embedding          
        dir = state[:,2,:, :]
        row = agent_position[:, 0]
        col = agent_position[:, 1]
        batch_indices = torch.arange(state.size(0))
        dir = dir[batch_indices, row, col] 
        dir_emdedding = self.dir_embedding(dir.unsqueeze(-1)).unsqueeze(1)

        # position embedding
        position = torch.arange(state_embed.size(1), device=state.device).unsqueeze(1)  
        position_encoding = self.generate_positional_encoding(position, 2) 
        position_encoding = position_encoding.unsqueeze(0).expand(state.size(0), -1, -1)  

        # action embedding
        action_embed = self.action_embedding(action.unsqueeze(1)).unsqueeze(1)  

        # action + position + direction
        action_embed = torch.cat([action_embed, agent_position_embed], dim=-1) 
        action_embed = torch.cat([action_embed, dir_emdedding], dim=-1)

        #  state + position 
        state_embed = torch.cat([state_embed, position_encoding], dim=-1)  

        # attention
        attention_output, attention_weights = self.attention(query=action_embed, key=state_embed, value=state_embed)
        return attention_output.squeeze(1), attention_weights.squeeze(1)

class PredictionModule(nn.Module):
    def __init__(self, embed_dim, state_dim, hidden_dim=128):
        super(PredictionModule, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        # self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_dim, state_dim)

    def forward(self, extracted_features):
        x = self.fc1(extracted_features)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        output = self.bn2(x)
        x = torch.nn.functional.relu(x)
        output = self.fc3(x)
        return torch.nn.functional.softplus(output)

class IntegratedModel(nn.Module):
    """
    integrate extraction module and prediction module
    """
    def __init__(self, state_dim, action_dim, embed_dim, num_heads, freeze_weight=False, weight_path=''):
        super(IntegratedModel, self).__init__()
        self.extraction_module = ExtractionModule(action_dim, embed_dim, num_heads)
        self.prediction_module = PredictionModule(embed_dim, state_dim)
        ## Freaze the extraction module & Local test
        if freeze_weight:
            checkpoint = torch.load(weight_path)
            self.extraction_module.load_state_dict(checkpoint['state_dict'])
            self.extraction_module.eval()
            for param in self.extraction_module.parameters():
                param.requires_grad = False # Freeze the model

    def forward(self, state, action):
        # extract features from state and action
        extracted_features, attentionWeight = self.extraction_module(state, action)
        # predict next state from extracted features
        next_state_pred = self.prediction_module(extracted_features)
        
        return next_state_pred, attentionWeight
    

class IntegratedPredictionModel(pl.LightningModule):
    def __init__(self, hparams):
        super(IntegratedPredictionModel, self).__init__()
        self.state_dim = hparams.obs_size  
        self.action_dim = hparams.action_size
        self.embed_dim = hparams.embed_dim
        self.num_heads = hparams.num_heads
        self.learning_rate= hparams.lr
        self.weight_decay = hparams.wd
        self.visualization_seperate = hparams.visualization_seperate
        self.visualization_together = hparams.visualization_together
        self.visualize_every = hparams.visualize_every
        self.save_path = hparams.save_path
        self.step_counter = 0  # init step counter 
        self.action_map = hparams.action_map 
        self.direction_map = hparams.direction_map
        if self.visualization_together:
            self.visualization_seperate = False
        self.model = IntegratedModel(self.state_dim, self.action_dim, self.embed_dim, self.num_heads, hparams.freeze_weight, hparams.weight_path)
        self.criterion = nn.MSELoss()


    def forward(self, state, action):
        """
        Forward pass: Get next state prediction and attention weights
        """
        next_state_pred, attentionWeight = self.model(state, action)
        return next_state_pred, attentionWeight

    def loss_function(self, next_observations_predict, next_observations_true):
        loss = nn.MSELoss()
        loss_obs = loss(next_observations_predict, next_observations_true)
        loss = {'loss_obs':loss_obs}
        return loss
    
    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-6, weight_decay=self.weight_decay)
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
        ## load data
        obs = batch['obs']
        act = batch['act']
        obs_next = batch['obs_next']

        ## Reshape date format from (batch_size, 12, 6, 3) to (batch_size, 3, 6, 12) 
        obs_temp = obs.view(obs.size(0), 12, 6, 3)  # convert to (batch_size, width, height, channels)
        obs = obs_temp.permute(0, 3, 2, 1)  # convert to (batch_size, channels, height, width)
        obs_next_temp = obs_next.view(obs_next.shape[0], 12, 6, 3)   # convert to (batch_size, width, height, channels)
        obs_next = obs_next_temp.permute(0, 3, 2, 1).flatten(start_dim=1) # convert to (batch_size, channels, height, width) then flatten

        ## transform prediction
        obs_pred, attentionWeight = self(obs, act)
        if obs_next.dtype != obs_pred.dtype:
            obs_next = obs_next.float()

        ## calculate loss
        loss = self.loss_function(obs_pred, obs_next)
        self.log_dict(loss)

        ## visualization
        self.step_counter += 1
        if (self.visualization_together or self.visualization_seperate )and self.step_counter % self.visualize_every == 0:
            """
            obs: curent state
            action = action taken
            attentionWeight = attention weight 
            obs_next = next state
            obs_pred = predicted next state
            """
            self.visualization(obs, act, attentionWeight, obs_next, obs_pred)

        return loss['loss_obs']

    def validation_step(self, batch, batch_idx):
        obs = batch['obs']
        act = batch['act']
        obs_next = batch['obs_next']
        obs_temp = obs.view(obs.size(0), 12, 6, 3)  # convert to (batch_size, width, height, channels)
        obs = obs_temp.permute(0, 3, 2, 1)  # convert to (batch_size, channels, height, width)
        obs_next_temp = obs_next.view(obs_next.shape[0], 12, 6, 3)  # convert to (batch_size, width, height, channels)
        obs_next = obs_next_temp.permute(0, 3, 2, 1).flatten(start_dim=1)

        obs_pred, _ = self(obs, act)
        if obs_next.dtype != obs_pred.dtype:
            obs_next = obs_next.float()
        loss = self.loss_function(obs_pred, obs_next)
        self.log_dict(loss)
        return {"loss_wm_val": loss['loss_obs']}

    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        avg_loss = torch.stack([x["loss_wm_val"] for x in outputs]).mean()
        self.log("avg_val_loss_wm", avg_loss)
        return {"avg_val_loss_wm": avg_loss}

    def on_save_checkpoint(self, checkpoint):
        # Example checkpoint customization: removing specific keys if needed
        t = checkpoint['state_dict']
        pass  # No specific filtering needed for a simple NN

    def visualization(self, obs, act, attentionWeight, obs_next, obs_pred):
        ## preporcessing data
        state_image = obs[-1, 0, :, :].detach().cpu().numpy() * 10  # convert tensor to numpy
        direction = self.direction_map[round(obs[-1, 2, :, :].detach().cpu().numpy().max() * 3)]
        action = self.action_map[round(act[-1].item() * 6)]
        heat_map = attentionWeight[-1, :].reshape(6, 12).detach().cpu().numpy()  #  convert tensor to numpy

        obs_next_temp = obs_next.view(obs_next.shape[0], 3, 6, 12)   # convert to (batch_size, width, height, channels)
        next_direction = self.direction_map[round(obs_next_temp[-1, 2, :, :].detach().cpu().numpy().max() * 3)]
        obs_next = obs_next_temp[-1, 0, :, :].detach().cpu().numpy() * 10  # convert tensor to numpy
        dir = round(obs_pred[-1, :].reshape(3, 6, 12)[2, :, :].detach().cpu().numpy().max() * 3)
        if dir not in self.direction_map.keys():
            pre_direction = "Unknown"
        else:
            pre_direction = self.direction_map[dir]
        obs_pred = np.round(obs_pred[-1, :].reshape(3, 6, 12)[0, :, :].detach().cpu().numpy() * 10)  #  convert tensor to numpy
        

        num_colors = 13 
        custom_cmap = plt.cm.get_cmap('gray', num_colors)
        ## visualization
        if self.visualization_seperate:
            plt.figure(figsize=(18, 6)) 
            plt.subplot(1, 3, 1)  
        else:
            plt.figure(figsize=(18, 10)) 
            plt.subplot(2, 2, 1)  
        obs_fig =plt.imshow(state_image, cmap=custom_cmap, interpolation='nearest')
        plt.colorbar(obs_fig, shrink=0.7, label='State Value')
        plt.title(f"State   Dir: {direction}   Next Action: {action}")

        if self.visualization_seperate:
            plt.subplot(1, 3, 2)  
        else:
            plt.subplot(2, 2, 3)  
        weight = plt.imshow(heat_map, cmap='viridis', interpolation='nearest')
        plt.colorbar(weight, shrink=0.7, label='Attention Weight')
        plt.title("Attention Heatmap")

        # if self.visualization_seperate:
        #     plt.subplot(1, 3, 3)  

        # else:
        #     plt.subplot(2, 3, 5)  
        # overlay = plt.imshow(state_image * heat_map, cmap='viridis', interpolation='nearest')  
        # plt.colorbar(overlay, shrink=0.48, label='Attention Overlay')
        # plt.title("State and Attention Overlay")

        if self.visualization_seperate:
            plt.tight_layout()
            save_file = os.path.join(self.save_path, f"visualization_step_{self.step_counter}.png")
            plt.savefig(save_file)  
            plt.close()

        if self.visualization_seperate:
            plt.figure(figsize=(12, 6)) 
            plt.subplot(1, 2, 1)  

        else:
            plt.subplot(2, 2, 2)  
        
        next = plt.imshow(obs_next, cmap=custom_cmap, interpolation='nearest')
        plt.colorbar(next, shrink=0.7, label='Next State')
        plt.title(f"Next State  Next Dir:{next_direction}")

        if self.visualization_seperate:
            plt.subplot(1, 2, 2)  

        else:
            plt.subplot(2, 2, 4)  
        
        prefig = plt.imshow(obs_pred, cmap=custom_cmap, interpolation='nearest')
        plt.colorbar(prefig, shrink=0.7, label='Predicted State')
        plt.title(f"Predicted State Next Dir: {pre_direction}")
        plt.tight_layout()

        if self.visualization_seperate:
            save_file = os.path.join(self.save_path, f"Obs_Vs_Predicetion_{self.step_counter}.png")
        else:
            save_file = os.path.join(self.save_path, f"Obs_Attention_{self.step_counter}.png")
        plt.savefig(save_file)  # save the figure to file
        plt.close()
    



