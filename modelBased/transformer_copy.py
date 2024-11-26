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

class ExtractionModule(nn.Module):
    def __init__(self, state_dim, action_dim, embed_dim, num_heads):
        super(ExtractionModule, self).__init__()
        self.state_embedding = nn.Linear(state_dim, embed_dim)
        self.action_embedding = nn.Linear(action_dim, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)


    def forward(self, state, action):
        # agent_position = torch.argwhere(state[:, :, :, ] == 1)[:,1:]
        state_embed = self.state_embedding(state) # (batch_size, 12*6, embed_dim)

        # state_embed = torch.cat([state_embed, position_embedding_expanded], dim=-1)  # (128, 72, 64)

        # action embedding
        action_embed = self.action_embedding(action.unsqueeze(1)).unsqueeze(1)  # (128, 1, 62)
        # concatenate state and action embeddings

        # attention mechanism
        attention_output, attention_weights = self.attention(query=action_embed, key=state_embed, value=state_embed)
        return attention_output.squeeze(1), attention_weights.squeeze(1)

class PredictionModule(nn.Module):
    def __init__(self, embed_dim, state_dim, hidden_dim=72):
        super(PredictionModule, self).__init__()
        # Input dimension is `embed_dim`, which will match the flattened input size
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 216)  # Output dimension = state_dim (216)

    def forward(self, extracted_features):
        x = self.fc1(extracted_features)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        output = self.fc3(x)  # Output shape: [batch_size, 216]
        
        return torch.nn.functional.softplus(output)  # Apply softplus activation

class IntegratedModel(nn.Module):
    """
    integrate extraction module and prediction module
    """
    def __init__(self, state_dim, action_dim, embed_dim, num_heads):
        super(IntegratedModel, self).__init__()
        self.extraction_module = ExtractionModule(state_dim, action_dim, embed_dim, num_heads)
        self.prediction_module = PredictionModule(embed_dim, state_dim)

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
        if self.visualization_together:
            self.visualization_seperate = False
        self.model = IntegratedModel(self.state_dim, self.action_dim, self.embed_dim, self.num_heads)
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
        obs = self.data_preprocess(obs)
        act = batch['act']
        obs_next = batch['obs_next']

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
        obs = self.data_preprocess(obs)
        act = batch['act']
        obs_next = batch['obs_next']
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
        direction = obs[-1, self.pos[-1], 2].item() * 3
        state_image = obs[-1, :, 0].view(12,6).detach().cpu().numpy() * 10  # convert tensor to numpy
        next_action = self.action_map[round(act[-1].item() * 6)]
        heat_map = attentionWeight[-1, :].reshape(12, 6).detach().cpu().numpy()  #  convert tensor to numpy

        obs_next_temp = obs_next.view(obs_next.shape[0], 12, 6, 3)   # convert to (batch_size, width, height, channels)
        obs_next = obs_next_temp[-1, :, :, 0].detach().cpu().numpy() * 10  # convert tensor to numpy
        obs_pred = np.round(obs_pred[-1, :].reshape(12, 6, 3)[:, :, 0].detach().cpu().numpy() * 10)  #  convert tensor to numpy

        num_colors = 13 
        custom_cmap = plt.cm.get_cmap('gray', num_colors)
        ## visualization
        if self.visualization_seperate:
            plt.figure(figsize=(18, 6)) 
            plt.subplot(1, 3, 1)  
        else:
            plt.figure(figsize=(18, 10)) 
            plt.subplot(2, 3, 1)  
        obs_fig =plt.imshow((state_image.T), cmap=custom_cmap, interpolation='nearest')
        plt.colorbar(obs_fig, shrink=0.48, label='State Value')
        plt.title(f"State   Dir: {direction}   Action: {next_action}")

        if self.visualization_seperate:
            plt.subplot(1, 3, 2)  
        else:
            plt.subplot(2, 3, 4)  
        weight = plt.imshow(heat_map.T, cmap='viridis', interpolation='nearest')
        plt.colorbar(weight, shrink=0.48, label='Attention Weight')
        plt.title("Attention Heatmap")

        if self.visualization_seperate:
            plt.subplot(1, 3, 3)  

        else:
            plt.subplot(2, 3, 6)  
        overlay = plt.imshow((state_image * heat_map).T, cmap='viridis', interpolation='nearest')  
        plt.colorbar(overlay, shrink=0.48, label='Attention Overlay')
        plt.title("State and Attention Overlay")

        if self.visualization_seperate:
            plt.tight_layout()
            save_file = os.path.join(self.save_path, f"visualization_step_{self.step_counter}.png")
            plt.savefig(save_file)  
            plt.close()

        if self.visualization_seperate:
            plt.figure(figsize=(12, 6)) 
            plt.subplot(1, 2, 1)  

        else:
            plt.subplot(2, 3, 2)  
        
        next = plt.imshow(obs_next.T, cmap=custom_cmap, interpolation='nearest')
        plt.colorbar(next, shrink=0.48, label='Next State')
        plt.title("Next State")

        if self.visualization_seperate:
            plt.subplot(1, 2, 2)  

        else:
            plt.subplot(2, 3, 5)  
        
        prefig = plt.imshow(obs_pred.T, cmap=custom_cmap, interpolation='nearest')
        plt.colorbar(prefig, shrink=0.48, label='Predicted State')
        plt.title("Predicted State")
        plt.tight_layout()

        if self.visualization_seperate:
            save_file = os.path.join(self.save_path, f"Obs_Vs_Predicetion_{self.step_counter}.png")
        else:
            save_file = os.path.join(self.save_path, f"Obs_Attention_{self.step_counter}.png")
        plt.savefig(save_file)  # save the figure to file
        plt.close()
        
    
    def data_preprocess(self, state):
        """
        concat agent position to state
        """
        reshaped_tensor = state.view(-1, 12, 6, 3)
        batch_size = reshaped_tensor.size(0)
        reshaped_tensor = state.view(batch_size, -1, 3)
        # agent_postion embedding
        agent_position = torch.argwhere(reshaped_tensor[:, :, 0] == 1)[:, 1:]
        self.pos = agent_position
        # convert to one-hot
        # position_embedding = torch.zeros((batch_size, 72, 1)).cuda()
        # position_embedding[torch.arange(batch_size).unsqueeze(1), agent_position] = 1
        position_values = torch.arange(1, 73).view(72, 1)  # 形状 [72, 1]

        # 扩展到 [128, 72, 1]
        position_embedding = position_values.unsqueeze(0).repeat(batch_size, 1, 1).cuda() # 形状 [128, 72, 1]
        reshaped_state = torch.cat([reshaped_tensor, position_embedding], dim=2)
        return reshaped_tensor


def extract_topk_regions(state, attention_weights, topk=5):
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

    # extract top-k regions
    extracted_regions = torch.gather(state, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, state.size(-1)))

    return extracted_regions, topk_indices
