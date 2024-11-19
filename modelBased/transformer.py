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
    def __init__(self, action_dim, embed_dim, num_heads):
        super(ExtractionModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, embed_dim -2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim -2, embed_dim -2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.Dropout(0.3)
        )
        self.action_embedding = nn.Linear(action_dim, embed_dim - 2)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)


    def forward(self, state, action):
        ## ********* Move this Part to the IntegratedModel for visualization 2024_11_17 **********
        # batch_size = state.size(0)
        # stateTemp = state.view(batch_size, 12, 6, 3)  # convert to (batch_size, width, height, channels)
        # state = stateTemp.permute(0, 3, 2, 1)
        ## ***************************************************************************************
        batch_size = state.size(0)
        state_embed = self.conv(state)  # (batch_size, embed_dim, grid_height, grid_width)
        state_embed = state_embed.flatten(2).permute(0, 2, 1)  # (128, 72, 62)

        # generate positional embeddings
        # agent position embedding
        agent_position = torch.argwhere(state[:, 0, :, :] == 1)[:,1:]
        position_embedding_expanded = agent_position.unsqueeze(1).expand(-1, state_embed.size(1), -1)  # (128, 72, 2)
        state_embed = torch.cat([state_embed, position_embedding_expanded], dim=-1)  # (128, 72, 64)

        # action embedding
        action_embed = self.action_embedding(action.unsqueeze(1)).unsqueeze(1)  # (128, 1, 62)
        position_embeddin_temp= agent_position.unsqueeze(1)  # (128, 1, 2)
        action_embed = torch.cat([action_embed, position_embeddin_temp], dim=-1) # (128, 1, 64)

        # attention mechanism
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
    def __init__(self, state_dim, action_dim, embed_dim, num_heads, visualizationFlag=False, visualize_every=1000, save_path='', action_map={}):
        super(IntegratedModel, self).__init__()
        self.extraction_module = ExtractionModule(action_dim, embed_dim, num_heads)
        self.prediction_module = PredictionModule(embed_dim, state_dim)
        self.visualizationFlag = visualizationFlag
        self.visualize_every = visualize_every  # visualize every 1000 steps
        self.save_path = save_path
        self.action_map = action_map

    def forward(self, state, action, step_counter):
        # extract features from state and action
        ## ********* Reshape date format from (batch_size, 12, 6, 3) to (batch_size, 3, 6, 12) 2024_11_17 **********
        batch_size = state.size(0)
        stateTemp = state.view(batch_size, 12, 6, 3)  # convert to (batch_size, width, height, channels)
        state = stateTemp.permute(0, 3, 2, 1)
        ## **********************************************************************************************************
        extracted_features, attentionWeight = self.extraction_module(state, action)


        if self.visualizationFlag and step_counter % self.visualize_every == 0:
            self.Visualization(state, attentionWeight, step_counter, action, self.action_map)
        
        # predict next state from extracted features
        next_state_pred = self.prediction_module(extracted_features)
        
        return next_state_pred
    
    def Visualization(self, state, attentionWeight, step_counter, action, action_map):
        state_image = state[-1, 0, :, :].detach().cpu().numpy() * 10  # convert tensor to numpy
        heat_map = attentionWeight[-1, :].reshape(6, 12).detach().cpu().numpy()  #  convert tensor to numpy

        plt.figure(figsize=(18, 6)) 
        num_colors = 13  # 灰度层级
        custom_cmap = plt.cm.get_cmap('gray', num_colors)

        plt.subplot(1, 3, 1)  
        plt.imshow(state_image, cmap=custom_cmap, interpolation='nearest')
        plt.colorbar(label='State Value')
        plt.title(f"Original State & Action: {action_map[round(action[-1].item() * 6)]}")
        plt.xlabel("Width (Columns)")
        plt.ylabel("Height (Rows)")

        plt.subplot(1, 3, 2)  
        plt.imshow(heat_map, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Attention Weight')
        plt.title("Attention Heatmap")
        plt.xlabel("Width (Columns)")
        plt.ylabel("Height (Rows)")

        plt.subplot(1, 3, 3)  
        plt.imshow(state_image * heat_map, cmap='viridis', interpolation='nearest')  
        plt.colorbar(label='Attention Overlay')
        plt.title("State and Attention Overlay")
        plt.xlabel("Width (Columns)")
        plt.ylabel("Height (Rows)")

        plt.tight_layout()
        os.makedirs(self.save_path, exist_ok=True)
        save_file = os.path.join(self.save_path, f"visualization_step_{step_counter}.png")
        plt.savefig(save_file)  
        plt.close()
        # print(f"Visualization saved to {save_file}")



class IntegratedPredictionModel(pl.LightningModule):
    def __init__(self, hparams):
        super(IntegratedPredictionModel, self).__init__()
        self.state_dim = hparams.obs_size  
        self.action_dim = hparams.action_size
        self.embed_dim = hparams.embed_dim
        self.num_heads = hparams.num_heads
        self.learning_rate= hparams.lr
        self.weight_decay = hparams.wd
        self.visualizationFlag = hparams.visualizationFlag
        self.visualize_every = hparams.visualize_every
        self.save_path = hparams.save_path
        self.step_counter = 0  # init step counter 
        self.action_map = hparams.action_map 
        self.model = IntegratedModel(self.state_dim, self.action_dim, self.embed_dim, self.num_heads, self.visualizationFlag, self.visualize_every, self.save_path, self.action_map)
        self.criterion = nn.MSELoss()


    def forward(self, state, action):
        """
        Forward pass: Get next state prediction and attention weights
        """
        self.step_counter += 1
        next_state_pred = self.model(state, action, self.step_counter)
        return next_state_pred

    def loss_function(self, next_observations_predict, next_observations_true):
        loss = nn.MSELoss()
        loss_obs = loss(next_observations_predict, next_observations_true)
        
        if self.visualizationFlag and self.step_counter % self.visualize_every == 0:
            self.Visualization(next_observations_true, next_observations_predict, self.step_counter)

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
        obs = batch['obs']
        # import matplotlib.pyplot as plt
        # for i in range(64):
        #     plt.imshow(obs[i, :].reshape(12, 6, 3)[:,:,0].cpu().numpy())
        #     plt.show()
        #     plt.close()
        act = batch['act']
        obs_pred = self(obs, act)
        obs_next = batch['obs_next']
        obs_next_temp = obs_next.view(obs_next.shape[0], 12, 6, 3)   # convert to (batch_size, width, height, channels)
        obs_next = obs_next_temp.permute(0, 3, 2, 1).flatten(start_dim=1)
        if obs_next.dtype != obs_pred.dtype:
            obs_next = obs_next.float()
        loss = self.loss_function(obs_pred, obs_next)
        self.log_dict(loss)
        return loss['loss_obs']

    def validation_step(self, batch, batch_idx):
        obs = batch['obs']
        act = batch['act']
        obs_pred = self(obs, act)
        obs_next = batch['obs_next']
        obs_next_temp = obs_next.view(obs_next.shape[0], 12, 6, 3)  # convert to (batch_size, width, height, channels)
        obs_next = obs_next_temp.permute(0, 3, 2, 1).flatten(start_dim=1)
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

    def Visualization(self, nextObs, prediction, step_counter):
        obs_next_temp = nextObs.view(nextObs.shape[0], 3, 6, 12)   # convert to (batch_size, width, height, channels)

        obs_next = obs_next_temp[-1, 0, :, :].detach().cpu().numpy() * 10  # convert tensor to numpy
        prediction = np.round(prediction[-1, :].reshape(3, 6, 12)[0, :, :].detach().cpu().numpy() * 10)  #  convert tensor to numpy

        plt.figure(figsize=(12, 6)) 

        num_colors =  13 # 灰度层级
        custom_cmap = plt.cm.get_cmap('gray', num_colors)
        plt.subplot(1, 2, 1)  
        plt.imshow(obs_next, cmap=custom_cmap, interpolation='nearest')
        plt.colorbar(label='Next State')
        plt.title("Next State")
        plt.xlabel("Width (Columns)")
        plt.ylabel("Height (Rows)")

        plt.subplot(1, 2, 2)  
        plt.imshow(prediction, cmap=custom_cmap, interpolation='nearest')
        plt.colorbar(label='Predicted State')
        plt.title("Predicted State")
        plt.xlabel("Width (Columns)")
        plt.ylabel("Height (Rows)")

        plt.tight_layout()
        os.makedirs(self.save_path, exist_ok=True)
        save_file = os.path.join(self.save_path, f"Obs_Vs_Predicetion_{step_counter}.png")
        plt.savefig(save_file)  # save the figure to file
        plt.close()


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
