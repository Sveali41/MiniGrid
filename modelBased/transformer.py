import torch
from torch import nn
import pytorch_lightning as pl
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Sequence, List, Dict, Tuple, Optional, Any, Set, Union, Callable, Mapping

# class ExtractionModule(nn.Module):
#     def __init__(self, state_dim, action_dim, embed_dim, num_heads, max_seq_len):
#         super(ExtractionModule, self).__init__()
#         self.state_embedding = nn.Linear(state_dim, embed_dim)
#         self.action_embedding = nn.Linear(action_dim, embed_dim)
#         self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
#         self.positional_embedding = nn.Parameter(torch.randn(max_seq_len, embed_dim))  # Learnable embeddings

#     def forward(self, state, action):
#         batch_size, state_dim = state.shape
#         state_embed = self.state_embedding(state).unsqueeze(1)  # (batch_size, seq_len, embed_dim)
#         positional_embedding = self.positional_embedding[:state_dim].unsqueeze(0).expand(batch_size, state_dim, -1)
#         # Add positional embeddings (broadcast to batch size)
#         state_embed = state_embed + positional_embedding  # (batch_size, seq_len, embed_dim)
        
#         action_embed = self.action_embedding(action.unsqueeze(1)).unsqueeze(1)  # (batch_size, 1, embed_dim)
        
#         # Action as query, state as key and value
#         attention_output, attention_weights = self.attention(query=action_embed, key=state_embed, value=state_embed)
#         return attention_output.squeeze(1), attention_weights.squeeze(1)

class ExtractionModule(nn.Module):
    def __init__(self, action_dim, embed_dim, num_heads):
        super(ExtractionModule, self).__init__()
        self.conv = nn.Conv2d(3, embed_dim, kernel_size=3, stride=1, padding=1)
        self.action_embedding = nn.Linear(action_dim, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.row_embedding = nn.Parameter(torch.randn(6, embed_dim))
        self.col_embedding = nn.Parameter(torch.randn(12, embed_dim))

    def forward(self, state, action):
        batch_size = state.size(0)
        stateTemp = state.view(batch_size, 12, 6, 3)  # convert to (batch_size, width, height, channels)
        state = stateTemp.permute(0, 3, 2, 1)
        state_embed = self.conv(state)  # (batch_size, embed_dim, grid_height, grid_width)
        grid_height, grid_width = state_embed.shape[2], state_embed.shape[3]
        state_embed = state_embed.flatten(2).permute(0, 2, 1)  # (batch_size, seq_len, embed_dim)

        # generate positional embeddings
        row_pos = self.row_embedding.unsqueeze(1).expand(-1, grid_width, -1)  # (grid_height, grid_width, embed_dim)
        col_pos = self.col_embedding.unsqueeze(0).expand(grid_height, -1, -1)  # (grid_height, grid_width, embed_dim)
        pos_embedding = (row_pos + col_pos).flatten(0, 1).unsqueeze(0).expand(batch_size, -1, -1)

        state_embed = state_embed + pos_embedding  # (batch_size, seq_len, embed_dim)

        # action embedding
        action_embed = self.action_embedding(action.unsqueeze(1)).unsqueeze(1)  # (batch_size, 1, embed_dim)

        # attention mechanism
        attention_output, attention_weights = self.attention(query=action_embed, key=state_embed, value=state_embed)
        return attention_output.squeeze(1), attention_weights.squeeze(1)


# class PredictionModule(nn.Module):
#     """
#     predict next state from extracted features
#     """
#     def __init__(self, embed_dim, state_dim):
#         super(PredictionModule, self).__init__()
#         self.fc = nn.Linear(embed_dim, state_dim)  # fully connected layer

#     def forward(self, extracted_features):
#         return self.fc(extracted_features)  # (batch_size, state_dim)
class PredictionModule(nn.Module):
    def __init__(self, embed_dim, state_dim):
        super(PredictionModule, self).__init__()
        self.fc = nn.Linear(embed_dim, state_dim)

    def forward(self, extracted_features):
        output = self.fc(extracted_features)  # (batch_size, state_dim)
        return torch.relu(output)  # Ensures non-negative outputs

class IntegratedModel(nn.Module):
    """
    integrate extraction module and prediction module
    """
    def __init__(self, state_dim, action_dim, embed_dim, num_heads):
        super(IntegratedModel, self).__init__()
        # max_seq_len = state_dim
        self.extraction_module = ExtractionModule(action_dim, embed_dim, num_heads)
        # self.extraction_module = ExtractionModule(state_dim, action_dim, embed_dim, num_heads, max_seq_len)
        self.prediction_module = PredictionModule(embed_dim, state_dim)

    def forward(self, state, action):
        # extract features from state and action
        extracted_features, _ = self.extraction_module(state, action)
        
        # predict next state from extracted features
        next_state_pred = self.prediction_module(extracted_features)
        
        return next_state_pred



class IntegratedPredictionModel(pl.LightningModule):
    def __init__(self, hparams):
        super(IntegratedPredictionModel, self).__init__()
        self.state_dim = hparams.obs_size  
        self.action_dim = hparams.action_size
        self.embed_dim = hparams.embed_dim
        self.num_heads = hparams.num_heads
        self.learning_rate= hparams.lr
        self.weight_decay = hparams.wd
        self.model = IntegratedModel(self.state_dim, self.action_dim, self.embed_dim, self.num_heads)
        self.criterion = nn.MSELoss()


    def forward(self, state, action):
        """
        Forward pass: Get next state prediction and attention weights
        """
        next_state_pred = self.model(state, action)
        return next_state_pred

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
        obs = batch['obs']
        act = batch['act']
        obs_pred = self(obs, act)
        obs_next = batch['obs_next']
        if obs_next.dtype != obs_pred.dtype:
            obs_next = obs_next.float()
        loss = self.loss_function(obs_pred, obs_next)
        self.log_dict(loss)
        return loss['loss_obs']

    def validation_step(self, batch, batch_idx):
        obs = batch['obs']
        act = batch['act']
        obs_pred = self(obs, act)[0]
        obs_next = batch['obs_next']
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
