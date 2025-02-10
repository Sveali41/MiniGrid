import torch
from torch import nn
import pytorch_lightning as pl
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List, Dict, Union
from common import utils
import transformer_support

class IntegratedModel(nn.Module):
    def __init__(self, feature_extration_mode, batch_size, grid_shape, embed_dim, num_heads, freeze_weight=False, weight_path=''):
        super().__init__()
        self.extraction_module = transformer_support.ExtractionModule(feature_extration_mode, batch_size, grid_shape, embed_dim, num_heads)
        self.prediction_module = transformer_support.PredictionModule(embed_dim, grid_shape)
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
        super().__init__()
        self.channel, self.row, self.col = hparams.grid_shape
        self.lr= hparams.lr
        self.weight_decay = hparams.wd
        self.visualizationFlag = hparams.visualization
        self.visualize_every = hparams.visualize_every
        self.step_counter = 0  
        self.model = IntegratedModel(hparams.feature_extration_mode, hparams.batch_size, hparams.grid_shape, hparams.embed_dim, hparams.num_heads, hparams.freeze_weight, hparams.weight_path)
        # self.loss = nn.MSELoss()
        self.loss = nn.SmoothL1Loss()
        self.visual_func = utils.Visualization(hparams)

    def forward(self, state, action):
        """
        Forward pass: Get next state prediction and attention weights
        """
        next_state_pred, attentionWeight = self.model(state, action)
        return next_state_pred, attentionWeight

    def loss_function(self, next_observations_predict, next_observations_true):
        loss_obs = self.loss(next_observations_predict, next_observations_true)
        loss = {'loss_obs':loss_obs}
        return loss
    
    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=self.weight_decay)
        reduce_lr_on_plateau = ReduceLROnPlateau(optimizer, mode='min',verbose=True, min_lr=1e-8)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr_on_plateau,
                "monitor": 'avg_val_loss_wm',
                "frequency": 1
            },
        }

    def preprocess_batch(self, batch):
        obs = batch['obs']
        act = batch['act']
        obs_next = batch['obs_next']
        ## Reshape date format from (batch_size, col, row, channel) to (batch_size, channel, row, col) 
        obs_temp = obs.view(obs.size(0), self.col, self.row, self.channel)  
        obs = obs_temp.permute(0, 3, 2, 1)  
        obs_next_temp = obs_next.view(obs_next.shape[0], self.col, self.row, self.channel)   
        obs_next = obs_next_temp.permute(0, 3, 2, 1).flatten(start_dim=1) 
        return obs, act, obs_next

    def training_step(self, batch, batch_idx):
        ## load data
        obs, act, obs_next = self.preprocess_batch(batch)

        ## transform prediction
        obs_pred, attentionWeight = self(obs, act)
        if obs_next.dtype != obs_pred.dtype:
            obs_next = obs_next.float()

        ## calculate loss
        loss = self.loss_function(obs_pred, obs_next)
        # self.log_dict(loss)
        self.log("train_loss", loss['loss_obs'], on_step=True, on_epoch=True, prog_bar=True, logger=False)

        ## visualization
        self.step_counter += 1
        if self.visualizationFlag and self.step_counter % self.visualize_every == 0:
            self.visual_func.visualize_attention(obs, act, attentionWeight, obs_next, obs_pred, self.step_counter)

        return loss['loss_obs']

    def validation_step(self, batch, batch_idx):
        obs, act, obs_next = self.preprocess_batch(batch)
        obs_pred, _ = self(obs, act)
        if obs_next.dtype != obs_pred.dtype:
            obs_next = obs_next.float()
        loss = self.loss_function(obs_pred, obs_next)
        # self.log_dict(loss)
        self.log("val_loss", loss['loss_obs'], on_step=False, on_epoch=True, prog_bar=True, logger=False)
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

   



