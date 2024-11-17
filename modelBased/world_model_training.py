import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch
# from path import *
import os
import json
import numpy as np
from world_model import SimpleNN
from data.datamodule import WMRLDataModule
from modelBased.common.utils import PROJECT_ROOT, get_env
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers.wandb import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


use_wandb = False
if use_wandb:
    import wandb
    wandb.require("core")

@hydra.main(version_base=None, config_path= str(PROJECT_ROOT / "conf/model"), config_name="config")
def train(cfg: DictConfig):
    hparams = cfg
    # data
    dataloader = WMRLDataModule(hparams = hparams.world_model)
    # Get a single batch from the dataloader
    # dataloader.setup()
    # dataloader_train = dataloader.train_dataloader()
    # dataloader_val = dataloader.val_dataloader()

    net = SimpleNN(hparams=hparams)
    wandb_logger = WandbLogger(project="WM Training", log_model=True)
    # ## Currently it does not log the model weights, there is a bug in wandb and/or lightning.
    wandb_logger.experiment.watch(net, log='all', log_freq=1000)
    # Define the trainer
    metric_to_monitor = 'avg_val_loss_wm'#"loss"
    early_stop_callback = EarlyStopping(monitor=metric_to_monitor, min_delta=0.00, patience=10, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(
                            save_top_k=1,
                            monitor = metric_to_monitor,
                            mode = "min",
                            dirpath = get_env('PTH_FOLDER'),
                            filename ="wm-{epoch:02d}-{avg_val_loss_wm:.4f}",
                            verbose = True
                        )
    trainer = pl.Trainer(logger=wandb_logger,
                    max_epochs=hparams.world_model.n_epochs, 
                    gpus=1,
                    callbacks=[early_stop_callback, checkpoint_callback])     
    # Start the training
    trainer.fit(net,dataloader)
    # Log the trained model
    model_pth = hparams.world_model.pth_folder
    trainer.save_checkpoint(model_pth)
    wandb.save(str(model_pth))


@hydra.main(version_base=None, config_path= str(PROJECT_ROOT / "conf/model"), config_name="config")
def validate(cfg: DictConfig):
    hparams = cfg
    model = SimpleNN(hparams=hparams)
    # Load the checkpoint
    dataloader = WMRLDataModule(hparams = hparams.world_model)
    dataloader.setup()
    checkpoint = torch.load(hparams.world_model.pth_folder)
    # Load state_dict into the model
    model.load_state_dict(checkpoint['state_dict'])
    # Set the model to evaluation mode (optional, depends on use case)
    model.eval()
    # Assuming the rest of your code is already set up as provided
    batch_size = 64
    num_tests = 20

    # Loop over the first 10 observations
    for i in range(num_tests):
        # Collecting the first 64 samples for the current test observation
        obs_batch = torch.tensor([dataloader.data_test[j]['obs'] for j in range(batch_size)])
        act_batch = torch.tensor([dataloader.data_test[j]['act'] for j in range(batch_size)])

        # Predict using the model
        obs_pred = model(obs_batch, act_batch)
        # map the observation to the nearest valid value
        obs_pred_map = []
        for k in range(batch_size):
            mapped_obs = map_obs_to_nearest_value(cfg, obs_pred[k])
            obs_pred_map.append(mapped_obs)
        obs_pred_map = torch.stack(obs_pred_map)

        # Denormalize the actual observation and round it
        obs_real_map = []
        for h in range(batch_size):
            denorm_obs = denormalize(obs_batch[h])  # Use index i to get the current observation
            obs_real = torch.round(denorm_obs)
            obs_real_map.append(obs_real)
        obs_real_map = torch.stack(obs_real_map)

        # Print the difference between the real and predicted observation
        print(f"Test {i+1}: Difference between real and predicted observation")
        print(obs_real_map - obs_pred_map)
    pass


def denormalize(x, dimension=6):
    """Denormalize the obs dimensiondata from its flattened state.
        input: x: torch.tensor of shape (,54)
    """
    obs_norm_values = [10, 5, 3] # Example normalization values for 3 channels
    # Reshape the data to its original shape before flattening
    x = x.reshape(dimension,3,3)
    
    # Ensure that the norm_values is not None and has the correct length
    if obs_norm_values is None or len(obs_norm_values) != x.shape[-1]:
        raise ValueError("Normalization values must be provided and must match the number of channels in the data.")
    
    # Denormalize each channel using the provided norm values
    for i in range(x.shape[-1]):  # Loop over the last dimension (channels)
        max_value = obs_norm_values[i]
        if max_value != 0:  # Avoid multiplication by zero (though normally this wouldn't be an issue)
            x[:, :, i] = x[:, :, i] * max_value
    return x

def normalize(x):
    """Normalize the obs data and flatten it.
        input: x: np.array of shape (6,3,3)
    """
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float32) 
    obs_norm_values = [10, 5, 3] 
    # Ensure that the norm_values is not None and has the correct length
    if obs_norm_values is None or len(obs_norm_values) != x.shape[-1]:
        raise ValueError("Normalization values must be provided and must match the number of channels in the data.")
    # Normalize each channel using the provided norm values
    for i in range(x.shape[-1]):  # Loop over the last dimension (channels)
        max_value = obs_norm_values[i]
        if max_value != 0:  # Avoid division by zero
            x[:, :, i] = x[:, :, i] / max_value  
    # Flatten the data
    x = x.reshape(-1)
    x = torch.tensor(x)
    return x


def map_to_nearest_value(tensor, valid_values):
    valid_values = torch.tensor(valid_values, dtype=torch.float32).to(tensor.device)
    tensor = tensor.unsqueeze(-1)  # Add a dimension to compare with valid values
    differences = torch.abs(tensor - valid_values)  # Calculate differences
    indices = torch.argmin(differences, dim=-1)  # Get index of nearest value
    nearest_values = valid_values[indices]  # Get nearest values using indices
    return nearest_values

def map_obs_to_nearest_value(cfg, obs):
    """
    Maps each value in the tensor to the nearest valid value from the valid_values list.
    
    Args:
        tensor (torch.Tensor): Input tensor with shape (,54).
        valid_values (list): From config read list of valid values to map to.
        
    Returns:
        torch.Tensor: 1.Denorm the tensor 2.Each element is replaced 
                      by the nearest valid value.
    """
    # Denormalize the tensor
    obs_denorm = denormalize(obs.clone())
    # load the valid values from the config
    hparams = cfg
    valid_values_obj = hparams.world_model.valid_values_obj
    valid_values_color = hparams.world_model.valid_values_color
    valid_values_state = hparams.world_model.valid_values_state
    # Map each channel to the nearest valid value
    obs_denorm[:, :, 0] = map_to_nearest_value(obs_denorm[:, :, 0], valid_values_obj)
    obs_denorm[:, :, 1] = map_to_nearest_value(obs_denorm[:, :, 1], valid_values_color)
    obs_denorm[:, :, 2] = map_to_nearest_value(obs_denorm[:, :, 2], valid_values_state)
    return obs_denorm

if __name__ == "__main__":
    # validate()
    train()
