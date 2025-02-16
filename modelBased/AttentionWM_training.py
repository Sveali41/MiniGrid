from AttentionWM import AttentionWorldModel
from data.datamodule import WMRLDataModule
from common.utils import PROJECT_ROOT, get_env
import hydra
from omegaconf import DictConfig
from pytorch_lightning.loggers.wandb import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping    
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import os
import torch

@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "modelBased/config"), config_name="config")
def train(cfg: DictConfig):
    use_wandb = cfg.attention_model.use_wandb

    # data
    dataloader = WMRLDataModule(hparams=cfg.attention_model)
    # dataloader.setup()

    # Model initialization
    net = AttentionWorldModel(hparams=cfg.attention_model)
 
    # Set up logger
    wandb_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(project="Local Attention Training", log_model=True)
        wandb_logger.experiment.watch(net, log='all', log_freq=1000)
    # else:
    #     print("Debug mode enabled. WandB logging is disabled.")

    # Define the trainer
    metric_to_monitor = 'avg_val_loss_wm'
    early_stop_callback = EarlyStopping(monitor=metric_to_monitor, min_delta=0.00, patience=5, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor=metric_to_monitor,
        mode="min",
        dirpath=os.path.dirname(cfg.attention_model.model_save_path),
        filename="att-{epoch:02d}-{avg_val_loss_wm:.5f}",
        verbose=True
    )

    trainer = pl.Trainer(
        logger=wandb_logger if use_wandb else None,
        max_epochs=cfg.attention_model.n_epochs,
        gpus=1,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    # Start the training
    if cfg.attention_model.freeze_weight:
        trainer.validate(net, dataloader)
    else:
        trainer.fit(net, dataloader)

    print(type(net.model))
    savePath = cfg.attention_model.weight_save_path
    torch.save({
        'model': net.model.state_dict(),
    }, savePath)


    model_pth = cfg.attention_model.weight_save_path
    # trainer.save_checkpoint(model_pth)
    if use_wandb:
        wandb.save(str(model_pth))
        wandb.save(savePath)
        
if __name__ == "__main__":
    train()
