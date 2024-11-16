from transformer import IntegratedPredictionModel
from data.datamodule import WMRLDataModule
from modelBased.common.utils import PROJECT_ROOT, get_env
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers.wandb import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "conf/transformer"), config_name="config")
def train(cfg: DictConfig):
    use_wandb = cfg.use_wandb
    hparams = cfg
    # data
    dataloader = WMRLDataModule(hparams=hparams.attention_model)

    # Model initialization
    net = IntegratedPredictionModel(hparams=hparams.attention_model)

    # Set up logger
    wandb_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(project="Attention Training", log_model=True)
        wandb_logger.experiment.watch(net, log='all', log_freq=1000)
    else:
        print("Debug mode enabled. WandB logging is disabled.")

    # Define the trainer
    metric_to_monitor = 'avg_val_loss_wm'
    early_stop_callback = EarlyStopping(monitor=metric_to_monitor, min_delta=0.00, patience=10, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor=metric_to_monitor,
        mode="min",
        dirpath=hparams.attention_model.pth_folder,
        filename="att-{epoch:02d}-{avg_val_loss_wm:.4f}",
        verbose=True
    )
    trainer = pl.Trainer(
        logger=wandb_logger if use_wandb else None,
        max_epochs=hparams.attention_model.n_epochs,
        gpus=1,
        callbacks=[early_stop_callback, checkpoint_callback]
    )

    # Start the training
    trainer.fit(net, dataloader)
    if use_wandb:
        # Log the trained model only when wandb is used
        model_pth = hparams.attention_model.pth_folder
        trainer.save_checkpoint(model_pth)
        wandb.save(str(model_pth))
        

if __name__ == "__main__":
    train()
