import os
ROOTPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys
sys.path.append(ROOTPATH)
from modelBased.AttentionWM import AttentionWorldModel
from modelBased.data.datamodule import WMRLDataModule
from modelBased.common.utils import PROJECT_ROOT, get_env
import hydra
from omegaconf import DictConfig
from pytorch_lightning.loggers.wandb import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping    
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import numpy as np

@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "modelBased/config"), config_name="config")
def train(cfg: DictConfig):
    run(cfg)

def run(cfg: DictConfig, old_params=None, fisher=None, layout=None, replay_data=None):
    print(f'*************************Data set: {cfg.attention_model.data_dir}************************')
    use_wandb = cfg.attention_model.use_wandb
    lambda_ewc = cfg.attention_model.lambda_ewc
    ewc_decay = 0.1

    # data
    if cfg.attention_model.continue_learning:
        dataloader = WMRLDataModule(hparams=cfg.attention_model, replay_data=replay_data)
    else:
        dataloader = WMRLDataModule(hparams=cfg.attention_model, replay_data=None)
    # dataloader.setup()

    # Model initialization
    net = AttentionWorldModel(hparams=cfg.attention_model)
 
    # Set up logger
    wandb_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(project="Local_Attention_Training", log_model=True, reinit=True)
        wandb_logger.experiment.watch(net, log='all', log_freq=1000)
        # wandb.log({"env_heatmap": wandb.Image((255*(layout/8)).astype(np.uint8))})

        
        
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
        return None, None
    else:
        net.fisher = fisher
        net.old_params = old_params
        if old_params is not None:
            net.load_old_params(old_params)
            
        trainer.fit(net, dataloader)
        # trainer.fit(net, dataloader)
        old_params = net.save_old_params()
        new_fisher = net.compute_fisher(dataloader.train_dataloader(), samples=3000)

        if fisher is not None:
            fisher = {
                k: ewc_decay * fisher[k] + new_fisher[k] for k in new_fisher
            }
        else:
            fisher = new_fisher

        print(type(net.model))
        # savePath = cfg.attention_model.weight_save_path
        # torch.save({
        #     'model': net.model.state_dict(),
        # }, savePath)


        model_pth = cfg.attention_model.model_save_path
        trainer.save_checkpoint(model_pth)
        if use_wandb:
            wandb.save(str(model_pth))
            wandb.save(model_pth)
        return old_params, fisher
        
def train_api(cfg: DictConfig, old_params=None, fisher=None, env_layout=None, replay_data=None):
    old_params, fisher = run(cfg, old_params, fisher, env_layout, replay_data)
    return old_params, fisher

if __name__ == "__main__":
    train()
