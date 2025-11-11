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
from modelBased.common.utils import TRAINER_PATH


@hydra.main(version_base=None, config_path=str(TRAINER_PATH / "conf"), config_name="config_test")
def train(cfg: DictConfig):
    run(cfg)


def run(cfg: DictConfig, old_params=None, fisher=None, layout=None, replay_data=None):
    print(f'*************************Data set: {cfg.attention_model.data_dir}************************')

    use_wandb = cfg.attention_model.use_wandb
    # 旧代码中的 ewc_decay 方式会几乎覆盖旧 Fisher，改为标准 EMA 系数（新 Fisher 的占比）
    fisher_beta = float(getattr(cfg.attention_model, "fisher_beta", 0.5))  # 0.3~0.7 可试

    # ===========
    # 数据模块
    # ===========
    if cfg.attention_model.continue_learning:
        datamodule = WMRLDataModule(hparams=cfg.attention_model, replay_data=replay_data)
    else:
        datamodule = WMRLDataModule(hparams=cfg.attention_model, replay_data=None)

    # ===========
    # 模型
    # ===========
    net = AttentionWorldModel(hparams=cfg.attention_model)

    # ===========
    # Logger
    # ===========
    wandb_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(project="Local_Attention_Training", log_model=True, reinit=True)
        wandb_logger.experiment.watch(net, log='all', log_freq=1000)
        # 如果需要也可以记录关卡可视化：
        # if layout is not None:
        #     wandb.log({"env_heatmap": wandb.Image((255*(layout/8)).astype(np.uint8))})

    # ===========
    # 回调
    # ===========
    metric_to_monitor = 'avg_val_loss_wm'
    early_stop_callback = EarlyStopping(
        monitor=metric_to_monitor,
        min_delta=0.00,
        patience=15,
        verbose=True,
        mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor=metric_to_monitor,
        mode="min",
        dirpath=os.path.dirname(cfg.attention_model.model_save_path),
        filename="att-{epoch:02d}-{avg_val_loss_wm:.5f}",
        verbose=True
    )

    # ===========
    # Trainer（使用新写法 + 梯度裁剪）
    # ===========
    trainer = pl.Trainer(
        precision=16,
        logger=wandb_logger if use_wandb else None,
        max_epochs=cfg.attention_model.n_epochs,
        accelerator="gpu",
        devices=1,
        gradient_clip_val=1.0,   # 防止新任务初期梯度尖刺
        callbacks=[early_stop_callback, checkpoint_callback],
        deterministic=False,
    )

    # ===========
    # 仅验证 or 训练
    # ===========
    if cfg.attention_model.freeze_weight:
        avg_val_loss = trainer.validate(net, datamodule)
        return avg_val_loss, net
    else:
        # 用统一入口设置“旧参数 + Fisher”锚点（内部已转到 CPU/float32）
        net.set_consolidation(old_params, fisher)

        # 训练
        trainer.fit(net, datamodule)

        # 导出旧参数（作为下一个任务的锚点）
        old_params = net.save_old_params()

        # 计算新的 Fisher（样本量可配，默认 3000）
        fisher_samples = int(getattr(cfg.attention_model, "fisher_samples", 3000))
        new_fisher = net.compute_fisher(
            datamodule.train_dataloader(),
            samples=fisher_samples,
            scale_factor=10  # 仅为兼容旧签名，函数内部已不使用
        )

        # 用标准 EMA 融合 Fisher：f_new = (1-β)*f_old + β*f_task
        if fisher is not None:
            fisher = {k: (1.0 - fisher_beta) * fisher[k] + fisher_beta * new_fisher[k] for k in new_fisher}
        else:
            fisher = new_fisher

        print(type(net.model))

        # 保存 checkpoint
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
