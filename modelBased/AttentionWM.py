import torch
from torch import nn
import pytorch_lightning as pl
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List, Dict, Union
from modelBased.common import utils
from . import AttentionWM_support
from . import Embedding_support
from . import MLP_support
import pandas as pd

    
class AttentionWorldModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.mask_size = hparams.attention_mask_size
        self.channel, self.row, self.col = hparams.grid_shape
        self.lr= hparams.lr
        self.weight_decay = hparams.wd
        self.visualizationFlag = hparams.visualization
        self.visualize_every = hparams.visualize_every
        self.step_counter = 0  
        self.data_type = hparams.data_type
        self.fisher = 0
        self.old_params = None
        MODEL_MAPPING = {
            'attention': AttentionWM_support.AttentionModule,
            'embedding': Embedding_support.EmbeddingModule,
            'mlp': MLP_support.SimpleNNModule
        }
        # 初始化模型
        module_class = MODEL_MAPPING.get(hparams.model_type.lower())
        if module_class is not None:
            self.model = module_class(
                hparams.data_type, 
                hparams.grid_shape, 
                hparams.attention_mask_size, 
                hparams.embed_dim, 
                hparams.num_heads
            )
        else:
            print(f"Model type: {hparams.model_type} not supported")
            exit()
        

        if hparams.freeze_weight:
            utils.load_model_weight(self.model, hparams.model_save_path)
        self.loss = nn.MSELoss() # nn.SmoothL1Loss()
        self.visual_func = utils.Visualization(hparams)

    def save_old_params(self):
        """保存当前模型参数作为旧任务的参数，用于EWC"""
        old_params = {n: p.clone().detach() for n, p in self.named_parameters() if p.requires_grad}
        return old_params
    
    # def load_old_params(self, old_params):
    #     """加载旧参数并应用到当前模型"""
    #     for n, p in self.named_parameters():
    #         if n in old_params:
    #             p.data.copy_(old_params[n])  # 使用旧参数的值替换当前参数


    def load_old_params(self, old_params):
        self.old_params = old_params

    def compute_fisher(self, dataloader, samples=100):
        fisher = {n: torch.zeros_like(p) for n, p in self.named_parameters() if p.requires_grad}
        self.eval()
        device = next(self.parameters()).device
        count = 0
        for i, batch in enumerate(dataloader):
            if count >= samples:
                break
            self.zero_grad()
            obs, act, obs_next = self.preprocess_batch(batch)
            obs = obs.to(device).float()
            act = act.to(device)
            obs_next = obs_next.to(device).float()
            obs_pred, _ = self(obs, act)
            loss = self.loss_function(obs_pred, obs_next)['loss_obs']
            loss.backward()
            for n, p in self.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach().pow(2)
            count += 1
        for n in fisher:
            fisher[n] /= count
        return fisher

    def ewc_loss(self, lambda_ewc=10.0):
        if self.fisher is None:
            return 0.0
        loss = 0.0
        device = next(self.parameters()).device
        for n, p in self.named_parameters():
                fisher = self.fisher[n].to(device)
                p_old = self.old_params[n].to(device)
                loss += (fisher * (p - p_old).pow(2)).sum()
        return lambda_ewc * loss



    def forward(self, state, action):
        next_state_pred, attentionWeight = self.model(state, action)
        
        return next_state_pred, attentionWeight

    def loss_function(self, next_observations_predict, next_observations_true):
        loss_obs = self.loss(next_observations_predict.flatten(1), next_observations_true.flatten(1))
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
        agent_postion_yx_batch = utils.get_agent_position(obs)
        obs_masked = utils.extract_masked_state(obs, self.mask_size, agent_postion_yx_batch)
        return obs_masked, act, obs_next

    def training_step(self, batch, batch_idx):
        ## load data
        obs, act, obs_next = self.preprocess_batch(batch)

        ## transform prediction
        obs_pred, attentionWeight = self(obs, act)
        if obs_next.dtype != obs_pred.dtype:
            obs_next = obs_next.float()

        ## calculate loss
        loss = self.loss_function(obs_pred, obs_next)
        self.log_dict(loss)

        # 添加 EWC 正则项
        ewc_loss_tensor = torch.tensor(self.ewc_loss(), device=loss['loss_obs'].device) 
        loss['loss_obs'] += ewc_loss_tensor
        # print("EWC Loss:", ewc_loss_tensor.item(), "Total Loss:", loss['loss_obs'].item())

        # self.log("train_loss", loss['loss_obs'], on_step=True, on_epoch=True, prog_bar=True, logger=False)
        
        ## visualization
        self.step_counter += 1
        if self.visualizationFlag and self.step_counter % self.visualize_every == 0:
            self.visual_func.visualize_attention(obs, act, attentionWeight, obs_next, obs_pred, self.step_counter)

        return loss['loss_obs']

    def validation_step(self, batch, batch_idx):
        obs, act, obs_next = self.preprocess_batch(batch)
        obs_pred, attention_weight = self(obs, act)
        if obs_next.dtype != obs_pred.dtype:
            obs_next = obs_next.float()
        loss = self.loss_function(obs_pred, obs_next)
        # print(torch.round(obs_pred))
        self.log_dict(loss)
                ## visualization
        self.step_counter += 1
        if self.visualizationFlag and self.step_counter % self.visualize_every == 0:
            self.visual_func.visualize_attention(obs, act, attention_weight, obs_next, obs_pred, self.step_counter)
        # return {"batch_idx": batch_idx, "val_loss": loss['loss_obs']}
        return {"loss_wm_val": loss['loss_obs']}

    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:

        losses = [x["loss_wm_val"].item() for x in outputs]
        df = pd.DataFrame(losses, columns=["loss_wm_val"])

        # 保存为 CSV（不包含 index）
        df.to_csv("validation_21*21_emb_mask5.csv", index=False, header=False)
        # 绘制 loss 变化曲线
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(8, 5))
        # plt.plot(batch_indices, losses, marker="o", linestyle="-")
        # plt.xlabel("Batch Index")
        # plt.ylabel("Validation Loss")
        # plt.title("Validation Loss per Batch")
        # plt.grid(True)

        avg_loss = torch.stack([x["loss_wm_val"] for x in outputs]).mean()
        self.log("avg_val_loss_wm", avg_loss)
        return {"avg_val_loss_wm": avg_loss}

    def on_save_checkpoint(self, checkpoint):
        # Example checkpoint customization: removing specific keys if needed
        t = checkpoint['state_dict']
        pass  # No specific filtering needed for a simple NN

   



