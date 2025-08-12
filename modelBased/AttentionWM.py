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
        self.lambda_ewc = getattr(hparams, "lambda_ewc", 10.0)
        self.lambda_ewc_min = 1.0
        self.lambda_ewc_max = 500.0
        self.drift_threshold = getattr(hparams, "drift_threshold", 1e-3)
        self.fisher = 0
        self.old_params = None
        self.env_type = hparams.env_type
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
        self.save_hyperparameters(hparams)


    def save_old_params(self):
        """Save current model parameters for EWC, moved to model's device."""
        device = next(self.parameters()).device
        old_params = {n: p.clone().detach().to(device) for n, p in self.named_parameters() if p.requires_grad}
        return old_params
    
    # def load_old_params(self, old_params):
    #     """加载旧参数并应用到当前模型"""
    #     for n, p in self.named_parameters():
    #         if n in old_params:
    #             p.data.copy_(old_params[n])  # 使用旧参数的值替换当前参数

    def compute_avg_param_drift(self) -> float:
        drift_sum = 0.0
        num = 0
        for name, param in self.named_parameters():
            if self.old_params is not None and name in self.old_params:
                drift = (param - self.old_params[name].to(param.device)).pow(2).mean()
                drift_sum += drift.item()
                num += 1
        avg_drift = drift_sum / max(num, 1)
        return avg_drift


    def update_lambda_ewc(self, avg_drift: float):
        # 初始化 drift history 缓存
        if not hasattr(self, "_drift_values"):
            self._drift_values = []
            self.warmup_steps = getattr(self.hparams, "warmup_steps", 100)

        # 自动 warmup 阈值
        if self.drift_threshold is None:
            self._drift_values.append(avg_drift)
            if len(self._drift_values) == self.warmup_steps:
                self.drift_threshold = sum(self._drift_values) / len(self._drift_values)
                print(f"[Auto-tuned] drift_threshold set to {self.drift_threshold:.6f}")
        else:
            # 正常 drift 控制 λ
            if avg_drift > self.drift_threshold:
                self.lambda_ewc *= 1.2
            elif avg_drift < self.drift_threshold / 2:
                self.lambda_ewc *= 0.9
            self.lambda_ewc = float(torch.clamp(torch.tensor(self.lambda_ewc), self.lambda_ewc_min, self.lambda_ewc_max))
        
        # 日志记录
        self.log("train/avg_param_drift", avg_drift)
        self.log("train/lambda_ewc", self.lambda_ewc)

    def load_old_params(self, old_params):
        device = next(self.parameters()).device
        self.old_params = {k: v.to(device) for k, v in old_params.items()}

    def compute_fisher(self, dataloader, samples=1000, scale_factor=10):
        fisher = {n: torch.zeros_like(p) for n, p in self.named_parameters() if p.requires_grad}
        self.eval()
        device = next(self.parameters()).device
        count = 0
        for i, batch in enumerate(dataloader):
            if count >= samples:
                break
            self.zero_grad()

            obs, act, obs_next, info = self.preprocess_batch(batch)
            obs = obs.to(device).float()
            act = act.to(device)
            obs_next = obs_next.to(device).float()
            obs_pred, _ = self(obs, act, info)
            loss = scale_factor * self.loss_function(obs_pred, obs_next)['loss_obs']
            loss.backward(retain_graph=False)
            for n, p in self.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach().pow(2)
            count += 1
        for n in fisher:
            fisher[n] /= count
        # for k in fisher:
        #     fisher[k] = torch.sqrt(fisher[k] + 1e-8)
        #     fisher[k] *= 5
        all_f = torch.cat([f.flatten() for f in fisher.values()])
        print(f"[Fisher] mean={all_f.mean():.3e}, max={all_f.max():.3e}, min={all_f.min():.3e}")
        return fisher
    

    def ewc_loss(self, lambda_ewc=10):
        if self.fisher is None or self.old_params is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        device = next(self.parameters()).device
        loss = torch.tensor(0.0, device=device)  
        for n, p in self.named_parameters():
            if n in self.fisher and n in self.old_params:
                fisher = self.fisher[n].to(device)
                p_old = self.old_params[n].to(device)
                loss += (fisher * (p - p_old).pow(2)).sum()

        return lambda_ewc * loss

    def forward(self, state, action, info):
        next_state_pred, attentionWeight = self.model(state, action, info)
        
        return next_state_pred, attentionWeight

    def loss_function(self, next_observations_predict, next_observations_true):
        loss_obs = self.loss(next_observations_predict.flatten(1), next_observations_true.flatten(1))
        loss = {'loss_obs':loss_obs}
        return loss
    

    def loss_function_weight(self, next_observations_predict, next_observations_true):
        # 变化掩码：只要该像素任一通道非 0，就算变化
        change_mask = (next_observations_true.abs() > 1e-6).any(dim=1, keepdim=True)  # (B,1,H,W)

        # 基础逐像素 MSE（注意 reduction='none' 保持形状）
        base = F.mse_loss(
            next_observations_predict,
            next_observations_true,
            reduction='none'   # 保持 (B,C,H,W)
        )

        # 权重矩阵：默认 1.0，变化处加权
        change_mask_full = change_mask.expand_as(base)   # (B,C,H,W)
        w = torch.ones_like(base)
        w[change_mask_full] = 5.0  # 变化处 ×5

        # 最终加权 MSE
        loss = (base * w).mean()

        return {"loss_obs": loss}


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

    def preprocess_batch(self, batch, training=False):
        obs = batch['obs']
        act = batch['act']
        obs_next = batch['obs_next']
        if self.env_type == 'with_obj':
            info = batch['info']
        else:
            info = None
        

        agent_postion_yx_batch = utils.get_agent_position(obs)
        obs_masked = utils.extract_masked_state(obs, self.mask_size, agent_postion_yx_batch)
        obs_next_masked = utils.extract_masked_state(obs_next, self.mask_size, agent_postion_yx_batch)
        
        ## visualization
        self.step_counter += 1
        if self.visualizationFlag and self.step_counter % self.visualize_every == 0 and training:
            next_masked = obs_next_masked + obs_masked 
            obs_next_all = obs + obs_next
            self.visual_func.visualize_data(obs, obs_next_all, act, obs_masked, next_masked, info, self.step_counter, agent_postion_yx_batch)
        return obs_masked, act, obs_next_masked, info


    def training_step(self, batch, batch_idx):
        obs, act, obs_next, info = self.preprocess_batch(batch, True)
        obs_pred, attentionWeight = self(obs, act, info)

        if obs_next.dtype != obs_pred.dtype:
            obs_next = obs_next.float()

        loss = self.loss_function_weight(obs_pred, obs_next)

        # 计算 EWC 正则项，并加入主损失
        ewc_loss_tensor = self.ewc_loss(self.lambda_ewc)
        loss_total = loss['loss_obs'] + ewc_loss_tensor

        # --- Logging ---
        self.log_dict(loss)
        self.log("train/ewc_loss", ewc_loss_tensor)
        self.log("train/loss_total", loss_total)

        # if self.global_step % 1000 == 0:
        #     print(f"[Step {self.global_step}] loss_obs: {loss['loss_obs'].item():.6f}, "
        #           f"ewc_loss: {ewc_loss_tensor.item():.6f}, total: {loss_total.item():.6f}")


        if self.old_params is not None:
            avg_drift = self.compute_avg_param_drift()
            self.update_lambda_ewc(avg_drift)

        # --- 可视化 ---
        # self.step_counter += 1
        # if self.visualizationFlag and self.step_counter % self.visualize_every == 0:
        #     next = obs_next + obs 
        #     pre = obs_pred + obs
        #     self.visual_func.visualize_attention(obs, act, attentionWeight, next, pre, self.step_counter, info)

        return loss_total
    
   
    def validation_step(self, batch, batch_idx):
        obs, act, obs_next, info = self.preprocess_batch(batch)
        obs_pred, attention_weight = self(obs, act, info)
        if obs_next.dtype != obs_pred.dtype:
            obs_next = obs_next.float()
        loss = self.loss_function(obs_pred, obs_next)
        # print(torch.round(obs_pred))
        self.log_dict(loss)
        ## visualization
        # self.step_counter += 1
        # if self.visualizationFlag and self.step_counter % self.visualize_every == 0:
        #     next = obs_next + obs 
        #     pre = obs_pred + obs
        #     self.visual_func.visualize_attention(obs, act, attention_weight, next, pre, self.step_counter, info)
        # return {"batch_idx": batch_idx, "val_loss": loss['loss_obs']}
        return {"loss_wm_val": loss['loss_obs']}

    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:

        losses = [x["loss_wm_val"].item() for x in outputs]
        df = pd.DataFrame(losses, columns=["loss_wm_val"])

        # 保存为 CSV（不包含 index）
        # df.to_csv("validation_21*21_emb_mask5.csv", index=False, header=False)
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

   



