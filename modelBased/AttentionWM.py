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
        self.ewc_ratio = getattr(hparams, "ewc_ratio", 0.2)           # 目标占比：EWC ≈ 20% * obs_loss；想手动控制就在 yaml 里设为 null
        self.lambda_ema = getattr(hparams, "lambda_ema", 0.1)         # λ 的 EMA 平滑系数
        self.lambda_ewc_min = getattr(hparams, "lambda_ewc_min", 1e-4)
        self.lambda_ewc_max = getattr(hparams, "lambda_ewc_max", 1e3)
        self.lambda_ewc = float(getattr(hparams, "lambda_ewc", 1.0))
        self.keep_cell_loss = getattr(hparams, "keep_cell_loss", False)  # whether or not to keep cell loss
        self.loss_accumulator = [[[] for _ in range(self.col)] for _ in range(self.row)]



        # 慢速外环（漂移）相关
        self.warmup_steps = getattr(hparams, "warmup_steps", 100)
        self.drift_cooldown = getattr(hparams, "drift_cooldown", 200)  # 多久允许调整一次
        self._last_drift_update_step = -10**9
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
        old_params = {k: v.clone().detach().cpu() 
              for k, v in self.state_dict().items()}
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

    def update_lambda_ewc_by_ratio(self, obs_loss: torch.Tensor, ewc_raw: torch.Tensor, r: float | None):
        """
        快速控制器：让 ewc_term ≈ r * obs_loss
        - r=None 或 r<=0 则不启用（使用固定 self.lambda_ewc）
        - 使用 EMA 平滑避免抖动，并限制在 [lambda_ewc_min, lambda_ewc_max]
        """
        if r is None or r <= 0:
            return
        val = ewc_raw.detach()
        if val.item() <= 0:
            return

        target_lambda = (r * obs_loss.detach()) / (val + 1e-12)
        new_lambda = (1 - self.lambda_ema) * self.lambda_ewc + self.lambda_ema * target_lambda.item()
        self.lambda_ewc = float(torch.clamp(torch.tensor(new_lambda), self.lambda_ewc_min, self.lambda_ewc_max))

    def update_lambda_ewc(self, avg_drift: float):
        """
        慢速外环：基于参数漂移 avg_drift 的 λ 调整（与快速占比控制器互补）
        - warmup 自动学习 drift_threshold（当 hparams.drift_threshold 为 None 时）
        - 冷却时间 cooldown：减少与快速控制器的相互干扰
        - 滞回区间 hysteresis：超出阈值范围才调整
        - 对数域微调：上下调更平滑，再做 EMA 平滑与边界裁剪
        """
        import math
        import torch

        # —— 保护性默认值（若没设定）——
        if not hasattr(self, "lambda_ewc"):
            self.lambda_ewc = float(getattr(self.hparams, "lambda_ewc", 1.0))
        if not hasattr(self, "_drift_values"):
            self._drift_values = []
        if not hasattr(self, "_last_drift_update_step"):
            self._last_drift_update_step = -10**9

        # —— 读超参（允许在 hparams 或实例属性上覆盖）——
        warmup_steps = getattr(self.hparams, "warmup_steps", getattr(self, "warmup_steps", 100))
        cooldown = getattr(self.hparams, "drift_cooldown", getattr(self, "drift_cooldown", 200))
        hi_ratio = getattr(self.hparams, "drift_hi", 1.10)  # 高于阈值 110% 才上调
        lo_ratio = getattr(self.hparams, "drift_lo", 0.70)  # 低于阈值 70% 才下调
        up_step = getattr(self.hparams, "drift_up_step", 0.10)  # 对数域上调步长
        down_step = getattr(self.hparams, "drift_down_step", 0.05)  # 对数域下调步长
        lam_min = getattr(self.hparams, "lambda_ewc_min", getattr(self, "lambda_ewc_min", 1e-4))
        lam_max = getattr(self.hparams, "lambda_ewc_max", getattr(self, "lambda_ewc_max", 1e3))
        lam_ema = getattr(self.hparams, "lambda_ema", getattr(self, "lambda_ema", 0.1))

        # —— 记录漂移日志 —— 
        self.log("train/avg_param_drift", avg_drift)

        # —— warmup：若阈值为 None，则用前 warmup_steps 个漂移的均值做阈值 —— 
        if getattr(self, "drift_threshold", None) is None:
            self._drift_values.append(float(avg_drift))
            if len(self._drift_values) >= warmup_steps:
                self.drift_threshold = float(sum(self._drift_values) / len(self._drift_values))
                print(f"[Auto-tuned] drift_threshold set to {self.drift_threshold:.6f}")
            # warmup 期间不调 λ
            return

        # —— 冷却：避免频繁与快速控制器打架 —— 
        if self.global_step - self._last_drift_update_step < cooldown:
            return

        # —— 滞回区间：超过才调 —— 
        hi = self.drift_threshold * hi_ratio
        lo = self.drift_threshold * lo_ratio

        lam = float(max(self.lambda_ewc, lam_min))
        lam_log = math.log(lam)
        changed = False

        if avg_drift > hi:
            lam_log += up_step
            changed = True
        elif avg_drift < lo:
            lam_log -= down_step
            changed = True

        if changed:
            lam_new = math.exp(lam_log)
            # 边界裁剪
            lam_new = float(torch.clamp(torch.tensor(lam_new), lam_min, lam_max))
            # EMA 平滑
            self.lambda_ewc = (1.0 - lam_ema) * self.lambda_ewc + lam_ema * lam_new
            # 更新时间戳
            self._last_drift_update_step = self.global_step

    def load_old_params(self, old_params):
        device = next(self.parameters()).device
        self.old_params = {k: v.to(device) for k, v in old_params.items()}

    # def compute_fisher(self, dataloader, samples, scale_factor):
    #     fisher = {n: torch.zeros_like(p) for n, p in self.named_parameters() if p.requires_grad}
    #     self.eval()
    #     device = next(self.parameters()).device
    #     count = 0
    #     for i, batch in enumerate(dataloader):
    #         if count >= samples:
    #             break
    #         self.zero_grad()

    #         obs, act, obs_next, info, obs_masked = self.preprocess_batch(batch)
    #         obs = obs.to(device).float()
    #         act = act.to(device)
    #         obs_next = obs_next.to(device).float()
    #         obs_pred, _ = self(obs, act, info)
    #         loss = scale_factor * self.loss_function_weight(obs_pred, obs_next, obs_masked)['loss_obs']
    #         loss.backward(retain_graph=False)
    #         for n, p in self.named_parameters():
    #             if p.grad is not None:
    #                 fisher[n] += p.grad.detach().pow(2)
    #         count += 1
    #     for n in fisher:
    #         fisher[n] /= count
    #     # for k in fisher:
    #     #     fisher[k] = torch.sqrt(fisher[k] + 1e-8)
    #     #     fisher[k] *= 5
    #     # Fisher normalization by mean
    #     for k in fisher:
    #         mean_val = fisher[k].mean()
    #         fisher[k] = scale_factor * fisher[k] / (mean_val + 1e-8)

    #     all_f = torch.cat([f.flatten() for f in fisher.values()])
    #     print(f"[Fisher] mean={all_f.mean():.3e}, max={all_f.max():.3e}, min={all_f.min():.3e}")
    #     return fisher

    def compute_fisher(self, dataloader, samples, scale_factor):
        """
        稳定的对角 Fisher 估计（与原签名兼容）
        - 不使用 scale_factor（仅保留兼容）
        - 关闭 AMP，fp32 反传
        - 对 batch 求“均值”（/count）
        - 99.9% 分位截断 + 全局均值归一化（mean≈1）
        """
        import torch
        self.eval()  # 固定 Dropout/BN 的行为（注意：BN 统计需要单独处理，见下文）
        device = next(self.parameters()).device

        fisher = {n: torch.zeros_like(p, dtype=torch.float32, device=device)
                for n, p in self.named_parameters() if p.requires_grad}

        count = 0
        for i, batch in enumerate(dataloader):
            if count >= int(samples):
                break

            self.zero_grad(set_to_none=True)

            # === 预处理到 fp32 ===
            obs, act, obs_next, info, obs_masked, _ = self.preprocess_batch(batch)
            obs      = obs.to(device, dtype=torch.float32)
            act      = act.to(device)
            obs_next = obs_next.to(device, dtype=torch.float32)

            # === 显式关闭 autocast，确保梯度为 fp32 ===
            with torch.cuda.amp.autocast(enabled=False):
                pred, _ = self(obs, act, info)
                loss_obs = self.loss_function_weight(pred, obs_next, obs_masked)['loss_obs']

            loss_obs.backward()

            # === 累积 grad^2 ===
            for n, p in self.named_parameters():
                if p.requires_grad and p.grad is not None:
                    g2 = p.grad.detach().float().pow(2)
                    fisher[n] += g2

            count += 1

        if count == 0:
            raise RuntimeError("compute_fisher: 'samples' 为 0 或 dataloader 为空。")

        # === 对 batch 做均值（/count），不是 /sqrt(count) ===
        for n in fisher:
            fisher[n] /= float(count)

        # === 分位数截断（更稳妥的 outlier 处理） ===
        with torch.no_grad():
            flat = torch.cat([t.flatten() for t in fisher.values()])
            q = torch.quantile(flat, 0.999)  # 99.9% 分位
            for n in fisher:
                fisher[n].clamp_(max=q.item())

        # === 全局均值归一化（让 mean≈1，便于 lambda_ewc 稳定） ===
        with torch.no_grad():
            flat = torch.cat([t.flatten() for t in fisher.values()])
            mean_val = flat.mean().clamp_min(1e-12)
            for n in fisher:
                fisher[n] /= mean_val

            # 诊断
            flat_norm = flat / mean_val
            p99 = torch.quantile(flat_norm, 0.99).item()
            print(f"[Fisher] batches={count}, mean≈1.0, p99={p99:.3e}, max={flat_norm.max().item():.3e}")

        # 放到 CPU 便于后续持久化
        fisher = {k: v.detach().cpu() for k, v in fisher.items()}
        return fisher


    

    # def ewc_loss(self, lambda_ewc):
    #     if self.fisher is None or self.old_params is None:
    #         return torch.tensor(0.0, device=next(self.parameters()).device)
        
    #     device = next(self.parameters()).device
    #     loss = torch.tensor(0.0, device=device)  
    #     for n, p in self.named_parameters():
    #         if n in self.fisher and n in self.old_params:
    #             fisher = self.fisher[n].to(device)
    #             p_old = self.old_params[n].to(device)
    #             loss += (fisher * (p - p_old).pow(2)).sum()

    #     return lambda_ewc * loss

    def set_consolidation(self, old_params: dict, fisher: dict, load_weights: bool = True):
        """
        设置 EWC 的“锚点”信息（旧参数 + Fisher），并可选地加载旧参数权重到当前模型。

        Args:
            old_params (dict): 上一阶段保存的模型参数 (state_dict)
            fisher (dict): Fisher 信息矩阵
            load_weights (bool): 是否将旧参数直接加载到当前模型
        """
        # ----------------------------------------------------------
        # (1) 旧参数部分
        # ----------------------------------------------------------
        if old_params is not None:
            # 保存旧参数为 CPU 版本 (float32)
            self.old_params = {k: v.detach().cpu().float() for k, v in old_params.items()}

            if load_weights:
                # 尝试将旧参数加载进当前模型
                current_state = self.state_dict()
                updated_state = {}

                loaded_keys, skipped_keys = [], []
                for k, v in old_params.items():
                    if k in current_state and current_state[k].shape == v.shape:
                        updated_state[k] = v.clone().detach()
                        loaded_keys.append(k)
                    else:
                        skipped_keys.append(k)

                # 执行加载（严格性关闭以防形状不匹配）
                current_state.update(updated_state)
                self.load_state_dict(current_state, strict=False)

                print(f"[EWC] Loaded {len(loaded_keys)} parameters from previous task "
                    f"(skipped {len(skipped_keys)} mismatched keys).")
            else:
                print("[EWC] old_params received but model weights not loaded (load_weights=False).")

        else:
            self.old_params = None
            print("[EWC] No old_params provided — starting from scratch.")

        # ----------------------------------------------------------
        # (2) Fisher 信息矩阵部分
        # ----------------------------------------------------------
        if fisher is not None:
            self.fisher = {k: v.detach().cpu().float() for k, v in fisher.items()}
            print(f"[EWC] Fisher matrix loaded with {len(self.fisher)} entries.")
        else:
            self.fisher = None
            print("[EWC] No Fisher matrix provided — no EWC regularization will be applied.")


    def ewc_loss(self):
        """
        返回“原始 EWC 值”（未乘 lambda_ewc），在 fp32 中计算。
        这里做了两个稳定化：
        1) 显式 to(device)+float()，避免半精度参与
        2) 按参数规模做平均（/count），让尺度与模型大小无关
        """
        device = next(self.parameters()).device
        if self.fisher is None or self.old_params is None:
            return torch.zeros((), device=device, dtype=torch.float32)

        total = torch.zeros((), device=device, dtype=torch.float32)
        count = 0

        # 关闭 autocast，确保 fp32
        import torch.cuda.amp as amp
        with amp.autocast(enabled=False):
            for n, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                if n not in self.fisher or n not in self.old_params:
                    continue

                f = self.fisher[n].to(device=device, dtype=torch.float32)
                d = (p.float() - self.old_params[n].to(device).float())
                total = total + (f * d.pow(2)).sum()
                count += d.numel()

            if count > 0:
                total = total / count

        return total  # 注意：这里不乘 lambda
    
    def accumulate_loss(self, loss_map, agent_pos):
        """
        loss_map: (mask_size, mask_size)  局部loss
        agent_pos: (y, x) 智能体在全局地图中的位置
        """
        ay, ax = agent_pos
        half = self.mask_size // 2

        for dy in range(self.mask_size):
            for dx in range(self.mask_size):
                global_y = ay + (dy - half)
                global_x = ax + (dx - half)

                # 边界检查，防止越界
                if 0 <= global_y < self.row and 0 <= global_x < self.col:
                    value = loss_map[dy, dx].item()
                    self.loss_accumulator[global_y][global_x].append(value)

    def compute_cell_loss(self, next_pred, next_true):
        """
        Compute per-cell MSE loss.
        Args:
            next_pred: Tensor (B, C, H, W)
            next_true: Tensor (B, C, H, W)
        Returns:
            loss_map: Tensor (B, H, W) - mean squared error per cell
        """
        # 计算每个位置的误差 (B, C, H, W)
        error = torch.abs(next_pred - next_true)
        
        # 在通道维度求平均 => (B, H, W)
        loss_map = error.mean(dim=1)

        return loss_map



    def forward(self, state, action, info):
        next_state_pred, attentionWeight = self.model(state, action, info)
        
        return next_state_pred, attentionWeight

    def loss_function(self, next_observations_predict, next_observations_true):
        loss_obs = self.loss(next_observations_predict.flatten(1), next_observations_true.flatten(1))
        loss = {'loss_obs':loss_obs}
        return loss
    

    def loss_function_weight(self, next_observations_predict, next_observations_true, obs_masked=None):
        device = next_observations_predict.device  # 获取模型所在设备（保险起见）


        # 1. 所有通道中任意变化都算作变化（当前已有）
        change_mask = (next_observations_true.abs() > 1e-6).any(dim=1, keepdim=True)  # (B,1,H,W)
        if obs_masked is not None:
            obs_masked = obs_masked.to(device)
            change_mask = change_mask.to(device)

        # 2. 基础 MSE（不减维度）
        base = F.mse_loss(
            next_observations_predict,
            next_observations_true,
            reduction='none'   # (B,C,H,W)
        )

        # 3. 初始权重：默认 1，变化位置 ×5
        w = torch.ones_like(base)
        w[change_mask.expand_as(base)] *= 5.0

        # 4. 如果提供了 obs_masked，额外对门/钥匙区域加权（×3）
        if obs_masked is not None:
            combined_mask = obs_masked & change_mask.squeeze(1)  # 同时必须有变化

            # 扩展为所有通道
            elements_mask_full = combined_mask.unsqueeze(1).expand_as(base)  # (B,C,H,W)
            w[elements_mask_full] *= 2.0  # 基于已有权重再叠加 +5 → 总共 ×10

        # 5. 最终加权 loss
        loss = (base * w).mean()
        return {"loss_obs": loss}



    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=self.weight_decay)
        # reduce_lr_on_plateau = ReduceLROnPlateau(optimizer, mode='min',verbose=True, min_lr=1e-8)
        reduce_lr_on_plateau = ReduceLROnPlateau(optimizer, mode='min', min_lr=1e-8)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr_on_plateau,
                "monitor": 'avg_val_loss_wm',
                "frequency": 1
            },
        }

    def preprocess_batch(self, batch, training=False):
        '''
        Preprocess the batch data: extract masked observations and object positions.
        batch['obs']: (B, C, H, W)
        '''
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

        # extract positions where objects are located
        object_map = obs_masked[:, 0]  # 取第0通道 (B,H,W)
        key_mask = (object_map == 5)
        door_mask = (object_map == 4)
        lava_mask = (object_map == 9)
        elements_mask = key_mask | door_mask | lava_mask  # (B,H,W)
        
        ## visualization
        self.step_counter += 1
        if self.visualizationFlag and self.step_counter % self.visualize_every == 0 and training:
            next_masked = obs_next_masked + obs_masked 
            obs_next_all = obs + obs_next
            self.visual_func.visualize_data(obs, obs_next_all, act, obs_masked, next_masked, info, self.step_counter, agent_postion_yx_batch)
        return obs_masked, act, obs_next_masked, info, elements_mask, agent_postion_yx_batch


    def training_step(self, batch, batch_idx):
        # —— 前向 & 主损失 —— #
        obs, act, obs_next, info, elements_mask, _ = self.preprocess_batch(batch, True)
        obs_pred, attentionWeight = self(obs, act, info)

        if obs_next.dtype != obs_pred.dtype:
            obs_next = obs_next.float()

        loss_dict = self.loss_function_weight(obs_pred, obs_next, elements_mask)
        obs_loss = loss_dict['loss_obs']

        # —— raw EWC（未乘 λ）—— #
        ewc_raw = self.ewc_loss()

        # —— 快控制：把 EWC 项对齐到 obs_loss 的固定占比（如 20%）—— #
        self.update_lambda_ewc_by_ratio(obs_loss, ewc_raw, self.ewc_ratio)

        # —— 合成总损失 —— #
        ewc_term = self.lambda_ewc * ewc_raw
        loss_total = obs_loss + ewc_term

        # —— 统一日志（用标量 tensor 更稳）—— #
        self.log_dict(loss_dict, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/ewc_raw", ewc_raw.detach(), on_step=True, on_epoch=True)
        self.log("train/lambda_ewc", torch.tensor(self.lambda_ewc, device=obs_loss.device),
                on_step=True, on_epoch=True)
        self.log("train/ewc_term", ewc_term.detach(), on_step=True, on_epoch=True)
        self.log("train/loss_total", loss_total.detach(), on_step=True, on_epoch=True)

        if self.global_step % 1000 == 0:
            print(f"[Step {self.global_step}] "
                f"loss_obs: {obs_loss.item():.6f}, "
                f"ewc_raw: {ewc_raw.item():.6f}, "
                f"lambda: {self.lambda_ewc:.4f}, "
                f"ewc_term: {ewc_term.item():.6f}, "
                f"total: {loss_total.item():.6f}")

        # # —— 慢速外环（有旧参数时才调；函数内部有 cooldown）—— #
        # if self.old_params is not None:
        #     avg_drift = self.compute_avg_param_drift()
        #     self.update_lambda_ewc(avg_drift)

        return loss_total

    
   
    def validation_step(self, batch, batch_idx):
        obs, act, obs_next, info, _, agent_position = self.preprocess_batch(batch)
        obs_pred, attention_weight = self(obs, act, info)
        # if self.hparams.freeze_weight:
        #     diff = torch.abs(obs_pred - obs_next)  # (128, 3, 3, 3)
        #     max_diff_per_group, max_indices = diff.reshape(diff.shape[0], -1).max(dim=1)  
        #     mask = max_diff_per_group > 0.1
        #     indices = torch.nonzero(mask, as_tuple=True)[0]
        #     for idx in indices:
        #         flat_idx = max_indices[idx].item()
        #         pred_val = obs_pred[idx].reshape(-1)[flat_idx].item()
        #         true_val = obs_next[idx].reshape(-1)[flat_idx].item()
        #         print(f"索引 {idx.item()} 最大差值: {max_diff_per_group[idx].item():.4f}, "
        #             f"pred={pred_val:.4f}, true={true_val:.4f}")
        
        # 将loss映射回全局并保存到列表中
        if self.keep_cell_loss:
            loss_map = self.compute_cell_loss(obs_pred, obs_next)
            batch_size = loss_map.shape[0]
            for i in range(batch_size):
                agent_pos = agent_position[i].tolist()  # (y, x)
                self.accumulate_loss(loss_map[i], agent_pos)
  
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

        if self.keep_cell_loss:
            avg_loss_map = torch.zeros((self.row, self.col), device=self.device)
            for y in range(self.row):
                for x in range(self.col):
                    vals = self.loss_accumulator[y][x]
                    avg_loss_map[y, x] = sum(vals) / len(vals) if vals else 0

            self.loss_map_result = avg_loss_map.cpu().numpy()
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

   



