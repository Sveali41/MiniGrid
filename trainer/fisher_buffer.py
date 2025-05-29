import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from modelBased.common import utils
import random


class FisherReplayBuffer:
    def __init__(self, max_size: int = 1000):
        self.buffer = []
        self.max_size = max_size
        self.mask_size = 3  # Cross mask size

    def compute_proxy_score_batch(
        self,
        model: torch.nn.Module,
        samples: List[Dict],
        top_k: int = 50
    ) -> List[Tuple[float, Dict]]:
        model.eval()
        device = next(model.parameters()).device
        scores = []

        with torch.no_grad():
            obs = torch.tensor(samples['obs']).to(device).float()
            act = torch.tensor(samples['act']).to(device)
            obs_next = torch.tensor(samples['obs_next']).to(device).float()
            if 'info' in samples:
                info = torch.tensor(samples['info']).to(device).float()
            else:
                info = None
            agent_postion_yx_batch = utils.get_agent_position(obs)
            agent_postion_yx_batch_next = utils.get_agent_position(obs_next)
            obs_masked = utils.extract_masked_state(obs, self.mask_size, agent_postion_yx_batch)
            obs_next_masked = utils.extract_masked_state(obs_next, self.mask_size, agent_postion_yx_batch_next)
            pred, _ = model(obs_masked, act, info)
            loss = [F.mse_loss(pred[i], obs_next_masked[i]).item() for i in range(len(pred))]

              # 计算每个样本的损失

            # 可选加权项，例如状态变化量
            delta = [(obs_next[i] - obs[i]).abs().mean().item() for i in range(len(obs_next))]
            score = [l + 0.1 * d for l, d in zip(loss, delta)] # 组合得分（你可以调系数）
        
            scored_samples = list(zip(score, [dict(obs=samples['obs'][i],
                                           act=samples['act'][i],
                                           obs_next=samples['obs_next'][i]) for i in range(len(score))]))
        scored_samples.sort(key=lambda x: -x[0])
        top_k_samples = [s for _, s in scored_samples[:top_k]]
        return top_k_samples


    def select_important_samples(
        self,
        samples: List[Dict],
        model: torch.nn.Module,
        fisher: Dict[str, torch.Tensor], 
        top_k: int = 50
    ) -> List[Dict]:
        scored = self.compute_proxy_score_batch(model, samples, top_k)
        return scored
    

    def update_with_top_k_recent(self, samples: Dict, model: torch.nn.Module, fisher: Dict[str, torch.Tensor], recent_k: int = 200, top_k: int = 50):
        samples['obs'] = samples['obs'][:recent_k]
        samples['obs_next'] = samples['obs_next'][:recent_k]
        samples['act'] = samples['act'][:recent_k]
        if 'info' in samples:
            samples['info'] = samples['info'][:recent_k]
        selected = self.select_important_samples(samples, model, fisher, top_k)
        self.buffer.extend(selected)
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]
    
    def update_with_random(
        self,
        samples: Dict,
        recent_k: int = 20000,
        random_k: int = 10000
    ):
        # 限制 recent_k 大小
        for k in ['obs', 'act', 'obs_next', 'info']:
            if k in samples:
                samples[k] = samples[k][:recent_k]

        # 获取所有样本的数量
        total_len = len(samples['obs'])
        indices = list(range(total_len))
        random.shuffle(indices)
        selected_indices = indices[:random_k]

        selected = []
        for i in selected_indices:
            sample = {
                'obs': samples['obs'][i],
                'act': samples['act'][i],
                'obs_next': samples['obs_next'][i]
            }
            if 'info' in samples:
                sample['info'] = samples['info'][i]
            selected.append(sample)

        self.buffer.extend(selected)
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]

    def export_dict(self) -> Dict[str, np.ndarray]:
        if not self.buffer:
            raise ValueError("Replay buffer is empty.")
        keys = self.buffer[0].keys()
        return {k: np.stack([s[k] for s in self.buffer]) for k in keys}

    def __len__(self):
        return len(self.buffer)
