import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple


class FisherReplayBuffer:
    def __init__(self, max_size: int = 1000):
        self.buffer = []
        self.max_size = max_size

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
            obs = torch.tensor(samples['obs'])
            act = torch.tensor(samples['act'])
            obs_next = torch.tensor(samples['obs_next'])
            for i in range(act.shape[0]):
                pred, _ = model(obs[i], act[i])
                loss = F.mse_loss(pred, obs_next[i]).item()

                # 可选加权项，例如状态变化量
                delta = (obs_next - obs).abs().mean().item()
                score = loss + 0.1 * delta  # 组合得分（你可以调系数）

                scores.append((score, sample))

        scores.sort(key=lambda x: -x[0])  # 按得分降序
        return scores[:top_k]


    def select_important_samples(
        self,
        samples: List[Dict],
        model: torch.nn.Module,
        fisher: Dict[str, torch.Tensor], 
        top_k: int = 50
    ) -> List[Dict]:
        scored = self.compute_proxy_score_batch(model, samples, top_k)
        return [s for _, s in scored]
    

    def update_with_top_k_recent(self, samples: Dict, model: torch.nn.Module, fisher: Dict[str, torch.Tensor], recent_k: int = 200, top_k: int = 50):
        samples['obs'] = samples['obs'][:recent_k]
        samples['obs_next'] = samples['obs_next'][:recent_k]
        samples['act'] = samples['act'][:recent_k]
        selected = self.select_important_samples(samples, model, fisher, top_k)
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
