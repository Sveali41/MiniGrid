import sys
from .common.utils import PROJECT_ROOT
from minigrid_custom_env import CustomMiniGridEnv
from minigrid.wrappers import FullyObsWrapper
import torch
import numpy as np
import networkx as nx
import pickle
from collections import deque
import random
from omegaconf import DictConfig
import hydra
from datetime import datetime
from .common import utils
from . import AttentionWM_support
from . import Embedding_support
from . import MLP_support
from PPO_world_training import find_position, process_data


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class GraphPlanner:
    def __init__(self, model, num_actions, mask_size):
        """
        model: world-model module, returns (delta_masked, _)
        num_actions: number of discrete actions
        mask_size: size used by process_data for masking
        """
        self.model = model
        self.num_actions = num_actions
        self.mask_size = mask_size
        self.G = nx.DiGraph()  # dynamic graph of masked states

    def _state_key(self, masked: np.ndarray) -> bytes:
        # convert masked-state array into a hashable bytes key
        return masked.astype(np.float32).tobytes()

    def expand_node(self, masked: np.ndarray):
        """
        For a given masked-state, run every action through the world model
        and add outgoing edges to the graph.
        """
        key = self._state_key(masked)
        if key in self.G:
            return
        self.G.add_node(key)
        for a in range(self.num_actions):
            with torch.no_grad():
                delta, _ = self.model(
                    torch.tensor(masked).unsqueeze(0).to(device),
                    torch.tensor([a]).to(device),
                    {'carrying_key': False}
                )
            next_masked = masked + delta.squeeze(0).cpu().numpy()
            cost = 1.0  # uniform edge cost; you can integrate reward or uncertainty
            self.G.add_edge(
                key,
                self._state_key(next_masked),
                action=a,
                weight=cost
            )

    def plan(self, start_masked: np.ndarray, goal_masked: np.ndarray, k: int = 1):
        """
        Online graph expansion + A* search between start and goal masked-states.
        Returns the first k actions on the shortest path.
        """
        # ensure both nodes exist and their edges are expanded
        self.expand_node(start_masked)
        self.expand_node(goal_masked)

        start_key = self._state_key(start_masked)
        goal_key = self._state_key(goal_masked)

        # admissible heuristic: L2 distance in masked-state space
        def heuristic(n1, n2):
            v1 = np.frombuffer(n1, dtype=np.float32)
            v2 = np.frombuffer(n2, dtype=np.float32)
            return np.linalg.norm(v1 - v2)

        path = nx.astar_path(self.G, start_key, goal_key, heuristic=heuristic, weight='weight')
        actions = [self.G.edges[(u, v)]['action'] for u, v in zip(path, path[1:])]
        return actions[:k]


class BCAgent(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, state):
        logits = self.net(state)
        return torch.argmax(logits, dim=-1).item()

    def update(self, batch, optimizer):
        states, actions = zip(*batch)
        s = torch.stack(states).to(device)
        a = torch.tensor(actions, dtype=torch.long, device=device)
        loss = torch.nn.functional.cross_entropy(self.net(s), a)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "modelBased/config"), config_name="config")
def training_agent_wm(cfg: DictConfig):
    run_planner_wm(cfg)

def run_planner_wm(cfg: DictConfig):
    # Unpack configs
    wm_cfg  = cfg.attention_model
    planner_cfg = cfg.planner

    # 1. Load world model
    MODEL_MAP = {
            'attention': AttentionWM_support.AttentionModule,
            'embedding': Embedding_support.EmbeddingModule,
            'mlp': MLP_support.SimpleNNModule
        }
    ModelClass = MODEL_MAP[wm_cfg.model_type.lower()]
    model = ModelClass(
        wm_cfg.data_type,
        wm_cfg.grid_shape,
        wm_cfg.attention_mask_size,
        wm_cfg.embed_dim,
        wm_cfg.num_heads
    ).to(device)
    utils.load_model_weight(model, wm_cfg.model_save_path)
    model.eval()
    mask_size = wm_cfg.attention_mask_size

    # 2. Create environment
    env = FullyObsWrapper(
        CustomMiniGridEnv(
            txt_file_path=planner_cfg.env_path,
            custom_mission="Find the key and open the door.",
            max_steps=planner_cfg.max_ep_len,
            render_mode=None
        )
    )
    state_dim  = int(np.prod(env.observation_space['image'].shape))
    action_dim = env.action_space.n

    # 3. Initialize planner and BC agent
    planner   = GraphPlanner(model, action_dim, mask_size)
    bc_agent  = BCAgent(state_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(bc_agent.parameters(), lr=planner_cfg.lr_actor)
    replay    = deque(maxlen=10000)
    batch_size = planner_cfg.get("batch_size", 512)

    max_steps = planner_cfg.max_training_timesteps
    max_len   = planner_cfg.max_ep_len

    time_step = 0
    # 4. Main training loop
    while time_step < max_steps:
        # 4.1 Reset real env to get initial raw observation
        init_state = env.reset()[0]['image']
        state_0 = utils.ColRowCanl_to_CanlRowCol(init_state)
        masked_state = process_data(state_0, mask_size)
        # Build masked goal (here reusing same masked; replace as needed)
        goal_pos    = find_position(state_0, (8,1,0))
        goal_masked = masked_state.copy()
        info = {'carrying_key': False}

        for t in range(1, max_len + 1):
            # 4.2 Planner → expert action
            expert_planner = planner.plan(masked_state, goal_masked, k=1)[0]

            # 4.3 Store for behavior cloning
            state_flat = torch.tensor(masked_state.flatten(), dtype=torch.float32)
            replay.append((state_flat, expert_planner))

            # 4.4 Roll out one step in the world model
            sm  = torch.tensor(masked_state, dtype=torch.float32).unsqueeze(0).to(device)
            a_tensor = torch.tensor([expert_planner], device=device)
            with torch.no_grad():
                delta_masked, _ = model(sm, a_tensor, info)
            masked_state = (masked_state + delta_masked.squeeze(0).cpu().numpy())

            # 4.5 Behavior-cloning update
            if len(replay) >= batch_size and time_step % batch_size == 0:
                batch = random.sample(replay, batch_size)
                bc_agent.update(batch, optimizer)
            time_step += 1

            # Optional: define your own done criterion in the world model
            done = False
            if done:
                break

    # 5. Save the cloned policy
    bc_path = planner_cfg.checkpoint_path_wm.replace('.pth', '_bc.pth')
    torch.save(bc_agent.state_dict(), bc_path)



if __name__ == "__main__":
    training_agent_wm()
