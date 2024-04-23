import matplotlib.pyplot as plt
from path import Paths
from minigrid_custom_env import CustomEnvFromFile
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, ImgObsWrapper
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import torch.distributions as td
from PPO import PPO, training_agent

use_wandb = True
if use_wandb:
    import wandb

    wandb.login(key="ae0b0db53ae05bebce869b5ccc77b9efd0d62c73")
    wandb.init(project='Policy_training', entity='svea41')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleNN(nn.Module):
    def __init__(self, obs_shape, next_obs_shape, action_shape, hidden_size):
        super(SimpleNN, self).__init__()
        # Calculate the total size of the flattened

        self.total_input_size = obs_shape + action_shape
        # Define the first dense layer to process the combined input
        self.shared_layers = nn.Sequential(
            nn.Linear(self.total_input_size, hidden_size),
            nn.ReLU()
        )
        self.state_head = nn.Sequential(
            nn.Linear(hidden_size, next_obs_shape),
            nn.ReLU()
        )
        self.reward_head = nn.Linear(hidden_size, 1)
        self.done_head = nn.Linear(hidden_size, 1)

    def forward(self, input_obs, input_action):
        combined_input = torch.cat((input_obs, input_action), dim=-1)
        combined_input = combined_input.float()
        out = self.shared_layers(combined_input)
        obs_out = self.state_head(out)
        reward_out = torch.sigmoid(self.reward_head(out))
        done_out = self.done_head(out)

        done_out = td.independent.Independent(
            td.Bernoulli(logits=done_out), 1
        )

        return obs_out, reward_out, done_out


class PPOPolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_space):
        super(PPOPolicyNetwork, self).__init__()
        self.input_dim = input_dim
        self.action_space = action_space

        # Fully connected layer for action logits
        self.fc_action = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, action_space),
        )

        # Fully connected layer for state value estimate
        self.fc_value = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
        )

    def forward(self, x):
        # Compute action logits
        action_logits = self.fc_action(x)

        # Compute state value estimate
        value = self.fc_value(x)

        return F.softmax(action_logits, dim=-1), value


class PPOAgent:
    def __init__(self, observation_channels, action_space, lr=1e-3, gamma=0.99, clip_param=0.2, update_interval=4000,
                 epochs=10):
        self.observation_channels = observation_channels
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.clip_param = clip_param
        self.update_interval = update_interval
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PPOPolicyNetwork(observation_channels, action_space).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        # These buffers store trajectories
        self.states = []
        self.actions = []
        self.state_values = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []

    def act(self, state):
        state = state.to(self.device)
        probs, state_value = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        # self.states.append(state)
        # self.actions.append(action)
        self.log_probs.append(m.log_prob(action))
        self.state_values.append(state_value)
        return action.item()

    def calculate_returns(self, rewards, gamma, normalization=True):
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        if normalization:
            returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        return returns

    def update(self):
        # Convert lists to tensors
        states = torch.stack(self.states).squeeze(1).to(self.device)
        actions = torch.tensor(self.actions, dtype=torch.int64).to(self.device)
        old_log_probs = torch.stack(self.log_probs).to(self.device)

        # Placeholder for rewards-to-go calculation; implement your method as needed
        rewards_to_go = self.calculate_returns(self.rewards, self.gamma).to(self.device)

        # Placeholder for advantage calculation; implement a more sophisticated method as needed
        advantages = rewards_to_go - torch.tensor(self.state_values).to(self.device).squeeze()

        # Calculate current log probs and state values for all stored states and actions
        probs, state_values = self.policy(states)
        dist = Categorical(probs)
        current_log_probs = dist.log_prob(actions)

        # Calculate the ratio (pi_theta / pi_theta_old)
        ratios = torch.exp(current_log_probs - old_log_probs)

        # Calculate surrogate loss
        surr1 = ratios * advantages.detach()
        surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages.detach()
        policy_loss = -torch.min(surr1, surr2).mean()

        # Placeholder for value loss; consider using rewards_to_go for more accurate value updates
        value_loss = F.mse_loss(torch.squeeze(state_values), rewards_to_go.detach())

        # Take gradient step
        self.optimizer.zero_grad()
        total_loss = policy_loss + value_loss
        total_loss.backward()
        self.optimizer.step()

        # Clear memory
        self.clear_memory()

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.state_values = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []


def preprocess_observation(obs: np.array) -> torch.Tensor:
    """
    Preprocess the observation obtained from the environment to be suitable for the CNN.
    This function extracts, randomly rotates, and normalizes the 'image' part of the observation.

    :param obs: dict, The observation dictionary received from the environment.
                Expected to have a key 'image' containing the visual representation.
    :return: torch.Tensor, The normalized and randomly rotated image observation.
    """
    # Extract the 'image' array from the observation dictionary
    obs[:, :, 0] = norm(obs[:, :, 0], 10)
    obs[:, :, 1] = norm(obs[:, :, 1], 5)
    obs[:, :, 2] = norm(obs[:, :, 2], 2)
    obs_flat = obs.reshape(-1)

    return obs_flat


def norm(x, max):
    norm_x = x / max
    return norm_x


def run_training(env: CustomEnvFromFile, world_model: SimpleNN, agent: PPOAgent, episodes: int = 100, ) -> None:
    """
    Runs the training loop for a specified number of episodes using PPO.

    Args:
        env (CustomEnvFromFile): The environment instance where the agent will be trained.
        agent (PPOAgent): The agent to be trained with PPO.
        episodes (int): The total number of episodes to run for training.
        env_name (str): A name for the environment, used for saving outputs.

    Returns:
        None
        :param episodes:
        :param agent:
        :param env:
        :param world_model:
    """
    for e in range(episodes):
        obs, _ = env.reset()  # Reset the environment at the start of each episode.
        state = preprocess_observation(torch.from_numpy(obs['image']).float())  # Preprocess the observation

        for time in range(env.max_steps):
            action = agent.act(state)  # Agent selects an action based on the current state.
            next_obs, reward, terminated = world_model(state, torch.tensor(action).unsqueeze(0))
            # Execute the action in world model.
            terminated = terminated.sample()
            agent.states.append(state)
            agent.actions.append(action)
            agent.rewards.append(float(reward))

            done = bool(terminated)  # Check if the episode has ended.

            state = next_obs  # Update the current state for the next iteration.

            if done:
                print(f"Episode: {e}/{episodes}, Score: {time}")
                if use_wandb:
                    cumulative_reward = 0
                    # Process the rewards starting from the end to simplify the computation
                    for reward in reversed(agent.rewards):
                        cumulative_reward = reward + 0.9 * cumulative_reward
                    wandb.log({"episode_reward": cumulative_reward})
                agent.update()
                break


def test_policy(policy, env, num_episodes=100):
    total_rewards = 0.0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        state = torch.FloatTensor(state[0]['image']).unsqueeze(0).to(device)
        while not done:
            with torch.no_grad():  # Important: do not calculate gradients
                action_prob, _ = policy(state.reshape(-1).to(device))

            # Assuming discrete action space here; modify as needed for continuous spaces
            action = action_prob.argmax().view(1, 1).item()

            next_state, reward, done, _, _ = env.step(action)

            episode_reward += reward
            state = torch.FloatTensor(next_state['image'])

        total_rewards += episode_reward

    average_reward = total_rewards / num_episodes
    print(f'Average Reward over {num_episodes} episodes: {average_reward}')
    return average_reward


if __name__ == "__main__":
    # test the model
    loaded_model = SimpleNN(54, 54, 1, 50)
    path = Paths()
    env_0 = FullyObsWrapper(
        CustomEnvFromFile(txt_file_path=path.LEVEL_FILE, custom_mission="Find the key and open the door.",
                          max_steps=2000,
                          render_mode="human"))
    # Please note that the default observation format is a partially observable view of the environment using a compact
    # and efficient encoding, with 3 input values per visible grid cell, 7x7x3 values total.
    loaded_model.load_state_dict(torch.load('env_model.pth'))

    image_shape = np.prod(env_0.observation_space['image'].shape)
    action_space = env_0.action_space.n

    agent = PPOAgent(observation_channels=image_shape, action_space=action_space, lr=1e-3, gamma=0.99)
    run_training(env_0, loaded_model, agent, episodes=10000)
    if use_wandb:
        wandb.finish()
    reward_result = test_policy(agent.policy, env_0, num_episodes=100)
