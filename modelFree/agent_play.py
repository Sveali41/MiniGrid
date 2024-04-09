import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from minigrid_custom_env import *
from stable_baselines3.common.callbacks import BaseCallback
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, ImgObsWrapper
import gym
import torch
import torch.nn as nn
import pandas as pd
import os


class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.factor = 0.99
        self.cumulate_reward = []
        self.cumulate_reward_factor = []
        self.episode_rewards = []
        self.timesteps = 1

    def _on_step(self) -> bool:
        # Access and print the reward
        # self.locals contains the rollout data
        reward = self.locals['rewards'][-1]  # Assuming a single environment
        self.cumulate_reward.append(reward)
        t = self.timesteps
        if t == 1:
            G = reward
        else:
            G = (self.factor ** (t - 1)) * reward + self.cumulate_reward_factor[-1]
        self.timesteps += 1
        self.cumulate_reward_factor.append(G)
        if self.locals['dones']:
            # if the episode finished, save G
            self.episode_rewards.append(sum(self.cumulate_reward_factor))
            self.cumulate_reward_factor = []
            self.timesteps = 1
        return True


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def simple_moving_average(values, window_size):
    """Calculate the simple moving average."""
    sma_s = []  # List to hold the simple moving averages
    # Calculate the moving average using a sliding window
    for i in range(len(values) - window_size + 1):
        sma = sum(values[i:i + window_size]) / window_size
        sma_s.append(sma)
    return sma_s


# create specific environment
path = Paths()
env = ImgObsWrapper(CustomEnvFromFile(txt_file_path=path.LEVEL_FILE, custom_mission="Find the key "
                                                                                    "and open the "
                                                                                    "door.",
                                      render_mode="human"))
policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

callback = RewardCallback()
model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=100000, callback=callback)
# save the experiment_MF data into csv file
path = Paths()
df = pd.DataFrame(index=range(100000))

df['Rewards'] = pd.Series(callback.cumulate_reward)
df['Factor rewards'] = pd.Series(callback.cumulate_reward_factor)
df['Episode rewards'] = pd.Series(callback.episode_rewards)

# Save the DataFrame to a CSV file
save_path = os.path.join(path.EXPERIMENT_RESULT, 'experiment_result.csv')
df.to_csv(save_path, index=False)
