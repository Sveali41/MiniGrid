import pandas as pd
from path import *
import matplotlib.pyplot as plt
import glob


#  1. mean_reward
def calculate_mean_reward(rewards):
    cumulative_mean_rewards = []
    cumulative_sum = 0
    for i, rewards in enumerate(rewards, start=1):
        cumulative_sum += rewards
        mean_so_far = cumulative_sum / i
        cumulative_mean_rewards.append(mean_so_far)
    return cumulative_mean_rewards


def calculate_max_rewards(rewards):
    cumulative_max_rewards = []
    max_reward = rewards[0] if rewards else 0  # Initialize with the first reward or 0 if list is empty

    for i in rewards:
        # Update the maximum reward if the current reward is greater
        max_reward = max(max_reward, i)
        # Append the current maximum reward to the max_rewards list
        cumulative_max_rewards.append(max_reward)
    return cumulative_max_rewards


def simple_moving_average(values, window_size):
    """Calculate the simple moving average."""
    sma_s = []  # List to hold the simple moving averages
    # Calculate the moving average using a sliding window
    for i in range(len(values) - window_size + 1):
        sma = sum(values[i:i + window_size]) / window_size
        sma_s.append(sma)
    return sma_s


def plot_rewards_over_time(path_visualize, reward_list, figure_title):
    """
    Plots the rewards at each timestep for three episodes on a single figure using a line plot.

    Parameters:
    reward1, reward2, reward3 (lists): Lists containing rewards at each timestep for three different episodes.
    """
    plt.figure(figsize=(18, 6))
    i = 1
    for rewards in reward_list:
        timesteps = range(1, len(rewards) + 1)
        plt.plot(timesteps, rewards, label=f'run {i}')
        i += 1
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.title(figure_title)
    plt.legend()
    plt.grid(True)
    filename = f'{figure_title}.png'
    file_path = os.path.join(path_visualize.EXPERIMENT_VISUALIZED, filename)
    plt.savefig(file_path)


def load_file(dfs):
    reward_factored_all = []
    mean_rewards_all = []
    max_rewards_all = []
    for df in dfs:
        reward = df['Rewards'].tolist()
        reward_factored = df['Episode rewards'].tolist()
        reward_factored_all.append(reward_factored)
        mean_rewards = calculate_mean_reward(reward)
        mean_rewards_all.append(mean_rewards)
        max_rewards = calculate_max_rewards(reward)
        max_rewards_all.append(max_rewards)

    return reward_factored_all, mean_rewards_all, max_rewards_all


path = Paths()
result_path = os.path.join(path.EXPERIMENT_RESULT, 'experiment_result_*.csv')
file_pattern = f"{result_path}"
# List all files in the folder that match the pattern
csv_files = glob.glob(file_pattern)
# Read each CSV file and store the DataFrames in a list
dataframes = [pd.read_csv(file) for file in csv_files]
reward_factored_real, mean_rewards_real, max_rewards_real = load_file(dataframes)
# plot and compare the outcome
# 1. factored_reward
reward_factored_smooth = []
for reward in reward_factored_real:
    smoothed = simple_moving_average(reward, 5)
    reward_factored_smooth.append(smoothed)
plot_rewards_over_time(path, reward_factored_smooth, 'reward_factored_real')
