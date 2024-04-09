import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from path import *
import os
import json
import numpy as np
import wandb
import torch.distributions as td

wandb.login(key="ae0b0db53ae05bebce869b5ccc77b9efd0d62c73")
wandb.init(project='MiniGrid', entity='svea41')


class SimpleNN(nn.Module):
    def __init__(self, obs_shape, next_obs_shape, action_shape, hidden_size):
        super(SimpleNN, self).__init__()
        # Calculate the total size of the flattened

        self.total_input_size = torch.prod(torch.tensor(obs_shape)) + action_shape
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
        combined_input = torch.cat((input_obs, input_action), dim=1)
        combined_input = combined_input.float()
        out = self.shared_layers(combined_input)
        obs_out = self.state_head(out)
        reward_out = torch.sigmoid(self.reward_head(out))
        done_out = self.done_head(out)

        # done_out = td.independent.Independent(
        #     td.Bernoulli(logits=done_out), len(self._output_shape)
        # )

        return obs_out, reward_out, done_out


class CustomDataset(Dataset):
    def __init__(self, observations, actions, next_observations, reward, terminated):
        """
        Initialize the dataset with observations, actions, and next observations.
        """
        self.observations = observations
        self.actions = actions
        # self.reward = reward
        self.next_observations = next_observations
        self.reward = reward
        self.terminated = terminated

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.observations)

    def __getitem__(self, idx):
        """
        Fetch the observation, action, and next observation at the specified index.
        """
        return self.observations[idx], self.actions[idx], self.next_observations[idx], self.reward[idx], \
            self.terminated[idx]


def training(loader, num_epochs, net, optimizer):
    net.to(device)
    for epoch in range(num_epochs):  # num_epochs is the number of epochs to train for
        loop = tqdm(loader, leave=True)
        for observations, actions, next_observations, reward, terminated in loop:  # Assume data_loader is defined
            observations = observations.to(device).float()
            # normalization the obs
            observations[:, :, :, 0] = norm(observations[:, :, :, 0], 10)
            observations[:, :, :, 1] = norm(observations[:, :, :, 1], 5)
            observations[:, :, :, 2] = norm(observations[:, :, :, 2], 2)

            actions = actions.to(device)
            next_observations = next_observations.to(device).float()
            reward = reward.to(device)
            terminated = terminated.to(device)
            observations = observations.view(observations.size(0), -1)

            actions = actions.unsqueeze(1)
            # Forward pass: Compute predicted next_observation by passing current observations and actions to the model
            predicted_next_observations, predicted_reward, done = net(observations, actions)
            # normalize the real_obs
            next_observations[:, :, :, 0] = norm(next_observations[:, :, :, 0], 10)
            next_observations[:, :, :, 1] = norm(next_observations[:, :, :, 1], 5)
            next_observations[:, :, :, 2] = norm(next_observations[:, :, :, 2], 2)
            next_observations = next_observations.view(next_observations.size(0), -1)
            # Compute loss
            loss_function = nn.MSELoss()
            loss_function_done = nn.BCEWithLogitsLoss()
            predicted_next_observations = predicted_next_observations.float()
            reward = reward.float()
            next_observations = next_observations.float()
            loss_obs = loss_function(predicted_next_observations, next_observations)
            loss_reward = loss_function(predicted_reward, reward)
            done = torch.tensor(done, dtype=torch.float32)
            terminated = torch.tensor(terminated, dtype=torch.float32).unsqueeze(1)
            loss_done = loss_function_done(done, terminated)

            loss_total = loss_obs + loss_reward + loss_done
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # Update the progress bar description with the current loss
            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(loss=loss_total.item(), loss_obs=loss_obs.item(), loss_reward=loss_reward.item(),
                             loss_done=loss_done.item())
            wandb.log({"epoch": epoch, "loss": loss_total.item(), 'loss_obs': loss_obs.item(),
                       'loss_reward': loss_reward.item(),
                       'loss_done': loss_done.item()})


def load_data(file_path, batch_size):
    with open(file_path, 'r') as file:
        data = json.load(file)

    dataset = CustomDataset(data['obs'], data['action'], data['next_obs'], data['reward'], data['terminated'])
    for i in range(len(data['obs'])):
        data['obs'][i] = np.array(data['obs'][i])
    for i in range(len(data['next_obs'])):
        data['next_obs'][i] = np.array(data['next_obs'][i])
    # Create a DataLoader for batching
    loader = DataLoader(dataset, batch_size, shuffle=True)
    return loader


def test_model(test_model, test_data):
    with torch.no_grad():
        for observations, actions, next_observations, reward, terminated in test_data:
            predicted_next_obs, predicted_rewards, predicted_done = test_model(observations, actions)
            # next_obs
            mse_loss = torch.nn.MSELoss()
            next_obs_accuracy = mse_loss(predicted_next_obs, next_observations)
            # done
            true_done_int = terminated.int()
            predicted_done_int = predicted_done.int()
            done_accuracy = (predicted_done_int == true_done_int).float().mean()
            # reward
            reward_accuracy = mse_loss(predicted_rewards, reward)
        return next_obs_accuracy, done_accuracy, reward_accuracy


def norm(x, max):
    norm_x = x / max
    return norm_x


def denorm(x, max):
    real = x * max
    return real


if __name__ == "__main__":
    # create a network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_shape_real = 6 * 3 * 3
    obs_next_shape_real = 6 * 3 * 3
    action_shape_real = 1
    model = SimpleNN(obs_shape=obs_shape_real, next_obs_shape=obs_next_shape_real, action_shape=1, hidden_size=50)
    model_optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # load the data
    path = Paths()
    data_save = os.path.join(path.MODEL_BASED_DATA, 'env_data.json')
    # Create the dataloader
    data_loader = load_data(data_save, 32)
    training(data_loader, 10000, model, model_optimizer)
    wandb.finish()
    torch.save(model.state_dict(), 'env_model.pth')
    # test the model
    test_path = os.path.join(path.MODEL_BASED_DATA, 'env_data_test.json')
    test_loader = load_data(test_path, 32)
    test_model(model, test_loader)
