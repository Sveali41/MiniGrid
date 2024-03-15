import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from path import *
import os


class SimpleNN(nn.Module):
    def __init__(self, obs_shape, next_obs_shape, action_shape, hidden_size):
        super(SimpleNN, self).__init__()
        # Calculate the total size of the flattened
        self.total_input_size = torch.prod(torch.tensor(obs_shape)) + action_shape
        # Define the first dense layer to process the combined input
        self.fc1 = nn.Linear(self.total_input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, next_obs_shape)

    def forward(self, input_obs, input_action):
        input_obs = torch.flatten(input_obs)
        combined_input = torch.cat((input_obs, input_action), dim=-1)
        out = self.fc1(combined_input)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class CustomDataset(Dataset):
    def __init__(self, observations, actions, next_observations):
        """
        Initialize the dataset with observations, actions, and next observations.
        """
        self.observations = observations
        self.actions = actions
        # self.reward = reward
        self.next_observations = next_observations

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.observations)

    def __getitem__(self, idx):
        """
        Fetch the observation, action, and next observation at the specified index.
        """
        return self.observations[idx], self.actions[idx], self.next_observations[idx]


def training(loader, num_epochs, net, optimizer):
    for epoch in range(num_epochs):  # num_epochs is the number of epochs to train for
        loop = tqdm(loader, leave=True)
        for observations, actions, next_observations in loop:  # Assume data_loader is defined

            # Forward pass: Compute predicted next_observation by passing current observations and actions to the model
            predicted_next_observations = net(observations, actions)

            # Compute loss
            loss_function = nn.MSELoss()
            loss = loss_function(predicted_next_observations, next_observations)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the progress bar description with the current loss
            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())


# create a network
obs_shape_real = 7 * 7 * 3
obs_next_shape_real = 7 * 7 * 3
action_shape_real = 1
model = SimpleNN(obs_shape=obs_shape_real, next_obs_shape=obs_next_shape_real, action_shape=1, hidden_size=50)
model_optimizer = optim.Adam(model.parameters(), lr=0.001)
# load the data
path = Paths()
data_save = os.path.join(path.MODEL_BASED_DATA, 'env_data.csv')
df_read = pd.read_csv(data_save)
# Create the dataset

dataset = CustomDataset(df_read['obs'], df_read['action'], df_read['next_obs'])
# Create a DataLoader for batching
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
training(data_loader, 10, model, model_optimizer)
