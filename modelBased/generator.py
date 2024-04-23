import torch
from torch import nn, optim
import numpy as np
import json
from path import Paths
import os


def parse_map_string(map_string):
    object_str, color_str = map_string.split('\n\n')
    object_map = [list(line) for line in object_str.split('\n')]
    return object_map


def one_hot_encode(map, category_dict):
    flat_list = [category_dict[item] for row in map for item in row]
    one_hot = np.eye(len(category_dict))[flat_list]
    return one_hot.reshape(-1)


def encode_maps(object_map):
    # Mapping characters to indices
    object_dict = {'W': 0, 'E': 1, 'G': 2, 'K': 3, 'D': 4}

    encoded_objects = one_hot_encode(object_map, object_dict)

    # Flatten each map row for use as VAE input
    vae_input = encoded_objects.reshape(encoded_objects.shape[0], -1)

    return vae_input


# read the training_data file
def load_json(save_path):
    with open(save_path, 'r') as json_file:
        environments_list = json.load(json_file)
    return environments_list


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # Using Sigmoid since we are assuming the data is normalized between 0 and 1
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu, log_var = torch.chunk(encoded, 2, dim=-1)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var


# load the data
path = Paths()
file_save = os.path.join(path.CHARACTOR_DATA, "training_data.json")
env_list = load_json(file_save)
# encode the map
for env in env_list:
    object_i = parse_map_string(env)
    input_charactor = encode_maps(object_i)
# Hyperparameters
latent_dim = 50
input_dim = input_charactor.size  # Ensure this matches the flattened input size
epochs = 50
learning_rate = 0.001

# Initialize VAE and optimizer
vae = VAE(input_dim, latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
loss_function = nn.BCELoss(reduction='sum')

# Convert data to PyTorch tensors
train_data = torch.tensor(input_charactor, dtype=torch.float)

# Training loop
for epoch in range(epochs):
    vae.train()
    optimizer.zero_grad()
    reconstruction, mu, log_var = vae(train_data)
    # Calculate loss
    bce_loss = loss_function(reconstruction, train_data)
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    loss = bce_loss + kld_loss
    # Backpropagation
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
