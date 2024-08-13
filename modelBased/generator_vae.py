import torch
from torch import nn, optim
import numpy as np
import json
from path import Paths
import os
from torch.utils.data import Dataset, DataLoader
from charactor_dataset import generate_maps
import torch.nn.functional as F
import torch.nn.init as init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

use_wandb = False
if use_wandb:
    import wandb

    wandb.login(key="ae0b0db53ae05bebce869b5ccc77b9efd0d62c73")
    wandb.init(project='Gen', entity='svea41')


def parse_map_string(map_string):
    object_str, color_str = map_string.split('\n\n')
    object_map = [list(line) for line in object_str.split('\n')]
    return object_map


def one_hot_encode(map, category_dict):
    flat_list = [category_dict[item] for row in map for item in row]
    one_hot = np.eye(len(category_dict))[flat_list]
    return one_hot.reshape(-1)


def decode_maps(object_map):
    # Mapping characters to indices
    object_dict = {'W': 0, 'E': 1, 'G': 2, 'K': 3, 'D': 4}
    reverse_object_dict = {value: key for key, value in object_dict.items()}
    mapped_data = [reverse_object_dict[value.item()] for value in object_map]
    return mapped_data


def encode_maps(object_map):
    # Mapping characters to indices
    object_dict = {'W': 0, 'E': 1, 'G': 2, 'K': 3, 'D': 4}

    vae_input = one_hot_encode(object_map, object_dict)

    # Flatten each map row for use as VAE input

    return vae_input


# read the training_data file
def load_json(save_path):
    with open(save_path, 'r') as json_file:
        environments_list = json.load(json_file)
    return environments_list


class RandomCharacterDataset(Dataset):
    def __init__(self, num_samples, width, height):
        self.num_samples = num_samples
        self.width = width
        self.height = height

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        map = generate_maps(self.width, self.height)
        map_onehot = encode_maps(map[0])
        return torch.tensor(map_onehot, dtype=torch.float32)


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2*latent_dim),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
            # Using Sigmoid since we are assuming the data is normalized between 0 and 1
        )
        # Call the custom initialization function
        self.initialize_weights()
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu, log_var = torch.chunk(encoded, 2, dim=-1)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)  # Set bias to a small positive value
            elif isinstance(m, nn.Embedding):
                # Initialize embeddings with a standard normal distribution
                init.normal_(m.weight, mean=0, std=1)





# generator
def generate_sample(vae, latent_dim, device):
    # Sample from the standard normal distribution
    z = torch.randn(1, latent_dim).to(device)
    # Decode the sample
    vae.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        generated = vae.decoder(z)
    return generated


if __name__ == "__main__":
    # load the data
    # path = Paths()
    # file_save = os.path.join(path.CHARACTOR_DATA, "training_data.json")
    # env_list = load_json(file_save)
    # encode the map
    # for env in env_list:
    #     object_i = parse_map_string(env)
    #     input_charactor = encode_maps(object_i)
    # Hyperparameters
    latent_dim = 50
    input_dim = 500  # Ensure this matches the flattened input size
    epochs = 10000
    learning_rate = 0.0003

    # Initialize VAE and optimizer
    vae = VAE(input_dim, latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    loss_function = nn.BCELoss(reduction='mean')

    # Dataset and DataLoader setup
    dataset = RandomCharacterDataset(1000, 10, 10)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training loop
    for epoch in range(epochs):
        for batch_data in dataloader:
            batch_data = batch_data.to(device)  # Move batch data to GPU
            vae.train()
            optimizer.zero_grad()
            reconstruction, mu, log_var = vae(batch_data)
            # Calculate loss
            bce_loss = loss_function(reconstruction, batch_data)
            # kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            # loss = bce_loss + kld_loss
            if use_wandb:
                wandb.log({"epoch": epoch, "loss": bce_loss.item()})
            # Backpropagation
            bce_loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}, Loss: {bce_loss.item()}')
    if use_wandb:
        wandb.finish()

    path = Paths()
    model_save = os.path.join(path.TRAINED_MODEL, 'generator.pth')
    torch.save(vae.state_dict(), model_save)

    # Example usage
    one_hot = generate_sample(vae, latent_dim=50, device='cuda').reshape(-1, 5)
    probabilities = F.softmax(one_hot, dim=-1)

    # Get the indices of the max values (one-hot positions)
    _, predicted_indices = torch.max(probabilities, -1)
    # Create a tensor for one-hot encoding
    one_hot_output = torch.zeros_like(probabilities)  # This will have the shape (100, 5)
    one_hot_output.scatter_(1, predicted_indices.unsqueeze(1), 1)
    # transfer the one_hot to str
    class_indices = torch.argmax(one_hot_output, dim=-1)
    env_str = np.array(decode_maps(class_indices)).reshape(10,10)
    print(env_str)
