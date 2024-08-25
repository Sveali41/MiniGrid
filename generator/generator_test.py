import torch.nn.functional as F
import torch
from path import Paths
import os
from generator_vae import VAE, decode_maps
import numpy as np
from minigrid_custom_env import CustomEnvFromFile
from minigrid.manual_control import ManualControl
from generator.basic_gen import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def generate_sample(vae, latent_dim, device):
    # Sample from the standard normal distribution
    z = torch.randn(1, latent_dim).to(device)
    # Decode the sample
    vae.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        generated = vae.decoder(z)
    return generated


def generate_color_map(objects):
    # Initialize color map with all 'W' (white)
    color_map = np.full_like(objects, 'W')

    # Find indices of keys (K) and doors (D)
    key_indices = np.where(objects == 'K')
    door_indices = np.where(objects == 'D')

    # Set color 'R' (red) for keys and doors
    color_map[key_indices] = 'R'
    color_map[door_indices] = 'R'

    return color_map


def format_and_concatenate_maps(obj_map, color_map):
    # Convert numpy arrays to string format
    obj_map_str = '\n'.join(''.join(row) for row in obj_map)
    color_map_str = '\n'.join(''.join(row) for row in color_map)

    # Concatenate maps with two newlines as separator
    combined_map_str = obj_map_str + '\n\n' + color_map_str
    return combined_map_str


def save_map_to_file(map_string, filename):
    with open(filename, 'w') as file:
        file.write(map_string)


def load_vae():
    latent_dim = 50
    input_dim = 500  # Ensure this matches the flattened input size

    # Initialize VAE and optimizer
    vae = VAE(input_dim, latent_dim).to(device)
    path = Paths()
    model_save = os.path.join(path.TRAINED_MODEL, 'generator.pth')
    vae.load_state_dict(torch.load(model_save))

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
    env_str = np.array(decode_maps(class_indices)).reshape(10, 10)
    print(env_str)

    # set the color map
    color_map = generate_color_map(env_str)
    map = format_and_concatenate_maps(env_str, color_map)
    path = Paths()
    name_string = 'test.txt'
    file_save = os.path.join(path.CHARACTOR_DATA, name_string)
    save_map_to_file(map, file_save)
    env = CustomEnvFromFile(txt_file_path=file_save, custom_mission="Find the key and open the door.",
                            render_mode="human")
    env.reset()
    ManualControl(env)

def load_gan():
    input_dim = 500  # Ensure this matches the flattened input size
    output_dim = 500
    # Initialize VAE and optimizer
    gan = Generator(input_dim, output_dim).to(device)
    path = Paths()
    model_save = os.path.join(path.TRAINED_MODEL, 'generator.pth')
    gan.load_state_dict(torch.load(model_save))

    # Example usage
    one_hot = generate_sample(gan, latent_dim=50, device='cuda').reshape(-1, 5)
    probabilities = F.softmax(one_hot, dim=-1)

    # Get the indices of the max values (one-hot positions)
    _, predicted_indices = torch.max(probabilities, -1)
    # Create a tensor for one-hot encoding
    one_hot_output = torch.zeros_like(probabilities)  # This will have the shape (100, 5)
    one_hot_output.scatter_(1, predicted_indices.unsqueeze(1), 1)
    # transfer the one_hot to str
    class_indices = torch.argmax(one_hot_output, dim=-1)
    env_str = np.array(decode_maps(class_indices)).reshape(10, 10)
    print(env_str)

    # set the color map
    color_map = generate_color_map(env_str)
    map = format_and_concatenate_maps(env_str, color_map)
    path = Paths()
    name_string = 'test.txt'
    file_save = os.path.join(path.CHARACTOR_DATA, name_string)
    save_map_to_file(map, file_save)
    env = CustomEnvFromFile(txt_file_path=file_save, custom_mission="Find the key and open the door.",
                            render_mode="human")
    env.reset()
    ManualControl(env)

if __name__ == "__main__":
    load_vae()
