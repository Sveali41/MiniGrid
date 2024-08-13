from path import Paths
from minigrid_custom_env import CustomEnvFromFile
from minigrid.wrappers import FullyObsWrapper
import torch
import os
from model_based import load_data, norm, SimpleNN

# set device to cpu or cuda
device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


def test_model(test_model, observations, actions):
    predicted_next_obs = test_model(observations, actions)
    return predicted_next_obs


def post_process_obs(predictions):
    predictions = predictions.reshape(-1, 6, 3, 3)
    predictions_denorm = denorm_and_round(predictions, (10, 5, 2))
    # for the env now: EKDGW element in the map (object, color, status)


def evaluate_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    criterion = torch.nn.MSELoss()  # Define the loss function

    with torch.no_grad():  # No need to track gradients
        for observations, actions, next_observations in test_loader:
            observations = norm(observations.float(), (10, 5, 2))
            observations = observations.view(observations.size(0), -1)
            next_observations = next_observations.float().to(device)
            # reward = reward.to(device)
            # terminated = terminated.to(device)
            actions = actions.unsqueeze(1).to(device)
            observations = observations.to(device).float()
            # Assuming your observations and next_observations are already flattened if needed
            # or adjust shape accordingly
            predictions = model(observations, actions)
            predictions = predictions.reshape(-1, 6, 3, 3)
            predictions_denorm = denorm_and_round(predictions, (10, 5, 2))
            loss = criterion(predictions_denorm, next_observations)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    return avg_loss


def norm(tensor, factors):
    for i in range(tensor.shape[-1]):
        tensor[:, :, :, i] = tensor[:, :, :, i] / factors[i]

    return tensor


def denorm_and_round(tensor, factors):
    """
    Denormalizes the tensor using factors provided and rounds the values.

    Args:
    - tensor: Input tensor to denormalize and round
    - factors: List of factors for denormalization, one for each channel

    Returns:
    - Rounded and denormalized tensor
    """
    # Iterate over each channel
    for i in range(tensor.shape[-1]):
        tensor = torch.clamp(tensor, min=-1e10, max=1e10)
        tensor[:, :, :, i] = torch.round(tensor[:, :, :, i] * factors[i])

    return tensor


if __name__ == "__main__":
    loaded_model = SimpleNN(54, 54, 1, 50).to(device)
    path = Paths()
    env_0 = FullyObsWrapper(
        CustomEnvFromFile(txt_file_path=path.LEVEL_FILE, custom_mission="Find the key and open the door.",
                          max_steps=2000,
                          render_mode="rgb"))
    # Please note that the default observation format is a partially observable view of the environment using a
    # compact and efficient encoding, with 3 input values per visible grid cell, 7x7x3 values total.
    loaded_model.load_state_dict(torch.load('env_model.pth'))
    data_save = os.path.join(path.MODEL_BASED_DATA, 'env_data.json')
    # Create the dataloader
    # data_loader = load_data(data_save, 1)
    # for observations, actions, next_observations in data_loader:
    #     observations = norm(observations, (10, 5, 2))
    #     next_observations = next_observations.float()
    #     # reward = reward.to(device)
    #     # terminated = terminated.to(device)
    #     actions = actions.unsqueeze(1).to(device)
    #     observations = observations.view(observations.size(0), -1).to(device)
    #
    #     predict_observations = test_model(loaded_model, observations, actions)
    #     predict_observations = predict_observations.view(1, 6, 3, 3)
    #     predict_observations = denorm_and_round(predict_observations, (10, 5, 2))

    # test the model
    test_data_path = os.path.join(path.MODEL_BASED_DATA, 'env_data_test.json')
    test_data = load_data(test_data_path, 32)
    test_accuracy = evaluate_model(loaded_model, test_data, device)
    print(f"Test MSE: {test_accuracy}")
