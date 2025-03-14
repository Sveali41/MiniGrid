import sys
sys.path.append('/home/siyao/project/rlPractice/MiniGrid')
from minigrid_custom_env import *
from minigrid.wrappers import FullyObsWrapper,  ImgObsWrapper
from path import *
import pandas as pd
from modelBased.common.utils import PROJECT_ROOT
from PPO import PPO
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import hydra
from modelBased.world_model_training import normalize, map_obs_to_nearest_value
import torch
from Rmax import RMaxExploration
from world_model_training import *
from modelBased.common.bin.PPO_world_training import get_destination, find_position
from data_collect import *
import wandb
from data.datamodule import extract_agent_cross_mask
from modelBased.AttentionWM import *


# set device to cpu or cuda
device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

def postprocess_delta_obs_batch(obs):
    # Round, convert to int, and reshape to (3, 3, 3)
    obs = obs.view(3, 3, 3)
    return obs

def replace_values(delta, target, position):
    """
    Replace the values in target tensor with the values in delta tensor.
    
    Args:
        delta (torch.Tensor): delta tensor of shape (3, 3, 3).
        target (torch.Tensor): Target tensor of shape (*, *, 3).
        position (tuple): Position (x, y) of the center in the target tensor.
        
    Returns:
        torch.Tensor: The target tensor with values from the delta tensor replaced at the specified position.
    """
    # Check if the delta tensor is of the correct shape
    if delta.shape != (3, 3, 3):
        raise ValueError("delta tensor must be of shape (3, 3, 3)")
    
    # Get the center position in the target tensor
    y, x = position
    
    target[:, y.item() - 1: y.item() + 2, x.item() - 1:x.item() + 2] += delta
    
    return target

def map_to_nearest_value_rmax(cfg, obs):
    hparams = cfg
    valid_values_obj = hparams.world_model.valid_values_obj
    valid_values_color = hparams.world_model.valid_values_color
    valid_values_state = hparams.world_model.valid_values_state
    # Map each channel to the nearest valid value
    obs[:, :, 0] = map_to_nearest_value(obs[:, :, 0], valid_values_obj)
    obs[:, :, 1] = map_to_nearest_value(obs[:, :, 1], valid_values_color)
    obs[:, :, 2] = map_to_nearest_value(obs[:, :, 2], valid_values_state)
    return obs

def map_to_nearest_value_att(cfg, obs):
    hparams = cfg
    valid_values_obj = hparams.world_model.valid_values_obj
    valid_values_color = hparams.world_model.valid_values_color
    valid_values_state = hparams.world_model.valid_values_state
    # Map each channel to the nearest valid value
    obs[0, :, :] = map_to_nearest_value(obs[0, :, :], valid_values_obj)
    obs[1, :, :] = map_to_nearest_value(obs[1, :, :], valid_values_color)
    obs[2, :, :] = map_to_nearest_value(obs[2, :, :], valid_values_state)
    return obs

def collect_data_from_env(cfg, env, policy, Rmax):
    """
    collect data from the environment using the given policy.
    
    Parameters:
        policy: policy to collect data from the environment.
        num_episodes: the number of episodes to collect.

    Returns:
        List of (state, action, next_state, reward) tuples.
    """
    obs, obs_next, act, rew, done = run_env(env, cfg.data_collect, policy, Rmax)
    save_experiments(cfg.data_collect, obs, obs_next, act, rew, done)
    # save count of the visit
    Rmax.save_count(cfg.collect.visit_count)

def load_attention_model(cfg):
    """
    Load the attention model.

    Parameters:
        cfg: The configuration object.

    Returns:
        The loaded attention model.
    """
    # hparams = cfg
    # model = ExtractionModule(hparams.attention_model.action_dim, hparams.attention_model.embed_dim, hparams.attention_model.num_heads)
    # # Load the checkpoint
    # checkpoint = torch.load(hparams.attention_model.pth_folder)
    # # Load state_dict into the model
    # model.load_state_dict(checkpoint['state_dict'])
    # # Set the model to evaluation mode (optional, depends on use case)
    # extraction_module = model.extraction_module

    hparams = cfg
    model = IntegratedModel(hparams.attention_model.obs_size, hparams.attention_model.action_size, hparams.attention_model.embed_dim, hparams.attention_model.num_heads)
    # Load the checkpoint
    checkpoint = torch.load(hparams.attention_model.pth_folder)
    # Load state_dict into the model
    model.load_state_dict(checkpoint['state_dict'])
    # Set the model to evaluation mode (optional, depends on use case)
    extraction_module = model.extraction_module
    return extraction_module

def train_world_model(cfg, data=None, model=None):
    """
    Train the world model using the given data.

    Parameters:
        data: List of (state, action, next_state, reward) tuples.
        model: The world model to train.
        optimizer: The optimizer to use for training.
        num_epochs: The number of epochs to train the model.

    Returns:
        The trained world model.
    """
    hparams = cfg
    use_wandb = cfg.world_model.use_wandb
    # data
    dataloader = WMRLDataModule(hparams = hparams.world_model, data=data)
    # Get a single batch from the dataloader
    # dataloader.setup()
    # dataloader.setup()dataloader_train = dataloader.train_dataloader()
    # dataloader_val = dataloader.val_dataloader()
    if model is None:
        # this line is to check if we need to load the model and fine-tune it
        net = SimpleNN(hparams=hparams)
    # Set up logger
    wandb_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(project="WM_Rmax", log_model=True)
        wandb_logger.experiment.watch(net, log='all', log_freq=1000)
    else:
        print("Debug mode enabled. WandB logging is disabled.")
    # Define the trainer
    metric_to_monitor = 'avg_val_loss_wm'#"loss"
    early_stop_callback = EarlyStopping(monitor=metric_to_monitor, min_delta=0.00, patience=15, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(
                            save_top_k=1,
                            monitor = metric_to_monitor,
                            mode = "min",
                            dirpath = get_env('PTH_FOLDER'),
                            filename ="wm_rmax-{epoch:02d}-{avg_val_loss_wm:.4f}",
                            verbose = True
                        )
    trainer = pl.Trainer(logger=wandb_logger if use_wandb else None,
                    max_epochs=hparams.world_model.n_epochs, 
                    gpus=1,
                    callbacks=[early_stop_callback, checkpoint_callback])     
    # Start the training
    trainer.fit(net,dataloader)
    if use_wandb:
        # Log the trained model only when wandb is used
        model_pth = hparams.world_model.pth_folder
        trainer.save_checkpoint(model_pth)
        wandb.save(str(model_pth))
    return net

def policy_initialization(cfg, env):
    """
    Initialize the policy.

    Parameters:
        cfg: The configuration object.

    Returns:
        The initialized policy.
    """
    
    # PPO hyperparameters
    lr_actor = cfg.PPO.lr_actor
    lr_critic = cfg.PPO.lr_critic
    gamma = cfg.PPO.gamma
    K_epochs = cfg.PPO.K_epochs
    eps_clip = cfg.PPO.eps_clip
    action_std = cfg.PPO.action_std
    has_continuous_action_space = cfg.PPO.has_continuous_action_space
    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space
    else:
        action_dim = env.action_space.n
    state_dim = np.prod(env.observation_space['image'].shape)
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)
    return ppo_agent

def update_policy_with_world_model(cfg, env, policy, R_max):
    """
    Update the policy using the given data.

    Parameters:
        policy: The policy to update.
        world model: The world model to use for updating the policy.
        R_max: The R_max value for the R_max exploration.
        env: The environment to reset.

    Returns:
        The updated policy.
    """
    hparams = cfg
    use_wandb = True
    # wandb.login(key="ae0b0db53ae05bebce869b5ccc77b9efd0d62c73")
    if use_wandb == True:
        wandb.init(project='WM_PPO_RMAX', entity='svea41')
    # PPO hyperparameters
    start_time = datetime.now().replace(microsecond=0)
    action_std_decay_rate = cfg.PPO.action_std_decay_rate
    min_action_std = cfg.PPO.min_action_std
    action_std_decay_freq = cfg.PPO.action_std_decay_freq
    save_model_freq = cfg.PPO.save_model_freq
    max_ep_len = cfg.PPO.max_ep_len
    has_continuous_action_space = cfg.PPO.has_continuous_action_space
    checkpoint_path = cfg.PPO.checkpoint_path
    max_training_timesteps = cfg.PPO.max_training_timesteps

    # 1. Initialize training PPO
    i_episode = 0
    update_timestep = max_ep_len * 4  # update policy every n timesteps
    print_freq = max_ep_len * 4
    print_running_reward = 0
    print_running_episodes = 0
    time_step = 0
    
    # 2. load the world model
    world_model = SimpleNN(hparams=hparams).to(device)
    checkpoint = torch.load(hparams.world_model.pth_folder)
    world_model.load_state_dict(checkpoint['state_dict'])
    # Set the model to evaluation mode (optional, depends on use case)
    world_model.eval()
    # 3. Initialize R_max exploration
    R_max.load_count(cfg.data_collect.collect.visit_count)
    
    # training loop
    while time_step <= max_training_timesteps:
        state = env.reset()[0]['image']
        goal_position = find_position(state, (8, 1, 0)) # find the goal position
        current_ep_reward = 0
        for t in range(1, max_ep_len + 1):
            # buffer.states = [state.squeeze(0) if state.dim() > 1 else state for state in buffer.states]
            if t>1:
                state = state.cpu().numpy()
            agent_pos = np.argwhere(state[:, :, 0] == 10)
            state_extract = extract_agent_cross_mask(state)
            state_extract_norm = normalize(state_extract).to(device)
            # else:
            #     agent_pos = np.argwhere(state[:, :, 0] == 10)
            #     state_extract = extract_agent_cross_mask(state)
            #     state_extract_norm = normalize(state_extract).to(device)
            state_norm = normalize(state).to(device)
            action = policy.select_action(state_norm)
            delta_state = postprocess_delta_obs_batch(world_model(state_extract_norm, torch.tensor(action/hparams.world_model.action_norm_values).to(device).unsqueeze(0)))
            state_change = torch.tensor(state_extract).to(device) + delta_state
            state_change = map_to_nearest_value_rmax(cfg, state_change)
            state_next = replace_values(state_change, torch.tensor(state).to(device), agent_pos[0])
            # obtain reward from the state representation & done
            done, reward = get_destination(state_next, t, max_ep_len, goal_position)
            # use previous state(flatten without normalize) and action to get the R_max reward
            reward = R_max.get_rmax_reward(state_extract, action, reward)
            # get the R_max value for the rewards
            # saving reward and is_terminals
            policy.buffer.rewards.append(reward)
            policy.buffer.is_terminals.append(done)
            state = state_next

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                policy.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                policy.decay_action_std(action_std_decay_rate, min_action_std)

            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                if use_wandb:
                    wandb.log({"average_reward": print_avg_reward})

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                policy.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        i_episode += 1

    wandb.finish()
    env.close()



@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "conf/Rmax"), config_name="config")
def training_agent_with_rmax(cfg: DictConfig):
    """This function trains the agent using PPO algorithm based on R_max concept
    1. collect env data from real env
    2. train world model
    3. collect data from world model
    4. train PPO agent
    """

    # Params for R_max
    path = Paths()
    env = FullyObsWrapper(CustomEnvFromFile(txt_file_path=path.LEVEL_FILE_Rmax, custom_mission="Find the key "
                                                                                      "and open the "
                                                                                      "door.",
                        max_steps=20000, render_mode=None))
    num_iterations = cfg.R_max.num_iterations
    exploration_policy = policy_initialization(cfg, env)
    world_model = None  
    rmax_exploration = RMaxExploration(cfg.R_max.R_max, cfg.R_max.exploration_threshold)
    for _ in range(num_iterations):
        # # Step 1: collect env data from real env
        # data = collect_data_from_env(cfg, env, exploration_policy, rmax_exploration)  

        # # Step 2: train world model (under abstracted features from attention model)
        train_world_model(cfg)  

        # # Step 3: train PPO agent udpate policy with world model
        # exploration_policy = update_policy_with_world_model(cfg, env, exploration_policy, rmax_exploration)  

# for shape (channels, height, width)
def denormalize_conv(x): 
    obs_norm_values = [10, 5, 3] 
    for i in range(x.shape[0]):  
        max_value = obs_norm_values[i]
        if max_value != 0: 
            x[i, :, :] = x[i, :, :] * max_value
    return x

    

@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "conf/Rmax"), config_name="config")
def validate(cfg: DictConfig):
    hparams = cfg
    model = SimpleNN(hparams=hparams)
    # Load the checkpoint
    dataloader = WMRLDataModule(hparams = hparams.world_model)
    dataloader.setup()
    checkpoint = torch.load(hparams.world_model.pth_folder)
    # Load state_dict into the model
    model.load_state_dict(checkpoint['state_dict'])
    # Set the model to evaluation mode (optional, depends on use case)
    model.eval()
    map_width = hparams.world_model.map_width
    map_height = hparams.world_model.map_height
    # Assuming the rest of your code is already set up as provided
    batch_size = 64
    num_tests = 20

    # Loop over the first 10 observations
    for i in range(num_tests):
        # Collecting the first 64 samples for the current test observation
        obs_batch = torch.tensor([dataloader.data_test[j]['obs'] for j in range(batch_size)])
        act_batch = torch.tensor([dataloader.data_test[j]['act'] for j in range(batch_size)])
        obs_real_batch = torch.tensor([dataloader.data_test[j]['obs_next'] for j in range(batch_size)])

        # Predict using the model
        if hparams.world_model.model == 'Attention':
            obs_temp = obs_batch.view(obs_batch.size(0), 12, 6, 3)  # convert to (batch_size, width, height, channels)
            obs_temp = obs_temp.permute(0, 3, 2, 1)  # convert to (batch_size, channels, height, width)
        obs_pred = model(obs_temp, act_batch)
        # map the observation to the nearest valid value
        obs_pred_map = []
        act_batch = act_batch * 6
        for k in range(batch_size):
            # denormalize the observation
            # add the denormalized and the delta observation
            delta_obs = postprocess_delta_obs_batch(obs_pred[k])
            obs_current = denormalize_conv(obs_temp[k])
            if hparams.world_model.model == 'Attention':
                agent_pos = np.argwhere(obs_current[0, :, :] == 10)
                obs_add =  replace_values(delta_obs.permute(2,1,0), obs_current.clone(), agent_pos)
                obs_add = map_to_nearest_value_att(cfg, obs_add)
                print("action:",act_batch[k])
                print("obs_change:", torch.round(delta_obs.permute(2,1,0)))
                print("obs_dir:",obs_current[2, :, :])
                print("obs_cur:", obs_current[0, :, :])
                print("obs_add:",obs_add[0, :, :])
                visualization(obs_current, act_batch[k], obs_add)

                
            else:
                obs_add = delta_obs + obs_current
            # print((delta_obs-obs_real_batch[k].view(3, 3, 3)).sum())
        print(obs_pred_map)
    pass



if __name__ == "__main__":
    # training_agent_with_rmax()
    validate()