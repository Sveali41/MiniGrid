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
from data_collect import *


# set device to cpu or cuda
device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

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
    save_experiments(cfg.data_collect, obs, obs_next, act, rew, done, Rmax)


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
    # data
    dataloader = WMRLDataModule(hparams = hparams.world_model, data=data)
    # Get a single batch from the dataloader
    # 
    # dataloader.setup()dataloader_train = dataloader.train_dataloader()
    # dataloader_val = dataloader.val_dataloader()
    if model is None:
        net = SimpleNN(hparams=hparams)
    wandb_logger = WandbLogger(project="WM Training", log_model=True)
    wandb_logger.experiment.watch(net, log='all', log_freq=1000)
    # Define the trainer
    metric_to_monitor = 'avg_val_loss_wm'#"loss"
    early_stop_callback = EarlyStopping(monitor=metric_to_monitor, min_delta=0.00, patience=10, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(
                            save_top_k=1,
                            monitor = metric_to_monitor,
                            mode = "min",
                            dirpath = get_env('PTH_FOLDER'),
                            filename ="wm-{epoch:02d}-{avg_val_loss_wm:.4f}",
                            verbose = True
                        )
    trainer = pl.Trainer(logger=wandb_logger,
                    max_epochs=hparams.world_model.n_epochs, 
                    gpus=1,
                    callbacks=[early_stop_callback, checkpoint_callback])     
    # Start the training
    trainer.fit(net,dataloader)
    # Log the trained model
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

def update_policy_with_world_model(cfg, env, policy, world_model, R_max):
    """
    Update the policy using the given data.

    Parameters:
        policy: The policy to update.
        data: List of (state, action, next_state, reward) tuples.

    Returns:
        The updated policy.
    """
    hparams = cfg
        # PPO hyperparameters
    start_time = datetime.now().replace(microsecond=0)
    action_std_decay_rate = cfg.PPO.action_std_decay_rate
    min_action_std = cfg.PPO.min_action_std
    action_std_decay_freq = cfg.PPO.action_std_decay_freq
    save_model_freq = cfg.PPO.save_model_freq
    max_ep_len = cfg.PPO.max_ep_len
    has_continuous_action_space = cfg.PPO.has_continuous_action_space
    checkpoint_path = cfg.PPO.checkpoint_path

    # 1. Initialize training PPO
    i_episode = 0
    update_timestep = max_ep_len * 4  # update policy every n timesteps
    print_freq = max_ep_len * 4
    print_running_reward = 0
    print_running_episodes = 0
    time_step = 0
   # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space
    else:
        action_dim = env.action_space.n
    state_dim = np.prod(env.observation_space['image'].shape)
    
    
    # 3. Initialize R_max exploration
    rmax_exploration = RMaxExploration(state_dim, action_dim, R_max=1.0, exploration_threshold=10)
    for time_step in range(1, hparams.R_max.max_training_timesteps + 1):
        if time_step <= hparams.R_max.exploration_timesteps:
            # when timestep < Rmax exploration threshold, collect data from real env
            state = env.reset()[0]['image']
            done = False
            while not done:
            # collect data from real env under the ppo policy to train World Model
                state = normalize(state).to(device)
                action = policy.select_action(state)
                next_state, reward, done, _, _ =  env.step(action)
                next_state = normalize(next_state['image']).to(device)
                # save this to data buffer to train world model
                # Count and state and action pair
                rmax_exploration.update_visit_count(state.cpu().numpy(), action, reward, next_state.cpu().numpy())
                state = next_state

        else:
            # collect data from world model and update PPO agent
            state = state.squeeze()
            action = ppo_agent.select_action(state)
            state = model(state, torch.tensor(action/hparams.world_model.action_norm_values).to(device).unsqueeze(0))
            state = state.to(dtype=torch.float32)
            # denorm the state
            state_denorm = map_obs_to_nearest_value(cfg, state)
            # obtain reward from the state representation & done
            done, reward = get_destination(state_denorm, t, max_ep_len, device)
            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

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
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            # if done:
            #     break

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            i_episode += 1

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
                        max_steps=2000, render_mode="human"))
    num_iterations = cfg.R_max.num_iterations
    exploration_policy = policy_initialization(cfg, env)
    world_model = None  
    rmax_exploration = RMaxExploration(cfg.R_max.R_max, cfg.R_max.exploration_threshold)
    for _ in range(num_iterations):
        # Step 1: collect env data from real env
        # data = collect_data_from_env(cfg, env, exploration_policy, rmax_exploration)  

        # Step 2: for iteration in range(num_iterations)
        train_world_model(cfg)  

        # Step 3: train PPO agent udpate policy with world model
        exploration_policy = update_policy_with_world_model(cfg, env, world_model, exploration_policy, rmax_exploration)  


if __name__ == "__main__":
    training_agent_with_rmax()