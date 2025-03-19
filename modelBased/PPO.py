import os
import glob
import time
from datetime import datetime
import sys
sys.path.append('/home/siyao/project/rlPractice/MiniGrid')
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
from minigrid_custom_env import CustomEnvFromFile
from minigrid.wrappers import FullyObsWrapper
from path import Paths
import random



# set device to cpu or cuda
device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    # def act(self, state):

    #     if self.has_continuous_action_space:
    #         action_mean = self.actor(state)
    #         cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
    #         dist = MultivariateNormal(action_mean, cov_mat)
    #     else:
    #         action_probs = self.actor(state)
    #         dist = Categorical(action_probs)

    #     action = dist.sample()
    #     action_logprob = dist.log_prob(action)
    #     state_val = self.critic(state)

    #     return action.detach(), action_logprob.detach(), state_val.detach()


    def act(self, state, epsilon=0.1, forward_bias=0.6):
        state = state.to(device)  # Ensure state is on GPU

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)

            # Normalize action probabilities
            action_probs = action_probs / action_probs.sum()

            # Move action_probs to GPU
            action_probs = action_probs.to(device)

            # Create probability distribution
            dist = Categorical(action_probs)

            # ---------------------------
            #  1️. Exploration: Sample a random action (favoring forward)
            # ---------------------------
            if random.random() < epsilon:
                num_actions = len(action_probs)
                forward_action_index = 2  # Assuming 2 represents "forward" in MiniGrid

                # Create an exploration probability distribution (fully on GPU)
                explore_probs = torch.ones(num_actions, device=device) * ((1 - forward_bias) / (num_actions - 1))
                explore_probs[forward_action_index] = forward_bias
                explore_probs = explore_probs / explore_probs.sum()  # Normalize

                # Sample a random action (EXPLORATION) - Now using `torch.multinomial()`
                action = torch.multinomial(explore_probs, 1).squeeze(0)  # Sample directly on GPU
            
            # ---------------------------
            #  2. Exploitation: Select action from trained PPO policy
            # ---------------------------
            else:
                action = dist.sample()  # Sample action from PPO-trained policy

        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()




    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # for single action continuous environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

        else:
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


def preprocess_observation(obs):
    obs = obs / np.array([10, 5, 2])
    return torch.from_numpy(obs.flatten()).float().to(device)


def training_agent(env, path):
    # parameters
    has_continuous_action_space = False
    max_ep_len = 400  # max timesteps in one episode
    max_training_timesteps = int(1e5)  # break training loop if timeteps > max_training_timesteps
    print_freq = max_ep_len * 4  # print avg reward in the interval (in num timesteps)
    save_model_freq = int(2e4)  # save model frequency (in num timesteps)
    update_timestep = max_ep_len * 4  # update policy every n timesteps
    print_running_reward = 0
    print_running_episodes = 0

    time_step = 0
    i_episode = 0
    action_std_decay_freq = int(2.5e5)
    action_std_decay_rate = 0.05
    min_action_std = 0.1
    checkpoint_path = os.path.join(path, 'PPO_model.pth')

    # param for agent
    K_epochs = 40  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor
    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network
    random_seed = 0  # set random seed if required (0 = no random seed)
    action_std = None
    has_continuous_action_space = False

    # state space dimension
    state_dim = np.prod(env_0.observation_space['image'].shape)

    # action space dimension
    if has_continuous_action_space:
        action_dim = env_0.action_space
    else:
        action_dim = env_0.action_space.n

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0
        state = preprocess_observation(state[0]['image']).to(device)

        for t in range(1, max_ep_len + 1):

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _, _ = env.step(action)
            state = preprocess_observation(state['image']).to(device)
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
                print_avg_reward = round(print_avg_reward, 2)
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
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        i_episode += 1

    env.close()


if __name__ == "__main__":
    use_wandb = True
    if use_wandb:
        import wandb

        wandb.login(key="ae0b0db53ae05bebce869b5ccc77b9efd0d62c73")
        wandb.init(project='PPO_test', entity='svea41')
    path = Paths()
    env_0 = FullyObsWrapper(
        CustomEnvFromFile(txt_file_path=path.LEVEL_FILE, custom_mission="Find the key and open the door.",
                          max_steps=1000,
                          render_mode="rgb"))

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    training_agent(env_0, path.TRAINED_MODEL)

    if use_wandb:
        wandb.finish()
