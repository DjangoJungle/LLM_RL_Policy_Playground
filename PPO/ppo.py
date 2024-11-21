import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import numpy as np
from network import FeedForwardNN


class PPO:
    def __init__(self, env):
        # Extravt environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # ALG STEP 1
        # Initialize actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)
        self._init_hyperparameters()

        # Create the optimizer
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Create variable for the matrix
        # Create the covariance matrix
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)  # Choose 0.5 for stdev arbitrarily
        self.cov_mat = torch.diag(self.cov_var)



    def learn(self, total_timesteps):
        t_so_far = 0  # Timesteps simulated so far

        while t_so_far < total_timesteps:  # ALG STEP 2
            # ALG STEP 3
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Print progress
            print(f"Timesteps so far: {t_so_far}/{total_timesteps}")
            print(f"Average batch reward: {batch_rtgs.mean().item():.2f}")

            # Calculate V_{pho, k}
            V, _ = self.evaluate(batch_obs, batch_acts)

            # ALG STEP 5
            # Calculate advantage
            A_k = batch_rtgs - V.detach()
            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                # Calculate pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network                
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()



    def evaluate(self, batch_obs, batch_acts):
        # Query critic network for a value V for each obs in batch_obs
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most
        # recent actor network
        # Similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs

    def rollout(self):
        # Batch data
        batch_obs = []          # batch observations
        batch_acts = []         # batch actions
        batch_log_probs = []    # log probs of each action
        batch_rews = []         # batch rewards
        batch_rtgs = []         # batch rewards-to-go
        batch_lens = []         # episodic lengths in batch

        # Number of timesteps run so far this batch
        t = 0

        while t < self.timesteps_per_batch:
            ep_rews = []

            obs, info = self.env.reset()
            
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                # Increment timesteps ran this batch so far
                t += 1

                # Collect observation
                batch_obs.append(obs)
                
                action, log_prob = self.get_action(obs)
                obs, rew, done, truncated, info = self.env.step(action)

                # Collect reward, action, and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)

        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        # ALG STEP 4
        batch_rtgs = self.compute_rtgs(batch_rews)

        print(f"Episode length: {ep_t + 1}, Total reward: {sum(ep_rews):.2f}")

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
    
    def get_action(self, obs):

        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()
    
    def compute_rtgs(self, batch_rews):
        """
        Compute rewards-to-go for the entire batch.
        Args:
            batch_rews (list of list): A list of episode reward lists.
        Returns:
            Tensor: A tensor of rewards-to-go for each timestep in the batch.
        """
        # Flatten the rewards-to-go into a single list
        batch_rtgs = []
    
        for ep_rews in batch_rews:  # Iterate through each episode
            discounted_reward = 0
            rtgs = []
            for rew in reversed(ep_rews):  # Compute rewards-to-go for one episode
                discounted_reward = rew + discounted_reward * self.gamma
                rtgs.insert(0, discounted_reward)  # Insert at the beginning
            batch_rtgs += rtgs  # Append this episode's rtgs to batch_rtgs
    
        return torch.tensor(batch_rtgs, dtype=torch.float)
    

    def _init_hyperparameters(self):
        # Default values for hyperparameters, will need to change later
        self.timesteps_per_batch = 1600         # timesteps per batch
        self.max_timesteps_per_episode = 400   # timesteps per episode
        self.gamma = 0.95                       # dicount factor
        self.n_updates_per_iteration = 5        # number of epochs per iteration
        self.clip = 0.2                         # clip threshold, recommended by the paper
        self.lr = 0.005                         # learning rate  


import gym

env = gym.make('Pendulum-v1')
model = PPO(env)
model.learn(10000)