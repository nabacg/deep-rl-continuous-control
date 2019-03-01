# coding: utf-8
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from model import CriticQNetwork, ActorQNetwork
from replaybuffers import ReplayBuffer
from utils import OUNoise

class DdpgAgent:
    
    def __init__(self, state_size, action_size, num_agents, 
                 seed, 
                 batch_size,
                 buffer_size,
                 actor_lr, 
                 critic_lr, 
                 weight_decay,
                 tau,
                 update_every, gamma, device, 
                 checkpoint_dir="."):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): how many agents are running in each step
            seed (int): random seed
            batch_size (int): how many experience tuples to process at once
            buffer_size (int): size of experience buffer
            actor_lr (float): actor learning rate alpha
            critic_lr (float): critic learning rate alpha
            weight_decay (float): rate of nn weight decay for critic network
            tau (float): soft update rate for synchronizing target and train network weights 
            update_every (int): how many env steps to train agent
            gamma (float): reward discount factor
            device (string): device to run PyTorch computation on (CPU, GPU)
            checkpoint_dir (string) : where to save checkpoints (trained weights)
        """
        self.num_agents = num_agents
        self.seed = torch.manual_seed(seed)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.tau = tau
        
        
        
        self.update_every = update_every
        self.gamma = gamma 
        self.device = device
       
        
        # NN models for 
#         network size 
        hidden_1_size = 256
        hidden_2_size = 128
        
        # Critic 
        self.critic_train     = CriticQNetwork(state_size, action_size, seed, hidden_1_size, hidden_2_size).to(device)
        self.critic_target    = CriticQNetwork(state_size, action_size, seed, hidden_1_size, hidden_2_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic_train.parameters(), lr=critic_lr, weight_decay=weight_decay)
        # Actor
        self.actor_train      = ActorQNetwork(state_size, action_size, seed, hidden_1_size, hidden_2_size).to(device)
        self.actor_target     = ActorQNetwork(state_size, action_size, seed, hidden_1_size, hidden_2_size).to(device)
        self.actor_optimizer  = optim.Adam(self.actor_train.parameters(), lr=actor_lr)
        
        # init Noise process
        self.noise = OUNoise((num_agents, action_size), seed)
        
        # init Replay Buffer
        self.memory = ReplayBuffer(action_size= action_size, 
                                   buffer_size=buffer_size,
                                   batch_size=batch_size, 
                                   seed=seed,
                                  device=device)
        
        self.step_counter = 0
        self.actor_loss = 0
        self.critic_loss = 0
        
        #checkpointing
        self.checkpoint_dir   = checkpoint_dir
        
        self.actor_weights = self.checkpoint_dir + "/" + "actor.pth"
        self.critic_weights = self.checkpoint_dir + "/" + "critic.pth"
        
        
    def load_checkpoint(self, file_prefix=None):
        actor_weights =  file_prefix + "_actor.pth"  if file_prefix else self.actor_weights
        critic_weights = file_prefix + "_critic.pth" if file_prefix else self.critic_weights

        if os.path.isfile(actor_weights) and os.path.isfile(critic_weights):
            self.actor_target.load_state_dict(torch.load(actor_weights))
            self.actor_train.load_state_dict(torch.load(actor_weights))
            self.critic_target.load_state_dict(torch.load(critic_weights))
            self.critic_train.load_state_dict(torch.load(critic_weights))
            
    def save_checkpoint(self, file_name=None):
        actor_weights =  file_name + "_actor.pth"  if file_name else self.actor_weights
        critic_weights = file_name + "_critic.pth" if file_name else self.critic_weights
        torch.save(self.actor_train.state_dict(), actor_weights)      
        torch.save(self.critic_train.state_dict(), critic_weights)  

    def act(self, states, add_noise = True):
        states = torch.from_numpy(states).float().to(self.device)
        self.actor_train.eval()
        with torch.no_grad():
            actions = self.actor_train(states).cpu().data.numpy()
        self.actor_train.train()
        
        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)
    
    def step(self, states, actions, rewards, next_states, dones):
        [self.memory.add(s, a, r, s_next, d) for (s, a, r, s_next, d) 
                in zip(states, actions, rewards, next_states, dones)]
        
        self.step_counter += 1
        if len(self.memory) >= self.batch_size:
            self.learn(self.memory.sample())
    
    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
       
        # Critic training
        target_actions_pred = self.actor_target(next_states)
        target_pred = self.critic_target(next_states, target_actions_pred)
        Q_target = rewards + (self.gamma * target_pred * (1 - dones))
        
        Q_pred = self.critic_train(states, actions)
        
        self.critic_optimizer.zero_grad()
        critic_loss = F.mse_loss(Q_pred, Q_target)
        self.critic_loss = critic_loss.data
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor training 
        actions_pred = self.actor_train(states)
        actor_loss = -self.critic_train(states, actions_pred).mean()
        self.actor_loss = actor_loss.data
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.critic_train, self.critic_target, self.tau)
        self.soft_update(self.actor_train, self.actor_target, self.tau)
        
        
        

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
        
