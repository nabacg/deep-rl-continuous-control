# coding: utf-8
import numpy as np
import random
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
from model import ActorNetwork, CriticNetwork
from replaybuffers import PrioretizedReplayBuffer



BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters               
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic

UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DdpgAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, learning_rate=LR_ACTOR, update_every=UPDATE_EVERY, discount_factor=GAMMA):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        #saving hyperparams
        self.update_every = update_every
        self.discount_factor = discount_factor

        # Actor-Network
        self.actor_train = ActorNetwork(state_size, action_size, seed, 128, 256).to(device)
        self.actor_target = ActorNetwork(state_size, action_size, seed, 128, 256).to(device)
        self.actor_optimizer = optim.Adam(self.actor_train.parameters(), lr=LR_ACTOR)

        # Critic Network
        self.critic_train = CriticNetwork(state_size, action_size, seed, 400, 300)
        self.critic_target = CriticNetwork(state_size, action_size, seed, 400, 300)
        self.critic_optimizer = optim.Adam(self.critic_train.parameters(),lr=LR_CRITIC)

        self.noise = OUNoise(action_size, self.seed)
        # Replay memory
        self.memory = PrioretizedReplayBuffer( BUFFER_SIZE, BATCH_SIZE, seed, device)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.loss_track = []
        self.actor_loss = 100
        self.critic_loss = 100

    def load_model_weights(self, actor_weights, critic_weights):
       
        self.actor_target.load_state_dict(torch.load(actor_weights))
        self.actor_train.load_state_dict(torch.load(actor_weights))
        self.critic_target.load_state_dict(torch.load(critic_weights))
        self.critic_train.load_state_dict(torch.load(critic_weights))

    def eval_actor(self, state, qnetwork):
        """ 
        TODO refactor eval_critic AND eval_actor to be one Function with VARARGS!!
        Helper method to evaluate model on given state and return action state values

        Params
        ==== 
            state (Torch tensor) - current env state
            model (QNetwork) - one of the Q networks (qnetwork_local, qnetwork_target)
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        qnetwork.eval() # setting  model to inference 
        with torch.no_grad():
            action = qnetwork(state)
        qnetwork.train() # setting model back to training
        return action

    def eval_critic(self, state, action, qnetwork):
        """
        TODO refactor eval_critic AND eval_actor to be one Function with VARARGS!!
         Helper method to evaluate model on given state and return action state values

        Params
        ==== 
            state (Numpy array) - current env state
            model (QNetwork) - one of the Q networks (qnetwork_local, qnetwork_target)
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action = torch.from_numpy(action).float().unsqueeze(0).to(device)
        qnetwork.eval() # setting  model to inference 
        with torch.no_grad():
            action_value = qnetwork(state, action)
        qnetwork.train() # setting model back to training
        return action_value
    
    def step(self, state, action, reward, next_state, done):

        # calculate TD error in order to save the experience with correct priority into PrioritiedReplayBuffer
        action_pred = self.eval_actor(next_state, self.actor_target).numpy()[0]
        Q_target = self.eval_critic(next_state, action_pred, self.critic_target).numpy()[0]
        # print ("STEP, action={}, action_pred={}, state={}, Q_target={}".format(action, action_pred, state, Q_target))

        Q_vals = self.eval_critic(state, action[0],  self.critic_train).numpy()[0]
        td_error = reward + GAMMA*Q_target  - Q_vals if done != 0 else reward - Q_vals

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, td_error)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.discount_factor)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor_train.eval()
        with torch.no_grad():
            action_values = self.actor_train(state).cpu().data.numpy()
        self.actor_train.train()

        if add_noise:
            action_values += self.noise.sample()
        return np.clip(action_values, -1, 1)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        ## DDPG  implementation 
        #### Critic network training

        # Calculate Q_Targets
        # first use target Actor to predict best next actions for next states S'
        target_actions_pred = self.actor_target(next_states)
        # Then use target critic to asses Q value of this (S', pred_action) tuple
        Q_target_preds = self.critic_target(next_states, target_actions_pred)
        # calculate the Q_target using TD error formula   
        Q_target = rewards + gamma * Q_target_preds * (1 - dones)

        # find what Q value does Critic train network assign to this (state, action) - current state, actual action performed
        Q_pred = self.critic_train(states, actions)

        
        # Minimize critic loss
        # do Gradient Descent step on Critic train network by minimizing diff between (Q_pred, Q_target)
        self.critic_optimizer.zero_grad()
        critic_loss = F.mse_loss(Q_pred, Q_target)
        self.critic_loss = critic_loss.item
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_train.parameters(), 1)
        self.critic_optimizer.step()

        #### Actor networ training
        # find wich action does Actor train predict
        actions_pred = self.actor_train(states)
        # Loss is negative of Critic_train Q estimate of (S,  actions_pred)
        # i.e. we want to maximize (minimize the negative) of action state Value function (Q) prediction by critic_train 
        # for current state and next action predicted by actor_train
        actor_loss = -self.critic_train(states, actions_pred).mean()
        self.actor_loss = actor_loss.item
       
        # print("Actor_loss={}, critic_loss={}".format(actor_loss.item(), critic_loss.item()))
        # minimize Actor loss
        # do Gradient Descent step on Actor train network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        # ------------------- update target network ------------------- #
        self.soft_update(self.critic_target, self.critic_train, TAU)
        self.soft_update(self.actor_train, self.actor_target, TAU)

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

def plot_scores_losses(scores, actor_losses, critic_losses):
    import matplotlib.pyplot as plt

    # len(agent.losses)

    fig = plt.figure()
    ax = fig.add_subplot(212)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    ax2 = fig.add_subplot(122)
    plt.plot(np.arange(len(actor_losses)), actor_losses)
    plt.ylabel('Actor Losses')
    plt.xlabel('Episode #')
    plt.show()

    ax3 = fig.add_subplot(122)
    plt.plot(np.arange(len(critic_losses)), critic_losses)
    plt.ylabel('Critic Losses')
    plt.xlabel('Episode #')
    plt.show()

def train_agent(agent, env, output_weights, target_mean_score=13.0, n_episodes = 2000, eps_decay=0.995, eps_end=0.01, input_weights = None):
    eps = 1.0
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    if input_weights:
        print(input_weights)
        agent.load_model_weights(input_weights)
    
    scores = []
    actor_losses = []
    critic_losses = []
    scores_window = deque(maxlen=100)  # last 100 scores
    for i_episode in range(1,n_episodes+1):

        env_info = env.reset(train_mode=True)[brain_name]   # reset the environment
        state = env_info.vector_observations[0]             # get the current state
        score = 0                                           # initialize the score
        done = False                                          
        while not(done):                                   # exit loop if episode finished
            action = agent.act(state, eps)                 # select an action
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            actor_losses.append(agent.actor_loss)
            critic_losses.append(agent.critic_loss)
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step

        scores_window.append(score)
        scores.append(score)

        eps = max(eps_end, eps_decay*eps)
        mean_score = np.mean(scores_window)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score))
            plot_scores_losses(scores, actor_losses, critic_losses)
        if mean_score >= target_mean_score:
            print("Target mean score of {:.2f} achived at {:.2f} after {} episodes.".format(target_mean_score, mean_score, i_episode))
                #     print("Score: {}".format(score))
            print("Saving model weights to {}".format(output_weights))
            torch.save(agent.qnetwork_local.state_dict(), output_weights)
            break
    return scores

def test_agent(agent, env, input_weights, n_episodes):
    brain_name = env.brain_names[0]
    agent.load_model_weights(input_weights)

    scores = [] 
    for i_episode in range(1,n_episodes+1):

        env_info = env.reset(train_mode=True)[brain_name]   # reset the environment
        state = env_info.vector_observations[0]             # get the current state
        score = 0                                           # initialize the score
        done = False                                          
        while not(done):                                   # exit loop if episode finished
            action = agent.act(state)                      # select an action
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
        scores.append(score)
        print('\rEpisode {}\ Score: {:.2f}'.format(i_episode, score))
    return scores


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state