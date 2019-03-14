import random
import copy
import numpy as np
from collections import deque

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size        
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
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
    
def plot_scores_losses(scores, mean_scores, actor_losses, critic_losses):
    import matplotlib.pyplot as plt

    # len(agent.losses)

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    ax11 = fig.add_subplot(2, 1, 2)
    plt.plot(np.arange(len(mean_scores)), mean_scores)
    plt.ylabel('Mean Score')
    plt.xlabel('Episode #')
    plt.show()

    ax2 = fig.add_subplot(2, 2, 3)
    plt.plot(np.arange(len(actor_losses)), actor_losses)
    plt.ylabel('Actor Losses')
    plt.xlabel('Steps #')
    plt.show()

    ax3 = fig.add_subplot(2, 2, 4)
    plt.plot(np.arange(len(critic_losses)), critic_losses)
    plt.ylabel('Critic Losses')
    plt.xlabel('Steps #')
    plt.show()


def train_agent(num_agents, agent, env, file_prefix, print_metrics_every=10,
        target_mean_score=13.0, n_episodes = 1000, eps_decay=0.999, 
        eps_end=0.01, input_weights = None, score_aggregate = np.mean):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    if input_weights:
        print(input_weights)
        agent.load_model_weights(input_weights)
    
    scores = []
    mean_scores = []
    actor_losses = []
    critic_losses = []
    eps = 0.9
    max_score = 0
    step_counter = 0
    # ToDo save weights for the model with best score so far

    scores_window = deque(maxlen=100)  # last 100 scores
    for i_episode in range(1,n_episodes+1):
        # print("Episonde {} start".format(i_episode), end="")

        env_info = env.reset(train_mode=True)[brain_name]   # reset the environment
        states = env_info.vector_observations               # get the current state
        score = np.zeros(num_agents)                                           # initialize the score
        done = False                                          
        while not(done):                                 # exit loop if episode finished
            if i_episode < 0.1*n_episodes:
                actions = np.random.standard_normal((num_agents,  agent.action_size))
            else:
                actions =  agent.act(states, add_noise=True, noise_decay=eps)                 # select an action
            # print(actions, actions.shape)
            env_info = env.step(actions)[brain_name]      # send the action to the environment
            next_states = env_info.vector_observations   # get the next state
            rewards = env_info.rewards                   # get the reward
            dones = env_info.local_done                  # see if episode has finished
            agent.step(states, actions, rewards, next_states, dones) 
                    
            actor_losses.append(agent.actor_loss)
            critic_losses.append(agent.critic_loss)
            score += rewards                              # update the score
            states = next_states                             # roll over the state to next time step
            done = np.any(dones)
            step_counter += 1
        
        score = score_aggregate(score)
        scores_window.append(score)
        scores.append(score)

        if step_counter % 10 ==0:
            eps = max(eps_end, eps_decay*eps)
        mean_score = np.mean(scores_window)
        mean_scores.append(mean_score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon:: {:.4f}'.format(i_episode, mean_score, eps), end="")
        agent.save_checkpoint()
        if i_episode % print_metrics_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score))
            plot_scores_losses(scores, mean_scores, actor_losses, critic_losses)
        if mean_score >= target_mean_score:
            print("Target mean score of {:.2f} achived at {:.2f} after {} episodes.".format(target_mean_score, mean_score, i_episode))
                #     print("Score: {}".format(score))
            agent.save_checkpoint(file_name=file_prefix)
            
            break
    return scores

def test_agent(agent, env, weight_file_prefix, n_episodes):
    brain_name = env.brain_names[0]
    agent.load_checkpoint(weight_file_prefix)

    scores = [] 
    for i_episode in range(1,n_episodes+1):

        env_info = env.reset(train_mode=False)[brain_name]   # reset the environment
        states = env_info.vector_observations                # get the current state
        score = 0                                           # initialize the score
        done = False                                                                                   
        while not(done):                                 # exit loop if episode finished
            actions =  agent.act(states)                 # select an action
            # print(actions, actions.shape)
            env_info = env.step(actions)[brain_name]      # send the action to the environment
            next_states = env_info.vector_observations   # get the next state
            rewards = env_info.rewards                   # get the reward
            dones = env_info.local_done                  # see if episode has finished
            agent.step(states, actions, rewards, next_states, dones) 

            score += np.mean(rewards)                           # update the score
            states = next_states                             # roll over the state to next time step
            done = np.any(dones)                          # roll over the state to next time step
        scores.append(score)
        print('\rEpisode {}\ Score: {:.2f}'.format(i_episode, score))
    return scores