# Report


## Plot of Rewards

Plot from [Training](Training.ipynb) showing scores (total reword per episode) on agent trained from scratch until average score over 20 agents for 100 episode window reached 30.0 . This was achived after 130 episodes, more details in Training notebook.

### Scores 
![Plot of agent scores by episode](https://github.com/nabacg/deep-rl-continuous-control/blob/master/images/score_solution.png?raw=true)

### Mean Scores over 100 episode window
![Plot of agent mean scores in 100 episode window ](https://github.com/nabacg/deep-rl-continuous-control/blob/master/images/mean_score_solution.png?raw=true))

### Actor Loss function

![Plot of Actor network loss function](https://github.com/nabacg/deep-rl-continuous-control/blob/master/images/actor_loss.png?raw=true))

### Critic Loss function
![Plot of Critic network loss function](https://github.com/nabacg/deep-rl-continuous-control/blob/master/images/critic_loss.png?raw=true))


## Learning Algorithm 

This repository, specifically files below:
 - src/
     - ddpg_agent.py
     - model.py 
     - replaybuffers.py

Contain implementation of Deep Deterministic Policy Gradient or DDPG to solve the 20 agent version of this environment. 


### Neural Network architecture 

File src/model.py contains PyTorch implementation of two small Neural Networks, for approximating the action value function Q (the Critic) and the Policy function (Actor). 


#### Critic network 
Consists of 3 fully connected layers of following size:
 - Input layer (state_size, 256), ReLU activation 
 - Hidden layer (256, 128), ReLU activation
 - Output layer (128, action_size), Tanh activation

 #### Actor network
Consists of 3 fully connected layers of following size:
 - Input layer (state_size, 256), ReLU activation 
 - Hidden layer (256+action_size, 128), ReLU activation
 - Output layer (128, 1), Linear activation

For Reacher environment 

state_size = 33
action_size = 4

The size and number of hidden layers was chosen experimentally. It's quite interesting that it was possibly to solve this environment reliably with Neural Network much smaller than one used by original publication authors. They used 2 layers of 400 and 300 neurons, compared with 256 and 128 used in this work. 

### NN Training
Network above is trained by minimizing [Mean Squared Error Loss function](https://pytorch.org/docs/stable/nn.html?highlight=mseloss#torch.nn.MSELoss) with help of [ADAM optimizer](https://pytorch.org/docs/stable/optim.html?highlight=mseloss#torch.optim.Adam).  

Learning rate of  LR = 1e-4 for both Actor and Critic network, was chosen after some experimentation with values between (1e-2, 1e-5). All chosen hyperparameter values are listed below. [Training notebook](Training.ipynb) contains an executable demonstration of training process and results.

### Model weights
Pretrained weights used to generate results presented here are part of this repository
and can be found in folder model_weights:
- solution_actor_weights.pth
- solution_critic_weights.pth


## Hyperparameters used

```python

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters               
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
UPDATE_EVERY = 1        # how often to update the network
WEIGHT_DECAY = 0        # Critic network weight decay 


```

Also Ornstein-Uhlenbeck process was used to add randomness to agent actions to enable exploration. Process parameters were set to 
```python
theta = 0.15 
sigma = 0.2 

```


#  Ideas for Future Work
- test this solution on more environments with continuous control
- implement [D4PG](https://openreview.net/pdf?id=SyZipzbCb) to leverage distributed computing 
- implementing algorithms below to compare their performance to DDPG
    - Trust Region Policy Optimization (TRPO)  
    - Truncated Natural Policy Gradient (TNPG) 
    - Proximal Policy Optimization (PPO)