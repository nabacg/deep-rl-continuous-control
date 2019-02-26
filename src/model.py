import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class ActorNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, layer_1_size=64, layer_2_size=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.model = nn.Sequential(
            nn.Linear(state_size, layer_1_size),
            nn.ReLU(),
            nn.Linear(layer_1_size, layer_2_size),
            nn.ReLU(),
            nn.Linear(layer_2_size, layer_2_size),
            nn.ReLU(),
            nn.Linear(layer_2_size, action_size),
            nn.Tanh()
        )
        self.reset_parameters()

    def reset_parameters(self):
        [n.weight.data.uniform_(*hidden_init(n)) for n in self.model[:-2] if isinstance(n, nn.Linear)]  
        self.model[-2].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.model(state)


class CriticNetwork(nn.Module):
    """Critic (Q-state action value function) Model """

    def __init__(self, state_size, action_size, seed, layer_1_size=64, layer_2_size=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.input_layer = nn.Linear(state_size, layer_1_size)
        self.hidden_1    = nn.Linear(layer_1_size+action_size, layer_2_size)
        self.hidden_2    = nn.Linear(layer_2_size, 1)
        self.reset_parameters()

                # ToDo check if the action concat at 2 layer can be done with Sequential
        # self.model = nn.Sequential(
        #     nn.Linear(state_size, layer_1_size),
        #     nn.ReLU(),
        #     nn.Linear(layer_1_size+action_size, layer_2_size),
        #     nn.ReLU(),
        #     nn.Linear(layer_2_size, 1)
        # )

    
    def forward(self, state, action):
        x = F.relu(self.input_layer(state))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.hidden_1(x))
        return self.hidden_2(x)

    def reset_parameters(self):
        self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))
        self.hidden_1.weight.data.uniform_(*hidden_init(self.hidden_1))
        self.hidden_2.weight.data.uniform_(-3e-3, 3e-3)