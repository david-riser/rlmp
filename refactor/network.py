import torch
import torch.nn as nn

class Network(nn.Module):
    """ A deep Q-network model implemented in PyTorch.  This 
        implementation makes use of the advantage estimate.
    """
    def __init__(self, obs_shape, num_actions, hidden_dim):
        super(Network, self).__init__()
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        # Network definitions in PyTorch.
        self.features = nn.Sequential(
            nn.Linear(self.obs_shape, self.hidden_dim), 
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        self.advantage = nn.Sequential(
            nn.Linear(self.hidden_dim, self.num_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(self.hidden_dim, 1)
        )


    def forward(self, state):
        """ Action-value function prediction from input observation 
            to the network.  Makes use of advantage estimate.
        """
        features = self.features(state)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean()


class ConvNetwork(nn.Module):
    """ A deep convolutionsal Q-network model implemented in PyTorch.  
        This implementation makes use of the advantage estimate.
    """
    def __init__(self, obs_shape, num_actions, hidden_dim):
        super(ConvNetwork, self).__init__()
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        self.features = nn.Sequential(
            nn.Conv2d(self.obs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.value = nn.Sequential(
            nn.Linear(self.feature_size(), self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(self.feature_size(), self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_actions)
        )


    def forward(self, state):
        """ Action-value function prediction from input observation 
            to the network.  Makes use of advantage estimate.
        """
        features = self.features(state)
        #features = features.view(features.size(), -1)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean()

    
    def feature_size(self):
        """ I took this from somewhere. """
        shape = self.features(torch.autograd.Variable(torch.zeros(1, *self.obs_shape))).view(1, -1).size(1)
        return shape
