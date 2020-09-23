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
