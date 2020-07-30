""" 
Network architecture functions from:
https://github.com/higgsfield/RL-Adventure/blob/master/1.dqn.ipynb

"""

import numpy as np
import random 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class DDQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DDQN, self).__init__()

        self.num_actions = num_actions
        self.num_inputs = num_inputs
        
        self.layers = nn.Sequential(
            nn.Linear(self.num_inputs, 128),
            nn.ReLU()
        )

        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )

        
    def forward(self, x):
        z = self.layers(x)
        value = self.value(z)
        advantage = self.advantage(z)
        return value + advantage - advantage.mean()
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0].cpu().numpy()
        else:
            action = random.randrange(self.num_actions)
        return action


class CnnDDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDDQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions

        print("CnnDDQN got input shape: ", input_shape)
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU()
        )
        
        self.value = nn.Sequential(
            nn.Linear(self.feature_size(), 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(self.feature_size(), 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + advantage - advantage.mean()
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0))
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0].cpu().numpy()
        else:
            action = random.randrange(self.num_actions)
        return action
    

