"""  State and action transformers for the networks. 
"""

import numpy as np
import torch


def cnn_state_transformer(state, device):
    """ Transformer the numpy state into a torch 
        tensor state.
    """
    state = np.swapaxes(state, 2, 0)
    state = torch.FloatTensor(state).to(device)
    if len(state.size()) == 3:
        state = state.unsqueeze(0)
    return state


def flat_state_transformer(state, device):
    """ A basic transformer used with fully 
        connected networks. 
    """
    return torch.FloatTensor(state).to(device)


def action_transformer(action):
    action = action.detach().cpu().numpy()
    action = np.argmax(action)
    return action
