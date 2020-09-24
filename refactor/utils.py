import numpy as np
import torch

from collections import namedtuple


Transition = namedtuple('Transition', 'state action next_state nth_state reward discounted_reward done n')


def rolling(array, operation, window, pad=False):
    """ Apply an operation to each window of values on 
        the input array and return the result.
    """
    output_size = len(array) - window if not pad else len(array)
    output = np.zeros(output_size)

    for i in range(output_size):
        output[i] = operation(array[i : i + window])

    return output



def expand_transitions(transitions, torchify=True):
    """ A list of transition objects is expanded into 
        several lists of the component tensors for
        calculation in batch mode.
    """
    states, actions, rewards, next_states, dones = [], [], [], [], []
    nth_states, discounted_rewards, ns = [], [], []
    for trans in transitions:
        states.append(trans.state)
        actions.append(trans.action)
        rewards.append(trans.reward)
        next_states.append(trans.next_state)
        dones.append(trans.done)
        nth_states.append(trans.nth_state)
        discounted_rewards.append(trans.discounted_reward)
        ns.append(trans.n)
        
    if torchify:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(device)
        nth_states = torch.FloatTensor(nth_states).to(device)
        ns = torch.LongTensor(ns).to(device)
        
    return states, actions, rewards, next_states, discounted_rewards, nth_states, dones, ns
