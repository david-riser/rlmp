import numpy as np
import random
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



def expand_transitions(transitions, torchify=True, state_transformer=None):
    """ A list of transition objects is expanded into 
        several lists of the component tensors for
        calculation in batch mode.
    """
    states, actions, rewards, next_states, dones = [], [], [], [], []
    nth_states, discounted_rewards, ns = [], [], []
    for trans in transitions:
        states.append(trans.state if not state_transformer else state_transformer(trans.state))
        actions.append(trans.action)
        rewards.append(trans.reward)
        next_states.append(trans.next_state if not state_transformer else state_transformer(trans.next_state))
        dones.append(trans.done)
        nth_states.append(trans.nth_state if not state_transformer else state_transformer(trans.nth_state))
        discounted_rewards.append(trans.discounted_reward)
        ns.append(trans.n)
        
    if torchify:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(device)
        ns = torch.LongTensor(ns).to(device)

        if state_transformer and len(states[0].shape) > 1:
            states = torch.cat(states).to(device)
            next_states = torch.cat(next_states).to(device)
            nth_states = torch.cat(nth_states).to(device)
        else:
            states = torch.stack(states).to(device)
            next_states = torch.stack(next_states).to(device)
            nth_states = torch.stack(nth_states).to(device)
            
                
    return states, actions, rewards, next_states, discounted_rewards, nth_states, dones, ns



def play_evaluation_games(model, env_builder, state_transformer,
                          action_transformer, num_games=20, epsilon=0.05):
    """ Play some evaluation games and return the
        scores.  
    """
    
    env = env_builder()
    
    scores = []
    actions = []
    with torch.no_grad():
        for eval_game in range(num_games):
            score = 0
            state = env.reset()
            state = state_transformer(state)

            done = False
            while not done:
                if random.random() > epsilon:
                    action = model(state)
                    action = action_transformer(action)
                else:
                    action = env.action_space.sample()

                actions.append(action)    
                next_state, reward, done, _ = env.step(action)
                score += reward
                state = state_transformer(next_state)

            scores.append(score)
            env.close()


    return scores, actions
