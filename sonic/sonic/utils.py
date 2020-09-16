import gym
import numpy as np
import os
import retro
import torch
import torch.autograd as autograd 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from gym import wrappers


def extract_human_data(movie_path):
    """ Given a path to a replay file, load it and 
        extract the series of state-action pairs.
    """
    movie = retro.Movie(movie_path)
    movie.step()

    env = retro.make(game=movie.get_game(), state=retro.State.NONE, 
                     use_restricted_actions=retro.Actions.ALL)
    env.initial_state = movie.get_state()
    state = env.reset()
    state = np.swapaxes(state,0,2)
    state = np.swapaxes(state,1,2)
    states, actions, next_states, rewards, dones = [], [], [], [], []
    while movie.step():
        keys = []
        for i in range(len(env.buttons)):
            keys.append(movie.get_key(i, 0))
            
        next_state, reward, done, info = env.step(keys)
        
        # Switch the channels to be first for pytorch
        next_state = np.swapaxes(next_state, 0, 2)
        next_state = np.swapaxes(next_state, 1, 2)
        
        actions.append(np.int8(keys))
        states.append(state)
        next_states.append(next_state)
        rewards.append(reward)
        dones.append(done)
        
        state = next_state
        
    return states, actions, next_states, rewards, dones


def torchify_state(state):
    """ Convert a single state to a shape and data type that 
        can be injested by a pytoch model.
    """
    reshaped_state = np.swapaxes(state, 0, 2)
    reshaped_state = np.swapaxes(reshaped_state, 1, 2)
    reshaped_state = np.expand_dims(reshaped_state, 0)
    torchified_state = autograd.Variable(torch.FloatTensor(reshaped_state))
    return torchified_state.cuda()


def rounding_action_transformer(action):
    """ A transformer for the actions produced in a torch.Tensor
        format.  The action that comes out is compatible with the 
        gym.env object step() function.
    """
    action = action.detach().cpu().numpy()
    action = np.round(action)[0]
    action = np.array(action, dtype=np.int8)
    return action


def record_imitation_game(env, model, action_transformer, state_transformer,
                          rnd_steps=50, max_frames=1000):
    """ Record an imitation game from the provided environment which 
        should output states that are compatible with the model input
        and receive actions which are compatible with the model output.
    """
    
    state = env.reset()
    frames = []
    frames.append(env.render(mode='rgb_array'))
    for step in range(rnd_steps):
        state, _, _, _ = env.step(env.action_space.sample())
        frames.append(env.render(mode='rgb_array'))
        
    done = False
    while not done:
        
        # Act according to our policy most of the time.  It 
        # seems to get stuck so I am trying to add some random
        # elements to get Sonic unstuck.
        if step % 5 == 0:
            new_state, reward, done, info = env.step(env.action_space.sample())
            state = new_state
        
        else:
            state = state_transformer(state)
            action = model(state)
            action = action_transformer(action)
            new_state, reward, done, info = env.step(action)
            state = new_state
            
        frames.append(env.render(mode='rgb_array'))
        if len(frames) % 100 == 0:
            print("Collected {} frames".format(len(frames)))
        
        if len(frames) >= max_frames:
            return frames
        
    return frames


def play_evaluation_games(env, model, action_transformer, state_transformer, n_games=1, rnd_steps=50, max_frames=1000):
    """ Play evaluation games and return the scores.
    """
    
    scores = []
    
    for game in range(n_games):
        state = env.reset()
        for step in range(rnd_steps):
            state, _, _, _ = env.step(env.action_space.sample())

        step = 0
        score = 0
        done = False
        while not done:

            # Act according to our policy most of the time.  It 
            # seems to get stuck so I am trying to add some random
            # elements to get Sonic unstuck.
            if step % 5 == 0:
                new_state, reward, done, info = env.step(env.action_space.sample())
                state = new_state

            else:
                state = state_transformer(state)
                action = model(state)
                action = action_transformer(action)
                new_state, reward, done, info = env.step(action)
                state = new_state
                
            step += 1 
            score += reward
            
            if step > max_frames:
                done = True
        
        print("Finished game {0} with score {1:6.0f}.".format(
            game, score
        ))
        scores.append(score)

    return scores



def torchify_float(x):
    """ Convert x from numpy array to torch.FloatTensor. """
    return autograd.Variable(torch.FloatTensor(x))


def torchify_long(x):
    """ Convert x from numpy array to torch.LongTensor. """
    return autograd.Variable(torch.LongTensor(x))


def encode_actions(actions, encoding):
    """ Using the encoder dictionary, encode all of the actions
        provided. 
    """
    
    encoded = np.zeros(len(actions), dtype=np.int8)
    for i, action in enumerate(actions):
        
        encoded[i] = encoding[tuple(action)]
        
    return encoded


def decode_actions(actions, decoding):
    """ Using the decoder dictionary, decode all of the actions
        provided. 
    """
    
    decoded = []
    for i, action in enumerate(actions):
        
        decoded.append(decoding[action])
        
    return np.array(decoded, np.int8)


def build_nstep_transitions(states, actions, next_states, rewards, dones, n, gamma):
    """ Assuming that the inputs are sorted temporally from early to late, 
        build n-step transitions and return them for training DQfD.
    """
    
    _states, _actions, _nth_states, _discounted_rewards, _dones = [], [], [], [], []
    _next_states, _rewards = [], []
    
    # Iterate up to the point where there are fewer than
    # n states left to process.
    for i in range(len(states) - n):
        
        # Most of these could just be list copies.  The only thing we need
        # to build on the fly is the nth_state and the discounted reward.
        _states.append(states[i])
        _actions.append(actions[i])
        _next_states.append(next_states[i])
        _nth_states.append(next_states[i + n - 1])
        _rewards.append(rewards[i])
        
        discounted_reward = 0.
        for j in range(n - 1):
            
            if not dones[i + j]:
                discounted_reward += gamma**j * rewards[i + j]
                done = False
            else:
                done = True
                break
                
        _discounted_rewards.append(discounted_reward)
        _dones.append(done)
    
    return _states, _actions, _next_states, _nth_states, _rewards, _discounted_rewards, _dones


def decoding_action_transformer(action, decoding):
    """ Transform the action predicted by the torch network into 
        something that can be consumed by gym.env.step().  In this 
        case, we're transforming a probability vector over actions 
        into the full 12 action space.
    """
    action = action.detach().cpu().numpy()
    action = np.argmax(action)
    action = decoding[action]
    return action
