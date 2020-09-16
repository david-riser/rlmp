""" 

Training code for DQfD adapted from the notebook.  This file
enables longer runtimes without interruption.

"""

import argparse
import gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import retro
import sys
import torch
import torch.autograd as autograd 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import wandb

from gym import wrappers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add the utilities in the sonic project folder
# to the path so that we can call upon them
project_dir = os.path.abspath('.')
sys.path.append(os.path.join(project_dir, 'sonic'))
from sonic import utils

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batches_per_epoch', type=int, default=10)
    parser.add_argument('--update_freq', type=int, default=25)
    parser.add_argument('--coef1', type=float, default=1.)
    parser.add_argument('--coef2', type=float, default=1.)
    parser.add_argument('--coef3', type=float, default=1e-5)
    parser.add_argument('--margin', type=float, default=0.8)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n_step', type=int, default=10)
    parser.add_argument('--level', type=str, default='GreenHillZone.Act1')
    return parser.parse_args()


def setup_wandb(args):
    config = dict(
        level = args.level,
        gamma = args.gamma,
        batch_size = args.batch_size,
        lr = args.lr,
        update_freq = args.update_freq,
        epochs = args.epochs,
        batches_per_epoch = args.batches_per_epoch,
        coef1 = args.coef1,
        coef2 = args.coef2,
        coef3 = args.coef3,
        margin = args.margin,
        n_step = args.n_step
    )
    
    wandb.init(
        project='rlmp',
        notes='DQfD',
        tags=['DQfD'],
        config=config
    )


def torch_td_loss(online_model, target_model, states, actions, 
                  next_states, rewards, dones, gamma=0.99):
    """ Compute the TD-error using pytorch for a set of transitions. """
    
    # Predict the value of the current and next state using the 
    # online and target networks respectively. 
    q_values = online_model(states)
    next_q_values = target_model(next_states)
    
    # Mask those states for which the next state is terminal.
    q_values = q_values.gather(1, actions.view(-1,1)).squeeze(1)
    next_q_values = next_q_values.max(1).values

    qhat = (rewards + gamma * next_q_values * (1 - dones))
    loss = (q_values - qhat.detach()).pow(2)
    loss = loss.mean()
    return loss


def torch_margin_loss(q_values, expert_actions, margin):
    """ Margin loss in torch. """
    
    # Calculate the margins and set them to zero where
    # expert has chosen action. 
    margins = torch.ones_like(q_values) * margin
    
    for i, action in enumerate(expert_actions):
        margins[i, action] = 0.
    
    loss_term1 = torch.max(q_values + margins, axis=1)[0]
    loss_term2 = torch.take(q_values, expert_actions)
    
    return loss_term1 - loss_term2


def torch_nstep_td_loss(online_model, target_model, actions, states, nth_states, 
                        discounted_rewards, dones, n, gamma):
    """ Calculate the n-step TD-loss using pytorch.  We assume that the discounted sum over
        rewards up to the n-th state has already been performed.  
    """
    
    # Predict the value of the current and next state using the 
    # online and target networks respectively. 
    q_values = online_model(states)
    nth_q_values = target_model(nth_states)
    
    # Mask those states for which the next state is terminal.
    q_values = q_values.gather(1, actions.view(-1,1)).squeeze(1)
    nth_q_values = nth_q_values.max(1).values

    qhat = (discounted_rewards + gamma**n * nth_q_values * (1 - dones))
    loss = (q_values - qhat.detach()).pow(2)
    loss = loss.mean()
    return loss


def torch_l2_penalty(parameters):
    
    loss = 0.
    for p in parameters:
        loss += torch.sum(p ** 2)
        
    return loss


def torch_dqfd_loss(online_model, target_model, states, next_states, nth_states, 
                    rewards, discounted_rewards, dones, gamma, n, coef1, coef2, coef3,
                    margin, expert_actions
                   ):
    """ Construct the full DQfD loss from the four component loss functions.
    """
    
    # Predict which action to take from the online network.
    q_values = online_model(states)
    actions = torch.argmax(q_values, axis=1)
    
    td_loss = torch_td_loss(
        online_model=online_model, 
        target_model=target_model, 
        actions=actions, states=states, 
        next_states=next_states, rewards=rewards, 
        dones=dones, gamma=gamma)
    
    ntd_loss = torch_nstep_td_loss(
        online_model=online_model, target_model=target_model, 
        actions=actions, states=states, 
        nth_states=nth_states, 
        discounted_rewards=discounted_rewards, 
        dones=dones, gamma=gamma, n=n)
    
    q_values = online_model(states)
    margin_loss = torch_margin_loss(q_values, expert_actions, margin)
    margin_loss = torch.mean(margin_loss)
    l2_loss = torch_l2_penalty(online_model.parameters())
    
    loss = td_loss + coef1 * ntd_loss + coef2 * margin_loss + coef3 * l2_loss
    
    return loss, td_loss, ntd_loss * coef1, margin_loss * coef2, l2_loss * coef3


class SonicNet(nn.Module):
    """ A simple deep q-network architecture to be used for predicing 
        action-values from states.
    """
    
    def __init__(self, input_shape, output_shape):
        super(SonicNet, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        channels, height, width = self.input_shape
        self.features = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=3),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=3),
            nn.ReLU()
        )
        
        self.value = nn.Sequential(
            nn.Linear(self._feature_size(), 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(self._feature_size(), 128),
            nn.ReLU(),
            nn.Linear(128, self.output_shape)
        )
        
    def _feature_size(self):
        """ I took this from somewhere. """
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + advantage - advantage.mean()

    

if __name__ == "__main__":

    config = {}
    config['data_path'] = os.path.abspath("./data/human")
    args = setup_args()
    setup_wandb(args)

    
    # Get a batch of states to extract the action distribution from for encoding.
    states, actions, next_states, rewards, dones = utils.extract_human_data(
        "data/human/SonicTheHedgehog-Genesis-{}-0000.bk2".format(args.level))
    unique_actions, counts = np.unique(actions, axis=0, return_counts=True)
    n_actions = len(unique_actions)

    # Setup an encoding and decoding for actions.
    encoding = {}
    for i, action in enumerate(unique_actions):
        encoding[tuple(action)] = i

    decoding = { value:key for key, value in encoding.items() }


    # Initialize our models. 
    online_model, target_model = SonicNet(states[0].shape, n_actions), SonicNet(states[0].shape, n_actions)

    states, actions, next_states, rewards, dones = utils.extract_human_data(
        "data/human/SonicTheHedgehog-Genesis-{}-0000.bk2".format(args.level))
    _states, _actions, _next_states, _nth_states, _rewards, _discounted_rewards, _dones = utils.build_nstep_transitions(
        states, actions, next_states, rewards, dones, n=args.n_step, gamma=args.gamma)
    _actions = utils.encode_actions(_actions, encoding)

    # Cast to a reasonable type
    _states = np.array(_states)
    _actions = np.array(_actions)
    _next_states = np.array(_next_states)
    _nth_states = np.array(_nth_states)
    _rewards = np.array(_rewards)
    _discounted_rewards = np.array(_discounted_rewards)
    _dones = np.array(_dones)

    online_model = online_model.to(device='cuda:0')
    target_model = target_model.to(device='cuda:0')
    target_model.load_state_dict(online_model.state_dict())
    optimizer = optim.Adam(online_model.parameters(), lr=args.lr)

    n_epochs = args.epochs
    n_batches_per_epoch = args.batches_per_epoch
    batch_size = args.batch_size
    index_pool = np.arange(len(_states))

    history = {}
    loggables = ["loss", "td-loss", "ntd-loss", "margin-loss", "l2-loss"]
    for loggable in loggables:
        history[loggable + "-mu"] = np.zeros(n_epochs)
        history[loggable + "-std"] = np.zeros(n_epochs)
    
    for epoch in range(n_epochs):
    
        if epoch % args.update_freq == 0:
            target_model.load_state_dict(online_model.state_dict())

    
        epoch_loss = []
        epoch_td_loss, epoch_ntd_loss, epoch_margin_loss, epoch_l2_loss = [], [], [], []
        for batch in range(n_batches_per_epoch):
        
            # Generate a batch of transitions from the data pool.
            batch_indices = np.random.choice(index_pool, batch_size, replace=False)
            b_states = utils.torchify_float(_states[batch_indices]).to(device='cuda:0')
            b_actions = utils.torchify_long(_actions[batch_indices]).to(device='cuda:0')
            b_next_states = utils.torchify_float(_next_states[batch_indices]).to(device='cuda:0')
            b_nth_states = utils.torchify_float(_nth_states[batch_indices]).to(device='cuda:0')
            b_rewards = utils.torchify_float(_rewards[batch_indices]).to(device='cuda:0')
            b_discounted_rewards = utils.torchify_float(_discounted_rewards[batch_indices]).to(device='cuda:0')
            b_dones = utils.torchify_float(_dones[batch_indices]).to(device='cuda:0')
        
            optimizer.zero_grad()
            loss, td_loss, ntd_loss, margin_loss, l2_loss = torch_dqfd_loss(
                online_model=online_model,
                target_model=target_model,
                states=b_states,
                next_states=b_next_states, nth_states=b_nth_states,
                rewards=b_rewards, discounted_rewards=b_discounted_rewards,
                dones=b_dones, gamma=args.gamma, n=args.n_step,
                coef1=args.coef1, coef2=args.coef2, coef3=args.coef3,
                margin=args.margin, expert_actions=b_actions
            )

            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.detach().cpu().numpy())
            epoch_td_loss.append(td_loss.detach().cpu().numpy())
            epoch_ntd_loss.append(ntd_loss.detach().cpu().numpy())
            epoch_margin_loss.append(margin_loss.detach().cpu().numpy())
            epoch_l2_loss.append(l2_loss.detach().cpu().numpy())
        
        print("Epoch {0}, Loss {1:6.4f}, TD-Loss {2:6.4f}, NTD-Loss {3:6.4f}, Margin Loss {4:6.4f}, L2 Loss {5:6.4f}".format(
            epoch, np.mean(epoch_loss), np.mean(epoch_td_loss), 
            np.mean(epoch_ntd_loss), np.mean(epoch_margin_loss), np.mean(epoch_l2_loss))
        )

        wandb.log({
            'loss':np.mean(epoch_loss), 'td-loss':np.mean(epoch_td_loss),
            'ntd-loss':np.mean(epoch_ntd_loss), 'l2-loss':np.mean(epoch_l2_loss),
            'margin-loss':np.mean(epoch_margin_loss)
        })
        
        history["loss-mu"][epoch] = np.mean(epoch_loss)
        history["loss-std"][epoch] = np.std(epoch_loss)
        history["td-loss-mu"][epoch] = np.mean(epoch_td_loss)
        history["td-loss-std"][epoch] = np.std(epoch_td_loss)
        history["ntd-loss-mu"][epoch] = np.mean(epoch_ntd_loss)
        history["ntd-loss-std"][epoch] = np.std(epoch_ntd_loss)
        history["margin-loss-mu"][epoch] = np.mean(epoch_margin_loss)
        history["margin-loss-std"][epoch] = np.std(epoch_margin_loss)
        history["l2-loss-mu"][epoch] = np.mean(epoch_l2_loss)
        history["l2-loss-std"][epoch] = np.std(epoch_l2_loss)

    # Save finished model.
    torch.save(online_model,'./dqfd_{}.pkl'.format(args.level))
    wandb.save('./dqfd_{}.pkl'.format(args.level))
    
    env = retro.make('SonicTheHedgehog-Genesis', state=args.level)
    scores = utils.play_evaluation_games(env, online_model, 
                                         state_transformer=utils.torchify_state, 
                                         action_transformer=lambda x: utils.decoding_action_transformer(x, decoding),
                                         n_games=20, rnd_steps=50, max_frames=1000)
    wandb.log({
        'eval_mean':np.mean(scores), 'eval_std':np.std(scores),
        'eval_min':np.min(scores), 'eval_max':np.max(scores)
    })

    frames = utils.record_imitation_game(env=env, model=online_model, 
                               action_transformer=lambda x: utils.decoding_action_transformer(x, decoding),
                               state_transformer=utils.torchify_state,
                               rnd_steps=50, max_frames=5000)
    imageio.mimwrite(
        "torch_dqfd_agent_{}.mp4".format(args.level), 
        frames, fps=30)
    wandb.log(
        {'video':wandb.Video("torch_dqfd_agent_{}.mp4".format(args.level))}
    )
