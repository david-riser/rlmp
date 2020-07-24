import argparse
import glob
import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
import wandb

from gym import wrappers
from pyvirtualdisplay import Display
from common.replay import ReplayBuffer
from common.dqn import Variable, DQN, CnnDQN, USE_CUDA



def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', type=str, default='CartPole-v0')
    ap.add_argument('--max_frames', type=int, default=10000)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--eps_start', type=float, default=1.)
    ap.add_argument('--eps_final', type=float, default=0.1)
    ap.add_argument('--eps_decay', type=float, default=2500)
    return ap.parse_args()

def setup_wandb(args):
    config = dict(
        env = args.env,
        max_frames = args.max_frames
    )
    wandb.init(
        project='rlmp',
        notes='Basic DQN',
        tags=['DQN'],
        config=config
    )


if __name__ == "__main__":

    args = get_args()
    setup_wandb(args)
    video_path = 'tmp/video/{}'.format(wandb.run.id)
    
    env = gym.make(args.env)
    env = wrappers.Monitor(
        gym.make(args.env),
        video_path,
        video_callable=lambda x: x % 20 == 0
    )

    # Configure display
    virtual_display = Display(visible=0, size=(320,240))
    virtual_display.start()


    # Setup agent
    if len(env.observation_space.shape) == 1:
        model = DQN(
            num_inputs=env.observation_space.shape[0],
            num_actions=env.action_space.n
        )
    else:
        model = CnnDQN(
            input_shape=env.observation_space.shape,
            num_actions=env.action_space.n
        )

    if USE_CUDA:
        model = model.cuda()
        
    optimizer = optim.Adam(model.parameters())
    replay_buffer = ReplayBuffer(10000)
    num_frames = 0
    while num_frames < args.max_frames:

        state = env.reset()
        done = False
        ep_reward = 0 
        while not done:
            epsilon = args.eps_final + (args.eps_start - args.eps_final) * np.exp(-1. * num_frames / args.eps_decay)
            action = model.act(state,epsilon)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            wandb.log({'reward':reward, 'epsilon':epsilon})
            num_frames += 1
            ep_reward += reward
            state = next_state

            # Update network
            states, actions, rewards, next_states, dones = replay_buffer.sample(args.batch_size)

            # Make into torch variables to calculate
            # the loss with
            states = Variable(torch.FloatTensor(np.float32(states)))
            actions = Variable(torch.LongTensor(actions))
            rewards = Variable(torch.FloatTensor(rewards))
            next_states = Variable(torch.FloatTensor(np.float32(next_states)))
            dones = Variable(torch.FloatTensor(dones))

            # Predict value for these state/action pairs
            # and next state/action pairs. 
            q_values = model(states)
            next_q_values = model(next_states)

            # Calculate value for actions we took
            q_values = q_values.gather(1, actions.view(-1,1))
            next_q_values = q_values.max(1).values

            qhat = (rewards + args.gamma * next_q_values)
            loss = (q_values - qhat.detach()).pow(2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if num_frames % 100 == 0:
                print("Frame {0}, Ep. Reward {1:6.4f}, Eps. {2:6.4f}, Q-avg: {3:6.4f}".format(
                    num_frames, ep_reward, epsilon, q_values.mean().detach().cpu().numpy()
                ))
            
        env.close()
        wandb.log({'ep_reward':ep_reward})
        
    # Upload the video 
    for movie in glob.glob(video_path + '/*.mp4'):
        wandb.log({'Video':wandb.Video(movie)})
