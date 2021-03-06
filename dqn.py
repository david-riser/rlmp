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
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', type=str, default='CartPole-v0')
    ap.add_argument('--max_frames', type=int, default=200000)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--eps_start', type=float, default=1.00)
    ap.add_argument('--eps_final', type=float, default=0.01)
    ap.add_argument('--eps_decay', type=float, default=2500)
    ap.add_argument('--lr', type=float, default=0.00001)
    ap.add_argument('--replay_size', type=int, default=10000)
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


def setup_env(args, train=True):
    if args.env == "CartPole-v0":
        env = gym.make(args.env)
    else:
        env = make_atari(args.env)
        if train:
            env = wrap_deepmind(env, episode_life=True, clip_rewards=True,
                                frame_stack=True, scale=True)    
        else:
            env = wrap_deepmind(env, episode_life=False, clip_rewards=False,
                                frame_stack=True, scale=True)    

    return env


if __name__ == "__main__":

    args = get_args()
    setup_wandb(args)
    video_path = 'tmp/video/{}'.format(wandb.run.id)
    
    env = setup_env(args, True)
    
    # Configure display
    virtual_display = Display(visible=0, size=(320,240))
    virtual_display.start()

    # Print shapes
    print("Obs shape:", env.observation_space.shape)
    print("Actions: ", env.action_space.n)

    # Setup agent
    num_eval_frames = 30    
    if len(env.observation_space.shape) == 1:
        model = DQN(
            num_inputs=env.observation_space.shape[0],
            num_actions=env.action_space.n
        )
        eval_frames = np.zeros((num_eval_frames,env.observation_space.shape[0]))
    else:
        # Pytorch is channels first
        height, width, channels = env.observation_space.shape
        model = CnnDQN(
            input_shape=(channels, height, width),
            num_actions=env.action_space.n
        )
        eval_frames = np.zeros((num_eval_frames,channels,height,width))

    if USE_CUDA:
        model = model.cuda()


    # Gather evaluation frames
    i = 0
    while i < num_eval_frames:
        state = np.array(env.reset())
        if len(state.shape) > 1:
            state = np.swapaxes(state,2,0)

        done = False
        while not done and i < num_eval_frames:
            next_state, reward, done, _ = env.step(env.action_space.sample())
            next_state = np.array(next_state)
            if len(next_state.shape) > 1:
                next_state = np.swapaxes(next_state,2,0)
            eval_frames[i] = state
            state = next_state
            i += 1

    eval_frames = Variable(torch.FloatTensor(np.float32(eval_frames)))
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    replay_buffer = ReplayBuffer(args.replay_size)
    num_frames = 0
    while num_frames < args.max_frames:

        state = env.reset()
        state = np.array(state)
        if len(state.shape) > 1:
            state = np.swapaxes(state,2,0)

        done = False
        ep_reward = 0 
        while not done:
            epsilon = args.eps_final + (args.eps_start - args.eps_final) * np.exp(-1. * num_frames / args.eps_decay)
            action = model.act(state,epsilon)
            next_state, reward, done, _ = env.step(action)
            next_state = np.array(next_state)
            if len(next_state.shape) > 1:
                next_state = np.swapaxes(next_state,2,0)

            replay_buffer.add(state, action, reward, next_state, done)
            wandb.log({'return':ep_reward, 'epsilon':epsilon})
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
            q_values = q_values.gather(1, actions.view(-1,1)).squeeze(1)
            next_q_values = next_q_values.max(1).values

            qhat = (rewards + args.gamma * next_q_values * (1 - dones))
            loss = (q_values - qhat.detach()).pow(2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            q_avg = model(eval_frames)
            q_avg = q_avg.cpu().detach().numpy().mean()
            wandb.log({'q_avg':q_avg})
            
            if num_frames % 100 == 0:
                print("Frame {0}, Ep. Reward {1:6.4f}, Eps. {2:6.4f}, Q-avg: {3:6.4f}".format(
                    num_frames, ep_reward, epsilon, q_avg
                ))
            
        env.close()
        wandb.log({'score':ep_reward})
        

    # Wrap to record the evaulations, a testing
    # env does not clip rewards nor does it
    # have episodic resets
    env = setup_env(args=args, train=False)
    env = wrappers.Monitor(
        env,
        video_path,
        video_callable=lambda x: True
    )

    eval_games = 10
    epsilon = 0.05
    scores = []
    with torch.no_grad():
        for eval_game in range(eval_games):
            score = 0 
            state = env.reset()
            state = np.array(state)
            if len(state.shape) > 1:
                state = np.swapaxes(state, 2, 0)
            done = False
            while not done:
                action = model.act(state,epsilon)
                next_state, reward, done, _ = env.step(action)
                next_state = np.array(next_state)
                if len(next_state.shape) > 1:
                    next_state = np.swapaxes(next_state, 2, 0)

                score += reward
                state = next_state
                
            scores.append(score)
            env.close()
            
    # Upload the video 
    for movie in glob.glob(video_path + '/*.mp4'):
        wandb.log({'Video':wandb.Video(movie)})

    # Ending metrics for the test games 
    wandb.log(
        {
            'eval_max':np.max(scores),
            'eval_mean':np.mean(scores),
            'eval_std':np.std(scores),
            'eval_median':np.median(scores)
        }
    )
