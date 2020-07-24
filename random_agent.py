import argparse
import glob
import gym
import numpy as np
import random
import wandb
from gym import wrappers
from pyvirtualdisplay import Display
from common.replay import ReplayBuffer


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', type=str, default='CartPole-v0')
    ap.add_argument('--max_frames', type=int, default=10000)
    return ap.parse_args()

def setup_wandb(args):
    config = dict(
        env = args.env,
        max_frames = args.max_frames
    )
    wandb.init(
        project='rlmp',
        notes='Random Agent',
        tags=['Random'],
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


    num_frames = 0
    while num_frames < args.max_frames:

        state = env.reset()
        done = False
        ep_reward = 0 
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            state = next_state 
            wandb.log({'reward':reward})
            num_frames += 1
            ep_reward += reward
            
        env.close()
        wandb.log({'ep_reward':ep_reward})
        
    # Upload the video 
    for movie in glob.glob(video_path + '/*.mp4'):
        wandb.log({'Video':wandb.Video(movie)})
