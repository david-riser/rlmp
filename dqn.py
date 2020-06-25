import argparse
import glob
import gym
import wandb
from gym import wrappers
from pyvirtualdisplay import Display

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', type=str, default='CartPole-v0')
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--steps', type=int, default=100)
    return ap.parse_args()

def setup_wandb(args):
    config = dict(
        env = args.env,
        episodes = args.episodes,
        max_steps = args.steps
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
    
    for episode in range(args.episodes):
        observation = env.reset()
        
        for timestep in range(args.steps):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            
            log_data = dict(
                reward = reward
            )
            wandb.log(log_data)
            
            if done:
                break

        env.close()

    # Upload the video
    for movie in glob.glob(video_path + '/*.mp4'):
        wandb.log({'Video':wandb.Video(movie)})
