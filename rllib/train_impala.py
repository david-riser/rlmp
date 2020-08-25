import argparse
import glob
import os
import ray
import ray.rllib.agents.impala as impala
import time
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="MsPacmanNoFrameskip-v4")
parser.add_argument("--iterations", default=30, type=int)
parser.add_argument("--video_log_freq", default=10, type=int)
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--num_workers", default=32, type=int)
parser.add_argument("--num_envs_per_worker", default=5, type=int)
parser.add_argument("--timesteps_per_iteration", default=38000, type=int)
parser.add_argument("--rollout_fragment_length", default=50, type=int)
parser.add_argument("--batch_size", default=500, type=int)


def build_training_config(args):
    config = impala.DEFAULT_CONFIG.copy()
    config['framework'] = 'tf'
    config['num_gpus'] = args.num_gpus
    config['num_workers'] = args.num_workers
    config['num_envs_per_worker'] = args.num_envs_per_worker
    config['rollout_fragment_length'] = args.rollout_fragment_length
    config['train_batch_size'] = args.batch_size
    config['monitor'] = 'true'
    config['clip_rewards'] = True
    config['lr_schedule'] = [[0., 0.0005], [20000000, 0.000000000001]]
    config['model']['use_lstm'] = True
    return config


def build_log_config(args, config):
    config_loggables = [
        'num_gpus', 'num_workers',
        'num_envs_per_worker', 'rollout_fragment_length',
        'train_batch_size',
        'timesteps_per_iteration'
    ]
    log_config = { var:config[var] for var in config_loggables }
    return log_config
    

def setup_wandb(args, log_config):
    wandb.init(
        project='pacman', notes='rllib',
        tags=['impala', 'rllib', 'dev run'],
        config=log_config
    )
    
    
if __name__ == "__main__":
    args = parser.parse_args()
    config = build_training_config(args)
    log_config = build_log_config(args, config)
    setup_wandb(args, log_config)

    # Start ray and load a training instance.
    ray.init()
    trainer = impala.ImpalaTrainer(config=config, env=args.env)

    # Find the new folder and make sure we can upload videos
    base_dir = '/home/ubuntu/ray_results/'
    expdir = max([base_dir + d for d in os.listdir(base_dir)], key=os.path.getmtime)
    print("Exp dir detected: {}".format(expdir))
    
    # Begin training
    timesteps = 0 
    for i in range(args.iterations):
        start_time = time.time()

        result = trainer.train()
        print("Finished iter {}".format(i), result)

        elapsed_time = time.time() - start_time
        current_steps = result['timesteps_total'] - timesteps
        
        # Monitor results
        wandb.log(
            {
                'episode_reward_mean':result['episode_reward_mean'],
                'episode_len_mean':result['episode_len_mean'],
                'timesteps_total':result['timesteps_total'],
                'elapsed_time':elapsed_time,
                'steps_per_unit_time':float(current_steps / elapsed_time)
            }
        )

        timesteps = result['timesteps_total']
        
        # Upload the new videos
        videos = glob.glob(expdir + "/*.mp4")
        print("Videos:", videos)
        if len(videos) > 0 and i % args.video_log_freq == 0:
            wandb.log({'Video {}'.format(i):wandb.Video(videos[0])})

        for video in videos:
            os.system('rm {}'.format(video))
    

