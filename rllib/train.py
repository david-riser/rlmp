import argparse
import glob
import os
import ray
import ray.rllib.agents.dqn.apex as apex
import time
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="MsPacmanNoFrameskip-v4")
parser.add_argument("--iterations", default=30, type=int)
parser.add_argument("--video_log_freq", default=10, type=int)
parser.add_argument("--num_gpus", default=2, type=int)
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--num_envs_per_worker", default=8, type=int)
parser.add_argument("--n_step", default=4, type=int)
parser.add_argument("--lr", default=0.00016, type=float)
parser.add_argument("--eps_final", default=0.015, type=float)
parser.add_argument("--eps_timesteps", default=200000, type=int)
parser.add_argument("--timesteps_per_iteration", default=38000, type=int)
parser.add_argument("--target_network_update_freq", default=48000, type=int)
parser.add_argument("--rollout_fragment_length", default=64, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--buffer_size", default=1000000, type=int)


def build_training_config(args):
    config = apex.APEX_DEFAULT_CONFIG.copy()
    config['framework'] = 'tf'
    config['double_q'] = 'true'
    config['dueling'] = 'true'
    config['num_atoms'] = 1
    config['noisy'] = 'false'
    config['n_step'] = args.n_step
    config['lr'] = args.lr
    config['adam_epsilon'] = 0.00015
    config['hiddens'] = [512]
    config['buffer_size'] = args.buffer_size
    config['exploration_config'] = {'final_epsilon':args.eps_final, 'epsilon_timesteps':args.eps_timesteps}
    config['prioritized_replay_alpha'] = 0.5
    config['final_prioritized_replay_beta'] = 1.0
    config['prioritized_replay_beta_annealing_timesteps'] = 2000000
    config['num_gpus'] = args.num_gpus
    config['num_workers'] = args.num_workers
    config['num_envs_per_worker'] = args.num_envs_per_worker
    config['rollout_fragment_length'] = args.rollout_fragment_length
    config['train_batch_size'] = args.batch_size
    config['target_network_update_freq'] = args.target_network_update_freq
    config['timesteps_per_iteration'] = args.timesteps_per_iteration
    config['monitor'] = 'true'
    return config


def build_log_config(args, config):
    config_loggables = [
        'double_q', 'dueling', 'num_atoms',
        'noisy', 'n_step', 'lr', 'adam_epsilon',
        'buffer_size', 'num_gpus', 'num_workers',
        'num_envs_per_worker', 'rollout_fragment_length',
        'train_batch_size', 'target_network_update_freq',
        'timesteps_per_iteration', 'prioritized_replay_alpha',
        'final_prioritized_replay_beta', 'prioritized_replay_beta_annealing_timesteps',
    ]
    log_config = { var:config[var] for var in config_loggables }
    log_config['final_episilon'] = config['exploration_config']['final_epsilon']
    log_config['episilon_timesteps'] = config['exploration_config']['epsilon_timesteps']
    return log_config
    

def setup_wandb(args, log_config):
    wandb.init(
        project='pacman', notes='rllib',
        tags=['apex', 'rllib', 'dev run'], config=log_config
    )
    
    
if __name__ == "__main__":
    args = parser.parse_args()
    config = build_training_config(args)
    log_config = build_log_config(args, config)
    setup_wandb(args, log_config)

    # Start ray and load a training instance.
    ray.init()
    trainer = apex.ApexTrainer(config=config, env=args.env)

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
                'learner_dequeue_time_ms':result['timers']['learner_dequeue_time_ms'],
                'learner_grad_time_ms':result['timers']['learner_grad_time_ms'],
                'learner_overall_time_ms':result['timers']['learner_overall_time_ms'],
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
    

