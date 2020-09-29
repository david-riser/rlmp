"""
No time to fix right now but the issue is that next_state
has the wrong shape when it goes into the model in ntd_loss
because the state transformer is not applied.

"""
import argparse
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import wandb

from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind

from network import ConvNetwork, Network
from replay import PrioritizedReplayBuffer
from schedules import BetaSchedule, EpsilonSchedule
from trainer import NStepTrainer
from transformers import action_transformer, cnn_state_transformer, flat_state_transformer
from utils import rolling



def get_args():
    """ Build a configuration from the command 
        line arguments. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=400)
    parser.add_argument('--n_batches_per_epoch', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--update_interval', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_steps', type=int, default=1)
    parser.add_argument('--eps_end', type=float, default=0.01)
    parser.add_argument('--eps_decay', type=int, default=5000)
    parser.add_argument('--beta_frames', type=int, default=50000)
    parser.add_argument('--beta_start', type=float, default=0.4)
    parser.add_argument('--buffer_capacity', type=int, default=100000)
    parser.add_argument('--hidden_dims', type=int, default=32)
    parser.add_argument('--env', type=str, default='CartPole-v0')
    return parser.parse_args()


def setup_wandb(config):
    """  Use the command line arguments to initialize this run
         with weights and biases.
    """
    wandb.init(
        project="rlmp",
        tags=["DQN", "n-step"],
        config=config
    )


def setup_env(env_name, train=True):
    if env_name == "CartPole-v0":
        env = gym.make(env_name)
    else:
        env = make_atari(env_name)
        if train:
            env = wrap_deepmind(env, episode_life=True, clip_rewards=False,
                                frame_stack=True, scale=True)    
        else:
            env = wrap_deepmind(env, episode_life=False, clip_rewards=False,
                                frame_stack=True, scale=True)    

    return env
    
    

if __name__ == "__main__":

    args = get_args()
    config = dict(
        n_epochs = args.n_epochs,
        n_batches_per_epoch = args.n_batches_per_epoch,
        batch_size = args.batch_size,
        update_interval = args.update_interval,
        gamma = args.gamma,
        lr = args.lr,
        n_steps = args.n_steps,
        eps_end = args.eps_end,
        eps_decay = args.eps_decay,
        beta_frames = args.beta_frames,
        beta_start = args.beta_start,
        buffer_capacity = args.buffer_capacity,
        hidden_dims = args.hidden_dims
    )

    setup_wandb(config)
    
    env_builder = lambda: setup_env(args.env)
    env = env_builder()
    
    use_cnn = len(env.observation_space.shape) > 2
    if use_cnn:
        input_shape = env.observation_space.shape
        input_shape = (input_shape[2], input_shape[1], input_shape[0])
        online_network = ConvNetwork(input_shape, env.action_space.n, args.hidden_dims)
        target_network = ConvNetwork(input_shape, env.action_space.n, args.hidden_dims)
    else:
        online_network = Network(env.observation_space.shape[0], env.action_space.n, args.hidden_dims)
        target_network = Network(env.observation_space.shape[0], env.action_space.n, args.hidden_dims)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print("Using GPU for training.")
        online_network = online_network.to(device)
        target_network = target_network.to(device)

    if use_cnn:
        state_transformer = lambda s: cnn_state_transformer(s, device)
    else:
        state_transformer = lambda s: flat_state_transformer(s, device)
        
    target_network.load_state_dict(online_network.state_dict())
    env.close()

    optimizer = optim.Adam(online_network.parameters(), lr=config['lr'])

    epsilon_schedule = EpsilonSchedule(1., args.eps_end, args.eps_decay)
    beta_schedule = BetaSchedule(args.beta_start, args.beta_frames)
    buffer = PrioritizedReplayBuffer(args.buffer_capacity, 0.6)

    trainer = NStepTrainer(config, online_network, target_network, optimizer,
                           buffer, epsilon_schedule, beta_schedule,
                           env_builder, action_transformer, state_transformer)
    trainer.train()
    torch.save(online_network, "network.pkl")
    
    # Plot something to investigate
    mean_returns = rolling(trainer.episodic_reward, np.mean, 4, pad=True)
    std_returns = rolling(trainer.episodic_reward, np.std, 4, pad=True)
    steps = np.arange(mean_returns.shape[0])
    plt.plot(steps, mean_returns, color="purple")
    plt.fill_between(steps, mean_returns - std_returns, mean_returns + std_returns,
                     alpha=0.3, color="purple")
    plt.grid(alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.savefig("rewards.pdf", bbox_inches="tight")

    wandb.save("rewards.pdf")
    wandb.save("network.pkl")