import gym
import torch
import torch.optim as optim

from network import Network
from replay import PrioritizedReplayBuffer
from schedules import BetaSchedule, EpsilonSchedule
from trainer import NStepTrainer

if __name__ == "__main__":
    config = dict(
        n_epochs = 5,
        n_batches_per_epoch = 1000,
        batch_size = 32,
        update_interval = 10000,
        gamma = 0.99,
        lr = 1e-3,
        n_steps = 3
    )

    env_builder = lambda: gym.make("CartPole-v0")
    env = env_builder()
    online_network = Network(env.observation_space.shape[0], env.action_space.n, 64)
    target_network = Network(env.observation_space.shape[0], env.action_space.n, 64)

    if torch.cuda.is_available():
        print("Using GPU for training.")
        online_network = online_network.to('cuda:0')
        target_network = target_network.to('cuda:0')

    target_network.load_state_dict(online_network.state_dict())
    env.close()

    optimizer = optim.Adam(online_network.parameters(), lr=config['lr'])

    epsilon_schedule = EpsilonSchedule(1., 0.01, 5000)
    beta_schedule = BetaSchedule(0.4, 1000)
    buffer = PrioritizedReplayBuffer(10000, 0.6)

    trainer = NStepTrainer(config, online_network, target_network, optimizer,
                           buffer, epsilon_schedule, beta_schedule,
                           env_builder)
    trainer.train()
