import argparse
import glob
import gym
import numpy as np
import random
import wandb
from gym import wrappers
from pyvirtualdisplay import Display
from utils.replay import ReplayBuffer

from tensorflow.keras import Model
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Flatten,
                                     Input, MaxPooling2D)
from tensorflow.keras.optimizers import Adam


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', type=str, default='CartPole-v0')
    ap.add_argument('--episodes', type=int, default=100)
    return ap.parse_args()

def setup_wandb(args):
    config = dict(
        env = args.env,
        episodes = args.episodes
    )
    wandb.init(
        project='rlmp',
        notes='Random Agent',
        tags=['Random'],
        config=config
    )


def build_network(input_shape, output_shape):
    """ Build a fully connected network or 
    a convolutional network based on the shape
    of the observation space. """
    if len(input_shape) > 1:
        print("Building convnet with shapes: {} -> {}".format(
            input_shape, output_shape
        ))
        return _build_convnet(input_shape, output_shape)
    else:
        print("Building net with shapes: {} -> {}".format(
            input_shape, output_shape
        ))
        return _build_fc(input_shape, output_shape)

    
def _build_convnet(input_shape, output_shape):
    inputs = Input(input_shape)

    x = Conv2D(8, 3)(inputs)
    x = Activation('relu')(x)
    x = Conv2D(8, 3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(8, 3)(x)
    x = Activation('relu')(x)
    x = Conv2D(8, 3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(8, 3)(x)
    x = Activation('relu')(x)
    x = Conv2D(8, 3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(64)(x)
    
    outputs = Dense(output_shape)(x)
    
    return Model(inputs, outputs)


def _build_fc(input_shape, output_shape):
        inputs = Input(input_shape)
        x = Dense(64)(inputs)
        x = Dense(64)(x)
        outputs = Dense(output_shape)(x)
        return Model(inputs, outputs)


class Agent:

    def __init__(self, action_space, observation_space, max_size, batch_size):
        self.batch_size = batch_size
        self.action_space = action_space
        self.observation_space = observation_space
        self.memory = ReplayBuffer(n_actions=self.action_space.n,
                                   obs_shape=self.observation_space.shape,
                                   max_size=max_size)
        self.policy = build_network(self.observation_space.shape, self.action_space.n)
        self.target = build_network(self.observation_space.shape, self.action_space.n)
        self._compile_models()


    def _compile_models(self):
        optimizer = Adam(0.001)
        self.policy.compile(optimizer=optimizer, loss='mse')
        self.target.compile(optimizer=optimizer, loss='mse')
        self.target.set_weights(self.policy.get_weights())


    def choose_action(self, observation):
         
        if random.random() < 0.9:
            print("Taking best action")
            action = self._choose_best_action(observation)
        else:
            print("Taking random action")
            action = self._choose_random_action(observation)

    def _choose_random_action(self, observation):
        return self.action_space.sample()


    def _choose_best_action(self, observation):
        return np.argmax(self.policy.predict(observation.reshape(
            1, *observation.shape
        )))


    def store(self, action, state, next_state, reward, done):
        self.memory.store(action, state, next_state, reward, done)
    
    def learn(self, i):

        # Time to do the updates
        if i % 5 == 0:
            self.target.set_weights(self.policy.get_weights())

        if len(self.memory) >= self.batch_size:
            print("Learning step" )
            actions, states, next_states, rewards, dones = self.memory.sample(self.batch_size)
            print(actions)
            print(states)
            print(next_states)
            print(rewards)
            print(dones)
            
            # Get the states and predict the Q-values
            # from the batch.
            policy_preds = self.policy.predict(states.reshape(1,*states.shape))
            target_preds = self.target.predict(next_states.reshape(1,*next_states.shape))
            rewards *= (1. - dones)

            # This might not be the correct update procedure yet,
            # but you have to start somewhere!
            self.policy.train_on_batch(rewards, target_preds)        

            
        
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

    
    agent = Agent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        max_size=10000,
        batch_size=32
    )


    while not agent.memory.is_full:
        done = False
        observation = env.reset()        
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            agent.store(action, observation, obs, reward, done)
            observation = obs
        
    for episode in range(args.episodes):
        observation = env.reset()

        done = False
        i = 0
        scores = []
        while not done:
            # action = env.action_space.sample()
            action = agent.choose_action(observation)
            print("Action ", action)

            next_observation, reward, done, _ = env.step(action)

            agent.store(action, observation, next_observation, reward, done)
            agent.learn(i)
            
            log_data = dict(
                reward = reward
            )
            wandb.log(log_data)
            i += 1

            scores.append(reward)
                    
            observation = next_observation
                    
        print(np.mean(scores))
            
        env.close()

    # Upload the video
    for movie in glob.glob(video_path + '/*.mp4'):
        wandb.log({'Video':wandb.Video(movie)})
