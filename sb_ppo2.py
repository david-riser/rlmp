import gym

from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import PPO2

# multiprocess environment
# env = make_vec_env('MsPacmanNoFrameskip-v4', n_envs=4)

env = make_atari_env('PongNoFrameskip-v4', num_env=4, seed=0)
env = VecFrameStack(env, n_stack=4)

model = PPO2(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo2_pacman")

#del model # remove to demonstrate saving and loading

#model = PPO2.load("ppo2_cartpole")

# Enjoy trained agent
#obs = env.reset()
#while True:
#    action, _states = model.predict(obs)
#    obs, rewards, dones, info = env.step(action)
    # env.render()
