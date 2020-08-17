import gym
from gym import wrappers
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines import PPO2


def play_evaluation_games(model, env_name, video_path, num_games=20):
    """ Play some evaluation games and return the
    scores.  Also take a video. """

    env = make_atari_env(env_name, num_env=1, seed=0)
    #env = wrappers.Monitor(
    #    env,
    #    video_path,
    #    video_callable=lambda x: True
    #)
    #env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)

    scores = []
    for eval_game in range(num_games):
        score = 0
        state = env.reset()
        
        done = False
        while not done:
            action, _ = model.predict(state)
            next_state, rewards, dones, info = env.step(action)
            score += rewards
            state = next_state

        scores.append(score)
        env.close()


    return scores

if __name__ == "__main__":
    # multiprocess environment
    # env = make_vec_env('MsPacmanNoFrameskip-v4', n_envs=4)

    env = make_atari_env('PongNoFrameskip-v4', num_env=32, seed=0)
    env = VecFrameStack(env, n_stack=4)

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=25000)
    # model.save("ppo2_pacman")

    scores = play_evaluation_games(model, 'PongNoFrameskip-v4', video_path='/tmp/vid', num_games=10)
    print(scores)
    # Enjoy trained agent
    #obs = env.reset()
    #while True:
    #    action, _states = model.predict(obs)
    #    obs, rewards, dones, info = env.step(action)
    # env.render()
