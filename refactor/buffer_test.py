import gym

from replay import PrioritizedReplayBuffer
from utils import Transition


if __name__ == "__main__":

    n_trans = 100000

    env = gym.make("CartPole-v0")
    state = env.reset()
    done = False

    replay = PrioritizedReplayBuffer(n_trans)
    while len(replay) < n_trans:
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        trans = Transition(state=state, action=action, next_state=next_state, reward=reward,
                           done=done, nth_state=None, discounted_reward=None, n=1)
        state = next_state
        replay.add(trans)
        if len(replay) % 2000 == 0:
            print("Transitions: ", len(replay))


        if done:
            done = False
            state = env.reset()
        

    replay.save("test_buffer.pkl", 1000)
    
