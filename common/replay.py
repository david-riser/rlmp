import numpy as np


class ReplayBuffer:
    
    """ Simple replay buffer to hold maxsize transitions in a list. """
    def __init__(self, maxsize):
        self.buffer = []
        self.maxsize = maxsize

    def add(self, state, action, reward, next_state, done):
        """ Add to the buffer and ensure that it is 
        not too full. """

        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

        if len(self.buffer) > self.maxsize:
            self.buffer.pop(0)

    def sample(self, batch_size):
        """ Get a batch of transitions. """
        idxes = np.random.choice(np.arange(len(self.buffer)), batch_size)

        states, actions, rewards = [], [], []
        next_states, dones = [], []
        for idx in idxes:
            states.append(self.buffer[idx][0])
            actions.append(self.buffer[idx][1])
            rewards.append(self.buffer[idx][2])
            next_states.append(self.buffer[idx][3])
            dones.append(self.buffer[idx][4])

        return (
            np.array(states), np.array(actions),
            np.array(rewards), np.array(next_states),
            np.array(dones)
        )
