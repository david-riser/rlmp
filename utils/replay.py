import numpy as np


class ReplayBuffer:
    def __init__(self, max_size=10000,
                 n_actions=1, obs_shape=1):
        self.max_size = max_size
        self.n_actions = n_actions
        self.obs_shape = obs_shape
        self.actions = np.empty((self.max_size, self.n_actions))
        self.states = np.empty((self.max_size, *self.obs_shape))
        self.next_states = np.empty((self.max_size, *self.obs_shape))
        self.rewards = np.empty((self.max_size, 1))
        self.dones = np.empty((self.max_size, 1))
        self.index_pool = np.arange(self.max_size)
        self.index = 0
        
    def store(self, action, state, next_state, reward, done):
        current_index = self.index % self.max_size
        self.actions[current_index] = action
        self.states[current_index] = state
        self.next_states[current_index] = next_state
        self.dones[current_index] = done
        self.rewards[current_index] = reward
        self.index += 1

    def sample(self, batch_size):
        if self.index >= self.max_size:
            indices = np.random.choice(self.index_pool, batch_size, replace=False)

            return (self.actions[indices], self.states[indices], self.next_states[indices], 
                    self.rewards[indices], self.dones[indices])
        else:
            return None

    @property
    def is_full(self):
        return self.index >= self.max_size
