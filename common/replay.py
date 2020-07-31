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


class NaivePrioritizedReplayBuffer:
    
    """ Prioritized replay buffer following the RL-Adventure example
    and the paper but written in my own style. """
    
    def __init__(self, maxsize, alpha=0.6):
        self.buffer = []
        self.maxsize = maxsize
        self.alpha = alpha
        self.priorities = []
        self.last_batch_indices = None
        
    def add(self, state, action, reward, next_state, done):
        """ Add to the buffer and ensure that it is 
        not too full. """

        max_prio = np.max(self.priorities) if self.buffer else 1.0
        
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)
        self.priorities.append(max_prio)
        
        if len(self.buffer) > self.maxsize:
            self.buffer.pop(0)
            self.priorities.pop(0)
            
            
    def sample(self, batch_size, beta=0.4):
        """ Get a batch of transitions. """
        probs = np.array(self.priorities) ** self.alpha
        probs /= np.sum(probs)
        idxes = np.random.choice(np.arange(len(self.buffer)), batch_size, p=probs)
        total = len(self.buffer)

        states, actions, rewards = [], [], []
        next_states, dones, weights = [], [], []
        for idx in idxes:
            states.append(self.buffer[idx][0])
            actions.append(self.buffer[idx][1])
            rewards.append(self.buffer[idx][2])
            next_states.append(self.buffer[idx][3])
            dones.append(self.buffer[idx][4])
            weights.append((total * probs[idx]) ** (-beta))

        self.last_batch_indices = idxes
        return (
            np.array(states), np.array(actions),
            np.array(rewards), np.array(next_states),
            np.array(dones), np.array(weights) / np.max(weights)
        )


    def update_priorities(self, prios):
        if self.last_batch_indices is not None:
            for index, prio in zip(self.last_batch_indices, prios):
                self.priorities[index] = prio
