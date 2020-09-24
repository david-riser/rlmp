import numpy as np
import torch


class PrioritizedReplayBuffer:
    """ Prioritized replay buffer following the RL-Adventure example
    and the paper but written in my own style. """
    def __init__(self, maxsize, alpha=0.6):
        self.maxsize = maxsize
        self.alpha = alpha
        self.priorities, self.buffer = [], []
        self.index_pool = np.arange(maxsize)
        

    def __len__(self):
        return len(self.buffer)

        
    def add(self, transition):
        """ Add to the buffer and ensure that it is 
            not too full. 
        """
        max_prio = np.max(self.priorities) if self.buffer else 1.0
        self.buffer.append(transition)
        self.priorities.append(max_prio)
        
        if len(self.buffer) > self.maxsize:
            self.buffer.pop(0)
            self.priorities.pop(0)
            
            
    def sample(self, batch_size, beta=0.4):
        """ Get a batch of transitions. """
        probs = np.array(self.priorities) ** self.alpha
        probs /= np.sum(probs)
        total = len(self.buffer)
        
        # Sample indices for this batch of transitions.  If the buffer is 
        # full it is much slower to re-create the index pool everytime so
        # instead sample a static array.
        if total == self.maxsize:
            indices = np.random.choice(self.index_pool, batch_size, p=probs)
        else:
            indices = np.random.choice(np.arange(total), batch_size, p=probs)

        transitions = []
        weights = np.zeros(batch_size)
        for i, index in enumerate(indices):
            transitions.append(self.buffer[index])
            weights[i] = (probs[index] * total) ** (-1 * beta)

        # Normalize the maximum weight for this batch to 1
        weights /= np.max(weights)

        return transitions, weights, indices


    def update_priorities(self, prios, indices):
        for index, prio in zip(indices, prios):
            self.priorities[index] = prio
