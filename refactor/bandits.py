import numpy as np
import random


class Bandit:
    """ An upper confidence bound bandit that uses an 
        exponentially weighted moving average to estimate
        action values.
    """
    def __init__(self, arms, alpha, c):
        self.arms = arms
        self.alpha = alpha
        self.c = c
        self.steps = 0
        self.n_arms = len(self.arms)
        self.values = np.zeros(self.n_arms)
        self.counts = np.zeros(self.n_arms)
        self.last_action = None


    def step(self, reward):
        self.steps += 1 
        self.counts[self.last_action] += 1
        self.values[self.last_action] = self.alpha * reward + (1. - self.alpha) * self.values[self.last_action]
        print("Stepping bandit with values: ")
        print(self.values)
        
        
    def sample(self):
        ucb = self.values + self.c * np.sqrt(np.log(1. + self.steps) / (1. + self.counts))
        print("UCB: ", ucb)
        self.last_action = np.argmax(ucb)
        print("Sampling action: ", self.last_action)
        return self.arms[self.last_action]

class EGBandit:
    """ An epsilon-greedy bandit that uses an 
        exponentially weighted moving average to estimate
        action values.
    """
    def __init__(self, arms, alpha, eps):
        self.arms = arms
        self.alpha = alpha
        self.eps = eps
        self.steps = 0
        self.n_arms = len(self.arms)
        self.values = np.zeros(self.n_arms)
        self.counts = np.zeros(self.n_arms)
        self.last_action = None


    def step(self, reward):
        self.steps += 1 
        self.counts[self.last_action] += 1
        self.values[self.last_action] = self.alpha * reward + (1. - self.alpha) * self.values[self.last_action]
        print("Stepping bandit with values: ")
        print(self.values)
        
        
    def sample(self):

        if random.random() < self.eps:
            print("Sampling a random action")
            self.last_action = np.random.randint(self.n_arms) 
        else:
            print("Sampling best action")
            self.last_action = np.argmax(self.values)


        return self.arms[self.last_action]
