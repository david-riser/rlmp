import numpy as np


class EpsilonSchedule:
    """ A simple exponentially decaying exploration rate 
        schedule based on the number of steps taken.
    """
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay


    def value(self, step):
        return self.end + (self.start - self.end) * np.exp(-1. * step / self.decay)


class FixedSchedule:
    """ A constant value regardless of step. 
    """
    def __init__(self, const):
        self.const = const


    def value(self, step):
        return self.const

    
class BetaSchedule: 
    """ Beta annealing schedule for the prioritized replay 
        buffer importance sampling. 
    """
    def __init__(self, start, frames):
        self.start = start
        self.frames = frames


    def value(self, step):
        return min(1.0, self.start + step * (1. - self.start) / self.frames)
