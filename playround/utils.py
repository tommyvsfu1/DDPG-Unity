import torch
import random
import numpy as np
import copy
from collections import namedtuple
import scipy.signal

def soft_update(target, source, tau):
    # code from https://github.com/ghliu/pytorch-ddpg/blob/master/util.py
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

class OUNoise:
    # code from https://github.com/tommytracey/DeepRL-P2-Continuous-Control/blob/master/ddpg_agent.py
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process.
        Params
        ======
            mu: long-running mean
            theta: the speed of mean reversion
            sigma: the volatility parameter
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


def discount(rewards):
    """
    Input : 
        1.reward
            type(reward) = np.array 
            reward.shape = (N,)
    
    Output :
        1.discounted reward
            type(reward) = np.array
            reward.shape = (N,1)
    """
    gamma = 0.99
    factor = 0.1
    x = rewards
    discounted = scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
    discounted *= factor

    return np.expand_dims(rewards,axis=1)