import torch
from collections import namedtuple
import random
random.seed(11037)

Transition = namedtuple('Transition',
                        ('state', 'action','reward', 'next_state', 'done'))

class ReplayBuffer(object):
    """
    Replay Buffer for Q function
        default size : 20000 of (s_t, a_t, r_t, s_t+1)
    Input : (capacity)
    """
    def __init__(self, capacity=20000):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        random.seed(11037)

    def push(self, *args):
        """
        Push (s_t, a_t, r_t, s_t+1) into buffer
            Input : s_t, a_t, r_t, s_t+1, done
            Output : None
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, device):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        next_state_batch = torch.cat(batch.next_state).to(device)
        dones_batch = torch.cat(batch.done).to(device)
        return (state_batch, action_batch, reward_batch, next_state_batch, dones_batch)

    def __len__(self):
        return len(self.memory)


def soft_update(target, source, tau):
    # code from https://github.com/ghliu/pytorch-ddpg/blob/master/util.py
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )