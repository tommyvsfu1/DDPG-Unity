# code reference:
# 1.https://blog.csdn.net/blanokvaffy/article/details/86232658 for debuging
# 2.https://github.com/liampetti/DDPG/blob/master/ddpg.py 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time
from noise import Noise
from collections import namedtuple
import random
from utils import soft_update
from model import Actor, Critic
#####################  hyper parameters  ####################
seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

MAX_EPISODES = 400
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
TAU = 0.01
RENDER = False
ENV_NAME = 'Pendulum-v0'


DELTA = 0.5 # The rate of change (time)
SIGMA = 0.5 # Volatility of the stochastic processes
OU_A = 3. # The rate of mean reversion
OU_MU = 0. # The long run average interest rate



Transition = namedtuple('Transition',
                        ('state', 'action','reward', 'next_state', 'done'))

class ReplayBuffer(object):
    """
    Replay Buffer for Q function
        default size : 20000 of (s_t, a_t, r_t, s_t+1)
    Input : (capacity)
    """
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

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

    def sample(self, batch_size, device='cpu'):
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


class Agent(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        #self.sess = tf.Session()
        self.P_online = Actor(s_dim,a_dim)
        self.P_target = Actor(s_dim,a_dim)
        self.P_target.load_state_dict(self.P_online.state_dict())
        self.Q_online = Critic(s_dim,a_dim)
        self.Q_target = Critic(s_dim,a_dim)
        self.Q_target.load_state_dict(self.Q_online.state_dict())
        self.q_optimizer = torch.optim.Adam(self.Q_online.parameters(),lr=LR_C)
        self.p_optimizer = torch.optim.Adam(self.P_online.parameters(),lr=LR_A)
        self.loss_td = nn.MSELoss()
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 32

        self.discrete = False
        self.ep_step = 0
        # noise
        self.noise = Noise(DELTA, SIGMA, OU_A, OU_MU)
        # Initialize noise
        self.ou_level = 0.
        self.action_low = -2
        self.action_high = 2
    def act(self, state, test=False):
        if not test:
            with torch.no_grad():
                # boring type casting
                state = ((torch.from_numpy(state)).unsqueeze(0)).float().to('cpu')
                action = self.P_online(state) # continuous output
                a = action.data.cpu().numpy()   
                if self.discrete:
                    action = np.argmax(a)
                    return a, action
                else :
                    if self.ep_step < 200:
                        self.ou_level = self.noise.ornstein_uhlenbeck_level(self.ou_level)
                    action = np.clip(a + self.ou_level,self.action_low,self.action_high)
                    return (torch.from_numpy(action)).view(-1)  

    def collect_data(self, state, action, reward, next_state, done):
        self.replay_buffer.push(torch.from_numpy(state).float().unsqueeze(0), 
                                torch.from_numpy(action).float().unsqueeze(0), 
                                torch.tensor([reward]).float().unsqueeze(0), 
                                torch.from_numpy(next_state).float().unsqueeze(0),
                                torch.tensor([done]).float().unsqueeze(0))
    def clear_data(self):
        raise NotImplementedError("Circular Queue don't need this function")

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size=self.batch_size, device='cpu')

        #===============================Critic Update===============================
        with torch.no_grad():
            target = rewards+ GAMMA* (1-dones) * self.Q_target((next_states, self.P_target(next_states)))  
        Q = self.Q_online((states,actions))
        td_error = self.loss_td(target, Q)
        self.q_optimizer.zero_grad()
        td_error.backward()
        self.q_optimizer.step()

        #===============================Actor Update===============================
        q = self.Q_online((states, self.P_online(states)))  
        loss_a = -torch.mean(q) 
        self.p_optimizer.zero_grad()
        loss_a.backward()
        self.p_optimizer.step()

        #===============================Target Update===============================
        soft_update(self.Q_target, self.Q_online, tau=1e-2)
        soft_update(self.P_target, self.P_online, tau=1e-2)


def train(env, agent):
    episode_rewards = []
    for i_episode in range(60):
        state = env.reset()
        done = False
        last_reward = 0
        while (not done):
            # env.render()
            # agent use policy to choose action
            a = agent.act(state)
            # action = env.action_space.sample()
            # agent <-> environment
            next_state, reward, done, info = env.step(a)
            # agent collect data
            agent.collect_data(state.reshape(-1), a.numpy(), reward, next_state.reshape(-1), done)
            # environment move to next state
            state = next_state.reshape(-1)
            last_reward += reward
            if done:
                break
            agent.update()
        agent.ep_step += 1
        episode_rewards.append(last_reward)
        print("Avg rewards", np.mean(episode_rewards[-10:]))
def run():
    env = gym.make('Pendulum-v0')
    env.seed(seed) 
    agent = Agent(a_dim=1, s_dim=3, a_bound=2)
    train(env, agent)
    env.close()

if __name__ == '__main__':
    run()