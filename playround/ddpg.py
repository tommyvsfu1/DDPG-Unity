import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from model import Actor, Critic
from utils import soft_update, discount
from noise import Noise
from collections import namedtuple
import random
seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
EPSILON_DECAY = 1e-3
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

class Agent():
    def __init__(self, test=False):
        # device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else :
            self.device = torch.device('cpu')    
        #########################################
        """
        Some hand tune config(for developing)
        """
        self.discrete = False
        self.action_dim = 1
        self.state_dim = 3
        self.batch_size = 100
        self.action_low = -2
        self.action_high = 2
        ##########################################
        self.P_online = Actor(state_dim=self.state_dim, action_size=self.action_dim).to(self.device)
        self.P_target = Actor(state_dim=self.state_dim, action_size=self.action_dim).to(self.device)
        self.P_target.load_state_dict(self.P_online.state_dict())
        self.Q_online = Critic(state_size=self.state_dim, action_size=self.action_dim).to(self.device)
        self.Q_target = Critic(state_size=self.state_dim, action_size=self.action_dim).to(self.device)
        self.Q_target.load_state_dict(self.Q_online.state_dict())
        # discounted reward
        self.gamma = 0.99 
        self.eps = 0.25
        # optimizer
        self.q_optimizer = torch.optim.Adam(self.Q_online.parameters(), lr=1e-3)    
        self.p_optimizer = torch.optim.Adam(self.P_online.parameters(), lr=1e-3)    
        # saved rewards and actions
        self.replay_buffer = ReplayBuffer()
    
        # noise
        self.noise = Noise(DELTA, SIGMA, OU_A, OU_MU)
        # Initialize noise
        self.ou_level = 0.

        self.ep_step = 0
    def act(self,state , test=False):
        if not test:
            with torch.no_grad():
                # boring type casting
                state = ((torch.from_numpy(state)).unsqueeze(0)).float().to(self.device)
                action = self.P_online(state) # continuous output
                a = action.data.cpu().numpy()    
                # if self.ep_step < 200:     
                    # self.ou_level = self.noise.ornstein_uhlenbeck_level(self.ou_level)
                    # a = a + self.ou_level
                if self.discrete:
                    action = np.argmax(a)
                    return a, action
                else :
                    if self.ep_step < 200:
                        self.ou_level = self.noise.ornstein_uhlenbeck_level(self.ou_level)
                    action = np.clip(a+self.ou_level,self.action_low,self.action_high)
                    return action, action    

    def collect_data(self, state, action, reward, next_state, done):
        self.replay_buffer.push(torch.from_numpy(state).float().unsqueeze(0), 
                                torch.from_numpy(action).float(), 
                                torch.tensor([reward]).float().unsqueeze(0), 
                                torch.from_numpy(next_state).float().unsqueeze(0),
                                torch.tensor([done]).float().unsqueeze(0))
    def clear_data(self):
        raise NotImplementedError("Circular Queue don't need this function")

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size=self.batch_size, device=self.device)
        # discounted rewards
        # rewards = torch.from_numpy(discount((rewards.view(rewards.shape[0])).cpu().numpy())).float().to(self.device)

        ### debug shape : ok                
        #===============================Critic Update===============================
        self.Q_online.train()
        Q = self.Q_online( (states, actions) )

        with torch.no_grad(): # don't need backprop for target value
            self.Q_target.eval()
            self.P_target.eval()
            target = rewards + self.gamma * (1-dones) * self.Q_target( (next_states, self.P_target(next_states)))
        critic_loss_fn = torch.nn.MSELoss()
        critic_loss = critic_loss_fn(Q, target).mean()
        # update
        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()
        # print("critic loss", critic_loss.item())
        
        
        #===============================Actor Update===============================
        # fix online_critic , update online_actor
        self.Q_online.eval()
        for p in self.Q_online.parameters():
            p.requires_grad = False
        for p in self.P_online.parameters():
            p.requires_grad = True
        policy_loss = - self.Q_online( (states, self.P_online(states)) )
        policy_loss = policy_loss.mean()
        self.p_optimizer.zero_grad()
        policy_loss.backward()
        self.p_optimizer.step()
        # print("policy loss", policy_loss.item())
        for p in self.Q_online.parameters():
            p.requires_grad = True
        #===============================Target Update===============================
        soft_update(self.Q_target, self.Q_online, tau=1e-3)
        soft_update(self.P_target, self.P_online, tau=1e-3)
        self.eps -= EPSILON_DECAY
        if self.eps <= 0:
            self.eps = 0
def train(env, agent):
    episode_rewards = []
    for i_episode in range(60):
        state = env.reset()
        done = False
        last_reward = 0
        while (not done):
            # env.render()
            # agent use policy to choose action
            a, action = agent.act(state)
            # action = env.action_space.sample()
            # agent <-> environment
            next_state, reward, done, info = env.step(action)
            # agent collect data
            agent.collect_data(state.reshape(-1), a, reward, next_state.reshape(-1), done)
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
    # env = gym.make('CartPole-v1') # use latest version
    env = gym.make('Pendulum-v0')
    env.seed(seed) 
    agent = Agent()
    train(env, agent)
    env.close()

if __name__ == '__main__':
    run()

