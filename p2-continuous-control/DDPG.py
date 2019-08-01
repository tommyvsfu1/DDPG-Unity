import torch
from model import Actor, Critic
from utils import ReplayBuffer, soft_update
from noise import Noise
import numpy as np
from logger import TensorboardLogger

DELTA = 0.5 # The rate of change (time)
SIGMA = 0.5 # Volatility of the stochastic processes
OU_A = 3. # The rate of mean reversion
OU_MU = 0. # The long run average interest rate

class Agent(object):
    def __init__(self, a_dim, s_dim, clip_value, device):
        self.device = device
        self.a_dim, self.s_dim  = a_dim, s_dim
        self.P_online = Actor(s_dim,a_dim).to(device)
        self.P_target = Actor(s_dim,a_dim).to(device)
        self.P_target.load_state_dict(self.P_online.state_dict())
        self.Q_online = Critic(s_dim,a_dim).to(device)
        self.Q_target = Critic(s_dim,a_dim).to(device)
        self.Q_target.load_state_dict(self.Q_online.state_dict())
        self.q_optimizer = torch.optim.Adam(self.Q_online.parameters(),lr=1e-3)
        self.p_optimizer = torch.optim.Adam(self.P_online.parameters(),lr=1e-3)


        self.loss_td = torch.nn.MSELoss()
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 128
        self.gamma = 0.99
        self.discrete = False
        self.ep_step = 0
        # noise
        self.noise = Noise(DELTA, SIGMA, OU_A, OU_MU)
                
        # Initialize noise
        self.ou_level = 0.
        self.action_low = -clip_value
        self.action_high = clip_value

        # log
        self.tensorboard = TensorboardLogger('./p2')

    def act(self, state, test=False):
        """
        Return : (1, action_dim) np.array
        """
        if not test:
            with torch.no_grad():
                # boring type casting
                self.P_online.eval()
                state = ((torch.from_numpy(state)).unsqueeze(0)).float().to(self.device)
                action = self.P_online(state) # continuous output
                self.P_online.train()
                a = action.data.cpu().numpy()   
                if self.ep_step < 200:
                    self.ou_level = self.noise.ornstein_uhlenbeck_level(self.ou_level)
                action = np.clip(a + self.ou_level,self.action_low,self.action_high)
                self.tensorboard.scalar_summary("action_0", action[0][0], self.tensorboard.time_step)
                self.tensorboard.scalar_summary("action_1", action[0][1], self.tensorboard.time_step)
                self.tensorboard.scalar_summary("action_2", action[0][2], self.tensorboard.time_step)
                self.tensorboard.scalar_summary("action_3", action[0][3], self.tensorboard.time_step)
                self.tensorboard.step_update()
                return action

    def collect_data(self, state, action, reward, next_state, done):
        self.replay_buffer.push(torch.from_numpy(state).float().unsqueeze(0), 
                                torch.from_numpy(action).float().unsqueeze(0), 
                                torch.tensor([reward]).float().unsqueeze(0), 
                                torch.from_numpy(next_state).float().unsqueeze(0),
                                torch.tensor([done]).float().unsqueeze(0))
    def clear_data(self):
        raise NotImplementedError("Circular Queue don't need this function")

    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.P_online.state_dict(), save_path + '_ponline.cpt')
        torch.save(self.P_target.state_dict(), save_path + '_ptarget.cpt')
        torch.save(self.Q_online.state_dict(), save_path + '_qonline.cpt')
        torch.save(self.Q_target.state_dict(), save_path + '_qtarget.cpt')

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        if len(self.replay_buffer) <= (self.replay_buffer.capacity * 0.8):
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size=self.batch_size, device=self.device)

        #===============================Critic Update===============================
        with torch.no_grad():
            self.P_target.eval()
            self.Q_target.eval()
            target = rewards+ self.gamma * (1-dones) * self.Q_target((next_states, self.P_target(next_states)))  
        Q = self.Q_online((states,actions))
        td_error = self.loss_td(Q, target)
        self.q_optimizer.zero_grad()
        td_error.backward()
        torch.nn.utils.clip_grad_norm_(self.Q_online.parameters(), 1)
        self.q_optimizer.step()

        #===============================Actor Update===============================
        q = self.Q_online((states, self.P_online(states)))  
        loss_a = -torch.mean(q) 
        self.p_optimizer.zero_grad()
        loss_a.backward()
        self.p_optimizer.step()

        #===============================Target Update===============================
        soft_update(self.Q_target, self.Q_online, tau=1e-3)
        soft_update(self.P_target, self.P_online, tau=1e-3)