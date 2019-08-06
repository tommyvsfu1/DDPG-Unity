import torch
from model import Actor, Critic
import numpy as np
from logger import TensorboardLogger
from collections import namedtuple, deque
import random
import copy
seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

DELTA = 0.5 # The rate of change (time)
SIGMA = 0.5 # Volatility of the stochastic processes
OU_A = 3. # The rate of mean reversion
OU_MU = 0. # The long run average interest rate

EPSILON = 1.0           # explore->exploit noise process added to act step
EPSILON_DECAY = 1e-6    # decay rate for noise process
Transition = namedtuple('Transition',
                        ('state', 'action','reward', 'next_state', 'done'))
class OUNoise:
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
        self.seed = random.seed(11037)
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

class Noise(object):

    def __init__(self, delta, sigma, ou_a, ou_mu):
        # Noise parameters
        self.delta = delta
        self.sigma = sigma
        self.ou_a = ou_a
        self.ou_mu = ou_mu

    def brownian_motion_log_returns(self):
        """
        This method returns a Wiener process. The Wiener process is also called Brownian motion. For more information
        about the Wiener process check out the Wikipedia page: http://en.wikipedia.org/wiki/Wiener_process
        :return: brownian motion log returns
        """
        sqrt_delta_sigma = np.sqrt(self.delta) * self.sigma
        return np.random.normal(loc=0, scale=sqrt_delta_sigma, size=None)

    def ornstein_uhlenbeck_level(self, prev_ou_level):
        """
        This method returns the rate levels of a mean-reverting ornstein uhlenbeck process.
        :return: the Ornstein Uhlenbeck level
        """
        drift = self.ou_a * (self.ou_mu - prev_ou_level) * self.delta
        randomness = self.brownian_motion_log_returns()
        return prev_ou_level + drift + randomness
class ReplayBuffer(object):
    """
    Replay Buffer for Q function
        default size : 20000 of (s_t, a_t, r_t, s_t+1)
    Input : (capacity)
    """
    def __init__(self, capacity=int(1e6)):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.position = 0
        self.seed = random.seed(11037)
    def push(self, *args):
        """
        Push (s_t, a_t, r_t, s_t+1) into buffer
            Input : s_t, a_t, r_t, s_t+1, done
            Output : None
        """
        # if len(self.memory) < self.capacity:
        #     self.memory.append(None)
        # self.memory[self.position] = Transition(*args)
        # self.position = (self.position + 1) % self.capacity
        e = Transition(*args)
        self.memory.append(e)

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
        
        self.noise = OUNoise(4)
        self.epsilon = EPSILON
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
                state = torch.from_numpy(state).float().to(self.device)
                actions = self.P_online(state) # continuous output
                self.P_online.train()
                a = actions.data.cpu().numpy()   
                actions = np.clip(a + self.epsilon * self.noise.sample(),self.action_low,self.action_high)
                # self.tensorboard.scalar_summary("action_0", action[0][0], self.tensorboard.time_step)
                # self.tensorboard.scalar_summary("action_1", action[0][1], self.tensorboard.time_step)
                # self.tensorboard.scalar_summary("action_2", action[0][2], self.tensorboard.time_step)
                # self.tensorboard.scalar_summary("action_3", action[0][3], self.tensorboard.time_step)
                # self.tensorboard.step_update()
                return actions

    def collect_data(self, state, action, reward, next_state, done):
        # print("state", state.shape)
        # print("action", action.shape)
        # print("reward", reward)
        # print("next state", next_state.shape)
        # print("done", done)
        self.replay_buffer.push(torch.from_numpy(state).float().unsqueeze(0), 
                                torch.from_numpy(action).float().unsqueeze(0), 
                                torch.tensor([reward]).float().unsqueeze(0), 
                                torch.from_numpy(next_state).float().unsqueeze(0),
                                torch.tensor([done]).float().unsqueeze(0))
    
    def reset(self):
        self.noise.reset()

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
        # if len(self.replay_buffer) <= (10000):
        #     return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size=self.batch_size, device=self.device)
        # print("state", states.shape)
        # print("action", actions.shape)
        # print("reward", rewards.shape)
        # print("next state", next_states.shape)
        # print("done", dones.shape)
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
        # for p in self.Q_online.named_parameters():
        #     layer_name, parameter = p
        #     if layer_name[0:2] != "bn":
        #         norm = torch.norm(parameter.grad)
        #         self.tensorboard.scalar_summary("Q"+layer_name, norm, self.tensorboard.time_train)
        self.q_optimizer.step()

        #===============================Actor Update===============================
        q = self.Q_online((states, self.P_online(states)))  
        loss_a = -torch.mean(q) 
        self.p_optimizer.zero_grad()
        loss_a.backward()
        # torch.nn.utils.clip_grad_norm_(self.P_online.parameters(), 0.5)
        # for p in self.P_online.named_parameters():
        #     layer_name, parameter = p
        #     if layer_name[0:2] != "bn":
        #         norm = torch.norm(parameter.grad)
        #         self.tensorboard.scalar_summary("P"+layer_name, norm, self.tensorboard.time_train)
        self.p_optimizer.step()

        #===============================Target Update===============================
        soft_update(self.Q_target, self.Q_online, tau=1e-3)
        soft_update(self.P_target, self.P_online, tau=1e-3)
        self.epsilon -= EPSILON_DECAY
        self.tensorboard.train_update()
        self.noise.reset()