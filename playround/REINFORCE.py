import gym
import numpy as np
import torch
import torch.nn as nn
from model import MLP
from logger import TensorboardLogger

seed = 11037
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.action_prob = []
        self.values = []
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.action_prob[:]
        del self.values[:]

class Agent():
    def __init__(self, test=False):
        # device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else :
            self.device = torch.device('cpu')
        
        self.model = MLP(state_dim=4,action_num=2,hidden_dim=256).to(self.device)  
        if test:
            self.load('./pg_best.cpt')        
        # discounted reward
        self.gamma = 0.99 
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        # saved rewards and actions
        self.memory = Memory()
        self.tensorboard = TensorboardLogger('./')
    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)
    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))
    def act(self,x,test=False):
        if not test:
            # boring type casting
            x = ((torch.from_numpy(x)).unsqueeze(0)).float().to(self.device)
            # stochastic sample
            action_prob = self.model(x)
            dist = torch.distributions.Categorical(action_prob)
            action = dist.sample()
            # memory log_prob
            self.memory.logprobs.append(dist.log_prob(action))
            return action.item()    
        else :
            self.model.eval()
            x = ((torch.from_numpy(x)).unsqueeze(0)).float().to(self.device)
            with torch.no_grad():
                action_prob = self.model(x)
                # a = np.argmax(action_prob.cpu().numpy())
                dist = torch.distributions.Categorical(action_prob)
                action = dist.sample()
                return action.item()
    def collect_data(self, state, action, reward):
        self.memory.actions.append(action)
        self.memory.rewards.append(torch.tensor(reward))
        self.memory.states.append(state)
    def clear_data(self):
        self.memory.clear_memory()

    def update(self):
        R = 0
        advantage_function = []        
        for t in reversed(range(0, len(self.memory.rewards))):
            R = R * self.gamma + self.memory.rewards[t]
            advantage_function.insert(0, R)

        # turn rewards to pytorch tensor and standardize
        advantage_function = torch.Tensor(advantage_function).to(self.device)
        advantage_function = (advantage_function - advantage_function.mean()) / (advantage_function.std() + np.finfo(np.float32).eps)

        policy_loss = []
        for log_prob, reward in zip(self.memory.logprobs, advantage_function):
            policy_loss.append(-log_prob * reward)
        # Update network weights
        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        self.optimizer.step() 
        # boring log
        self.tensorboard.scalar_summary("loss", loss.item())
        self.tensorboard.update()
    
def train(env, agent):
    best_avg_reward = float('-inf')
    episode_rewards = []
    for i_episode in range(300):
        state = env.reset()
        done = False
        while (not done):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.collect_data(state, action, reward)
            # move to next state
            state = next_state
            if done:
                #print("Episode finished after {} timesteps".format(t+1))
                break
        agent.update()
        last_reward = np.sum(agent.memory.rewards)
        episode_rewards.append(last_reward)
        agent.clear_data()
        avg_reward = np.mean(episode_rewards[-10:])
        print("avg rewards", avg_reward)
        # save the best model
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            agent.save('pg_best.cpt')
            
def test(env, agent):
    episode_rewards = []
    for i_episode in range(200):
        state = env.reset()
        rewards = []
        done = False
        while(not done):
            env.render()
            action = agent.act(state,test=True)
            next_state, reward, done, info = env.step(action)
            # move to next state
            state = next_state
            rewards.append(reward)
            if done:
                break
        print("rewards", np.sum(rewards))
        episode_rewards.append(np.sum(rewards))
    print("Mean Rewards:", np.mean(episode_rewards))

def run():
    env = gym.make('CartPole-v1') # use latest version
    env.seed(seed)
    agent = Agent()
    train(env, agent)
    # test(env,agent)
    env.close()

if __name__ == '__main__':
    run()
