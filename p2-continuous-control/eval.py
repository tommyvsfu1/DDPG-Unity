from unityagents import UnityEnvironment
from collections import deque
import numpy as np
import torch
from DDPG import Agent
import time
import argparse
from logger import TensorboardLogger
"""
Single Agent
states = (1, 33) np.array
actions = (1, 4) np.array
rewards = [] list with length 1
dones = [] list with length 1
"""
"""
Multi Agents
states = (20, 33) np.array
actions = (20, 4) np.array
rewards = [] list with length 20
dones = [] list with length 20
"""

log = TensorboardLogger('./p2_log_test')

def act():
    action_size = 4
    actions = np.random.randn(20, action_size) # select an action (for each agent)
    actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1 
    return actions

def env_step(env, actions, brain_name):
    """
    Return next_states, rewards, dones
    """
    env_info = env.step(actions)[brain_name]           # send all actions to tne environment
    next_states = env_info.vector_observations         # get next state (for each agent)
    rewards = env_info.rewards                         # get reward (for each agent)
    dones = env_info.local_done                        # see if episode finished
    return next_states, rewards, dones



def eval(env, agent, brain_name):
    mean_scores = []                               # list of mean scores from each episode
    scores_window = deque(maxlen=100)  # mean scores from most recent episodes
    moving_avgs = []                               # list of moving averages    

    for _ in range(5):
        env_info = env.reset(train_mode=False)[brain_name]      
        states = env_info.vector_observations
        dones = env_info.local_done 
        scores = np.zeros(20) 
        start_time = time.time()   
        # agent.ep_step += 1       
        agent.reset() 
        while(True):
            # use policy make action
            actions = agent.act(states,test=True) 
            # agent <-> environment
            next_states, rewards, dones = env_step(env, actions, brain_name)
            # move to next states
            states = next_states           
            scores += rewards
            if (np.any(dones)):
                break
        #############################Boring Log#############################
        ####################################################################  
        duration = time.time() - start_time    
        mean_scores.append(np.mean(scores))           # save mean score for the episode
        scores_window.append(mean_scores[-1])         # save mean score to window
        moving_avgs.append(np.mean(scores_window))    # save moving average
        print("moving_avgs", moving_avgs[-1],"duration",duration)
        log.scalar_summary("Test_Avg_Rewards", moving_avgs[-1], log.time_ep)
        log.episode_update()

def parse():
    parser = argparse.ArgumentParser(description="p2")
    parser.add_argument('--machine', default="Mac", help='environment name')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args

def run(args):
    if args.machine == "Mac":
        env = UnityEnvironment(file_name='./Reacher.app',seed=1)
    else :
        env = UnityEnvironment(file_name='./Reacher_Linux_NoVis/Reacher.x86_64',seed=1)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else :
        device = torch.device('cpu')
    print("using device", device)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])
    #==========================my version=========================
    agent = Agent(a_dim=4, s_dim=33, clip_value=1, device=device) # continuous action clip
    agent.load("./pretrained/")
    eval(env, agent, brain_name)
    env.close()

if __name__ == '__main__':
    args = parse()
    run(args)