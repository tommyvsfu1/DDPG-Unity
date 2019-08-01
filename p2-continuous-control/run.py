from unityagents import UnityEnvironment
from collections import deque
import numpy as np
from DDPG import Agent
import time
import argparse
"""
states = (1, 33) np.array
actions = (1, 4) np.array
rewards = [] list with length 1
dones = [] list with length 1
"""
def act():
    action_size = 4
    actions = np.random.randn(1, action_size) # select an action (for each agent)
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

def train(env, agent, brain_name, train_mode=True):
    solved_score=30.0
    consec_episodes=100
    print_every=1
    mean_scores = []                               # list of mean scores from each episode
    min_scores = []                                # list of lowest scores from each episode
    max_scores = []                                # list of highest scores from each episode
    best_score = -np.inf
    scores_window = deque(maxlen=consec_episodes)  # mean scores from most recent episodes
    moving_avgs = []                               # list of moving averages    

    for i_episode in range(120):
        episode_max_frames = 1000 # debug using 1
        env_info = env.reset(train_mode=train_mode)[brain_name]      
        states = env_info.vector_observations 
        scores = 0      
        start_time = time.time()   
        agent.ep_step += 1        
        for _ in range(episode_max_frames):
            # use policy make action
            actions = agent.act(states.reshape(-1)) 
            # actions = act()
            # agent <-> environment
            next_states, rewards, dones = env_step(env, actions, brain_name)
            # collect data
            agent.collect_data(states.reshape(-1), 
                               actions.reshape(-1), 
                               rewards[0], 
                               next_states.reshape(-1), 
                               dones[0])
            
            # move to next states
            states = next_states           
            scores += rewards[0]                    
            if np.any(dones):                                  
                break

            agent.update()  

        #############################Boring Log#############################
        ####################################################################  
        duration = time.time() - start_time
        min_scores.append(np.min(scores))             # save lowest score for a single agent
        max_scores.append(np.max(scores))             # save highest score for a single agent        
        mean_scores.append(np.mean(scores))           # save mean score for the episode
        scores_window.append(mean_scores[-1])         # save mean score to window
        moving_avgs.append(np.mean(scores_window))    # save moving average
                
        if i_episode % print_every == 0:
            print('\rEpisode {} ({} sec)  -- \tMin: {:.1f}\tMax: {:.1f}\tMean: {:.1f}\tMov. Avg: {:.1f}'.format(\
                i_episode, round(duration), min_scores[-1], max_scores[-1], mean_scores[-1], moving_avgs[-1]))
        
        if train_mode and mean_scores[-1] > best_score:
            agent.save('./best')
            # print("****save model****")
                
        if moving_avgs[-1] >= solved_score and i_episode >= consec_episodes:
            print('\nEnvironment SOLVED in {} episodes!\tMoving Average ={:.1f} over last {} episodes'.format(\
                                    i_episode-consec_episodes, moving_avgs[-1], consec_episodes))            
            if train_mode:
                agent.save('./solved')
                # print("****save model****")
            break

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
        env = UnityEnvironment(file_name='./Reacher.app')
    else :
        env = UnityEnvironment(file_name='./Reacher_Linux_NoVis/Reacher.x86_64')
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    agent = Agent(a_dim=4, s_dim=33, clip_value=1) # continuous action clip
    train(env, agent, brain_name)
    env.close()

if __name__ == '__main__':
    args = parse()
    run(args)