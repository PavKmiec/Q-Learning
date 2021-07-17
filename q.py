
from collections import defaultdict, namedtuple
from sympy.matrices import Matrix, ImmutableMatrix
import gym
import numpy as np
import random
import pandas as pd
from random import random


# CartPole-v1

'''Other simple environments can also perform reasonably but require adjusting parameter values'''

# setting up environment using Gym
env = gym.make('CartPole-v1')

# checking the action space
print('Action Space: ', env.action_space.n)




# Defining storage for tables
# and parameters

# Q-table
Q_table = defaultdict(float)

# Frequency table
N_table = defaultdict(int)

# reward count
reward_util = 0 
# steps
steps  = 0

# value of future reward
# used to balance immediate and future reward
gamma= 0.99


# epsilon parameters
epsilon = 0.99
# decay rate
epsilon_decay = 0.999
# final epsilon - exploitation
final_epsilon = 0.01

# learning rate
alpha = 0.6
# learning rate over time
alpha_decay = 0.8


# state seve
# using named tuple
# function for creating tuple 
# subclasses with named fields
save_state = namedtuple('SavedState', ['state'])

# previous state 
prev_s = None
# previous reward
prev_r = 0
# previous action
prev_a = 0




# epsilon greedy function 
# adjust exploration/exploitation amount over time 
def epsilon_greedy(s, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    # choosing action with highest Q.
    qvals = {a: Q_table[s, a] for a in range(env.action_space.n)}
    max_q = max(qvals.values())

    # in case numerous actions have same max Q value.
    actions_with_max_q = [a for a, q in qvals.items() if q == max_q]
    return np.random.choice(actions_with_max_q)




# updating the Q-table
def update_q_val(prev_state, action, reward, next_state, done):
    # choosing best action next state
    max_q_next = max([Q_table[next_state, a] for a in range(env.action_space.n)])
    # updating N table
    N_table[prev_state, action] = N_table[prev_state, action] + 1
    # update Q table
    # isf state terminated value on next state is not included
    Q_table[prev_state, action] += alpha * (
        reward + gamma * max_q_next * (1 - done) - Q_table[prev_state, action])



# main loop
# running agent/game for number of runs
samples = 10000
# psilon drop rate
eps_drop = (epsilon - final_epsilon) / samples * 2


for sample in range(samples):    
    # initialize the environment
    curr_s = env.reset()
    reward_util = 0 
    steps  = 0
    curr_r = 0
    done = False
    step = 0
    #print(Q_table)
    while not done:
        # previous action        
        prev_a = epsilon_greedy(save_state(ImmutableMatrix(curr_s)), epsilon)
        # show visualisation
        env.render()
        # preforming action 
        # and getting reward, state
        next_state, reward, done, info = env.step(prev_a)
        # if not None set reward to previous rewerd
        if reward is not None:
            prev_r = reward
            ## add reward
            reward_util = reward_util + reward
                
        # set states
        prev_s = curr_s
        curr_s = next_state

        # update q values
        update_q_val(save_state(ImmutableMatrix(prev_s)), 
            prev_a, 
            reward, 
            save_state(ImmutableMatrix(curr_s)), 
            done)
        # add step
        steps = steps + 1

    # update epsilon
    if(epsilon > final_epsilon):
         epsilon -= eps_drop
         if(epsilon < final_epsilon):
            epsilon = final_epsilon
       

    # Print data
    print("epsilon:")
    print(epsilon)
    print("sample:")
    print(sample)
    print("Reward value: ")
    print(reward_util)

    # set data frame
    df = pd.DataFrame([[sample, reward_util]], columns=["Steps", "Reward"])
    # append data to file
    #print(df)
    df.to_csv("./q-learning_data.csv", header=None, mode="a")
    
