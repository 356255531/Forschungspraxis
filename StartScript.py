""" Trains a cross-entropy agent. Uses OpenAI Gym. """
import numpy as np
import cPickle as pickle
import gym
import sys

M = 4;
I = 300; # number of iterations
N = 100; # number of samples per iteration
render = False;
quant = 0.2;
NTOP = np.int(quant*N);

modelmu = [0,0,0,0];
modelvar = [1,1,1,1];

def logistic(x):
    return 1.0 / (1.0 + np.exp(-x));

def get_action(x):
    return 0 if np.dot(x,modelmu) < 0 else 1
    
env = gym.make("CartPole-v0");
observation = env.reset();
inum = 0; # iteration number
running_reward = None;
for inum in range(I):
    xs, rs = [], []; # sequences of samples and rewards
    for isam in range(N):
        observation = env.reset();
        model = np.sqrt(modelvar)*np.random.randn(M) + modelmu; # generate random sample
        done = False;
        reward_sum = 0;
        while not done: # play the game with the random sample
            if render:
                env.render();
            action = get_action(observation)
            observation, reward, done, info = env.step(action);
            reward_sum += reward;
        rs.append(reward_sum);
        xs.append(model);
        
    sort_indices = np.argsort(rs)[::-1];
    xs = np.array(xs)[sort_indices];
    rs = rs[:NTOP];
    xs = xs[:NTOP];
    modelmu = np.mean(xs, axis=0);
    modelvar = np.var(xs, axis=0);
    reward_avg = np.mean(rs);
    first_reward = reward_avg if running_reward is None else first_reward
running_reward = reward_avg if running_reward is None else running_reward * 0.99 + reward_avg * 0.01