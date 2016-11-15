import gym
from math import floor
import numpy as np
import matplotlib.pyplot as plt
#############################
precise_x = 20
precise_y = 20

learning_rate = 0.9
discount_factor = 0.9
correction_factor = 3
###################
Qfunc = np.random.rand(precise_x * precise_y * 2)
w = np.zeros(precise_x * precise_y * 2)
#################


def observation_to_feature(observation, action):
    """Transfer observation to feature space(only for Mountain Car!) """
    size_unit_in_x = 1.8 / (precise_x * 1.0)
    size_unit_in_y = 0.14 / (precise_y * 1.0)
    x = floor((observation[0] + 1.2) / size_unit_in_x)
    if x == precise_x:
        x = precise_x - 1
    y = floor((observation[1] + 0.07) / size_unit_in_y)
    if y == precise_y:
        y = precise_y - 1
    feature = np.zeros(precise_x * precise_y * 2)
    if action == 2:
        action = 1
    feature[int(y * precise_y + x) + action * precise_x * precise_y] = 1
    return feature


def epsilon_greddy_action_choose(observation, Qfunc):
    if np.random.random_sample() < 0.1:
        action = np.random.randint(2)
        return action
    if (np.dot(Qfunc, observation_to_feature(observation, 0)) >
            np.dot(Qfunc, observation_to_feature(observation, 1))):
        return 0
    else:
        return 1


def parameter_update(observation, observation_next, action,
                     action_next, learning_rate, discount_factor, reward,
                     Qfunc, w):
    """Run TDC"""
    feature = observation_to_feature(observation, action)
    feature_next = observation_to_feature(observation_next, action_next)
    TD_error = (reward + discount_factor * np.dot(Qfunc, feature_next) - np.dot(Qfunc, feature))

    Qfunc = Qfunc + learning_rate * (feature - discount_factor * feature_next) * np.dot(feature, w)
    w = w + correction_factor * learning_rate * (TD_error - np.dot(feature, w)) * feature
    return Qfunc, w


env = gym.make('MountainCar-v0')

total_reward = 0
Qfunc_difference = []
total_reward_episode = []
for i_episode in range(1000):
    observation = env.reset()
    total_reward = 0
    Qfunc_previous = Qfunc
    for t in range(200):
        # env.render()
        # print(observation)
        action = epsilon_greddy_action_choose(observation, Qfunc)
        # print action
        observation_next, reward, done, info = env.step(action)
        action_next = epsilon_greddy_action_choose(observation_next, Qfunc)
        Qfunc_previous = Qfunc
        Qfunc, w = parameter_update(observation, observation_next,
                                    action, action_next, learning_rate,
                                    discount_factor, reward, Qfunc, w)
        if done:
            learning_rate = learning_rate * 0.999
        # print np.dot(Qfunc_previous - Qfunc, Qfunc_previous - Qfunc)
        total_reward += reward
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
    total_reward_episode.append(total_reward)
    Qfunc_difference.append(np.dot(Qfunc_previous - Qfunc, Qfunc_previous - Qfunc))
    if i_episode % 100 == 0:
        print i_episode, "th episode completed"

# print Qfunc_difference
# print w

plt.figure(1)
plt.subplot(211)
plt.plot(Qfunc_difference)

plt.subplot(212)
plt.plot(total_reward_episode)
plt.show()

for i_episode in range(10):
    observation = env.reset()
    for t in range(100):
        env.render()
        # print(observation)
        action = epsilon_greddy_action_choose(observation, Qfunc)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
