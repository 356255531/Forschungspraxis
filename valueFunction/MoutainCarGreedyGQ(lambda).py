__auther__ = "Zhiwei"
############################
import gym
import math
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import sys
import itertools
#############################
precise = [20, 20]

learning_rate = 0.01
discount_factor = 0.9
eligibility_factor = 0.9
discount_of_learning_rate = 0.9
epsilon = 0.1
###################
class StateActionSpace(object):
    """docstring for StateActionSpace"""
    def __init__(self, continuous_space, precise, actions):
        super(StateActionSpace, self).__init__()
        self.continuous_space = continuous_space
        self.precise = precise
        self.actions = actions

        self.state_space, self.lower_band, self.upper_band, self.unit = self.__discret_space()
        self.action_space = self.__state_action_space_generation()


    def __discret_space(self):
        lower_band = list(self.continuous_space)[0]
        upper_band = list(self.continuous_space)[1]
        precise = [float(i) for i in self.precise]
        unit = (np.array(upper_band) - np.array(lower_band)) / np.array(precise)

        sub_precise = []
        for i in self.precise:
            sub_precise.append([j for j in xrange(0, i)])
        return itertools.product(*sub_precise), lower_band, upper_band, unit


    def __state_action_space_generation(self):
        """ Hard code:
            generate the action space only for mountain car """
        return {i : self.actions for i in self.state_space}


    def _m_state_continuous_to_discret(self, input_state):
        try:
            if len(self.upper_band) != len(input_state):
                raise ValueError("The continous state illegal")
            for i in xrange(0, len(input_state)):
                if not(self.lower_band[i] <= input_state[i] <= self.upper_band[i]):
                    raise ValueError("The continous state illegal")
        except ValueError as e:
            print e
            sys.exit()

        discret_state = input_state
        discret_state = (discret_state - np.array(self.lower_band)) / self.unit
        discret_state = [int(math.floor(i)) for i in discret_state]

        for i in xrange(0, len(input_state)):
            if input_state[i] == self.upper_band[i]:
                discret_state[i] = self.precise[i] - 1
        return discret_state


    def _m_state_to_feature(self, state, action):
        discret_state = self.__state_continuous_to_discret(state)
        num_zero_entry = reduce(lambda x, y : x * y, self.precise, 1) * len(self.actions)
        feature = np.zeros(num_zero_entry)
        num_nonzero_entry = 0

        new_state = discret_state
        new_state.append(action)
        new_precise = self.precise
        new_precise.append(len(self.actions))
        precise = new_precise

        count = 0
        for i, j in zip(new_precise, new_state):
            precise.pop(0)
            count += j * reduce(lambda x, y:x * y, precise, 1)
        feature[count] = 1
        return feature


class GQLambda(object):
    """docstring for GQLambda"""
    def __init__(self, alpha, gamma, lambda_back, eta, epsilon, action_space):
        super(GQLambda, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.lambda_back = lambda_back
        self.eta = eta
        self.epsilon = epsilon
        self.state_space = action_space.keys()
        self.action_space = action_space

        self.num_qfunc = reduce(lambda x, y: x + len(y), action_space.values(), 0)
        self.theta = np.random.rand(self.num_qfunc)
        self.w = np.zeros(self.num_qfunc)
        self.e = np.zeros(self.num_qfunc)

    def _m_Learn(self, phi, phi_bar, step_reward, rho, i_unknown):
        print total_reward, self.gamma, self.theta, phi_bar
        delta = step_reward + self.gamma * np.dot(self.theta, phi_bar)
        delta -= np.dot(self.theta, phi)

        self.e = rho * self.e + i_unknown * phi

        dot_w_e = np.dot(self.w, self.e);
        dot_w_phi = np.dot(self.w, phi);
        gradient = delta * self.e - self.gamma * (1 - self.lambda_back) * dot_w_e * phi_bar

        self.theta += self.alpha * gradient
        self.w += self.alpha * self.eta * (delta * self.e - dot_w_phi * phi)
        self.e *= self.gamma * self.lambda_back;

    def _m_GreedyPolicy(self, state):
        state = tuple(state)
        if np.random.random_sample() < self.epsilon:
            action = rd.choice(self.action_space[state])
            return action

        max_value = -float("inf")
        for action in self.action_space[state]:
            action_value = np.dot(self.theta, self.__state_action_to_feature(state, action))
            if max_value < action_value:
                return_action = action
                max_value = action_value

        return return_action
        
    def __state_action_to_feature(self, state, action):
        """Transfer observation to feature space(only for Mountain Car!) """
        count = 0
        for state_index in self.state_space:
            if state_index != state:
                count += len(list(self.action_space[state_index]))

        count += list(self.action_space[state]).index(action)
        feature = np.zeros(self.num_qfunc)
        feature[count] = 1
        return feature

state_action_space = StateActionSpace(([-1.2, -0.07], [0.6, 0.07]), [20, 20], [0, 1])

learning_agent = GQLambda(learning_rate, discount_factor, eligibility_factor, discount_of_learning_rate, epsilon, state_action_space.action_space)
# print learning_agent.state_space
env = gym.make('MountainCar-v0')

total_reward = 0
Qfunc_difference = []
total_reward_episode = []
for i_episode in range(200):
    observation = env.reset()
    observation_next = observation
    phi = state_action_space._m_state_continuous_to_discret(observation)
    rho = 1

    total_reward = 0
    Qfunc_previous = learning_agent.theta
    for t in range(200):
        # env.render()
        # print(observation)
        action = learning_agent._m_GreedyPolicy(phi)
        while set(observation_next) == set(observation):
            observation_next, step_reward, done, info = env.step(action)  
        phi_bar = state_action_space._m_state_continuous_to_discret(observation_next)
        action = learning_agent._m_GreedyPolicy(phi)
        print phi, phi_bar, step_reward, rho
        learning_agent._m_Learn(phi, phi_bar, step_reward, rho, 1)
        observation = observation_next
        phi = state_action_space._m_state_continuous_to_discret(observation)
#         Qfunc_previous = Qfunc;
#         # print np.dot(Qfunc_previous - Qfunc, Qfunc_previous - Qfunc)
#         total_reward += reward
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
#     total_reward_episode.append(total_reward)
#     Qfunc_difference.append(np.dot(Qfunc_previous - Qfunc, Qfunc_previous - Qfunc))
#     if i_episode % 100 == 0:
#             print i_episode, "th episode completed"

# print Qfunc_difference
# print w

# plt.figure(1)
# plt.subplot(211)
# plt.plot(Qfunc_difference)

# plt.subplot(212)
# plt.plot(total_reward_episode)
# plt.show()

# for i_episode in range(10):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         # print(observation)
#         action = epsilon_greddy_action_choose(observation, Qfunc)
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break