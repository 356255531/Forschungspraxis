__auther__ = "Zhiwei"
import numpy as np
import random as rd
import sys

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
        try:
            if len(self.theta) != len(phi_bar):
                raise ValueError("Dimension not same")
        except ValueError as e:
            print e
            print "theta dimension is", len(self.theta)
            print "phi_bar dimension is", len(phi_bar)
            sys.exit(0)
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