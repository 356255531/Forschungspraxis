import pdb
import numpy as np
import random as rd
from copy import deepcopy


class OSKQ_New(object):
    """docstring for OSKQ_New"""

    def __init__(self,
                 mu_1,
                 mu_2,
                 alpha,
                 gamma,
                 e_lambda,
                 epsilon,
                 action_space,
                 sigma
                 ):
        super(OSKQ_New, self).__init__()
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.alpha = alpha
        self.gamma = gamma
        self.e_lambda = e_lambda
        self.epsilon = epsilon
        self.action_space = action_space
        self.sigma = sigma

        self.dict = np.empty(
            shape=(0, 3)
        )
        self.theta = np.empty(
            shape=(0, 1)
        )
        self.eligibility_trace = np.empty(
            shape=(0, 1)
        )

    def _m_Learn(
            self,
            observation,
            action,
            observation_bar,
            action_bar,
            reward
    ):
        feature_vector = np.append(observation, action)

        feature_vector_bar = np.append(observation_bar, action_bar)

        delta = reward + self.gamma * self.get_Q_Func(feature_vector_bar) - \
            self.get_Q_Func(feature_vector)
        # print feature_vector
        # print "dict", self.dict
        # print self.dict
        # print self.eligibility_trace
        self.eligibility_trace = \
            self.gamma * self.e_lambda * self.eligibility_trace \
            + self.select_function(feature_vector, self.mu_2) * \
            self.gaussian_kernel(feature_vector)
        self.theta += self.alpha * delta * self.eligibility_trace
        self.dict_update(feature_vector)

    def get_Q_Func(
        self,
        feature_vector
    ):
        Q_func = np.sum(
            self.theta * self.select_function(feature_vector, self.mu_2) *
            self.gaussian_kernel(feature_vector)
        )
        return Q_func

    def dict_update(
        self,
        feature_vector
    ):
        already_in_dict = np.sum(self.select_function(feature_vector, self.mu_1))

        # pdb.set_trace()
        if already_in_dict == 0:
            self.dict = np.concatenate(
                (self.dict, feature_vector.reshape((1, 3))),
                axis=0
            )
            self.theta = np.concatenate(
                (self.theta, np.array(np.random.random_sample()).reshape((1, 1))),
                axis=0)
            self.eligibility_trace = np.concatenate(
                (self.eligibility_trace, np.zeros(shape=(1, 1))),
                axis=0)

    def _m_GreedyPolicy(self, observation):
        """
        feature_vector size (3,0)"""
        if np.random.sample < self.epsilon:
            return rd.sample(self.action_space, 1)[0]

        max_Q_func = -float("inf")
        for action in self.action_space:
            feature_vector = np.append(observation, action)
            if_exist_in_dict = np.sum(self.select_function(feature_vector, self.mu_1))
            if not if_exist_in_dict:
                return rd.sample(self.action_space, 1)[0]

            Q_func = self.get_Q_Func(feature_vector)
            if Q_func > max_Q_func:
                # pdb.set_trace()
                max_Q_func = Q_func
                greedy_action = action

        return greedy_action

    def gaussian_kernel(self, feature_vector):
        dict_feature_num = self.dict.shape[0]

        one_vector = np.ones((dict_feature_num, 1))

        minuser = np.dot(one_vector, feature_vector.reshape((1, 3)))

        differ_matrix = self.dict - minuser

        gaussian_kernel_input = np.dot(differ_matrix, differ_matrix.T)

        gaussian_kernel_input = np.diag(gaussian_kernel_input).reshape((dict_feature_num, 1))

        gaussian_kernel = np.exp(
            -gaussian_kernel_input / (self.sigma * self.sigma * 2.0)
        )

        return gaussian_kernel

    def select_function(self, feature_vector, mu):
        selection = (2 - 2 * self.gaussian_kernel(feature_vector)) < mu
        selection = selection.astype(int)
        return selection


if __name__ == '__main__':
    import gym
    from StateActionSpace_MountainCar import StateActionSpace_MountainCar

    # Learning Parameter
    precise = [8, 8]

    learning_rate = 0.1
    discount_factor = 0.9
    discount_of_learning_rate = 0.9
    eligibility_factor = 0.9
    regularize_factor = 0.00003  # 0.0001
    epsilon = 0.1

    # Parameter OSK-Q
    mu_1 = 0.04
    mu_2 = 0.08
    sigma = 1
    # Macro
    NUM_STEP = 200
    NUM_EPISODE = 500
    REWARD_THREASHOLD = -100
    # Definition of dependencies
    env = gym.make('MountainCar-v0')

    observation_space = (
        env.observation_space.low,
        env.observation_space.high
    )

    MountainCar_universal_action_space = [i for i in xrange(0, env.action_space.n)]
    state_action_space = StateActionSpace_MountainCar(
        observation_space,
        precise,
        MountainCar_universal_action_space
    )

    learning_agent_OSKQ = OSKQ_New(
        mu_1,
        mu_2,
        learning_rate,
        discount_factor,
        eligibility_factor,
        epsilon,
        state_action_space.action_space,
        sigma
    )

    # pdb.set_trace()
