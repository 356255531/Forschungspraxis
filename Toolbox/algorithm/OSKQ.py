import numpy as np
import random as rd
import sys
from copy import deepcopy
import pdb

__auther__ = "Zhiwei"

# from IPython.core.debugger import Tracer

class OSKQ(object):
    def __init__(self,
                 mu_1,
                 mu_2,
                 state_action_space,
                 env,
                 alpha=0.1,
                 gamma=0.9,
                 e_factor=0.9,
                 epsilon=0.1,
                 sigma=0.5,
                 num_episode=500,
                 max_step_per_episode=200,
                 reward_threashold=float("inf"),
                 ):
        """
            Input(constructor):
                alpha: float, learning rate,
                gamma: float, discount factor,
                e_factor: float, eligibility factor,
                eta: float,
                epsilon: float, greedy factor
                action_space: dict, {state1(tuple):actions1(list),....} """
        super(OSKQ, self).__init__()
        # Input parameter initialization
        self.__alpha = alpha
        self.__gamma = gamma
        self.__e_factor = e_factor
        self.__epsilon = epsilon
        self.__sigma = sigma
        self.__mu_1 = mu_1
        self.__mu_2 = mu_2
        self.__state_action_space = state_action_space
        self.__state_space = state_action_space.get_state_space()
        self.__action_space = state_action_space.get_action_space()
        self.__env = env
        self.__num_episode = num_episode
        self.__max_step_per_episode = max_step_per_episode
        self.__reward_threashold = reward_threashold

        # Build in paramater initialization
        self.__feature_vector_dim = state_action_space.get_state_dim() + \
            state_action_space.get_action_dim()

        self.__dict = self._init_dict()
        self.__theta = self._init_theta()

        self.__e = self._init_eligibility_trace()

        self.__total_reward_history = []
        self.__theta_diff_history = []
        self.__max_total_reward = -float("inf")

    def _init_dict(self):
        init_dict = np.empty(shape=(0, self.__feature_vector_dim))
        return init_dict

    def _init_theta(self):
        theta_size = self.__dict.shape[0]
        theta = np.empty((theta_size, 1))
        return theta

    def _init_eligibility_trace(self):
        dict_size = self.__dict.shape[0]
        eligitility_trace = np.zeros((dict_size, 1))
        return eligitility_trace

    def _trans_observation_to_state(self, observation):
        state = self.__state_action_space.trans_observation_to_state(
            observation
        )
        return state

    def _trans_state_action_to_feature_vector(
        self,
        state,
        action
    ):
        feature_vector = self.__state_action_space.trans_state_action_to_feature_vector(
            state,
            action
        )
        return feature_vector

    def _add_feature_vector_to_dict(
        self,
        feature_vector
    ):
        learned_dict = np.concatenate((self.__dict, feature_vector.T), axis=0)
        return learned_dict

    def _kernel_select_func(
        self,
        phi,
        phi_bar,
        mu
    ):
        phi_size = phi.shape[0]

        kernel_select_func_input = phi - np.ones((phi_size, 1)).dot(phi_bar.T)
        kernel_select_func_input = kernel_select_func_input.dot(
            kernel_select_func_input.T
        )
        kernel_select_func_input = np.array([np.diag(
            kernel_select_func_input
        )]
        ).T
        kernel_select_func = (
            kernel_select_func_input < (mu * mu)
        ).astype(int)
        return kernel_select_func

    def _gaussian_kernel(
        self,
        phi,
        phi_bar
    ):
        phi_size = phi.shape[0]

        one_vector = np.ones((phi_size, 1))
        gaussian_kernel_input = phi - one_vector.dot(phi_bar.T)
        gaussian_kernel_input = gaussian_kernel_input.dot(
            gaussian_kernel_input.T
        )
        gaussian_kernel_input = np.array([np.diag(
            gaussian_kernel_input
        )]
        ).T
        gaussian_kernel = np.exp(-self.__sigma * gaussian_kernel_input)
        return gaussian_kernel

    def _take_action(self, action):
        observation_bar, step_reward, done, info = self.__env.step(
            action
        )
        state_var = self._trans_observation_to_state(
            observation_bar
        )
        return state_var, step_reward, done, info

    def _if_dict_empty(self):
        if self.__dict.shape[0] == 0:
            return True
        else:
            return False

    def _get_e(self):
        return self.__e

    def get_epsilon_greedy_action(
        self,
        state,
        epsilon
    ):
        # pdb.set_trace()
        if self._if_dict_empty():
            epsilon = 1

        if np.random.random_sample() < epsilon:
            return_action = np.random.choice(self.__action_space)
            return return_action

        max_value = -float("inf")
        for action in self.__action_space:
            Qfunc = self._get_Qfunc(
                state,
                action
            )
            if max_value < Qfunc:
                return_action = action
                max_value = Qfunc

        if 'return_action' not in locals():
            return_action = np.random.choice(self.__action_space)

        return return_action

    def get_optimal_action(self, state):
        action = self.get_epsilon_greedy_action(state, 0)
        return action

    def _get_Qfunc(
            self,
            state,
            action
    ):
        # pdb.set_trace()
        feature_vector = self._trans_state_action_to_feature_vector(
            state,
            action
        )
        # pdb.set_trace()

        try:
            kernel_select_func = self._kernel_select_func(
                self.__dict,
                feature_vector,
                self.__mu_2
            )
        except ValueError:
            print self.__dict
            print feature_vector
            sys.exit(1)

        if np.count_nonzero(kernel_select_func) == 0:
            return -float("inf")

        try:
            Qfunc = kernel_select_func.T.dot(
                self.__theta * self._gaussian_kernel(
                    self.__dict,
                    feature_vector
                )
            )

            Qfunc = np.sum(Qfunc)
        except ValueError:
            pdb.set_trace()
        return Qfunc

    def _update_dict(
        self,
        state,
        action
    ):
        """
            Check whether the discrete state corresponding feature already
            exists in learning dictionary

            input:
                state(tuple)

            intermediate variable:
                feature_vector(np.array)

            no return
        """
        # pdb.set_trace()
        feature_vector = self._trans_state_action_to_feature_vector(
            state,
            action
        )

        kernel_select_func = self._kernel_select_func(
            self.__dict,
            feature_vector,
            self.__mu_1
        )

        if np.count_nonzero(kernel_select_func) == 0:
            self.__dict = self._add_feature_vector_to_dict(
                feature_vector
            )

            try:
                self.__theta = np.concatenate(
                    (self.__theta, np.random.random_sample((1, 1))),
                    axis=0
                )
            except ValueError:
                pdb.set_trace()
            self.__e = np.concatenate(
                (self.__e, np.zeros((1, 1))),
                axis=0
            )

    def _get_dict(self):
        return self.__dict

    def _get_theta(self):
        return self.__theta

    def _update_eligibility_trace(
        self,
        state,
        action
    ):
        """
            update eligitility trace in each step of episode

            input:
                state(tuple), action(tuple)

            return:
                new eligibility trace(np.ndarray)

        """
        feature_vector = self._trans_state_action_to_feature_vector(
            state,
            action
        )

        # pdb.set_trace()
        eligitility_trace = self.__gamma * self.__e_factor * self.__e + \
            self._kernel_select_func(
                self.__dict,
                feature_vector,
                self.__mu_2) * self._gaussian_kernel(
                self.__dict,
                feature_vector
            )
        # pdb.set_trace()
        return eligitility_trace

    def _update_weights(self, delta):
        theta = self.__theta + self.__alpha * delta * self.__e
        return theta

    def _init_enviroment(self):
        observation = self.__env.reset()
        state = self._trans_observation_to_state(observation)
        return state

    def _run_episode(
        self,
        state,
        greedy_action
    ):
        # pdb.set_trace()
        state_bar, step_reward, done, info = self._take_action(
            greedy_action
        )

        optimal_action = self.get_optimal_action(state_bar)

        delta = step_reward + self.__gamma * self._get_Qfunc(
            state,
            greedy_action
        ) - self._get_Qfunc(
            state_bar,
            optimal_action
        )

        self.__e = self._update_eligibility_trace(
            state,
            greedy_action

        )
        # pdb.set_trace()
        self.__theta = self._update_weights(delta)

        return state_bar, step_reward

    def learn(self):
        """ Update the learned parameters
            Input:
                phi: integer array, feature vector[0,0,0..1...0,0]
                phi_bar: integer array, feature vector[0,0,0..1...0,0]
                transient_reward:
                 float, reward each step
                rho: float, the probability expectation of action
                i: float, set of interest for s_t, a_t [0,1] """

        for num_episode in range(self.__num_episode):
            state = self._init_enviroment()
            state_bar = deepcopy(state)

            total_reward = 0
            theta_previous = deepcopy(self.__theta)
            for num_step in range(self.__max_step_per_episode):
                # pdb.set_trace()
                greedy_action = self.get_epsilon_greedy_action(
                    state,
                    self.__epsilon
                )
                # pdb.set_trace()
                self._update_dict(state, greedy_action)
                # pdb.set_trace()
                state_bar, step_reward = self._run_episode(
                    state,
                    greedy_action
                )

                # pdb.set_trace()
                state = state_bar

                total_reward += step_reward
            print self.__theta.shape[0]
            self.__total_reward_history.append(total_reward)
            if total_reward > self.__max_total_reward:
                self.__max_total_reward = total_reward

            # theta_diff = theta_previous - self.__theta
            # norm_theta_diff = np.linalg.norm(theta_diff)
            # self.__theta_diff_history.append(norm_theta_diff)
