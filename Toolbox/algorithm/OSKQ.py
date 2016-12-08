import numpy as np
import numpy.random as rd
import sys
from copy import deepcopy
import pdb

__auther__ = "Zhiwei"


class OSKQ(object):
    def __init__(self,
                 mu_1,
                 mu_2,
                 state_action_space,
                 env,
                 alpha=0.1,
                 gamma=0.9,
                 e_factor=0.9,
                 epsilon=0.3,
                 sigma=1,
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
        self.__state_space = \
            state_action_space.get_state_space()
        self.__action_space = \
            state_action_space.get_action_space()
        self.__env = env
        self.__num_episode = num_episode
        self.__max_step_per_episode = max_step_per_episode
        self.__reward_threashold = reward_threashold

        # Build in paramater initialization
        self.__feature_vector_dim = \
            state_action_space.get_feature_vector_dim()

        self.__dict = self._init_dict()
        self.__theta = self._init_theta()

        self.__e = self._init_eligibility_trace()

        self.__total_reward_history = []
        self.__theta_diff_history = []
        self.__max_total_reward = -float("inf")

    def _init_dict(self):
        init_dict = np.empty(
            shape=(0, self.__feature_vector_dim)
        )
        return init_dict

    def _init_theta(self):
        theta_size = self.__dict.shape[0]
        theta = np.empty((theta_size, 1))
        return theta

    def _init_eligibility_trace(self):
        dict_size = self.__dict.shape[0]
        eligitility_trace = np.zeros(
            (dict_size, 1)
        )
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
        self.__dict = np.concatenate(
            (self.__dict, feature_vector.T),
            axis=0
        )
        self.__e = np.concatenate(
            (self.__e, np.zeros((1, 1))),
            axis=0
        )
        self.__theta = np.concatenate(
            (self.__theta, rd.random_sample((1, 1))),
            axis=0
        )

    def _kernel_select_func(
        self,
        phi,
        phi_bar,
        mu
    ):
        dist = 2 - 2 * self._gaussian_kernel(phi, phi_bar)
        kernel_select_func = (dist < mu).astype(int)

        return kernel_select_func

    def _gaussian_kernel(
        self,
        phi,
        phi_bar
    ):
        phi_size = phi.shape[0]

        one_vector = np.ones((phi_size, 1))
        gaussian_kernel_input = phi - one_vector.dot(
            phi_bar.T
        )
        gaussian_kernel_input = gaussian_kernel_input.dot(
            gaussian_kernel_input.T
        )
        gaussian_kernel_input = np.array([np.diag(
            gaussian_kernel_input
        )]
        ).T
        gaussian_kernel = np.exp(
            -gaussian_kernel_input / (self.__sigma * self.__sigma * 2.0)
        )
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
        if rd.random_sample() < epsilon or \
                self._if_dict_empty():
            action = self._get_random_action()
            return action

        max_value = -float("inf")
        for action in self.__action_space:
            # pdb.set_trace()
            Qfunc = self._get_Qfunc(
                state,
                action
            )
            if Qfunc is None:
                action = self._get_random_action()
                return action
            if max_value < Qfunc:
                return_action = deepcopy(action)
                max_value = Qfunc

        if 'return_action' not in locals():
            return_action = rd.choice(
                self.__action_space
            )

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
        # pdb.set_trace()
        feature_vector = self._trans_state_action_to_feature_vector(
            state,
            action
        )
        # pdb.set_trace()

        kernel_select_func = self._kernel_select_func(
            self.__dict,
            feature_vector,
            self.__mu_2
        )

        if np.count_nonzero(kernel_select_func) == 0:
            self._add_feature_vector_to_dict(feature_vector)
            kernel_select_func = self._kernel_select_func(
                self.__dict,
                feature_vector,
                self.__mu_2
                )

        gaussian_kernel = self._gaussian_kernel(
            self.__dict,
            feature_vector
        )

        Qfunc = kernel_select_func.T.dot(
            self.__theta * gaussian_kernel
        )

        Qfunc = np.sum(Qfunc)

        return Qfunc

    def _update_dict(self, state, action):
        feature_vector = self._trans_state_action_to_feature_vector(
            state, action
        )

        if_feature_vector_redundent = self._kernel_select_func(
            self.__dict,
            feature_vector,
            self.__mu_1
        )
        # pdb.set_trace()
        if np.sum(if_feature_vector_redundent) == 0:
            for action in self.__action_space:
                feature_vector = self._trans_state_action_to_feature_vector(
                    state, action
                )
                self._add_feature_vector_to_dict(feature_vector)

    def _get_dict(self):
        return self.__dict

    def _get_theta(self):
        return self.__theta

    def _update_eligibility_trace(
        self,
        state,
        action
    ):
        feature_vector = self._trans_state_action_to_feature_vector(
            state,
            action
        )

        # pdb.set_trace()
        eligitility_trace = self.__gamma * self.__e_factor * self.__e + \
            self._kernel_select_func(
                self.__dict,
                feature_vector,
                self.__mu_2) * \
            self._gaussian_kernel(
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
        epsilon_greedy_action
    ):
        # print epsilon_greedy_action
        # pdb.set_trace()
        # pdb.set_trace()
        state_bar, step_reward, done, info = self._take_action(
            epsilon_greedy_action
        )

        if done:
            return state_bar, step_reward, done

        # pdb.set_trace()
        optimal_action = self.get_optimal_action(state_bar)
        state_greedy_action_Qfunc = self._get_Qfunc(
            state,
            epsilon_greedy_action
        )

        state_optimal_action_Qfunc = self._get_Qfunc(
            state_bar,
            optimal_action
        )

        # pdb.set_trace()
        delta = step_reward + self.__gamma * state_optimal_action_Qfunc - \
            state_greedy_action_Qfunc

        self.__e = self._update_eligibility_trace(
            state,
            epsilon_greedy_action

        )
        # pdb.set_trace()
        self.__theta = self._update_weights(delta)

        return state_bar, step_reward, done

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
            print "Episode: ", num_episode

            state = self._init_enviroment()

            total_reward = 0
            theta_previous = deepcopy(self.__theta)
            # pdb.set_trace()

            for num_step in range(self.__max_step_per_episode):
                # pdb.set_trace()
                epsilon_greedy_action = self.get_epsilon_greedy_action(
                    state,
                    self.__epsilon
                )

                self._update_dict(state, epsilon_greedy_action)

                state_bar, step_reward, done = self._run_episode(
                    state,
                    epsilon_greedy_action
                )
                if done:
                    break

                # pdb.set_trace()
                state = state_bar

                total_reward += step_reward

            print total_reward
            if total_reward != -500:
                print "FuckFuckFuckFuckFuckFuckFuckFuckFuckFuckFuckFuckFuck"
            print self.__dict.shape[0]
            if (num_episode + 1) % 500 == 0:
                pdb.set_trace()
            # self.__total_reward_history.append(total_reward)
            # if total_reward > self.__max_total_reward:
                # self.__max_total_reward = total_reward

            if theta_previous.shape == self.__theta.shape:
                theta_diff = theta_previous - self.__theta
                norm_theta_diff = np.linalg.norm(theta_diff)
            if 'norm_theta_diff' in locals():
                print norm_theta_diff
            # print nortd
            # self.__theta_diff_history.append(norm_theta_diff)

    def _get_random_action(self):
        action = rd.choice(self.__action_space)
        return action
