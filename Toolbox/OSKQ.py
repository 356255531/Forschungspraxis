import numpy as np
import random as rd
import sys
from copy import deepcopy

__auther__ = "Zhiwei"

# from IPython.core.debugger import Tracer


class OSKQ(object):
    """ An agent for learning the parameter using GQ(lambda)
        and perform greedy action selection

      Member function:
            __m_if_learned_dict_legal(learned_dict)

            __m_init_learned_dict()

            __m_if_learned_theta_legal(learned_theta)

            __m_init_theta()

            __m_init_eligibility_trace()

            __m_if_state_legal(state)

            __m_state_illegal_exit(self, state, msg=None)

            __m_if_action_legal(self, action)

            __m_action_illegal_exit(self, action, msg=None)

            __m_if_feature_vector_legal(feature_vector)

            __m_feature_vector_illegal_exit(self, feature_vector, msg=None)

            __m_if_observation_legal(self, observation)

            __m_observation_illegal_exit(self, observation, msg=None)

            __m_trans_observation_to_state(observation)

            __m_trans_state_action_to_feature_vector(state, action)

            __m_add_feature_vector_to_dict(feature_vector)

            __m_kernel_select_func(state)

            __m_gaussian_kernel(phi, phi_bar)

            __m_take_action(action)

            get_epsilon_greedy_action(state, epsilon)

            get_optimal_action(self, state)

            __m_get_Qfunc(state)

            __m_update_dict(state)

            __m_update_eligibility_trace(state, action)

            __m_update_weights(self, delta)

            __m_init_enviroment()

            __m_run_episode(state, greedy, action)

            m_learn(step_reward, state, state_bar, action, greedy_action)

        Attributes:
            __alpha
            __gamma
            __e_factor
            __epsilon
            __mu_1
            __mu_2
            __state_action_space
            __state_space
            __action_space
            __env,
            __num_episode
            __max_step_per_episode
            __reward_threashold
            __alpha
            __gamma
            __e_factor
            __epsilon
            __mu_1
            __mu_2
            __state_action_space
            __state_space
            __action_space
            __env,
            __num_episode
            __max_step_per_episode
            __reward_threashold

            __dict
            __theta
            __feature_vector_dim
            __e
            __toal_reward_history
            __theta_diff_history
            __max_total_reward

        """

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
                 learned_dict=None,
                 learned_theta=None
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
        self.__sigma = self.__sigma
        self.__mu_1 = mu_1
        self.__mu_2 = mu_2
        self.__state_action_space = state_action_space
        self.__state_space = state_action_space.get_state_space()
        self.__action_space = state_action_space.get_actio_space()
        self.__env = env
        self.__num_episode = num_episode
        self.__max_step_per_episode = max_step_per_episode
        self.__reward_threashold = reward_threashold

        if learned_dict is not None:
            if self.__m_if_learned_dict_legal(learned_dict):
                self.__dict = learned_dict
            else:
                self.__dict = self.__m_init_learned_dict()
        else:
            self.__dict = self.__m_init_learned_dict()

        if learned_theta is not None:
            if self.__m_if_learned_theta_legal(learned_theta):
                self.__theta = learned_theta
            else:
                self.__theta = self.__m_init_theta()
        else:
            self.__theta = self.__m_init_theta()

        # Build in paramater initialization
        self.__feature_vector_dim = state_action_space.get_state_dim() + \
            state_action_space.get_action_dim()

        self.__e = self.__m_init_eligibility_trace()

        self.__toal_reward_history = []
        self.__theta_diff_history = []
        self.__max_total_reward = -float("inf")

    def __m_if_learned_dict_legal(self, learned_dict):
        """
            judge if a given dictionary is legal

            input:
                dictionary(np.ndarray)

            return:
                bool
        """
        if_learned_dict_legal = self.__state_action_space.if_dict_legal(
            learned_dict
        )
        return if_learned_dict_legal

    def __m_init_learned_dict(self):
        learned_dict = np.empty(shape=(0, 0))
        return learned_dict

    def __m_if_learned_theta_legal(self, learned_theta):
        if_learned_theta_legal = self.__state_action_space.if_learned_theta_legal(
            learned_theta,
            self.__dict
        )
        return if_learned_theta_legal

    def __m_init_theta(self):
        dict_size = self.__dict.shape[0]
        theta = np.random.random(dict_size)
        return theta

    def __m_init_eligibility_trace(self):
        dict_size = self.__dict.shape[0]
        eligitility_trace = np.zeros(dict_size)
        return eligitility_trace

    def __m_if_state_legal(
        self,
        state
    ):
        """
            judge if a given state is legal

            input:
                state(tuple)

            return:
                bool
        """
        if_legal = \
            self.__state_action_space.if_state_legal(state)

        if if_legal:
            return True
        else:
            return False

    def __m_state_illegal_exit(self, state, msg=None):
        try:
            if not self.__m_if_state_legal(state):
                raise ValueError("State illegal" + msg)
        except ValueError as v_e:
            sys.exit(v_e)

    def __m_if_action_legal(self, action):
        if_action_legal = self.__state_action_space.if_action_legal(
            action
        )
        if if_action_legal:
            return True
        else:
            return False

    def __m_action_illegal_exit(self, action, msg=None):
        try:
            if not self.__m_if_action_legal(action):
                raise ValueError("Action illegal" + msg)
        except ValueError as v_e:
            sys.exit(v_e)

    def __m_if_feature_vector_legal(
        self,
        feature_vector
    ):
        """
            judge if a given feature vector is legal

            input:
                feature_vector(np.ndarray)

            return:
                bool
        """
        if_legal = \
            self.__state_action_space.if_feature_vector_legal(
                feature_vector
            )

        if if_legal:
            return True
        else:
            return False

    def __m_feature_vector_illegal_exit(self, feature_vector, msg=None):
        try:
            if not self.__m_if_feature_vector_legal(feature_vector):
                raise ValueError("Feature vector illegal" + msg)
        except ValueError as v_e:
            sys.exit(v_e)

    def __m_if_observation_legal(self, observation):
        if_observation_legal = self.__state_action_space.if_observation_legal(
            observation
        )
        if if_observation_legal:
            return True
        else:
            return False

    def __m_observation_illegal_exit(self, observation, msg=None):
        try:
            if not self.__m_if_observation_legal(observation):
                raise ValueError("Observation illegal" + msg)
        except ValueError as v_e:
            sys.exit(v_e)

    def __m_trans_observation_to_state(self, observation):
        self.__m_observation_illegal_exit(observation)
        state = self.__state_action_space.trans_observation_to_state(
            observation
        )
        return state

    def __m_trans_state_action_to_feature(
        self,
        state,
        action
    ):
        """
            Use the transform rule defined in state_action_space to transform
            discret state state to feature vector

            input:
                state(tuple)

            return:
                feature_vector(np.ndarray)
        """
        self.__m_state_illegal_exit(state)
        self.__m_action_illegal_exit(action)

        feature_vector = self.__state_action_space.trans_state_to_feature(
            state,
            action
        )
        return feature_vector

    def __m_add_feature_vector_to_dict(
        self,
        feature_vector
    ):
        """
            Add the feature vector into learning dictionary

            input:
                feature vector(np.ndarray)

            transform feature vector to column vecotr(np.mat) and append to the end of
            learning dictionary
        """
        self.__m_feature_vector_illegal_exit(feature_vector)

        learned_dict = np.concatenate((self.__dict, feature_vector.T), axis=0)
        return learned_dict

    def __m_kernel_select_func(
        self,
        phi,
        phi_bar
    ):
        """
            input:
                phi(np.mat), phi_bar(np.mat)

            return:
                return np.mat

        """
        phi_size = phi.shape[0]
        if phi_size > 1:
            self.__m_feature_vector_illegal_exit(phi[0])
            self.__m_feature_vector_illegal_exit(phi_bar)
        else:
            self.__m_feature_vector_illegal_exit(phi)
            self.__m_feature_vector_illegal_exit(phi_bar)

        kernel_select_func_input = phi.dot(phi_bar)
        kernel_select_func = (kernel_select_func_input > self.__mu_1).astype(int)
        return kernel_select_func

    def __m_gaussian_kernel(
        self,
        phi,
        phi_bar
    ):
        phi_size = phi.shape[0]
        if phi_size > 1:
            self.__m_feature_vector_illegal_exit(phi[0])
            self.__m_feature_vector_illegal_exit(phi_bar)
        else:
            self.__m_feature_vector_illegal_exit(phi)
            self.__m_feature_vector_illegal_exit(phi_bar)

        one_vector = np.array([np.ones(self.__feature_vector_dim)]).T
        gaussian_kernel_input = phi - np.dot(one_vector, phi_bar)
        gaussian_kernel_input = np.dot(
            gaussian_kernel_input,
            gaussian_kernel_input.T
        )
        gaussian_kernel = np.exp(-self.__sigma * gaussian_kernel_input.diag)
        return gaussian_kernel

    def __m_take_action(self, action):
        observation_bar, step_reward, done, info = self.__env.step(
            action
        )
        state_var = self.__m_trans_observation_to_state(
            observation_bar
        )
        return state_var, step_reward, done, info

    def get_epsilon_greedy_action(
        self,
        state,
        epsilon
    ):
        """ greedy pick one action according to learned parameters
            with probability of epsilon to pick random action

            Input:
                descret_state: tuple,
                state_action_space: object """
        try:
            if not self.__m_if_state_legal(state):
                raise ValueError("State illegal when get \
                    epsilon-greedy policy")
        except ValueError as v_e:
            sys.exit(v_e)

        if np.random.random_sample() < epsilon:
            action = np.random.choice(self.__action_space)
            return action

        max_value = -float("inf")
        for action in self.__action_space:
            Qfunc = self.__m_get_Qfunc(
                state,
                action
            )
            if max_value < Qfunc:
                return_action = action
                max_value = Qfunc

        return return_action

    def get_optimal_action(self, state):
        action = self.get_epsilon_greedy_action(state, 0)
        return action

    def __m_get_Qfunc(
            self,
            state,
            action
    ):
        self.__m_state_illegal_exit(state)
        self.__m_action_illegal_exit(action)

        feature_vector = self.__m_trans_state_action_to_feature(
            state,
            action
        )

        Qfunc = self.__m_kernel_select_func(self.__dict, feature_vector) * \
            self.__theta * self.__m_gaussian_kernel(
                self.__dict,
                feature_vector
        )

        return Qfunc

    def __m_update_dict(
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
        self.__m_state_illegal_exit(state, "when updating dictionary!")
        self.__m_if_action_legal(action, "when updating dictionary!")

        feature = self.__m_trans_state_action_to_feature(
            state,
            action
        )

        if_feature_in_dict = False
        for dict_ele in self.__learned_dict:
            if self.__m_gaussian_kernel(feature, dict_ele) < self.__mu_1:
                if_feature_in_dict = True

        if not if_feature_in_dict:
            self.__learned_dict = self.__m_add_feature_vector_to_dict(
                self.__learned_dict,
                feature
            )

    def __m_update_eligibility_trace(
        self,
        state,
        action
    ):
        """
            update eligitility trace, usually used in each step

            input:
                state(tuple), action(tuple)

            return:
                new eligibility trace(np.narray)

        """
        self.__m_state_illegal_exit(state, "when updating eligibility trace!")
        self.__m_action_illegal_exit(action, "when updating eligibility trace!")

        feature_vector = self.__m_trans_state_action_to_feature_vector(
            state,
            action
        )

        eligitility_trace = self.__gamma * self.__e_factor * self.__e + \
            self.__m_kernel_select_func(
                self.__dict,
                feature_vector) * self.__m_gaussian_kernel(
                self.__dict,
                feature_vector
            )

        return eligitility_trace

    def __m_update_weights(self, delta):
        theta = self.__theta + self.__alpha * delta * self.__e
        return theta

    def __m_init_enviroment(self):
        observation = self.__env.reset()
        state = self.__m_trans_observation_to_state(observation)
        if_state_legal = self.__m_if_state_legal(state)

        try:
            if not if_state_legal:
                raise ValueError("State illegal in enviroment \
                    initialization")
        except ValueError as v_e:
            sys.exit(v_e)

        state = self.__m_observation_to_state(observation)
        return state

    def __m_run_episode(
        self,
        state,
        greedy_action
    ):
        state_bar, step_reward, done, info = self.__m_take_action(
            greedy_action
        )

        optimal_action = self.get_optimal_action(state_bar)

        delta = step_reward + self.__gamma * self.__m_get_Qfunc(
            state,
            greedy_action
        ) - self.__m_get_Qfunc(
            state_bar,
            optimal_action
        )

        self.__e = self.__m_update_eligibility_trace(delta)
        self.__theta = self.__m_update_weights()

        return state_bar, step_reward

    def m_learn(
        self,
        state,
        env
    ):
        """ Update the learned parameters
            Input:
                phi: integer array, feature vector[0,0,0..1...0,0]
                phi_bar: integer array, feature vector[0,0,0..1...0,0]
                transient_reward:
                 float, reward each step
                rho: float, the probability expectation of action
                i: float, set of interest for s_t, a_t [0,1] """

        for num_episode in self.__num_episode:
            state = self.__m_init_enviroment()
            state_bar = deepcopy(state)

            total_reward = 0
            theta_previous = deepcopy(self.__theta)
            for num_step in range(self.__max_step_per_episode):
                greedy_action = self.get_epsilon_greedy_action(
                    state,
                    self.__epsilon
                )

                self.__m_update_dict(state, greedy_action)

                state_bar, step_reward = self.__m_run_episode(
                    num_episode,
                    state,
                    greedy_action
                )

                state = state_bar

                total_reward += step_reward

            self.__total_reward_history.append(total_reward)
            if total_reward > self.__max_total_reward:
                self.__max_total_reward = total_reward

            theta_diff = theta_previous - self.__theta
            norm_theta_diff = np.linalg.norm(theta_diff)
            self.__theta_diff_history.append(norm_theta_diff)
