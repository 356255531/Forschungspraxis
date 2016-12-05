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
            __m_init_learned_dict()

            __m_init_theta()

            __m_init_eligibility_trace()

            __m_if_state_legal(state)

            __m_if_feature_vector_legal(feature_vector)

            __m_gaussian_kernel(phi, phi_bar)

            __m_trans_state_to_feature_vector(state, action)

            __m_add_feature_vector_to_dict(feature_vector)

            __m_update_dict(state)

            __m_kernel_select_func(state)

            __m_get_Qfunc(state)

            __m_update_parameters()

            __m_update_eligibility_trace(delta)

            __m_Learn(step_reward, state, state_bar, action, greedy_action)

            get_greedy_action(state, state_action_space)

        Input(constructor):
                 alpha: float, learning rate,
                 gamma: float, discount factor,
                 e_factor: float, eligibility factor,
                 eta: float,
                 epsilon: float, greedy factor
                 action_space: dict, {state1(tuple):actions1(list),....}

        Attributes:
                 alpha: float, learning rate,
                 gamma: float, discount factor,
                 e_factor: float, eligibility factor,
                 eta: float,
                 epsilon: float, greedy factor
                 action_space: dict, {state1(tuple):actions1(list),....}
                 num_element_qfunc: integer, number of elements in
                                    feature vector
                 theta: float array, learned parameters
                 w: float array, intermedia vector
                 e: float array, eligibility trace

        """

    def __init__(self,
                 alpha,
                 gamma,
                 e_factor,
                 epsilon,
                 mu_1,
                 mu_2,
                 state_action_space,
                 learned_dict=None,
                 theta=None
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
        self.__mu_1 = mu_1
        self.__mu_2 = mu_2
        self.__state_action_space = state_action_space
        self.__state_space = state_action_space.get_state_space()
        self.__action_space = state_action_space.get_actio_space()

        if learned_dict is not None:
            if self.__m_if_learned_dict_legal(learned_dict):
                self.__learned_dict = learned_dict
            else:
                self.__learned_dict = self.__m_init_learned_dict()
        else:
            self.__learned_dict = self.__m_init_learned_dict()

        if theta is not None:
            if self.__m_if_theta_legal(theta):
                self.__theta = theta
            else:
                self.__theta = self.__m_init_theta()
        else:
            self.__theta = self.__m_init_theta()

        # Build in paramater initialization
        self.__feature_vector_dim = state_action_space.get_state_dim() + \
            state_action_space.get_action_dim()

        self.__e = self.__m_init_eligibility_trace()

    def __m_init_learned_dict(self):
        learned_dict = np.empty(shape=(0, 0))
        return learned_dict

    def __m_init_theta(self):
        pass

    def __m_init_eligibility_trace(self):
        eligitility_trace_dim = self.__learned_dict.shape[0]
        eligitility_trace = np.zeros(eligitility_trace_dim)
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
        if not isinstance(feature_vector, np.ndarray):
            return False
        if len(feature_vector) != self.__learned_dict.shape[0]:
            return False
        return True

    def __m_gaussian_kernel(
            self,
            phi,
            phi_bar):
        """
            return gaussian kernel

            input:
                phi(np.array), phi(np.array), feature vector

            return:
                gaussian kernel value(float)
        """
        phi_diff = phi - phi_bar
        squared_norm = np.dot(phi_diff, phi_diff)

        gaussian_kernel = np.exp(-self.__sigma * squared_norm)
        return gaussian_kernel

    def __m_kernel_select_func(
        self,
        state
    ):

        if np.linalg.norm(state_diff) < self.__min_kernel_dist:
            return 1
        else:
            return 0

    def __m_trans_state_action_to_feature(
        self,
        state
    ):
        """
            Use the transform rule defined in state_action_space to transform
            discret state state to feature vector

            input:
                state(tuple)

            return:
                feature_vector(np.ndarray)
        """
        feature_vector = self.__state_action_space.trans_state_to_feature(state)
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
        try:
            if not self.__m_if_feature_vector_legal(feature_vector):
                raise IndexError("Feature vector illegal when adding \
                    feature vector to dictionary")
        except IndexError as index_error:
            print index_error
            sys.exit(1)

        feature_vector = np.mat(feature_vector).transpose()
        self.__learned_dict = np.append(
            self.__learned_dict,
            feature_vector,
            axis=1
        )

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
        try:
            if self.__m_if_state_legal(state):
                raise IndexError("State illegal when updating dictionary!")
        except IndexError as index_error:
            print index_error
            sys.exit(1)

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

    def __m_get_Qfunc(
            self,
            state,
            action
    ):
        pass

    def __m_update_parameter(
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
        feature_vector = self.__m_trans_state_action_to_feature_vector(
            state,
            action
        )

        eligitility_trace = self.__gamma * self.__e_factor * self.__e + \
            self.__m_kernel_select_func(self.__learned_dict, feature_vector) * \
            self.__m_gaussian_kernel(self.__learned_dict, feature_vector)

        return eligitility_trace

    def __m_update_eligibility_trace(self, delta):
        theta = self.__theta + self.__alpha * delta * self.__e
        return theta

    def _m_Learn(self,
                 step_reward,
                 state,
                 state_bar,
                 action,
                 greedy_action
                 ):
        """ Update the learned parameters
            Input:
                phi: integer array, feature vector[0,0,0..1...0,0]
                phi_bar: integer array, feature vector[0,0,0..1...0,0]
                transient_reward:
                 float, reward each step
                rho: float, the probability expectation of action
                i: float, set of interest for s_t, a_t [0,1] """
        self.__m_update_dict(state)

        greedy_action = self.get_greedy_action(state)

        state_bar, step_reward, done, info = self.__m_take_action(action)

        greedy_action_bar = self.get_greedy_action(state_bar)

        delta = step_reward + self.gamma * self.__m_get_Qfunc(
            state,
            greedy_action
        ) - self.__m_get_Qfunc(
            state_bar,
            greedy_action_bar
        )

        self.__e = self.__m_update_eligibility_trace(delta)
        self.__theta = self.__m_update_parameter()

    def get_greedy_action(
        self,
        state,
        state_action_space
    ):
        """ greedy pick one action according to learned parameters
            with probability of epsilon to pick random action

            Input:
                descret_state: tuple,
                state_action_space: object """
        state = tuple(state)
        if np.random.random_sample() < self.epsilon:
            action = rd.choice(
                state_action_space.action_space[state]
            )
            return action

        max_value = -float("inf")
        for action in state_action_space.action_space[state]:
            action_value = self.__m_get_Qfunc(
                state,
                action
            )
            if max_value < action_value:
                return_action = action
                max_value = action_value

        return return_action
