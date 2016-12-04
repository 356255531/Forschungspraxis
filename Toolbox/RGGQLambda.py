import numpy as np
import random as rd
import sys
from copy import deepcopy

__auther__ = "Zhiwei"

# from IPython.core.debugger import Tracer


class RGGQLambda(object):
    """ An agent for learning the parameter using GQ(lambda)
        and perform greedy action selection

        Input(constructor):
                 alpha: float, learning rate,
                 gamma: float, discount factor,
                 lambda_back: float, eligibility factor,
                 eta: float,
                 epsilon: float, greedy factor
                 action_space: dict, {state1(tuple):actions1(list),....}

        Attributes:
                 alpha: float, learning rate,
                 gamma: float, discount factor,
                 lambda_back: float, eligibility factor,
                 eta: float,
                 epsilon: float, greedy factor
                 action_space: dict, {state1(tuple):actions1(list),....}
                 num_element_qfunc: integer, number of elements in
                                    feature vector
                 theta: float array, learned parameters
                 w: float array, intermedia vector
                 e: float array, eligibility trace
        Function:
            _m_Learn: learning the parameters
            _m_GreedyPolicy: greedy pick one action according to
                             learned parameters with probability of
                             epsilon to pick random action
        """

    def __init__(self,
                 alpha,
                 eta,
                 gamma,
                 lambda_back,
                 reguarize_rho,
                 epsilon,
                 action_space
                 ):
        """
            Input(constructor):
                alpha: float, learning rate,
                gamma: float, discount factor,
                lambda_back: float, eligibility factor,
                eta: float,
                epsilon: float, greedy factor
                action_space: dict, {state1(tuple):actions1(list),....} """
        super(RGGQLambda, self).__init__()

        self.alpha = alpha
        self.eta = eta
        self.gamma = gamma
        self.lambda_back = lambda_back
        self.reguarize_rho = reguarize_rho
        self.epsilon = epsilon
        self.state_space = action_space.keys()
        self.action_space = deepcopy(action_space)

        self.num_element_qfunc = reduce(
            lambda x, y: x + len(y),
            self.action_space.values(),
            0
        )
        self.theta = np.random.rand(self.num_element_qfunc)
        # self.theta = np.zeros(self.num_element_qfunc)
        self.w, self.e, self.x, self.y = self.__parameter_init()

    def __parameter_init(self):
        w = np.zeros(self.num_element_qfunc)
        e = np.zeros(self.num_element_qfunc)
        x = np.array(list(w) + list(self.theta))
        y = np.zeros(self.num_element_qfunc * 2)
        return w, e, x, y

    def _m_Learn(self,
                 phi,
                 phi_bar,
                 step_reward,
                 rho,
                 i
                 ):
        """ Update the learned parameters
            Input:
                phi: integer array, feature vector[0,0,0..1...0,0]
                phi_bar: integer array, feature vector[0,0,0..1...0,0]
                transient_reward:
                 float, reward each step
                rho: float, the probability expectation of action
                i: float, set of interest for s_t, a_t [0,1] """
        try:
            if (
                len(self.theta) != len(phi_bar) or
                len(self.theta) != len(phi)
            ):
                raise ValueError("Dimension not same")
        except ValueError as e:
            print e
            print "theta dimension is", len(self.theta)
            print "phi_bar dimension is", len(phi_bar)
            sys.exit(0)

        delta = step_reward + self.lambda_back * np.dot(phi_bar, self.theta) - np.dot(phi, self.theta)
        # print delta
        self.e = rho * self.e + i * phi

        y_1, y_2 = np.split(self.y, 2)
        y_transposed_A_1 = self.eta * phi * np.dot(y_1, phi) + self.gamma * (1 - self.lambda_back) * self.e * np.dot(y_2, phi_bar)
        y_transposed_A_2 = (phi - self.gamma * phi_bar) * np.dot(
            self.eta * y_1 + y_2,
            self.e
        )
        y_transposed_A = np.array(
            list(y_transposed_A_1) + list(y_transposed_A_2)
        )

        A_x_minus_b_1 = -self.eta * (delta * self.e - np.dot(phi, self.w) * phi)
        A_x_minus_b_2 = self.gamma * (1 - self.lambda_back) * np.dot(self.e, self.w) * phi_bar - delta * self.e
        A_x_minus_b = np.array(
            list(A_x_minus_b_1) + list(A_x_minus_b_2)
        )

        x_half = self.x - self.alpha * y_transposed_A
        y_half = self.y + self.alpha * A_x_minus_b

        self.x = self.__proximal(x_half, A_x_minus_b)
        # print self.x
        # sys.exit(0)
        self.y = self.__update_y(y_half)

        self.w, self.theta = np.split(self.x, 2)

        # Tracer()()
        self.e *= self.gamma * self.lambda_back

    def __proximal(self, x_half, A_x_minus_b):
        x = x_half - self.alpha * A_x_minus_b
        x = [max(i - self.reguarize_rho, 0) - max(-i - self.reguarize_rho, 0) for i in x]
        return np.array(x)

    def __update_y(self, y_half):
        second_norm = np.linalg.norm(y_half, ord=2)
        return min(1, 1 / second_norm) * y_half

    def _m_GreedyPolicy(
        self,
        discret_state,
        state_action_space
    ):
        """ greedy pick one action according to learned parameters
            with probability of epsilon to pick random action

            Input:
                descret_state: tuple,
                state_action_space: object """
        discret_state = tuple(discret_state)
        if np.random.random_sample() < self.epsilon:
            action = rd.choice(
                state_action_space.action_space[discret_state]
            )
            return action

        max_value = -float("inf")
        for action in state_action_space.action_space[discret_state]:
            action_value = np.dot(
                self.theta,
                state_action_space._m_discrete_state_to_feature(
                    discret_state,
                    action
                )
            )
            if max_value < action_value:
                return_action = action
                max_value = action_value

        return return_action
