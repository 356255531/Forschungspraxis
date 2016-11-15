import numpy as np
import random as rd
import sys
from copy import deepcopy

__auther__ = "Zhiwei"

# from IPython.core.debugger import Tracer


class GQLambda(object):
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
                 gamma,
                 lambda_back,
                 eta,
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
        super(GQLambda, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.lambda_back = lambda_back
        self.eta = eta
        self.epsilon = epsilon
        self.state_space = action_space.keys()
        self.action_space = deepcopy(action_space)

        self.num_element_qfunc = reduce(
            lambda x, y: x + len(y),
            self.action_space.values(),
            0
        )
        self.theta = np.random.rand(self.num_element_qfunc)
        self.w = np.zeros(self.num_element_qfunc)
        self.e = np.zeros(self.num_element_qfunc)

    def _m_Learn(self,
                 phi,
                 phi_bar,
                 transient_reward,
                 outcome_reward,
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

        delta = transient_reward + (1 - self.gamma) * outcome_reward + self.gamma * np.dot(self.theta, phi_bar) - np.dot(self.theta, phi)

        self.e = rho * self.e + i * phi

        # Tracer()()
        self.theta += self.alpha * (delta * self.e - self.gamma * (1 - self.lambda_back) * np.dot(self.w, self.e) * phi_bar)
        self.w += self.alpha * self.eta * (delta * self.e - np.dot(self.w, phi) * phi)
        self.e *= self.gamma * self.lambda_back

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
