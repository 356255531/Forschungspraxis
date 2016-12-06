import numpy as np
import itertools
import sys
import math
import collections
from copy import deepcopy

import pdb

__auther__ = "Zhiwei"


class StateActionSpace_MountainCar_OSKQ(object):
    """
        Define a state-action space for Mountain Car problem

        Member Function:
            _m_observation_to_discrete_state(observation)

            _m_discrete_state_to_feature(state, action)

        Input(constructor):
            observation_bounds: tuple, = (lower_band(list), upper_band(list))

            precise: list,

            actions:list

        Attributes:
            observation_bounds: tuple, = (lower_band(list), upper_band(list))

            precise: list,

            actions:list,

            lower_band: list,

            upper_band: list,

            divide_unit: list,

            state_space: list, [state1(tuple), state2(tuple),.....]

            action_space: dict, {state1(tuple):actions(list), .....}

            element_num_afunc: integer

        """

    def __init__(
        self,
        observation_bounds,
        precise,
        action_space
    ):
        """
        Input:
            observation_bounds: tuple, = (lower_band(list), upper_band(list))
            precise: list,
            actions:list

        Attributes:
            observation_bounds: tuple, = (lower_band(list), upper_band(list))
            precise: list,
            actions:list,
            lower_band: list,
            upper_band: list,
            divide_unit: list,
            state_space: list, [state1(tuple), state2(tuple),.....]
            action_space: list, ununique
            element_num_afunc: integer """
        super(StateActionSpace_MountainCar_OSKQ, self).__init__()
        # Input attributes
        self.__continuous_space = observation_bounds
        self.__precise = precise
        self.__action_space = action_space

        # Build in attributes
        self.__lower_band = list(observation_bounds)[0]
        self.__upper_band = list(observation_bounds)[1]

        self.__divide_unit = self._get_divide_unit()

        self.__state_space = self._derive_state_space()

        self.__observation_dim = len(self.__lower_band)

        self.__state_dim = 2

        self.__action_dim = 1

        self.__feature_vector_dim = self.__state_dim + self.__action_dim

    def get_state_dim(self):
        return self.__state_dim

    def get_action_dim(self):
        return self.__action_dim

    def if_dict_legal(self, input_dict):
        try:
            if input_dict.shape[1] != self.__feature_vector_dim:
                return False
        except IndexError:
            return False
        except AttributeError:
            return False

        return True

    def if_learned_theta_legal(self, learned_theta, input_dict):
        try:
            if learned_theta.shape[1] != 1 or \
                    learned_theta.shape[0] != input_dict[0]:
                return False
        except AttributeError:
            return False
        except IndexError:
            return False

        return True

    def if_observation_legal(self, observation):
        """
            observation -> list
        """
        try:
            if len(observation) != self.__observation_dim or \
                    not isinstance(observation, list):
                return False
        except TypeError:
            return False

        for observation_index in range(len(observation)):
            if not (self.__lower_band[observation_index] <=
                    observation[observation_index] <=
                    self.__upper_band[observation_index]):
                return False

        return True

    def if_state_legal(self, state):
        if state in self.__state_space:
            return True
        else:
            return True

    def if_feature_vector_legal(self, feature_vector):
        try:
            if feature_vector.shape[1] != 1 or \
                    not isinstance(feature_vector, np.ndarray) or \
                    len(feature_vector.shape) != 2:
                return False
        except AttributeError:
            return False
        except IndexError:
            return False

        feature_vector_copy = feature_vector.T[0]
        feature_vector_length = len(feature_vector_copy)
        if feature_vector_length != \
                (self.__state_dim + self.__action_dim):
            return False

        state = list(feature_vector_copy)

        action = [state.pop() for i in xrange(0, self.__action_dim)]

        if len(state) == 1:
            state = state[0]
        else:
            state = tuple(state)

        if len(action) == 1:
            action = action[0]
        else:
            action = tuple(action)

        if (not self.if_state_legal(state)) or \
                (not self.if_action_legal(action)):
            return False

        return True

    def if_action_legal(self, action):
        if action in self.__action_space:
            return True
        else:
            return False

    def _get_divide_unit(self):
        precise = [float(i) for i in self.__precise]
        unit = (
            np.array(self.__upper_band) - np.array(self.__lower_band)
        ) / np.array(precise)
        return unit

    def _derive_state_space(self):
        sub_precise = []
        for i in self.__precise:
            sub_precise.append([j for j in xrange(0, i)])
        temp_discrete_state_space = [i for i in itertools.product(*sub_precise)]
        return temp_discrete_state_space

    def get_state_space(self):
        state_space = deepcopy(self.__state_space)
        return state_space

    def get_action_space(self):
        """
            Hard code:
                generate the action space only for mountain car
        """
        action_space = deepcopy(self.__action_space)
        return action_space

    def trans_observation_to_state(
        self,
        observation
    ):
        """
            Translate a continuous observation to a discrete state

            Input:
                observation: list

            Output:
                state: tuple
        """
        # try:
        #     if len(self.__upper_band) != len(observation):
        #         raise ValueError("The continous state illegal")
        #     for i in xrange(0, len(observation)):
        #         if not(
        #                 self.__lower_band[i] <=
        #                 observation[i] <=
        #                 self.__upper_band[i]
        #         ):
        #             ValueError("The continous state illegal")
        # except ValueError as v_e:
        #     print v_e
        #     sys.exit()

        state = np.array(observation)
        state -= np.array(self.__lower_band)
        state /= self.__divide_unit
        state = [int(math.floor(i)) for i in state]

        for i in xrange(0, len(observation)):
            if observation[i] == self.__upper_band[i]:
                state[i] = self.__precise[i] - 1

        state = tuple(state)

        return state

    def trans_state_action_to_feature_vector(
        self,
        state,
        action
    ):
        """
            Translate a discrete state to a feature vector in
            Mountain Car problem

            Input:
                    state: tuple,
                    action: tuple

            Output:
                feature_vector: np.array
        """
        # try:

        #     if state not in self.state_space:
        #         raise ValueError("Unavailable discrete state in feature vector transition")
        #     if action not in self.action_space[state]:
        #         raise ValueError("Unavailable action in feature vector transition")
        # except ValueError as v_e:
        #     print v_e
        #     sys.exit(0)

        # feature = np.zeros(self.element_num_qfunc)
        # count = 0
        # for discrete_state_index in self.action_space.keys():   # Count the position of state-action map
        #     if discrete_state_index != state:          # and assign 1 to it in sparse feature vector
        #         count += len(self.action_space[discrete_state_index])
        #     else:
        #         for action_index in self.action_space[discrete_state_index]:
        #             if action_index != action:
        #                 count += 1
        #             else:
        #                 feature[count] = 1
        #                 feature = np.array([int(i) for i in feature])
        #                 return feature
        try:
            if (not self.if_state_legal(state)) or \
                    (not self.if_action_legal(action)):
                raise ValueError("state or action illeagal \
                    in feature vector transformation!")
        except ValueError as v_e:
            sys.exit(v_e)

        feature_vector = []

        if isinstance(state, collections.Iterable):
            feature_vector += list(state)
        else:
            feature_vector.append(state)

        if isinstance(action, collections.Iterable):
            feature_vector += list(action)
        else:
            feature_vector.append(action)

        feature_vector = np.array([feature_vector]).T
        return feature_vector


if __name__ == '__main__':
    import gym
    state_action_space = StateActionSpace_MountainCar_OSKQ(
        ([-1.2, -0.07], [0.6, 0.07]),
        [8, 8],
        [0, 1, 2]
    )
    for state in itertools.product(
        [0, 1, 2],
        repeat=2
    ):
        print "discrete state is: ", state
        feature_vector = state_action_space.trans_state_action_to_feature_vector(
            state,
            1
        )
        print feature_vector
    print state_action_space.if_feature_vector_legal(np.array([[2], [4], [0]]))
    # env = gym.make('CartPole-v0')
    # print env.observation_space.low, env.observation_space.high
