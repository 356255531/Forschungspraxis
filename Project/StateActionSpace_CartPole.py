import numpy as np
import itertools
import sys
import math
import gym

__auther__ = "Zhiwei"


class StateActionSpace_MountainCar(object):
    """
        Define a state-action space for Mountain Car problem

        Member Function:
            _m_observation_to_discrete_state(observation)

            _m_discrete_state_to_feature(discrete_state, action)

        Input(constructor):
            continuous_space: tuple, = (lower_band(list), upper_band(list))

            precise: list,

            actions:list

        Attributes:
            continuous_space: tuple, = (lower_band(list), upper_band(list))

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
        continuous_space,
        precise,
        actions
    ):
        """
        Input:
            continuous_space: tuple, = (lower_band(list), upper_band(list))
            precise: list,
            actions:list

        Attributes:
            continuous_space: tuple, = (lower_band(list), upper_band(list))
            precise: list,
            actions:list,
            lower_band: list,
            upper_band: list,
            divide_unit: list,
            state_space: list, [state1(tuple), state2(tuple),.....]
            action_space: dict, {state1(tuple):actions(list), .....}
            element_num_afunc: integer """
        super(StateActionSpace_MountainCar, self).__init__()
        self.continuous_space = continuous_space
        self.precise = precise
        self.actions = actions

        self.lower_band = list(continuous_space)[0]
        self.upper_band = list(continuous_space)[1]
        self.divide_unit = self.__get_divide_unit()

        self.state_space = self.__get_discrete_stat_space()
        self.action_space = self.__get_action_space_()
        self.element_num_qfunc = reduce(
            lambda x, y: x + len(y),
            self.action_space.values(),
            0
        )

    def __get_divide_unit(self):
        precise = [float(i) for i in self.precise]
        unit = (
            np.array(self.upper_band) - np.array(self.lower_band)
        ) / np.array(precise)
        return unit

    def __get_discrete_stat_space(self):
        sub_precise = []
        for i in self.precise:
            sub_precise.append([j for j in xrange(0, i)])
        temp_discrete_state_space = [i for i in itertools.product(*sub_precise)]
        return temp_discrete_state_space

    def __get_action_space_(self):
        """
            Hard code:
                generate the action space only for mountain car
        """
        return {i: self.actions for i in self.state_space}

    def _m_observation_to_discrete_state(
        self,
        observation
    ):
        """
            Translate a continuous observation to a discrete state

            Input:
                observation: list

            Output:
                discrete_state: tuple
        """
        try:
            if len(self.upper_band) != len(observation):
                raise ValueError("The continous state illegal")
            for i in xrange(0, len(observation)):
                if not(
                        self.lower_band[i] <=
                        observation[i] <=
                        self.upper_band[i]
                ):
                    ValueError("The continous state illegal")
        except ValueError as v_e:
            print v_e
            sys.exit()

        discrete_state = np.array(observation)
        discrete_state -= np.array(self.lower_band)
        discrete_state /= self.divide_unit
        discrete_state = [int(math.floor(i)) for i in discrete_state]

        for i in xrange(0, len(observation)):
            if observation[i] == self.upper_band[i]:
                discrete_state[i] = self.precise[i] - 1

        discrete_state = tuple(discrete_state)

        return discrete_state

    def _m_discrete_state_to_feature(
        self,
        discrete_state,
        action
    ):
        """
            Translate a discrete state to a feature vector in
            Mountain Car problem

            Input:
                    discrete_state: tuple,
                    action: tuple

            Output:
                feature_vector: np.array
        """
        try:

            if discrete_state not in self.state_space:
                raise ValueError("Unavailable discrete state in feature vector transition")
            if action not in self.action_space[discrete_state]:
                raise ValueError("Unavailable action in feature vector transition")
        except ValueError as v_e:
            print v_e
            sys.exit(0)

        feature = np.zeros(self.element_num_qfunc)
        count = 0
        for discrete_state_index in self.action_space.keys():   # Count the position of state-action map
            if discrete_state_index != discrete_state:          # and assign 1 to it in sparse feature vector
                count += len(self.action_space[discrete_state_index])
            else:
                for action_index in self.action_space[discrete_state_index]:
                    if action_index != action:
                        count += 1
                    else:
                        feature[count] = 1
                        feature = np.array([int(i) for i in feature])
                        return feature


if __name__ == '__main__':
    # state_action_space = StateActionSpace_MountainCar(
    #     ([-1.2, -0.07], [0.6, 0.07]),
    #     [3, 3],
    #     [0, 1]
    # )
    # for discrete_state in itertools.product(
    #     [0, 1, 2],
    #     repeat=2
    # ):
    #     print "discrete state is: ", discrete_state
    #     feature_vector = state_action_space._m_discrete_state_to_feature(
    #         discrete_state,
    #         1
    #     )
    #     print list(feature_vector).index(1)

    env = gym.make('CartPole-v0')
    print env.observation_space.low, env.observation_space.high
