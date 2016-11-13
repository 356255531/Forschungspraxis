__auther__ = "Zhiwei"
import numpy as np
import itertools
import sys
import math
import copy

class StateActionSpace(object):
    """docstring for StateActionSpace"""
    def __init__(self, continuous_space, precise, actions):
        super(StateActionSpace, self).__init__()
        self.continuous_space = continuous_space
        self.precise = precise
        self.actions = actions
        self.num_zero_entry = reduce(lambda x, y : x * y, precise, 1) * len(actions)

        self.state_space, self.lower_band, self.upper_band, self.unit = self.__discret_space()
        self.action_space = self.__state_action_space_generation()


    def __discret_space(self):
        lower_band = list(self.continuous_space)[0]
        upper_band = list(self.continuous_space)[1]
        precise = [float(i) for i in self.precise]
        unit = (np.array(upper_band) - np.array(lower_band)) / np.array(precise)

        sub_precise = []
        for i in self.precise:
            sub_precise.append([j for j in xrange(0, i)])
        return itertools.product(*sub_precise), lower_band, upper_band, unit


    def __state_action_space_generation(self):
        """ Hard code:
            generate the action space only for mountain car """
        return {i : self.actions for i in self.state_space}


    def _m_state_continuous_to_discret(self, input_state):
        try:
            if len(self.upper_band) != len(input_state):
                raise ValueError("The continous state illegal")
            for i in xrange(0, len(input_state)):
                if not(self.lower_band[i] <= input_state[i] <= self.upper_band[i]):
                    raise ValueError("The continous state illegal")
        except ValueError as e:
            print e
            sys.exit()

        discret_state = input_state
        discret_state = (discret_state - np.array(self.lower_band)) / self.unit
        discret_state = [int(math.floor(i)) for i in discret_state]

        for i in xrange(0, len(input_state)):
            if input_state[i] == self.upper_band[i]:
                discret_state[i] = self.precise[i] - 1
        return discret_state


    def _m_discret_state_to_feature(self, discret_state, action):
        feature = np.zeros(self.num_zero_entry)
        new_state = copy.deepcopy(discret_state)
        new_state.append(action)
        new_precise = self.precise
        new_precise.append(len(self.actions))
        precise = new_precise

        count = 0
        for i, j in zip(new_precise, new_state):
            precise.pop(0)
            count += j * reduce(lambda x, y:x * y, precise, 1)
        try:
            if count > len(feature):
                raise ValueError("Index out of range")
        except ValueError as e:
            print e
            print "Count is ", count
            sys.exit(0)

        feature[count] = 1
        return feature


if __name__ == '__main__':
    state_action_space = StateActionSpace(([-1.2, -0.07], [0.6, 0.07]), [3, 3], [0, 1])
    discret_state = state_action_space._m_state_continuous_to_discret([0, 0.07])
    print discret_state
    feature_vector = state_action_space._m_discret_state_to_feature(discret_state, 0)
    print feature_vector