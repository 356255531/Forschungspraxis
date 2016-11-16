from abc import ABCMeta, abstractmethod


class T_LearningAgent(object):
    """docstring for T_sta
"""

    __metaclass__ = ABCMeta

    @abstractmethod
    def _m_learn(self):
        pass

    @abstractmethod
    def _m_get_action(self):
        pass

    @abstractmethod
    def _m_get_value_or_Q_function(self):
        pass

    @abstractmethod
    def _m_adjust_learning_parameters(
        self,
        name_parameter,
        expected_value
    ):
        pass
