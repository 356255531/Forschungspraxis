from abc import ABCMeta, abstractmethod


class T_StateActionSpace(object):
    """docstring for T_sta
"""

    __metaclass__ = ABCMeta

    @abstractmethod
    def _m_get_state_space(self):
        pass

    @abstractmethod
    def _m_get_action_space(self):
        pass

    @abstractmethod
    def _m_observation_to_discrete_state(
        self,
        observation
    ):
        pass

    @abstractmethod
    def _m_discrete_state_to_feature(
        self,
        discrete_state,
        action=None
    ):
        pass

    @abstractmethod
    def _m_observation_to_feature(
        self,
        observation,
        action=None
    ):
        pass
