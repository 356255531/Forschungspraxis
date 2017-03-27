import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.legend_handler import HandlerLine2D
from copy import deepcopy
import time
import pdb

from Toolbox.algorithm import RGGQLambda, GGQLambda, OSKQ_New
from Toolbox.StateActionSpace import StateActionSpace_MountainCar

__auther__ = "Zhiwei"


def same_parameter_run(eligibility_factor=0.9, mu_2=0.8):
    # Learning Parameter
    precise = [8, 8]

    learning_rate = 0.1
    discount_factor = 0.9
    discount_of_learning_rate = 0.999
    regularize_factor = 0.0003  # 0.0001
    epsilon = 0.1

    # Parameter OSK-Q
    mu_1 = 0.04
    sigma = 1
    # Macro
    NUM_STEP = 1000
    NUM_EPISODE = 500
    AVE_TIMES = 20
    REWARD_THREASHOLD = -100
    # Definition of dependencies
    env = gym.make('MountainCar-v0')

    observation_space = (
        env.observation_space.low,
        env.observation_space.high
    )

    MountainCar_universal_action_space = [i for i in xrange(0, env.action_space.n)]
    state_action_space = StateActionSpace_MountainCar(
        observation_space,
        precise,
        MountainCar_universal_action_space
    )

    # Run algorithm
    for ave_times in range(AVE_TIMES):
        learning_agent_RGGQLambda = RGGQLambda(
            learning_rate,
            discount_of_learning_rate,
            discount_factor,
            eligibility_factor,
            regularize_factor,
            epsilon,
            state_action_space.action_space
        )
        learning_agent = learning_agent_RGGQLambda
        # pdb.set_trace()

        Qfunc_error_history_2 = []
        total_reward_episode_2 = []
        max_reward = -float("inf")
        time_history_2 = []
        for i_episode in range(NUM_EPISODE):
            time_start = time.clock()
            observation = env.reset()
            observation_bar = deepcopy(observation)
            discret_state = state_action_space._m_observation_to_discrete_state(
                observation
            )
            discret_state_bar = deepcopy(discret_state)

            action = learning_agent._m_GreedyPolicy(
                discret_state,
                state_action_space
            )

            phi = state_action_space._m_discrete_state_to_feature(
                discret_state,
                action
            )

            rho = 1

            total_reward = 0
            Qfunc_previous = deepcopy(learning_agent.theta)
            learning_agent.e = np.zeros(learning_agent.num_element_qfunc)

            for t in range(NUM_STEP):
                while set(discret_state) == set(discret_state_bar):
                    observation_bar, step_reward, done, info = env.step(action)

                    if done:
                        if total_reward > REWARD_THREASHOLD:
                            learning_agent.epsilon *= 0.999
                        print "Episode finished after {} timesteps in RGGQ(lambda)".format(t + 1), "in ", ave_times + 1, "times"
                        break

                    discret_state_bar = state_action_space._m_observation_to_discrete_state(
                        observation_bar
                    )

                action_bar = learning_agent._m_GreedyPolicy(
                    discret_state_bar,
                    state_action_space
                )
                phi_bar = state_action_space._m_discrete_state_to_feature(
                    discret_state_bar,
                    action_bar
                )

                learning_agent._m_Learn(phi,
                                        phi_bar,
                                        step_reward,
                                        rho,
                                        0.5
                                        )

                observation = observation_bar
                phi = phi_bar
                action = action_bar
                discret_state = discret_state_bar
                total_reward += step_reward
                if done:
                    if total_reward > REWARD_THREASHOLD:
                        learning_agent.epsilon *= 0.999
                    print "Episode finished after {} timesteps in RGGQ(lambda)".format(t + 1), "in ", ave_times + 1, "times"
                    break
        return np.sum(np.array(learning_agent.theta == 0, int)) / (learning_agent.num_element_qfunc * 1.0)


def main():
    q_func = same_parameter_run()
    print q_func


if __name__ == '__main__':
    main()
# # Run learned policy an simulator
# for i_episode in range(10):
#     observation = env.reset()

#     for t in range(200):
#         env.render()

#         discret_state = state_action_space._m_observation_to_discrete_state(
#             observation
#         )
#         action = learning_agent._m_GreedyPolicy(
#             discret_state,
#             state_action_space
#         )

#         observation, reward, done, info = env.step(action)

#         if done:
#             print("Episode finished after {} timesteps".format(t + 1))
#             break
