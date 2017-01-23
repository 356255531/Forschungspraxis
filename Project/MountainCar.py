import gym
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import pdb

from Toolbox.algorithm import RGGQLambda, GGQLambda, OSKQ_New
from Toolbox.StateActionSpace import StateActionSpace_MountainCar

__auther__ = "Zhiwei"

# Learning Parameter
precise = [8, 8]

learning_rate = 0.1
discount_factor = 0.9
discount_of_learning_rate = 0.9
eligibility_factor = 0.9
regularize_factor = 0.0001  # 0.0001
epsilon = 0.1

# Parameter OSK-Q
mu_1 = 0.04
mu_2 = 0.08
sigma = 1
# Macro
NUM_STEP = 500
NUM_EPISODE = 500
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

learning_agent_RGGQLambda = RGGQLambda(
    learning_rate,
    discount_of_learning_rate,
    discount_factor,
    eligibility_factor,
    regularize_factor,
    epsilon,
    state_action_space.action_space
)

learning_agent_GGQLambda = GGQLambda(
    learning_rate,
    discount_factor,
    eligibility_factor,
    discount_of_learning_rate,
    epsilon,
    state_action_space.action_space
)
# learning_agent_OSKQ = OSKQ_New(
#     mu_1,
#     mu_2,
#     learning_rate,
#     discount_factor,
#     eligibility_factor,
#     epsilon,
#     [0, 1, 2],
#     sigma
# )

learning_agent = learning_agent_GGQLambda

# Run algorithm
Qfunc_error_history = []
total_reward_episode = []
time_history = []
max_reward = -float("inf")
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
                print "Episode finished after {} timesteps".format(t + 1)
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

        if isinstance(learning_agent, RGGQLambda):  # for adapting the
                                                    # parameter setting
                                                    # of different algorithm
            learning_agent._m_Learn(phi,
                                    phi_bar,
                                    step_reward,
                                    rho,
                                    0.5
                                    )
        else:
            learning_agent._m_Learn(phi,
                                    phi_bar,
                                    step_reward,
                                    step_reward,
                                    rho,
                                    1
                                    )

        observation = observation_bar
        phi = phi_bar
        action = action_bar
        discret_state = discret_state_bar
        total_reward += step_reward
        if done:
            if total_reward > REWARD_THREASHOLD:
                learning_agent.epsilon *= 0.999
            print "Episode finished after {} timesteps".format(t + 1)
            break
    if total_reward > max_reward:
        max_reward = total_reward

    total_reward_episode.append(total_reward)   # Add total reward to reward history

    delta_q_func = Qfunc_previous - learning_agent.theta
    Qfunc_difference_this_episode = np.dot(
        delta_q_func,
        delta_q_func
    )
    Qfunc_error_history.append(     # Add error to error history
        Qfunc_difference_this_episode
    )

    time_end = time.clock()
    time_consumed = time_end - time_start
    time_history.append(time_consumed)

    if i_episode % 10 == 0:
        print i_episode, "th episode completed"
        print "Q update is", Qfunc_difference_this_episode
        print "Maximal reward is", max_reward, "\n"


learning_agent = learning_agent_RGGQLambda

# Run algorithm
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
                print "Episode finished after {} timesteps".format(t + 1)
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

        if isinstance(learning_agent, RGGQLambda):  # for adapting the
                                                    # parameter setting
                                                    # of different algorithm
            learning_agent._m_Learn(phi,
                                    phi_bar,
                                    step_reward,
                                    rho,
                                    0.5
                                    )
        else:
            learning_agent._m_Learn(phi,
                                    phi_bar,
                                    step_reward,
                                    step_reward,
                                    rho,
                                    1
                                    )

        observation = observation_bar
        phi = phi_bar
        action = action_bar
        discret_state = discret_state_bar
        total_reward += step_reward
        if done:
            if total_reward > REWARD_THREASHOLD:
                learning_agent.epsilon *= 0.999
            print "Episode finished after {} timesteps".format(t + 1)
            break
    if total_reward > max_reward:
        max_reward = total_reward

    total_reward_episode_2.append(total_reward)   # Add total reward to reward history

    delta_q_func = Qfunc_previous - learning_agent.theta
    Qfunc_difference_this_episode = np.dot(
        delta_q_func,
        delta_q_func
    )
    Qfunc_error_history_2.append(     # Add error to error history
        Qfunc_difference_this_episode
    )

    time_end = time.clock()
    time_consumed = time_end - time_start
    time_history_2.append(time_consumed)

    if i_episode % 10 == 0:
        print i_episode, "th episode completed"
        print "Q update is", Qfunc_difference_this_episode
        print "Maximal reward is", max_reward, "\n"

# learning_agent = learning_agent_OSKQ
# # Run algorithm
# total_reward_episode_3 = []
# max_reward = -float("inf")
# time_history_3 = []
# for i_episode in range(NUM_EPISODE):
#     time_start = time.clock()
#     observation = env.reset()
#     observation_bar = deepcopy(observation)

#     total_reward = 0
#     # learning_agent.e = np.zeros(learning_agent.num_element_qfunc)
#     # learning_agent.eligibility_trace.fill(0)
#     # pdb.set_trace()
#     # print learning_agent.dict
#     for t in range(NUM_STEP):
#         action = learning_agent._m_GreedyPolicy(observation)
#         while set(observation) == set(observation_bar):
#             observation_bar, step_reward, done, info = env.step(action)
#             if done:
#                 break
#         if done:
#             if total_reward > REWARD_THREASHOLD:
#                 learning_agent.epsilon *= 0.999
#             print "Episode finished after {} timesteps".format(t + 1)
#             break

#         action_bar = learning_agent._m_GreedyPolicy(
#             observation_bar
#         )

#         learning_agent._m_Learn(
#             observation,
#             action,
#             observation_bar,
#             action_bar,
#             step_reward
#         )

#         observation = observation_bar
#         total_reward += step_reward
#         if done:
#             if total_reward > REWARD_THREASHOLD:
#                 learning_agent.epsilon *= 0.999
#             print "Episode finished after {} timesteps".format(t + 1)
#             break
#     if total_reward > max_reward:
#         max_reward = total_reward

#     total_reward_episode_3.append(total_reward)   # Add total reward to reward history

#     time_end = time.clock()
#     time_consumed = time_end - time_start
#     time_history_3.append(time_consumed)

#     if i_episode:
#         print i_episode, "th episode completed"
#         print "dict size", learning_agent_OSKQ.dict.shape
#         # print "Q update is", Qfunc_difference_this_episode
#         print "Maximal reward is", max_reward, "\n"
# Plot results
plt.figure(1)
# plt.subplot(311)
plt.plot(Qfunc_error_history)
plt.plot(Qfunc_error_history_2)
# plt.plot(Qfunc_error_history_2)
plt.show()

plt.figure(2)
# plt.subplot(312)
plt.plot(total_reward_episode)
plt.plot(total_reward_episode_2)
# plt.plot(total_reward_episode_3)
plt.show()

plt.figure(3)
# plt.subplot(313)
plt.plot(time_history)
plt.plot(time_history_2)
# plt.plot(time_history_3)
plt.show()

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
