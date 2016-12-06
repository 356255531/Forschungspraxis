import gym
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from Toolbox.StateActionSpace import StateActionSpace_MountainCar_OSKQ
from Toolbox.algorithm import OSKQ
__auther__ = "Zhiwei"

# Learning Parameter
precise = [8, 8]

mu_1 = 1
mu_2 = 2
learning_rate = 0.1
discount_factor = 0.9
discount_of_learning_rate = 0.9
eligibility_factor = 0.9
regularize_factor = 0.0001  # 0.0001
epsilon = 0.1
strandard_deviation = 1

NUM_EPISODE = 500
MAX_STEP_PER_EPISODE = 200
REWARD_THREASHOLD = -150
# Definition of dependencies
env = gym.make('MountainCar-v0')

observation_space = (
    env.observation_space.low,
    env.observation_space.high
)

MountainCar_universal_action_space = [i for i in xrange(0, env.action_space.n)]
# state_action_space = StateActionSpace_MountainCar_OSKQ(
#     observation_space,
#     precise,
#     MountainCar_universal_action_space
# )

# learning_agent_OSKQ = OSKQ(
#     mu_1,
#     mu_2,
#     state_action_space,
#     env,
#     learning_rate,
#     discount_factor,
#     eligibility_factor,
#     epsilon,
#     0.5 * strandard_deviation,
#     NUM_EPISODE,
#     MAX_STEP_PER_EPISODE,
#     REWARD_THREASHOLD
# )

# learning_agent = learning_agent_OSKQ
print env.reset() , " "
# learning_agent.learn()

# # Run algorithm
# Qfunc_error_history = []
# total_reward_episode = []
# max_reward = -float("inf")
# for i_episode in range(NUM_EPISODE):
#     observation = env.reset()
#     observation_bar = deepcopy(observation)
#     discret_state = state_action_space._m_observation_to_discrete_state(
#         observation
#     )
#     discret_state_bar = deepcopy(discret_state)

#     total_reward = 0
#     Qfunc_previous = deepcopy(learning_agent.theta)
#     # learning_agent.e = np.zeros(learning_agent.num_element_qfunc)

#     for t in range(200):
#         action = learning_agent._m_GreedyPolicy(
#             discret_state,
#             state_action_space
#         )

#         while set(discret_state) == set(discret_state_bar):
#             observation_bar, step_reward, done, info = env.step(action)

#             if done:
#                 if total_reward > REWARD_THREASHOLD:
#                     learning_agent.epsilon *= 0.999
#                 print "Episode finished after {} timesteps".format(t + 1)
#                 break

#             discret_state_bar = state_action_space._m_observation_to_discrete_state(
#                 observation_bar
#             )

#         learning_agent._m_Learn(
#             step_reward,
#             discret_state,
#             discret_state_bar
#         )

#         observation = observation_bar
#         action = action_bar
#         discret_state = discret_state_bar
#         total_reward += step_reward
#         if done:
#             if total_reward > REWARD_THREASHOLD:
#                 learning_agent.epsilon *= 0.999
#             print "Episode finished after {} timesteps".format(t + 1)
#             break
#     if total_reward > max_reward:
#         max_reward = total_reward

#     total_reward_episode.append(total_reward)   # Add total reward to reward history

#     delta_q_func = Qfunc_previous - learning_agent.theta
#     Qfunc_difference_this_episode = np.dot(
#         delta_q_func,
#         delta_q_func
#     )
#     Qfunc_error_history.append(     # Add error to error history
#         Qfunc_difference_this_episode
#     )

#     if i_episode % 10 == 0:
#         print i_episode, "th episode completed"
#         print "Q update is", Qfunc_difference_this_episode
#         print "Maximal reward is", max_reward, "\n"

# # Plot results
# plt.figure(1)
# plt.subplot(211)
# plt.plot(Qfunc_error_history)

# plt.subplot(212)
# plt.plot(total_reward_episode)
# plt.show()

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
