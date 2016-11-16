import gym
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import sys

from RGGQLambda import RGGQLambda
from StateActionSpace_MountainCar import StateActionSpace_MountainCar

__auther__ = "Zhiwei"

# from IPython.core.debugger import Tracer
############# Learning Parameter ##############
precise = [8, 8]

learning_rate = 0.1
discount_of_learning_rate = 0.9
discount_factor = 0.9
eligibility_factor = 0.9
regularize_factor = 0.0003
epsilon = 0.1
###############################################
state_action_space = StateActionSpace_MountainCar(
    ([-1.2, -0.07], [0.6, 0.07]),
    precise,
    [0, 2]
)

learning_agent = RGGQLambda(
    learning_rate,
    discount_of_learning_rate,
    discount_factor,
    eligibility_factor,
    regularize_factor,
    epsilon,
    state_action_space.action_space
)
# print learning_agent.state_space
env = gym.make('MountainCar-v0')

Qfunc_difference = []
total_reward_episode = []
max_reward = -float("inf")
for i_episode in range(2000):
    observation = env.reset()
    # count = 0
    # while 1:
    #     observation_bar, _, _, _ = env.step(1)
    #     print observation_bar
    #     count += 1
    #     if count == 100:
    #         sys.exit(0)
    observation_bar = deepcopy(observation)
    discret_state = state_action_space._m_observation_to_discrete_state(
        observation
    )
    discret_state_bar = discret_state

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
    # Tracer()()
    for t in range(500):
        # env.render()
        same_action_count = 0
        while set(discret_state) == set(discret_state_bar):
            observation_bar, step_reward, done, info = env.step(action)
            discret_state_bar = state_action_space._m_observation_to_discrete_state(
                observation_bar
            )
        # Tracer()()
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
#         # print np.dot(Qfunc_previous - Qfunc, Qfunc_previous - Qfunc)
        total_reward += step_reward
        if done:
            if total_reward > -150:
                learning_agent.epsilon *= 0.999
            print "Episode finished after {} timesteps".format(t + 1)
            break
    if total_reward > max_reward:
        max_reward = total_reward
    total_reward_episode.append(total_reward)
    Qfunc_difference_this_episode = np.dot(
        Qfunc_previous - learning_agent.theta,
        Qfunc_previous - learning_agent.theta
    )
    Qfunc_difference.append(
        Qfunc_difference_this_episode
    )

    Qfunc_previous = deepcopy(learning_agent.theta)
    if i_episode % 10 == 0:
        print i_episode, "th episode completed"
        print "Q update is", Qfunc_difference_this_episode
        print "Maximal reward is", max_reward, "\n"

# print Qfunc_difference
# print w

print learning_agent.theta
plt.figure(1)
plt.subplot(211)
plt.plot(Qfunc_difference)

plt.subplot(212)
plt.plot(total_reward_episode)
plt.show()

env = gym.make('MountainCar-v0')
for i_episode in range(10):
    observation = env.reset()

    for t in range(200):
        env.render()
        # print(observation)
        discret_state = state_action_space._m_observation_to_discrete_state(
            observation
        )
        action = learning_agent._m_GreedyPolicy(
            discret_state,
            state_action_space
        )
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
