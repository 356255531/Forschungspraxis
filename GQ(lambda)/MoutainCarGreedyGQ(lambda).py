import gym
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from GQLambda import GQLambda
from StateActionSpace import StateActionSpace
#############################
from IPython.core.debugger import Tracer
#############

__auther__ = "Zhiwei"

precise = [20, 20]

learning_rate = 0.01
discount_factor = 0.9
eligibility_factor = 0.9
discount_of_learning_rate = 0.9
epsilon = 0.1
###################
state_action_space = StateActionSpace(
    ([-1.2, -0.07], [0.6, 0.07]),
    precise,
    [0, 1]
)

learning_agent = GQLambda(
    learning_rate,
    discount_factor,
    eligibility_factor,
    discount_of_learning_rate,
    epsilon,
    state_action_space.action_space
)
# print learning_agent.state_space
env = gym.make('MountainCar-v0')

Qfunc_difference = []
total_reward_episode = []
for i_episode in range(50):
    observation = env.reset()
    observation_bar = deepcopy(observation)
    discret_state = state_action_space._m_state_continuous_to_discret(
        observation
    )
    discret_state_bar = discret_state

    action = learning_agent._m_GreedyPolicy(discret_state)

    phi = state_action_space._m_discret_state_to_feature(
        discret_state,
        action
    )

    rho = 1

    total_reward = 0
    Qfunc_previous = learning_agent.theta
    Tracer()()
    print i_episode
    for t in range(120):
        # env.render()
        same_action_count = 0
        while set(discret_state) == set(discret_state_bar):
            observation_bar, step_reward, done, info = env.step(action)
            discret_state_bar = state_action_space._m_state_continuous_to_discret(
                observation_bar
            )
        # Tracer()()
        action_bar = learning_agent._m_GreedyPolicy(discret_state_bar)
        phi_bar = state_action_space._m_discret_state_to_feature(
            discret_state_bar,
            action_bar
        )
        learning_agent._m_Learn(phi, phi_bar, step_reward, rho, 1)

        observation = observation_bar
        phi = phi_bar
        action = action_bar
        discret_state = discret_state_bar
#         # print np.dot(Qfunc_previous - Qfunc, Qfunc_previous - Qfunc)
        total_reward += step_reward
        if done:
            print "Episode finished after {} timesteps".format(t + 1)
            break
    total_reward_episode.append(total_reward)
    Qfunc_difference.append(
        np.dot(
            Qfunc_previous - learning_agent.theta,
            Qfunc_previous - learning_agent.theta
        )
    )
    Qfunc_previous = learning_agent.theta
    if i_episode % 100 == 0:
        print i_episode, "th episode completed"

# print Qfunc_difference
# print w

plt.figure(1)
plt.subplot(211)
plt.plot(Qfunc_difference)

plt.subplot(212)
plt.plot(total_reward_episode)
plt.show()

# for i_episode in range(10):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         # print(observation)
#         action = epsilon_greddy_action_choose(observation, Qfunc)
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
