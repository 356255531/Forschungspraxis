import gym
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from Toolbox.algorithm import RGGQLambda
from Toolbox.algorithm import GGQLambda
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

NUM_EPISODE = 500
REWARD_THREASHOLD = -150
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

learning_agent = learning_agent_RGGQLambda

# Run algorithm
Qfunc_error_history = []
total_reward_episode = []
max_reward = -float("inf")
for i_episode in range(NUM_EPISODE):
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

    for t in range(200):
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

    if i_episode % 10 == 0:
        print i_episode, "th episode completed"
        print "Q update is", Qfunc_difference_this_episode
        print "Maximal reward is", max_reward, "\n"

# Plot results
plt.figure(1)
plt.subplot(211)
plt.plot(Qfunc_error_history)

plt.subplot(212)
plt.plot(total_reward_episode)
plt.show()

# Run learned policy an simulator
for i_episode in range(10):
    observation = env.reset()

    for t in range(200):
        env.render()

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