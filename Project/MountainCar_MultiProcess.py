import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.legend_handler import HandlerLine2D
from copy import deepcopy
import time
import multiprocessing
import pickle

from Toolbox.algorithm import RGGQLambda, GGQLambda, OSKQ_New
from Toolbox.StateActionSpace import StateActionSpace_MountainCar

__auther__ = "Zhiwei"


def GGQLambda_same_parameter(mu_2=0.08,
                             ave_times=20,
                             learning_rate=0.1,
                             eligibility_factor=0.9):
    """
        ave_times=20,
        learning_rate=0.1,
        eligibility_factor=0.9,
        mu_2=0.8
    """
    precise = [8, 8]

    discount_factor = 0.9
    discount_of_learning_rate = 0.999
    epsilon = 0.1

    # Macro
    NUM_STEP = 1000
    NUM_EPISODE = 500
    AVE_TIMES = ave_times
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

    for ave_times in range(AVE_TIMES):
        learning_agent_GGQLambda = GGQLambda(
            learning_rate,
            discount_factor,
            eligibility_factor,
            discount_of_learning_rate,
            epsilon,
            state_action_space.action_space
        )
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
                        print "Episode finished after {} timesteps in GQ(lambda)".format(t + 1), "in ", ave_times + 1, "times"
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
                    print "Episode finished after {} timesteps in GQ(lambda)".format(t + 1), "in ", ave_times + 1, "times"
                    break
            time_end = time.clock()
            time_consumed = time_end - time_start
            time_history.append(time_consumed)

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

        Qfunc_error_history = np.array(Qfunc_error_history)
        if 'Qfunc_error_history_ave' not in locals():
            Qfunc_error_history_ave = Qfunc_error_history
        else:
            Qfunc_error_history_ave = Qfunc_error_history_ave + (Qfunc_error_history - Qfunc_error_history_ave) / (ave_times * 1.0)

        total_reward_episode = np.array(total_reward_episode)
        if 'total_reward_episode_ave' not in locals():
            total_reward_episode_ave = total_reward_episode
        else:
            total_reward_episode_ave = total_reward_episode_ave + (total_reward_episode - total_reward_episode_ave) / (ave_times * 1.0)

        time_history = np.array(time_history)
        if 'time_history_ave' not in locals():
            time_history_ave = time_history
        else:
            time_history_ave = time_history_ave + (time_history - time_history_ave) / (ave_times * 1.0)

    Qfunc_error_history = Qfunc_error_history_ave
    total_reward_episode = total_reward_episode_ave
    time_history = time_history_ave
    with open(
            "/home/zhiwei/Workspace/Forschungspraxis/Project/data/MountainCar/total_reward_GGQ-" + str(learning_rate) + "-" + str(eligibility_factor), 'wb') as f:
        pickle.dump(total_reward_episode, f)
    with open(
            "/home/zhiwei/Workspace/Forschungspraxis/Project/data/MountainCar/time_history_GGQ-" + str(learning_rate) + "-" + str(eligibility_factor), 'wb') as f:
        pickle.dump(time_history, f)


def RGGQLambda_same_parameter(mu_2=0.08,
                              ave_times=20,
                              learning_rate=0.1,
                              eligibility_factor=0.9):
    """
        ave_times=20,
        learning_rate=0.1,
        eligibility_factor=0.9,
        mu_2=0.8
    """
    # Learning Parameter
    precise = [8, 8]

    discount_factor = 0.9
    discount_of_learning_rate = 0.999
    regularize_factor = 0.0001  # 0.0001
    epsilon = 0.1

    # Macro
    NUM_STEP = 1000
    NUM_EPISODE = 500
    AVE_TIMES = ave_times
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

        Qfunc_error_history_2 = np.array(Qfunc_error_history_2)
        if 'Qfunc_error_history_ave_2' not in locals():
            Qfunc_error_history_ave_2 = Qfunc_error_history_2
        else:
            Qfunc_error_history_ave_2 = Qfunc_error_history_ave_2 + (Qfunc_error_history_2 - Qfunc_error_history_ave_2) / (ave_times * 1.0)

        total_reward_episode_2 = np.array(total_reward_episode_2)
        if 'total_reward_episode_ave_2' not in locals():
            total_reward_episode_ave_2 = total_reward_episode_2
        else:
            total_reward_episode_ave_2 = total_reward_episode_ave_2 + (total_reward_episode_2 - total_reward_episode_ave_2) / (ave_times * 1.0)

        time_history_2 = np.array(time_history_2)
        if 'time_history_ave_2' not in locals():
            time_history_ave_2 = time_history_2
        else:
            time_history_ave_2 = time_history_ave_2 + (time_history_2 - time_history_ave_2) / (ave_times * 1.0)

    Qfunc_error_history_2 = Qfunc_error_history_ave_2
    total_reward_episode_2 = total_reward_episode_ave_2
    time_history_2 = time_history_ave_2
    with open(
            "/home/zhiwei/Workspace/Forschungspraxis/Project/data/MountainCar/total_reward_RGGQ-" + str(learning_rate) + "-" + str(eligibility_factor), 'wb') as f:
        pickle.dump(total_reward_episode_2, f)
    with open(
            "/home/zhiwei/Workspace/Forschungspraxis/Project/data/MountainCar/time_history_RGGQ-" + str(learning_rate) + "-" + str(eligibility_factor), 'wb') as f:
        pickle.dump(time_history_2, f)


def OSK_Q_same_parameter(mu_2=0.08,
                         ave_times=20,
                         learning_rate=0.1,
                         eligibility_factor=0.9):
    """
        ave_times=20,
        learning_rate=0.1,
        eligibility_factor=0.9,
        mu_2=0.8
    """
    # Learning Parameter
    precise = [8, 8]

    discount_factor = 0.9
    discount_of_learning_rate = 0.999
    regularize_factor = 0.0001  # 0.0001
    epsilon = 0.1

    # Parameter OSK-Q
    mu_1 = 0.04
    sigma = 1
    # Macro
    NUM_STEP = 1000
    NUM_EPISODE = 500
    AVE_TIMES = ave_times
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
        learning_agent_OSKQ = OSKQ_New(
            mu_1,
            mu_2,
            learning_rate,
            discount_factor,
            eligibility_factor,
            epsilon,
            [0, 1, 2],
            sigma
        )
        learning_agent = learning_agent_OSKQ

        Qfunc_error_history_3 = []
        total_reward_episode_3 = []
        max_reward = -float("inf")
        time_history_3 = []
        for i_episode in range(NUM_EPISODE):
            time_start = time.clock()
            observation = env.reset()
            observation_bar = deepcopy(observation)

            total_reward = 0
            for t in range(NUM_STEP):
                action = learning_agent._m_GreedyPolicy(observation)

                while set(observation) == set(observation_bar):
                    observation_bar, step_reward, done, info = env.step(action)

                    if done:
                        if total_reward > REWARD_THREASHOLD:
                            learning_agent.epsilon *= 0.999
                        print "Episode finished after {} timesteps in OSK-Q".format(t + 1), "in ", ave_times + 1, "times"
                        break

                action_bar = learning_agent._m_GreedyPolicy(
                    observation_bar
                )

                learning_agent._m_Learn(
                    observation,
                    action,
                    observation_bar,
                    action_bar,
                    step_reward
                )

                observation = observation_bar
                action = action_bar
                total_reward += step_reward
                if done:
                    if total_reward > REWARD_THREASHOLD:
                        learning_agent.epsilon *= 0.999
                    print "Episode finished after {} timesteps in OSK-Q(lambda)".format(t + 1), "in ", ave_times + 1, "times"
                    break
            if total_reward > max_reward:
                max_reward = total_reward

            total_reward_episode_3.append(total_reward)   # Add total reward to reward history

            time_end = time.clock()
            time_consumed = time_end - time_start
            time_history_3.append(time_consumed)

            if i_episode % 10 == 0:
                print i_episode, "th episode completed"
                print "Maximal reward is", max_reward, "\n"

        total_reward_episode_3 = np.array(total_reward_episode_3)
        if 'total_reward_episode_ave_3' not in locals():
            total_reward_episode_ave_3 = total_reward_episode_3
        else:
            total_reward_episode_ave_3 = total_reward_episode_ave_3 + (total_reward_episode_3 - total_reward_episode_ave_3) / (ave_times * 1.0)

        time_history_3 = np.array(time_history_3)
        if 'time_history_ave_3' not in locals():
            time_history_ave_3 = time_history_3
        else:
            time_history_ave_3 = time_history_ave_3 + (time_history_3 - time_history_ave_3) / (ave_times * 1.0)

    total_reward_episode_3 = total_reward_episode_ave_3
    time_history_3 = time_history_ave_3
    with open(
            "/home/zhiwei/Workspace/Forschungspraxis/Project/data/MountainCar/total_reward_OSKQ-" + str(learning_rate) + "-" + str(eligibility_factor) + "-" + str(mu_2), 'wb') as f:
        pickle.dump(total_reward_episode_3, f)
    with open(
            "/home/zhiwei/Workspace/Forschungspraxis/Project/data/MountainCar/time_history_OSKQ-" + str(learning_rate) + "-" + str(eligibility_factor) + "-" + str(mu_2), 'wb') as f:
        pickle.dump(time_history_3, f)


def same_parameter_run(eligibility_factor=0.9, mu_2=0.8):
    # Learning Parameter
    precise = [8, 8]

    learning_rate = 0.1
    discount_factor = 0.9
    discount_of_learning_rate = 0.999
    regularize_factor = 0.0001  # 0.0001
    epsilon = 0.1

    # Parameter OSK-Q
    mu_1 = 0.04
    sigma = 1
    # Macro
    NUM_STEP = 1000
    NUM_EPISODE = 500
    AVE_TIMES = 1
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

    for ave_times in range(AVE_TIMES):
        learning_agent_GGQLambda = GGQLambda(
            learning_rate,
            discount_factor,
            eligibility_factor,
            discount_of_learning_rate,
            epsilon,
            state_action_space.action_space
        )
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
                        print "Episode finished after {} timesteps in GQ(lambda)".format(t + 1), "in ", ave_times + 1, "times"
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
                    print "Episode finished after {} timesteps in GQ(lambda)".format(t + 1), "in ", ave_times + 1, "times"
                    break
            time_end = time.clock()
            time_consumed = time_end - time_start
            time_history.append(time_consumed)

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

        Qfunc_error_history = np.array(Qfunc_error_history)
        if 'Qfunc_error_history_ave' not in locals():
            Qfunc_error_history_ave = Qfunc_error_history
        else:
            Qfunc_error_history_ave = Qfunc_error_history_ave + (Qfunc_error_history - Qfunc_error_history_ave) / (ave_times * 1.0)

        total_reward_episode = np.array(total_reward_episode)
        if 'total_reward_episode_ave' not in locals():
            total_reward_episode_ave = total_reward_episode
        else:
            total_reward_episode_ave = total_reward_episode_ave + (total_reward_episode - total_reward_episode_ave) / (ave_times * 1.0)

        time_history = np.array(time_history)
        if 'time_history_ave' not in locals():
            time_history_ave = time_history
        else:
            time_history_ave = time_history_ave + (time_history - time_history_ave) / (ave_times * 1.0)

    Qfunc_error_history = Qfunc_error_history_ave
    total_reward_episode = total_reward_episode_ave
    time_history = time_history_ave
    with open(
            "/home/zhiwei/Workspace/Forschungspraxis/Project/data/MountainCar/total_reward_episode-" + str(eligibility_factor) + "-" + str(mu_2), 'wb') as f:
        pickle.dump(total_reward_episode, f)
    with open(
            "/home/zhiwei/Workspace/Forschungspraxis/Project/data/MountainCar/time_history-" + str(eligibility_factor) + "-" + str(mu_2), 'wb') as f:
        pickle.dump(time_history, f)
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
            Qfunc_error_history = np.array(Qfunc_error_history)

        Qfunc_error_history_2 = np.array(Qfunc_error_history_2)
        if 'Qfunc_error_history_ave_2' not in locals():
            Qfunc_error_history_ave_2 = Qfunc_error_history_2
        else:
            Qfunc_error_history_ave_2 = Qfunc_error_history_ave_2 + (Qfunc_error_history_2 - Qfunc_error_history_ave_2) / (ave_times * 1.0)

        total_reward_episode_2 = np.array(total_reward_episode_2)
        if 'total_reward_episode_ave_2' not in locals():
            total_reward_episode_ave_2 = total_reward_episode_2
        else:
            total_reward_episode_ave_2 = total_reward_episode_ave_2 + (total_reward_episode_2 - total_reward_episode_ave_2) / (ave_times * 1.0)

        time_history_2 = np.array(time_history_2)
        if 'time_history_ave_2' not in locals():
            time_history_ave_2 = time_history_2
        else:
            time_history_ave_2 = time_history_ave_2 + (time_history_2 - time_history_ave_2) / (ave_times * 1.0)

    Qfunc_error_history_2 = Qfunc_error_history_ave_2
    total_reward_episode_2 = total_reward_episode_ave_2
    time_history_2 = time_history_ave_2
    with open(
            "/home/zhiwei/Workspace/Forschungspraxis/Project/data/MountainCar/total_reward_episode_2-" + str(eligibility_factor) + "-" + str(mu_2), 'wb') as f:
        pickle.dump(total_reward_episode_2, f)
    with open(
            "/home/zhiwei/Workspace/Forschungspraxis/Project/data/MountainCar/time_history_2-" + str(eligibility_factor) + "-" + str(mu_2), 'wb') as f:
        pickle.dump(time_history_2, f)

    # Run algorithm
    for ave_times in range(AVE_TIMES):
        learning_agent_OSKQ = OSKQ_New(
            mu_1,
            mu_2,
            learning_rate,
            discount_factor,
            eligibility_factor,
            epsilon,
            [0, 1, 2],
            sigma
        )
        learning_agent = learning_agent_OSKQ

        Qfunc_error_history_3 = []
        total_reward_episode_3 = []
        max_reward = -float("inf")
        time_history_3 = []
        for i_episode in range(NUM_EPISODE):
            time_start = time.clock()
            observation = env.reset()
            observation_bar = deepcopy(observation)

            total_reward = 0
            for t in range(NUM_STEP):
                action = learning_agent._m_GreedyPolicy(observation)

                while set(observation) == set(observation_bar):
                    observation_bar, step_reward, done, info = env.step(action)

                    if done:
                        if total_reward > REWARD_THREASHOLD:
                            learning_agent.epsilon *= 0.999
                        print "Episode finished after {} timesteps in OSK-Q".format(t + 1), "in ", ave_times + 1, "times"
                        break

                action_bar = learning_agent._m_GreedyPolicy(
                    observation_bar
                )

                learning_agent._m_Learn(
                    observation,
                    action,
                    observation_bar,
                    action_bar,
                    step_reward
                )

                observation = observation_bar
                action = action_bar
                total_reward += step_reward
                if done:
                    if total_reward > REWARD_THREASHOLD:
                        learning_agent.epsilon *= 0.999
                    print "Episode finished after {} timesteps in OSK-Q(lambda)".format(t + 1), "in ", ave_times + 1, "times"
                    break
            if total_reward > max_reward:
                max_reward = total_reward

            total_reward_episode_3.append(total_reward)   # Add total reward to reward history

            time_end = time.clock()
            time_consumed = time_end - time_start
            time_history_3.append(time_consumed)

            if i_episode % 10 == 0:
                print i_episode, "th episode completed"
                print "Maximal reward is", max_reward, "\n"

        total_reward_episode_3 = np.array(total_reward_episode_3)
        if 'total_reward_episode_ave_3' not in locals():
            total_reward_episode_ave_3 = total_reward_episode_3
        else:
            total_reward_episode_ave_3 = total_reward_episode_ave_3 + (total_reward_episode_3 - total_reward_episode_ave_3) / (ave_times * 1.0)

        time_history_3 = np.array(time_history_3)
        if 'time_history_ave_3' not in locals():
            time_history_ave_3 = time_history_3
        else:
            time_history_ave_3 = time_history_ave_3 + (time_history_3 - time_history_ave_3) / (ave_times * 1.0)

    total_reward_episode_3 = total_reward_episode_ave_3
    time_history_3 = time_history_ave_3
    with open(
            "/home/zhiwei/Workspace/Forschungspraxis/Project/data/MountainCar/total_reward_episode_3-" + str(eligibility_factor) + "-" + str(mu_2), 'wb') as f:
        pickle.dump(total_reward_episode_3, f)
    with open(
            "/home/zhiwei/Workspace/Forschungspraxis/Project/data/MountainCar/time_history_3-" + str(eligibility_factor) + "-" + str(mu_2), 'wb') as f:
        pickle.dump(time_history_3, f)


# plt.figure(1)
# plt.subplot(311)
# plt.plot(Qfunc_error_history)
# plt.plot(Qfunc_error_history_2)
# plt.plot(Qfunc_error_history_3)
# plt.show()


def main():
    ave_times = 20
    processes = []
    for learning_rate in [0.001, 0.003, 0.01, 0.03, 0.1]:
        for eligibility_factor in [0.2, 0.4, 0.6, 0.8]:
            processes.append(multiprocessing.Process(
                target=GGQLambda_same_parameter,
                args=(0.08, ave_times, learning_rate, eligibility_factor,)))

    for learning_rate in [0.001, 0.003, 0.01, 0.03, 0.1]:
        for eligibility_factor in [0.2, 0.4, 0.6, 0.8]:
            processes.append(multiprocessing.Process(
                target=RGGQLambda_same_parameter,
                args=(0.08, ave_times, learning_rate, eligibility_factor,)))

    for learning_rate in [0.001, 0.003, 0.01, 0.03, 0.1]:
        for eligibility_factor in [0.2, 0.4, 0.6, 0.8]:
            for mu_2 in [0.04, 0.08]:
                processes.append(multiprocessing.Process(
                    target=OSK_Q_same_parameter,
                    args=(mu_2, 20, learning_rate, eligibility_factor,)))
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    # font = {'family': 'normal',
    #         'size': 10}
    # matplotlib.rc('font', **font)

    # fig = plt.figure(figsize=(8, 14))
    # for line in range(2):
    #     for column in range(4):
    #         ax = fig.add_subplot(420 + line * 4 + column + 1)
    #         ax.set_xlabel('Num of episode')
    #         ax.set_ylabel('Sum of reward')

    #         line_1, = plt.plot(total_reward_episode_set[line * 4 + column], label='GQ(lambda)')
    #         line_2, = plt.plot(total_reward_episode_set_2[line * 4 + column], label='RGGQ(lambda)')
    #         line_3, = plt.plot(total_reward_episode_set_3[line * 4 + column], label='OKS-Q')
    # fig.legend((line_1, line_2, line_3), ("GQ(lambda)", "RGGQ(lambda)", "OKS-Q"), loc=8)
    # fig.savefig("Total Reward.jpg")

    # fig = plt.figure(figsize=(8, 14))
    # for line in range(2):
    #     for column in range(4):
    #         ax = fig.add_subplot(420 + line * 4 + column + 1)
    #         ax.set_xlabel('Num of episode')
    #         ax.set_ylabel('Total time needed')

    #         line_1, = plt.plot(time_history_set[line * 4 + column], label='GQ(lambda)')
    #         line_2, = plt.plot(time_history_set_2[line * 4 + column], label='RGGQ(lambda)')
    #         line_3, = plt.plot(time_history_set_3[line * 4 + column], label='OKS-Q')
    # fig.legend((line_1, line_2, line_3), ("GQ(lambda)", "RGGQ(lambda)", "OKS-Q"), loc=8)
    # fig.savefig("Total time consumed.jpg")


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
