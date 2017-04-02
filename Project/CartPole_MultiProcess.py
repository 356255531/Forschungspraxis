import gym
import numpy as np
from copy import deepcopy
import time
import multiprocessing
import pickle
from sys import platform
from os.path import isfile

from Toolbox.algorithm import RGGQLambda, GGQLambda, OSKQ
from Toolbox.StateActionSpace import StateActionSpace_CartPole

__auther__ = "Zhiwei"


if "darwin" == platform:
    path = "/Users/Finn/Dropbox/Lehrveranstaltungen/Forschungspraxis/Project/data/CartPole/"
else:
    path = "/home/zhiwei/Dropbox/Lehrveranstaltungen/Forschungspraxis/Project/data/CartPole/"


def GGQLambda_MultiProcess_Ave(ave_times=20,
                               learning_rate=0.1,
                               eligibility_factor=0.9):
    """
        ave_times=20,
        learning_rate=0.1,
        eligibility_factor=0.9,
        mu_2=0.8
    """
    # Learning parameters
    precise = [10, 8, 10, 8]

    discount_factor = 0.9
    discount_of_learning_rate = 0.999
    epsilon = 0.1

    # Macro
    NUM_EPISODE = 600
    AVE_TIMES = ave_times
    REWARD_THREASHOLD = 40
    # Definition of dependencies
    env = gym.make('CartPole-v0')

    observation_space = (
        env.observation_space.low,
        env.observation_space.high
    )

    CartPole_universal_action_space = [i for i in xrange(0, env.action_space.n)]
    state_action_space = StateActionSpace_CartPole(
        observation_space,
        precise,
        CartPole_universal_action_space
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

        Qfunc_error_history = []
        total_reward_episode = []
        time_history = []
        max_reward = -float("inf")
        for i_episode in range(NUM_EPISODE):
            time_start = time.clock()
            observation = env.reset()

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

            done = False
            step = 0
            while not done:
                step += 1
                while set(discret_state) == set(discret_state_bar):
                    observation_bar, step_reward, done, info = env.step(action)

                    if done:
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

                phi = phi_bar
                action = action_bar
                discret_state = discret_state_bar
                total_reward += step_reward
                if done:
                    break
            print "Episode finished after {} timesteps in GQ(lambda)".format(step), "in ", ave_times + 1, "times"
            time_end = time.clock()
            time_consumed = time_end - time_start
            time_history.append(time_consumed)

            if total_reward > max_reward:
                if total_reward > REWARD_THREASHOLD:
                    epsilon *= 0.999
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
            path + "total_reward_GGQ-" + str(learning_rate) + "-" + str(eligibility_factor), 'wb') as f:
        pickle.dump(total_reward_episode, f)
    with open(
            path + "time_history_GGQ-" + str(learning_rate) + "-" + str(eligibility_factor), 'wb') as f:
        pickle.dump(time_history, f)


def RGGQLambda_MultiProcess_Ave(ave_times=20,
                                learning_rate=0.1,
                                eligibility_factor=0.9,
                                regularize_factor=0.0001):
    """
        ave_times=20,
        learning_rate=0.1,
        eligibility_factor=0.9,
        mu_2=0.8
    """
    # Learning parameters
    precise = [10, 8, 10, 8]

    discount_factor = 0.9
    discount_of_learning_rate = 0.999
    epsilon = 0.1

    # Macro
    NUM_EPISODE = 600
    AVE_TIMES = ave_times
    REWARD_THREASHOLD = 40
    # Definition of dependencies
    env = gym.make('CartPole-v0')

    observation_space = (
        env.observation_space.low,
        env.observation_space.high
    )

    CartPole_universal_action_space = [i for i in xrange(0, env.action_space.n)]
    state_action_space = StateActionSpace_CartPole(
        observation_space,
        precise,
        CartPole_universal_action_space
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

        sparsity = []
        Qfunc_error_history_2 = []
        total_reward_episode_2 = []
        time_history_2 = []
        max_reward = -float("inf")
        for i_episode in range(NUM_EPISODE):
            time_start = time.clock()
            observation = env.reset()

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

            done = False
            step = 0
            while not done:
                step += 1
                while set(discret_state) == set(discret_state_bar):
                    observation_bar, step_reward, done, info = env.step(action)

                    if done:
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
                                        1
                                        )

                phi = phi_bar
                action = action_bar
                discret_state = discret_state_bar
                total_reward += step_reward
                if done:
                    break
            print "Episode finished after {} timesteps in RGGQ(lambda)".format(step), "in ", ave_times + 1, "times"
            time_end = time.clock()
            time_consumed = time_end - time_start
            time_history_2.append(time_consumed)

            if total_reward > max_reward:
                if total_reward > REWARD_THREASHOLD:
                    epsilon *= 0.999
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

            sparsity.append(np.sum(learning_agent.theta == 0) / (learning_agent.num_element_qfunc * 1.0))

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
            path + "total_reward_RGGQ-" +
            str(learning_rate) + "-" + str(eligibility_factor) + "-" + str(regularize_factor), 'wb') as f:
        pickle.dump(total_reward_episode_2, f)
    with open(
            path + "time_history_RGGQ-" +
            str(learning_rate) + "-" + str(eligibility_factor) + "-" + str(regularize_factor), 'wb') as f:
        pickle.dump(time_history_2, f)
    with open(
            path + "sparsity_RGGQ-" +
            str(learning_rate) + "-" + str(eligibility_factor) + "-" + str(regularize_factor), 'wb') as f:
        pickle.dump(sparsity, f)


def OSK_Q_MultiProcess_Ave(ave_times=20,
                           learning_rate=0.1,
                           eligibility_factor=0.9,
                           observation_dim=4,
                           mu_1=0.04,
                           mu_2=0.08):
    """
        ave_times=20,
        learning_rate=0.1,
        eligibility_factor=0.9,
        mu_2=0.8
    """
    # Learning Parameter
    discount_factor = 0.9
    discount_of_learning_rate = 0.999
    epsilon = 0.1

    # Parameter OSK-Q
    sigma = 1

    # Macro
    NUM_EPISODE = 600
    AVE_TIMES = ave_times
    REWARD_THREASHOLD = 40
    # Definition of dependencies
    env = gym.make('CartPole-v0')

    # Run algorithm
    for ave_times in range(AVE_TIMES):
        learning_agent_OSKQ = OSKQ(
            mu_1,
            mu_2,
            learning_rate,
            discount_factor,
            eligibility_factor,
            [0, 1],
            observation_dim,
            sigma
        )
        learning_agent = learning_agent_OSKQ

        Qfunc_error_history_3 = []
        total_reward_episode_3 = []
        time_history_3 = []
        max_reward = -float("inf")
        for i_episode in range(NUM_EPISODE):
            time_start = time.clock()
            observation = env.reset()
            action = learning_agent._m_GreedyPolicy(observation, epsilon)

            total_reward = 0
            done = False
            step = 0
            while not done:
                step += 1
                observation_bar, step_reward, done, info = env.step(action)

                action_bar = learning_agent._m_GreedyPolicy(
                    observation_bar,
                    epsilon
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
            print "Episode finished after {} timesteps in OSK-Q(lambda)".format(step), "in ", ave_times + 1, "times"
            time_end = time.clock()
            time_consumed = time_end - time_start
            time_history_3.append(time_consumed)

            if total_reward > max_reward:
                if total_reward > REWARD_THREASHOLD:
                    epsilon *= 0.999
                max_reward = total_reward

            total_reward_episode_3.append(total_reward)   # Add total reward to reward history

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
            path + "total_reward_OSKQ-" + str(learning_rate) + "-" +
            str(eligibility_factor) + "-" + str(mu_1) + "-" + str(mu_2), 'wb') as f:
        pickle.dump(total_reward_episode_3, f)
    with open(
            path + "time_history_OSKQ-" + str(learning_rate) + "-" +
            str(eligibility_factor) + "-" + str(mu_1) + "-" + str(mu_2), 'wb') as f:
        pickle.dump(time_history_3, f)


def main():
    ave_times = 5
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    for learning_rate in [0.03, 0.1]:
        for eligibility_factor in [0, 0.1]:
            if not isfile(path + "time_history_GGQ-" + str(learning_rate) +
                          "-" + str(eligibility_factor)):
                pool.apply_async(GGQLambda_MultiProcess_Ave,
                                 (ave_times, learning_rate, eligibility_factor,))

    for learning_rate in [0.03, 0.1]:
        for eligibility_factor in [0, 0.1]:
            for regularize_factor in [0.00003, 0.0001, 0.0003]:
                if not isfile(path + "time_history_RGGQ-" + str(learning_rate) + "-" +
                              str(eligibility_factor) + "-" + str(regularize_factor)):
                    pool.apply_async(RGGQLambda_MultiProcess_Ave,
                                     (ave_times, learning_rate, eligibility_factor, regularize_factor,))

    for learning_rate in [0.03, 0.1]:
        for eligibility_factor in [0, 0.1]:
            for mu_1 in [0.04]:
                for mu_2 in [0.08, 0.16]:
                    if not isfile(path + "time_history_OSKQ-" + str(learning_rate) + "-" +
                                  str(eligibility_factor) + "-" + str(mu_1) + "-" + str(mu_2)):
                        pool.apply_async(OSK_Q_MultiProcess_Ave,
                                         (ave_times, learning_rate, eligibility_factor, 4, mu_1, mu_2, ))

    pool.close()
    pool.join()


if __name__ == '__main__':
    GGQLambda_MultiProcess_Ave(100, 1, 0)
    # main()
