import gym
import time

env = gym.make('CartPole-v0')

######################### Choose Scenario ########################
for num_episode in range(10000):

    observation = env.reset()

    for t in range(1000):

        env.render()

        print observation

        action = env.action_space.sample()

        print action

        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break