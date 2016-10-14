###################################################################
# In this script two algortihm will be tested on their performance
# with two classic control problem, Cart Pole and Mountian Car.
# At the end, a conparison as summury with graphics will be given 
# as the result.
#
# And it's also allowed to run the problem after trainning with
# the trained value function.
#
# Code for Forschungspraxis
###################################################################

############## Global Learning Parameter Setting #################
epsilonGreedy = 0.5           # Epsilon used in epsilonGreedy method  
learningRate = 0.1            # learningRate
gamma = 0.7                  # Discount coefficient used in computation of TD error
numEpisodes = 1000            # Number of Episodes used in trainning
lambdaDiscount = 0.5          # Lambda in SarsaLambda algorithm
iterNumLimit = 50             # Iteration number Limit

############## Learning with GTD an Cart Pole   ###################

env = gym.make('CartPole-v0')

for num_episode in range(10000):

    observation = env.reset()

    for t in range(1000):

        env.render()

        print observation

        action = getAction()

        observation, reward, done, info = env.step(action)

        algorithm.m_RunStep(observation, reward);

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

############## Learning with my algorithm an Cart Pole   ###################

for num_episode in range(10000):

    observation = env.reset()

    for t in range(1000):

        env.render()

        print observation

        action = env.action_space.sample()

        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

############## Learning with GTD an Mountain Car   ###################

env = gym.make('MountainCar-v0')

for num_episode in range(10000):

    observation = env.reset()

    for t in range(1000):

        env.render()

        print observation

        action = env.action_space.sample()

        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

############## Learning with my algorithm an Mountain Car   ###################

env = gym.make('MountainCar-v0')

for num_episode in range(10000):

    observation = env.reset()

    for t in range(1000):

        env.render()

        print observation

        action = env.action_space.sample()

        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

############## Output the Comparision Result   ###################


############## Run Demo with trained value function   ###################