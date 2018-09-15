import gym
import numpy as np
from environment import Environment
from runner import Runner

env = gym.make("Breakout-v4")
num_actions = env.action_space.n

rows, cols, channels = env.observation_space.shape
print(rows, cols, channels)
observation_shape = np.zeros([rows, cols, 2]).reshape(-1).shape[0]
observation_shape

env = Environment()
runner = Runner(env, observation_shape, num_actions)

while True:
    runner.run(episodes=20, render=False)
    runner.run(episodes=20, render=True)

    runner.run(episodes=20, render=False)
    runner.run(episodes=20, render=True)

    runner.run(episodes=20, render=False)
    runner.run(episodes=20, render=True)

    runner.run(episodes=20, render=False)
    runner.run(episodes=20, render=True)