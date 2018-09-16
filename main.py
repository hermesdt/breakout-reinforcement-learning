import gym
import numpy as np
from environment import Environment
from runner import Runner

env = gym.make("Breakout-v4")
num_actions = env.action_space.n

rows, cols, channels = env.observation_space.shape
print("frame shape:", (rows, cols, channels))
observation_shape = np.zeros([1, 2, rows, cols])#.reshape(1, -1).shape
print("observation shape:", observation_shape.shape, observation_shape.reshape(-1).shape)
observation_shape = observation_shape.reshape(-1).shape[0]

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