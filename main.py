import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import gym
import numpy as np
from environment import Environment
from runner import Runner

env = gym.make("BreakoutDeterministic-v4")
num_actions = env.action_space.n
image_size = [80, 80]

rows, cols, channels = env.observation_space.shape
print("frame shape:", (rows, cols, channels))
observation_shape = np.zeros([1, 2, *image_size])#.reshape(1, -1).shape
print("observation shape:", observation_shape.shape, observation_shape.reshape(-1).shape)
observation_shape = observation_shape.reshape(-1).shape[0]

env = Environment(image_size=image_size)
runner = Runner(env, observation_shape, num_actions)

while True:
    runner.run(episodes=20, render=True, learn=False)
