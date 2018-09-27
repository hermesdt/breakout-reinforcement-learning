import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from environment import Environment
from runner import Runner
#from runner import Runner

env = Environment("PongNoFrameskip-v4")

print("frame shape:", env.observation_space.shape)#.reshape(1, -1).shape
print("action shape:", env.action_space.n)

runner = Runner(env)
runner.run(5, render=False)
runner.run(1, render=True)
# while True:
#     runner.run(episodes=20, render=True, learn=True)
