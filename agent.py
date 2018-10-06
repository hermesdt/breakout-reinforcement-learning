from collections import deque
import numpy as np
import random
from math import log
from dqn import DQN
from dqn_param_noise import DQNParamNoise
import os
from memory import Memory
import torch
from torch.distributions import Categorical

class Agent():
    def __init__(self, env):
        self.num_actions = env.action_space.n

        target_dqn = DQNParamNoise(env)
        self.algo = DQNParamNoise(env, target_dqn=target_dqn)

        self.num_steps = 0
        self.episode_steps = 0
        self.env = env
        self.memory = Memory(maxlen=10000)
        self.rewards = deque(maxlen=100)
        self.epsilon = 0.005
        self.total_reward = 0

    def reset(self, learn=True):
        self.algo.episode_start()
        self.total_reward = 0
    
    def store_in_memory(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))
    
    def learn_from_memory(self):
        self.algo.train(self.memory.get_batch(batch_size=32))
    
    def step(self, state, learn=True):
        action_probs = self.algo.eval(np.expand_dims(state, 0)).double()[0]
        # print(action_probs)
        action_probs = action_probs/action_probs.sum()

        if random.random() < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = action_probs.max(0)[1]
        
        new_state, reward, done, info = self.env.step(action)
        self.store_in_memory(state, action, reward, new_state, done)
        if self.num_steps >= 100 and learn:
            self.learn_from_memory()
        
        if done:
            print(f"[{self.num_steps}] reward {self.env.mean_reward}, steps {self.episode_steps}, speed {self.env.speed} f/s, epsilon {self.epsilon}")
            self.episode_steps = 0
            self.algo.save()
        
        self.num_steps += 1
        self.episode_steps += 1
        self.total_reward += reward

        self.epsilon -= 1e-04
        self.epsilon = max(0.005, self.epsilon)
        
        return new_state, reward, done, info 
