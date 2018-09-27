from collections import deque
import numpy as np
import random
from dqn import DQN
import os
from memory import Memory
import torch
from torch.distributions import Categorical

class Agent():
    def __init__(self, env):
        target_dqn = DQN(env)
        self.algo = DQN(env, target_dqn=target_dqn)

        self.num_actions = env.action_space.n
        self.num_steps = 0
        self.episode_steps = 0
        self.env = env
        self.memory = Memory(maxlen=10000)
        self.rewards = deque(maxlen=100)
        self.epsilon = 1
        self.total_reward = 0
    
    def reset(self, learn=True):
        print("Total reward:", np.mean(list(deque)), "Num steps:", self.episode_steps,
            "epsilon:", self.epsilon, flush=True)
        
        self.total_reward = 0
    
    def store_in_memory(self, state, action, reward, new_state, done):
        self.memory.append(state, action, reward, new_state, done)
    
    def learn_from_memory(self):
        self.algo.train(self.memory.get_batch(batch_size=32))
    
    def step(self, state):
        action_probs = self.algo.eval(np.expand_dims(state, 0)).double()[0]
        action_probs = action_probs/action_probs.sum()

        if random.random() < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = action_probs.max(0)[1]
        
        new_state, reward, done, info = self.env.step(action)
        self.store_in_memory(state, action, reward, new_state, done)
        if self.num_steps >= 100:
            self.learn_from_memory()
        
        if reward != 0:
            print(f"[{self.num_steps}] reward {reward}, steps {self.episode_steps}, epsilon {self.epsilon}")
            self.episode_steps = 0
        
        self.num_steps += 1
        self.episode_steps += 1
        self.total_reward += reward

        self.epsilon -= 1e-05
        self.epsilon = max(0.1, self.epsilon)
        
        return new_state, reward, done, info 
