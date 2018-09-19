from collections import deque
import numpy as np
import random
from ppo import PPO
from dqn import DQN
import os
from episode import Episode
import torch
from torch.distributions import Categorical

class Agent():
    def __init__(self, env, observation_shape, num_actions):
        self.num_actions = num_actions
        target_dqn = DQN(observation_shape, num_actions)
        self.algo = DQN(observation_shape, num_actions, target_dqn=target_dqn)

        self.num_steps = 0
        self.num_episodes = 0
        self.total_num_steps = 0
        self.env = env
        self.episodes = deque([Episode()], maxlen=500)
        self.total_reward = 0
        self.epsilon = 0.32
    
    def reset(self, learn=True):
        print("Total reward:", self.total_reward, "Num steps:", self.num_steps,
            "epsilon:", self.epsilon, flush=True)
        self.num_steps = 0
        self.total_reward = 0
        self.num_episodes += 1

        if len(self.episodes) >= 50 and learn:
            print("learning started")
            self.learn_from_memory()
            self.algo.save()
            print("finished learning")

        self.episodes.append(Episode())
        # self.memory.clear()
    
    def store_in_memory(self, state, action, reward, new_state, done):
        self.episodes[-1].store_step(state, action, reward, new_state, done)
    
    def learn_from_memory(self):
        self.algo.fit(self.episodes[-1], bs=32)

        indexes = np.random.choice(len(self.episodes), size=2, replace=False)
        for index in indexes:
            self.algo.fit(self.episodes[index], bs=32)
    
    def step(self, state):
        action_probs = self.algo.eval(state).double()[0]
        action_probs = action_probs/action_probs.sum()
        if random.random() < self.epsilon:
           action = random.randint(0, self.num_actions - 1)
        else:
           action = action_probs.max(0)[1]

        # dist = Categorical(action_probs)
        # action = dist.sample()
        
        new_state, reward, done, info = self.env.step(action)
        # print(f"info: {info}, num_steps: {self.num_steps}, reward: {reward}, done: {done}")
        
        self.store_in_memory(state, action, reward, new_state, done)
        
        self.num_steps += 1
        self.total_num_steps += 1
        self.total_reward += reward
        self.epsilon -= 5e-06
        self.epsilon = max(0.1, self.epsilon)
        
        return new_state, done
        
