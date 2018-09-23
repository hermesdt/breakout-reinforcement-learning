from collections import deque
import numpy as np
import random
from ppo import PPO
from dqn import DQN
import os
from episode import Episode
from memory import Memory
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
        self.memory = Memory(maxlen=1000)
        self.episode = Episode()
        self.total_reward = 0
        self.epsilon = 0.32
    
    def reset(self, learn=True):
        print("Total reward:", self.total_reward, "Num steps:", self.num_steps,
            "epsilon:", self.epsilon, flush=True)

        if len(self.memory) >= 100 and learn:
            print("learning started")
            self.learn_from_memory()
            self.algo.save()
            print("finished learning")
        
        self.num_steps = 0
        self.total_reward = 0
        self.num_episodes += 1
    
    def store_in_memory(self):
        self.memory.add_episode(self.episode)
        self.episode = Episode()
    
    def learn_from_memory(self):
        for _batch_id in range(self.num_steps):
            self.algo.train(self.memory.get_batch(batch_size=32))
    
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
        self.episode.store_step(state, action, reward, new_state, done)
        # print(f"info: {info}, num_steps: {self.num_steps}, reward: {reward}, done: {done}")
        
        if done:
            self.store_in_memory()
        
        self.num_steps += 1
        self.total_num_steps += 1
        self.total_reward += reward
        self.epsilon -= 5e-06
        self.epsilon = max(0.1, self.epsilon)
        
        return new_state, done
        
