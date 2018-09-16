from collections import deque
from ppo import PPO
import numpy as np
from episode import Episode

class Agent():
    def __init__(self, env, observation_shape, num_actions):
        self.num_actions = num_actions
        self.algo = PPO(observation_shape, num_actions)
        self.num_steps = 0
        self.total_num_steps = 0
        self.env = env
        self.episodes = deque([Episode()], maxlen=4)
        self.total_reward = 0
    
    def reset(self):
        print("Total reward:", self.total_reward, "Num steps:", self.num_steps)
        self.num_steps = 0
        self.total_reward = 0

        print("learning started")
        self.learn_from_memory()
        print("finished learning")

        self.episodes.append(Episode())
        # self.memory.clear()
    
    def store_in_memory(self, state, action, reward, new_state, done):
        self.episodes[-1].store_step(state, action, reward, new_state, done)
    
    def learn_from_memory(self):
        for episode in self.episodes:
            self.algo.fit(episode, bs=128)
    
    def step(self, state):
        action_probs = self.algo.eval(state).double()[0]
        action_probs = action_probs/action_probs.sum()
        action = np.random.choice(range(self.num_actions), p=action_probs)
        
        new_state, reward, done, info = self.env.step(action)
        # print(f"info: {info}, num_steps: {self.num_steps}, reward: {reward}, done: {done}")
        
        self.store_in_memory(state, action, reward, new_state, done)
        
        self.num_steps += 1
        self.total_num_steps += 1
        self.total_reward += reward
        
        return new_state, done
        