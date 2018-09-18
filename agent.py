from collections import deque
import numpy as np
from ppo import PPO
import numpy as np
import os
from episode import Episode
import torch

class Agent():
    def __init__(self, env, observation_shape, num_actions):
        self.num_actions = num_actions
        self.algo = PPO(observation_shape, num_actions)
        if os.path.exists("actor_weights"):
            self.algo.actor.load_state_dict(torch.load("actor_weights", map_location='cpu'))
        if os.path.exists("critic_weights"):
            self.algo.critic.load_state_dict(torch.load("critic_weights", map_location='cpu'))
        self.num_steps = 0
        self.num_episodes = 0
        self.total_num_steps = 0
        self.env = env
        self.episodes = deque([Episode()], maxlen=20)
        self.total_reward = 0
    
    def reset(self, learn=True):
        print("Total reward:", self.total_reward, "Num steps:", self.num_steps, flush=True)
        self.num_steps = 0
        self.total_reward = 0
        self.num_episodes += 1

        if len(self.episodes) >= 5 and learn:
            print("learning started")
            self.learn_from_memory()
            torch.save(self.algo.actor.state_dict(), "actor_weights")
            torch.save(self.algo.critic.state_dict(), "critic_weights")
            print("finished learning")

        self.episodes.append(Episode())
        # self.memory.clear()
    
    def store_in_memory(self, state, action, reward, new_state, done):
        self.episodes[-1].store_step(state, action, reward, new_state, done)
    
    def learn_from_memory(self):
        self.algo.fit(self.episodes[-1], bs=128)

        indexes = np.random.choice(len(self.episodes), size=3, replace=False)
        episodes = np.array(list(self.episodes))[indexes]
        for episode in episodes:
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
        
