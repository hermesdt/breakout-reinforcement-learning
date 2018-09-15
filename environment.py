import gym
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch import nn
import skimage

def rgb2grey(frame):
    return skimage.color.rgb2grey(frame)

class Environment():
    def __init__(self):
        self.env = gym.make("Breakout-v4")
        self.last_frame = self.env.reset()
        self.last_info = None
    
    def reset(self):
        self.last_frame = self.env.reset()
        self.last_frame, reward, done, info = self.env.step(0)
        self.last_frame, reward, done, info = self.env.step(1)
        self.last_frame, reward, done, info = self.env.step(2)
        frame, reward, done, self.last_info = self.env.step(3)
        
        state = self.build_state(self.last_frame, frame)
        self.last_frame = frame
        
        return state
    
    def render(self):
        self.env.render()
    
    def build_state(self, frame1, frame2):
        return np.stack([
            rgb2grey(frame1),
            rgb2grey(frame2)
        ], axis=2).reshape(-1)
    
    def step(self, action):
        frame, reward, done, info = self.env.step(action)
        state = np.stack([
            rgb2grey(self.last_frame),
            rgb2grey(frame)
        ], axis=2).reshape(-1)
        
        if self.last_info and self.last_info['ale.lives'] > info['ale.lives']:
            done = True
            reward = -1
            
        self.last_frame = frame
        self.last_info = info
        
        return state, reward, done, info