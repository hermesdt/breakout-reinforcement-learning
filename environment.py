import gym
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch import nn
import skimage
from skimage.transform import resize as skresize
from collections import deque

def rgb2grey(frame):
    return skimage.color.rgb2grey(frame)

def resize(frame, output_shape):
    return skresize(frame, output_shape, anti_aliasing=True)

class Environment():
    def __init__(self, image_size = [80, 80]):
        self.env = gym.make("BreakoutDeterministic-v4")
        self.image_size = image_size
        self.last_frame = self.env.reset()
        self.last_info = None
        self.frames = deque(maxlen=2)
    
    def reset(self):
        self.frames.clear()
        if self.last_info and self.last_info['ale.lives'] <= 0:
            self.frames.append(self.env.reset())
        frame, reward, done, info = self.env.step(1)
        self.frames.append(frame)
        
        state = self.build_state()
        return state
    
    def render(self):
        self.env.render()
    
    def build_state(self):
        state = np.zeros([1, 2, *self.image_size], dtype=np.float)
        for idx, frame in enumerate(list(self.frames)):
            state[0, idx] = rgb2grey(resize(frame[50:], self.image_size))
        #s = np.stack([
        #    rgb2grey(resize(frame1, self.image_size)),
        #    rgb2grey(resize(frame2, self.image_size))
        #], axis=0).reshape([1, 2, *self.image_size])
        return state
    
    def step(self, action):
        frame, reward, done, info = self.env.step(action)
        self.frames.append(frame)
        state = self.build_state()
        
        if self.last_info and self.last_info['ale.lives'] > info['ale.lives']:
            done = True
            reward = -0.1

            
        self.last_frame = frame
        self.last_info = info
        
        return state, reward, done, info
