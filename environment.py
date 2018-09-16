import gym
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch import nn
import skimage
from skimage.transform import resize as skresize

def rgb2grey(frame):
    return skimage.color.rgb2grey(frame)

def resize(frame, output_shape):
    return skresize(frame, output_shape)

class Environment():
    def __init__(self, image_size = [80, 80]):
        self.env = gym.make("Breakout-v4")
        self.image_size = image_size
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
        frame_shape = frame1.shape
        s = np.stack([
            rgb2grey(resize(frame1, self.image_size)),
            rgb2grey(resize(frame2, self.image_size))
        ], axis=0).reshape([1, 2, *self.image_size])
        return s
    
    def step(self, action):
        frame, reward, done, info = self.env.step(action)
        state = self.build_state(self.last_frame, frame)
        
        if self.last_info and self.last_info['ale.lives'] > info['ale.lives']:
            done = True
            reward = -1
            
        self.last_frame = frame
        self.last_info = info
        
        return state, reward, done, info