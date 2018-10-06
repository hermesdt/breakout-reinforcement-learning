import torch
from torch import nn
import numpy as np
import os, re
from math import log
from collections import OrderedDict
from param_noise_layer import ParamNoiseLayer
from contextlib import contextmanager

class DQNParamNoise(nn.Module):
    def __init__(self, env, gamma=0.99, lr=0.00001, target_dqn=None, sigma=0):
        super().__init__()

        self.sigma = 0.017
        self.delta = -log(1 - self.sigma + self.sigma / env.action_space.n)

        self.lr = lr
        self.num_actions = env.action_space.n
        self.gamma = gamma
        self.target_dqn = target_dqn
        self.num_iterations = 0
        self.sigma_update_steps = 50

        self.conv1 = nn.Conv2d(4, 16, 8, 4)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.linear1 = ParamNoiseLayer(2592, 256, sigma=self.sigma)
        self.layer_norm1 = nn.LayerNorm(256)
        self.linear2 = ParamNoiseLayer(256, env.action_space.n, sigma=self.sigma)
        self.noise_layers = [
            self.linear1, self.linear2
        ]
        
        self.optim = torch.optim.RMSprop(self.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

        if os.path.exists("dqn_weights") and self.target_dqn is not None:
            self.load_state_dict(torch.load("dqn_weights", map_location='cpu'))
    
    def forward(self, input):
        bs = input.size(0)
        x = nn.functional.leaky_relu(self.conv1(input))
        x = nn.functional.leaky_relu(self.conv2(x))
        x = x.view(bs, -1)
        x = nn.functional.leaky_relu(self.linear1(x))
        x = self.layer_norm1(x)
        x = nn.functional.leaky_relu(self.linear2(x))
        return x

    def eval(self, state):
        return self(torch.Tensor(state))
    
    def episode_start(self):
        for noise_layer in self.noise_layers:
            noise_layer.reset_noise()
    
    @contextmanager
    def non_perturbed(self):
        weight_noises = []
        bias_noises = []

        for layer in self.noise_layers:
            weight_noises.append(layer.weight_noise)
            bias_noises.append(layer.bias_noise)

            layer.weight_noise = torch.zeros(layer.weight_noise.size())
            layer.bias_noise = torch.zeros(layer.bias_noise.size())
        
        yield

        for idx, layer in enumerate(self.noise_layers):
            layer.weight_noise = weight_noises[idx]
            layer.bias_noise = bias_noises[idx]

    def train(self, data):
        (states, actions, rewards, new_states, dones) = data
        bs = len(states)

        self.optim.zero_grad()
        y_hat = self.eval(states).gather(1, torch.Tensor(actions).long().unsqueeze(-1))
        v_h = self.target_dqn.eval(torch.Tensor(new_states)).max(1)[0].view(-1, 1)
        v_h[np.argwhere(dones).reshape(-1)] = 0
        y = torch.Tensor(rewards.reshape(bs, -1)) + self.gamma*v_h
        y = y.detach()

        loss = self.criterion(y_hat, y)

        loss.backward()
        self.optim.step()
        self.num_iterations += 1

        states_t = torch.Tensor(states)
        q_perturbed_values = self(states_t)
        with self.non_perturbed():
            q_values = self(states_t)
        
        probs = torch.nn.functional.softmax(q_values, dim=-1)
        perturbed_probs = torch.nn.functional.softmax(q_perturbed_values, dim=-1)
        
        distance = nn.functional.kl_div(probs, perturbed_probs)
        if distance < self.delta:
            self.sigma *= 1.01
        else:
            self.sigma /= 1.01
        
        for layer in self.noise_layers:
            layer.sigma = self.sigma
            layer.reset_noise()

        if self.num_iterations % 100 == 0:
            state_dict = self.state_dict().copy()
            for key in self.state_dict().keys():
                if re.match(r"target_dqn", key):
                    del state_dict[key]

            self.target_dqn.load_state_dict(state_dict)
            for layer in self.target_dqn.noise_layers:
                layer.sigma = 0

    def save(self):
        torch.save(self.state_dict(), "dqn_weights")
