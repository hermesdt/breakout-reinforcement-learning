import torch
from torch import nn
import numpy as np
import math, os, re
from collections import OrderedDict

class DQN(nn.Module):
    def __init__(self, env, gamma=0.99, lr=0.00001, target_dqn=None):
        super().__init__()

        self.lr = lr
        self.num_actions = env.action_space.n
        self.gamma = gamma
        self.target_dqn = target_dqn
        self.num_iterations = 0

        self.conv1 = nn.Conv2d(4, 16, 8, 4)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.linear1 = nn.Linear(2592, 256)
        self.linear2 = nn.Linear(256, env.action_space.n)
        
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
        x = nn.functional.leaky_relu(self.linear2(x))
        return x

    def eval(self, state):
        return self.forward(torch.Tensor(state))
    
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

        if self.num_iterations % 100 == 0:
            state_dict = self.state_dict().copy()
            for key in self.state_dict().keys():
                if re.match(r"target_dqn", key):
                    del state_dict[key]

            self.target_dqn.load_state_dict(state_dict)

    def save(self):
        torch.save(self.state_dict(), "dqn_weights")
