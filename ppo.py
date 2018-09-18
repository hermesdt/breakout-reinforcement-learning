import torch
from torch import nn
import numpy as np
import math
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, observation_shape, num_actions, lr=0.001):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, 2)
        self.conv2 = nn.Conv2d(32, 16, 3, 2)
        self.linear1 = nn.Linear(5776, 256)
        self.linear2 = nn.Linear(256, num_actions)
        
        self.optim = torch.optim.SGD(self.parameters(), lr=lr)
    
    def forward(self, input):
        bs = input.size(0)
        x = nn.functional.leaky_relu(self.conv1(input))
        x = nn.functional.leaky_relu(self.conv2(x))
        x = x.view(bs, -1)
        x = nn.functional.leaky_relu(self.linear1(x))
        x = nn.functional.leaky_relu(self.linear2(x))
        #print("pre gumbel", x)
        x = nn.functional.softmax(x)
        # print("post gumbel", x)
        return x
    
    def train(self, data, td_error):
        (states, actions, rewards, new_states, dones) = data
        self.optim.zero_grad()
        x = torch.Tensor(states)
        td_error = td_error.detach()
        td_error.requires_grad=False

        #values = self.forward(x)[:,actions]*td_error
        dist = Categorical(self.forward(x))
        log_probs = -dist.log_prob(torch.Tensor(actions))*td_error
        loss = log_probs.sum()
        print("loss", loss.item())
        loss.backward()
        self.optim.step()

class Critic(nn.Module):
    def __init__(self, observation_shape, num_actions,
                lr=0.001):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, 2)
        self.conv2 = nn.Conv2d(32, 16, 3, 2)
        self.linear1 = nn.Linear(5776, 256)
        self.linear2 = nn.Linear(256, 1)
        
        self.optim = torch.optim.SGD(self.parameters(), lr=lr)
    
    def forward(self, input):
        x = nn.functional.leaky_relu(self.conv1(input))
        x = nn.functional.leaky_relu(self.conv2(x))
        bs = input.size(0)
        x = x.view(bs, -1)
        x = nn.functional.leaky_relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def train(self, data, gamma=0.99):
        (states, actions, rewards, new_states, dones) = data
        bs = len(states)
        x = torch.Tensor(states)
        y_hat = self.forward(x)
        v_h = self.forward(torch.Tensor(new_states))
        v_h[np.argwhere(dones)] = 0
        y = torch.Tensor(rewards.reshape(bs, -1)) + gamma*v_h
        # import pdb; pdb.set_trace()
        
        td_error = y - y_hat
        
        self.optim.zero_grad()
        td_error.pow(2).sum().backward()
        self.optim.step()
        
        return td_error

class PPO():
    def __init__(self, observation_shape, num_actions, gamma=0.99):
        self.num_actions = num_actions
        self.gamma = gamma
        self.actor = Actor(observation_shape, num_actions, lr=0.00005)
        self.critic = Critic(observation_shape, num_actions, lr=0.00005)
        self.epsilon = 0.2
    
    def eval(self, state):
        probs = self.actor(
            torch.autograd.Variable(torch.Tensor(state), requires_grad=False)
        ).detach()
        
        return probs
    
    def fit(self, episode, bs=64):
        for b in range(math.ceil(len(episode)/bs)):
            td_error = self.critic.train(episode[b:b+bs])
            self.actor.train(episode[b:b+bs], td_error)
