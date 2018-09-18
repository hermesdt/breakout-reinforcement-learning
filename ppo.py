import torch
from torch import nn
import numpy as np
import math

class Actor(nn.Module):
    def __init__(self, observation_shape, num_actions, lr=0.001):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, 2)
        self.conv2 = nn.Conv2d(32, 16, 3, 2)
        self.linear1 = nn.Linear(31824, 256)
        self.linear2 = nn.Linear(256, num_actions)
        
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, input):
        bs = input.size(0)
        x = nn.functional.relu(self.conv1(input))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(bs, -1)
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear2(x))

        x = nn.functional.gumbel_softmax(x, tau=1)
        return x
    
    def train(self, data, td_error):
        (states, actions, rewards, new_states, dones) = data
        self.optim.zero_grad()
        x = torch.Tensor(states)
        
        values = self.forward(x)[:,actions.max(1)]
        loss = torch.log(values).sum()
        loss.backward()
        self.optim.step()

class Critic(nn.Module):
    def __init__(self, observation_shape, num_actions,
                lr=0.001):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, 2)
        self.conv2 = nn.Conv2d(32, 16, 3, 2)
        self.linear1 = nn.Linear(31824, 256)
        self.linear2 = nn.Linear(256, 1)
        
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, input):
        x = nn.functional.relu(self.conv1(input))
        x = nn.functional.relu(self.conv2(x))
        bs = input.size(0)
        x = x.view(bs, -1)
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear2(x))
        return x
    
    def train(self, data, gamma=0.99, n_step=16):
        (states, actions, rewards, new_states, dones) = data
        x = torch.Tensor(states)

        y_hat = self.forward(x)
        v_h = self.forward(torch.Tensor(new_states))
        v_h[dones == True] = 0
        y = torch.Tensor(rewards.reshape(1, -1)) + gamma**n_step*v_h
        
        td_error = y - y_hat
        
        self.optim.zero_grad()
        td_error.sum().backward()
        self.optim.step()
        
        return td_error

class PPO():
    def __init__(self, observation_shape, num_actions, gamma=0.99, n_steps=16):
        self.num_actions = num_actions
        self.gamma = gamma
        self.n_steps = n_steps
        self.actor = Actor(observation_shape, num_actions, lr=0.00025)
        self.critic = Critic(observation_shape, num_actions, lr=0.00025)
        self.epsilon = 0.2
    
    def eval(self, state):
        probs = self.actor(
            torch.autograd.Variable(torch.Tensor(state), requires_grad=False)
        ).detach()
        
        return probs
    
    def fit(self, episode, bs=32):
        states, actions, rewards, new_states, dones = [], [], [], [], []
        
        for idx, (state, action, reward, new_state, done) in enumerate(episode):
            states.append(state)
            oh_action = [1 if i == action else 0 for i in range(self.num_actions)]
            actions.append(oh_action)

            reward = 0
            for df_idx, (_, _, r, _, _) in enumerate(episode[idx:min(idx+self.n_steps, len(episode))][::-1]):
                reward += r*self.gamma**df_idx

            rewards.append(reward)
            new_states.append(episode[min(idx+self.n_steps, len(episode)-1)][3])
            dones.append(done)
        
        states = np.array(states).reshape([len(states), *states[0].shape[1:]])
        actions = np.array(actions)
        rewards = np.array(rewards)
        new_states = np.array(new_states).reshape([len(new_states), *new_states[0].shape[1:]])
        
        data = (states, actions, rewards, new_states, dones)

        for b in range(math.ceil(len(data)/bs)):
            td_error = self.critic.train(data[b:b+bs])
            self.actor.train(data, td_error)
        
        
# env = Environment()
# ppo = PPO()
# data = [(env.reset(), 1, 0, env.reset(), False)]
# # PPO().fit(data)
# action_probs = PPO().eval(env.reset())
# probs_sum = sum(p for p in action_probs)
# action_probs = action_probs/probs_sum
# print(action_probs.type())
# np.random.choice(range(num_actions), p=action_probs)
        
