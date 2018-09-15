import torch
from torch import nn
import numpy as np

class Actor(nn.Module):
    def __init__(self, observation_shape, num_actions, lr=0.001):
        super().__init__()
        self.linear1 = nn.Linear(observation_shape, 256)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(256, num_actions)
        self.softmax = nn.Softmax()
        
        self.optim = torch.optim.RMSprop(self.parameters(), lr=lr)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid1(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
    
    def train(self, data, td_error):
        (states, actions, rewards, new_states, dones) = data
        self.optim.zero_grad()
        # loss = self.criterion(self.forward(torch.Tensor(states)), targets)
        x = torch.Tensor(states)
        
        values = self.forward(x)[:,actions.max(1)]
        loss = torch.log(values).sum()
        loss.backward()
        self.optim.step()

class Critic(nn.Module):
    def __init__(self, observation_shape, num_actions, lr=0.001):
        super().__init__()
        self.linear1 = nn.Linear(observation_shape, 256)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(256, 1)
        self.nonlinearity = nn.ReLU()
        
        self.optim = torch.optim.RMSprop(self.parameters(), lr=lr)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid1(x)
        x = self.linear2(x)
        x = self.nonlinearity(x)
        return x
    
    def train(self, data):
        (states, actions, rewards, new_states, dones) = data
        x = states
        y_hat = self.forward(torch.Tensor(x))
        v_h = self.forward(torch.Tensor(new_states))
        v_h[dones == True] = 0
        y = torch.Tensor(rewards.reshape(1, -1)) + 0.98**64*v_h
        
        td_error = y - y_hat
        
        self.optim.zero_grad()
        # loss = self.criterion(y_hat, y)
        td_error.sum().backward()
        self.optim.step()
        
        return td_error
        
        

class PPO():
    def __init__(self, observation_shape, num_actions):
        self.num_actions = num_actions
        self.actor = Actor(observation_shape, num_actions)
        self.critic = Critic(observation_shape, num_actions)
        self.epsilon = 0.2
    
    def eval(self, state):
        probs = self.actor(
            torch.autograd.Variable(torch.Tensor(state), requires_grad=False)
        ).detach()
        
        # probs = np.full(num_actions, self.epsilon/num_actions)
        # action = np.argmax(probs)
        # probs[action] = 1 - self.epsilon + self.epsilon / num_actions
        # return probs
        
        #probs = np.round(probs, decimals=4)
        #probs[0] -= 1 - np.round(probs, decimals=4).sum()
        return probs
    
    def fit(self, data):
        states, actions, rewards, new_states, dones = [], [], [], [], []
        n_steps = 64
        df = 0.98
        
        for idx, (state, action, reward, new_state, done) in enumerate(list(data)):
            states.append(state)
            oh_action = [1 if i == action else 0 for i in range(self.num_actions)]
            actions.append(oh_action)

            reward = 0
            for df_idx, (_, _, r, _, _) in enumerate(list(data)[::-1][idx:min(idx+n_steps, len(data))]):
                reward += r*df**df_idx

            rewards.append(reward)
            new_states.append(new_state)
            dones.append(done)
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        new_states = np.array(new_states)
        
        data = (states, actions, rewards, new_states, dones)
        
        td_error = self.critic.train(data)
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
        