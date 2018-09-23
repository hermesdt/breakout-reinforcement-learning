from collections import deque
import numpy as np

class Memory():
    def __init__(self, maxlen=1000):
        self.maxlen = maxlen
        self.states = None
        self.actions = np.array([])
        self.rewards = np.array([])
        self.new_states = None
        self.dones = np.array([])
        self.steps = np.array([])
    
    def append(self, state, action, reward, new_state, done, step):
        # import pudb; pudb.set_trace()
        if self.states is not None:
            self.states = np.concatenate([self.states, state])[-self.maxlen:]
        else:
            self.states = state
        self.actions = np.append(self.actions, action)[-self.maxlen:]
        self.rewards = np.append(self.rewards, reward)[-self.maxlen:]
        if self.new_states is not None:
            self.new_states = np.concatenate([self.new_states, new_state])[-self.maxlen:]
        else:
            self.new_states = new_state
        self.dones = np.append(self.dones, done)[-self.maxlen:]
        self.steps = np.append(self.steps, step)[-self.maxlen:]
    
    def add_episode(self, episode):
        self.append(*episode.data())

    def get_batch(self, batch_size=32):
        indexes = np.random.choice(len(self.states), size=batch_size, replace=False)
        return (
            self.states[indexes],
            self.actions[indexes],
            self.rewards[indexes],
            self.new_states[indexes],
            self.dones[indexes],
            self.steps[indexes]
        )
    
    def __len__(self):
        return len(self.states)