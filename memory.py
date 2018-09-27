from collections import deque
import numpy as np

class Memory():
    def __init__(self, maxlen=1000):
        self.maxlen = maxlen
        self.states = None
        self.actions = np.array([], dtype=np.uint8)
        self.rewards = np.array([])
        self.new_states = None
        self.dones = np.array([])
    
    def append(self, state, action, reward, new_state, done):
        if self.states is not None:
            np.concatenate([self.states, np.expand_dims(state, 0)])[-self.maxlen:]
        else:
            self.states = np.expand_dims(state, 0)
        self.actions = np.append(self.actions, action)[-self.maxlen:]
        self.rewards = np.append(self.rewards, reward)[-self.maxlen:]
        if self.new_states is not None:
            np.concatenate([self.new_states, np.expand_dims(new_state, 0)])[-self.maxlen:]
        else:
            self.new_states = np.expand_dims(new_state, 0)
        self.dones = np.append(self.dones, done)[-self.maxlen:]
    
    def add_episode(self, episode):
        self.append(*episode.data())

    def get_batch(self, batch_size=32):
        indexes = np.random.choice(len(self.states), size=batch_size, replace=True)
        return (
            self.states[indexes],
            self.actions[indexes],
            self.rewards[indexes],
            self.new_states[indexes],
            self.dones[indexes],
        )
    
    def __len__(self):
        return len(self.states)