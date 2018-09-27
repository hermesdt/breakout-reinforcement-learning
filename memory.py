from collections import deque
import numpy as np

class Memory():
    def __init__(self, maxlen=1000):
        self.maxlen = maxlen
        self.memory = []
    
    # state, action, reward, new_state, done
    def append(self, data):
        self.memory.append(data)
        self.memory = self.memory[-self.maxlen:]

    def get_batch(self, batch_size=32):
        indexes = np.random.choice(len(self.memory), size=batch_size, replace=True)
        memories = [self.memory[i] for i in indexes]

        states = [m[0] for m in memories]
        actions = [m[1] for m in memories]
        rewards = [m[2] for m in memories]
        new_states = [m[3] for m in memories]
        dones = [m[4] for m in memories]

        o = (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(new_states),
            np.array(dones)
        )
        return o
    
    def __len__(self):
        return len(self.memory)