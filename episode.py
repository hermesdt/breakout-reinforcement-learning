import numpy as np

class Episode():
    def __init__(self):
        self.memory = []
        self._processed_data = None
    
    def store_step(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    #def __iter__(self):
    #    if self._processed_data is None:
    #        self._process_data()
    #    
    #    for data in self._processed_data:
    #        yield data

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, key):
        if self._processed_data is None:
            self._process_data()
        
        return [d[key] for d in self._processed_data]
    
    def _process_data(self):
        states, actions, rewards, new_states, dones = [], [], [], [], []
        
        for idx, (state, action, reward, new_state, done) in enumerate(self.memory):
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            new_states.append(new_state)
            dones.append(done)
        
        states = np.array(states).reshape([len(states), *states[0].shape[1:]])
        actions = np.array(actions)
        rewards = np.array(rewards)
        new_states = np.array(new_states).reshape([len(new_states), *new_states[0].shape[1:]])
        
        self._processed_data = (states, actions, rewards, new_states, dones)