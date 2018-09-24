import numpy as np

class Episode():
    def __init__(self):
        self.memory = []
        self._processed_data = None
    
    @property
    def win(self):
        return self.memory[-1][2] == 1
    
    @property
    def num_steps(self):
        return len(self.memory)
    
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
        return [d[key] for d in self._process_data()]
    
    def data(self):
        return self._process_data()


    def _process_data(self):
        if self._processed_data is not None:
            return self._processed_data

        states, actions, rewards, new_states, dones, steps = [], [], [], [], [], []
        n = 8
        gamma = 0.99
        self._processed_data = []
        
        #import pudb; pudb.set_trace()
        for idx, (state, action, reward, new_state, done) in enumerate(self.memory):
            s = state
            a = action
            n_s = self.memory[min(idx + n, len(self.memory) - 1)][3]
            d = done
            G = 0

            # remaining = min(len(self.memory) - n, )
            for i,offset in enumerate(range(idx, min(len(self.memory), idx+n))):
                G += (gamma**i) * self.memory[offset][2]
                
        
            states.append(s)
            actions.append(a)
            rewards.append(G)
            new_states.append(n_s)
            dones.append(d)
            steps.append(n)
            # print("G", G, "idx", idx)

            # self._processed_data.append((s, a, G, n_s, d, n))
        
        states = np.array(states).reshape([len(states), *states[0].shape[1:]])
        actions = np.array(actions)
        rewards = np.array(rewards)
        steps = np.array(steps)
        dones = np.array(dones, dtype=np.uint8)
        new_states = np.array(new_states).reshape([len(new_states), *new_states[0].shape[1:]])
        
        self._processed_data = (states, actions, rewards, new_states, dones, steps)
        return self._processed_data