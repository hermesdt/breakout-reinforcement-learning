class Episode():
    def __init__(self):
        self.memory = []
    
    def store_step(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def __iter__(self):
        for data in self.memory:
            yield data

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, key):
        return self.memory[key]