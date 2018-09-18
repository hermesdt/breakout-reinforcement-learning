from agent import Agent

class Runner():
    def __init__(self, env, observation_shape, num_actions):
        self.env = env
        self.agent = Agent(env, observation_shape, num_actions)
    
    def run(self, episodes=10, render=False, learn=True):
        done = False
        state = self.env.reset()
        
        for i in range(episodes):
            while not done:
                state, done = self.agent.step(state)
                if render:
                    self.env.render()
            
            self.agent.reset(learn=learn)
            state = self.env.reset()
            
            done = False
            # print("done")
            