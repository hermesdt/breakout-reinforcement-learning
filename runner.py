from agent import Agent

class Runner():
    def __init__(self, env):
        self.env = env
        self.agent = Agent(env)
    
    def run(self, episodes=10, render=False):
        done = False
        state = self.env.reset()

        for i in range(episodes):
            while not done:
                state, _, done, _ = self.agent.step(state)
                if render:
                    self.env.render()
            
            self.agent.reset()
            state = self.env.reset()
            
            done = False
            # print("done")
            