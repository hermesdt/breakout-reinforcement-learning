from agent import Agent
import time

class Runner():
    def __init__(self, env):
        self.env = env
        self.agent = Agent(env)
    
    def run(self, episodes=10, render=False, learn=True):
        done = False
        state = self.env.reset()

        for i in range(episodes):
            while not done:
                state, _, done, _ = self.agent.step(state, learn=learn)
                if render:
                    self.env.render()
                    time.sleep(0.02)
            
            self.agent.reset()
            state = self.env.reset()
            
            done = False
            # print("done")
            