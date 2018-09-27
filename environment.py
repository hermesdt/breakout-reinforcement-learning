import gym
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np

class ImageToPyTorch(gym.ObservationWrapper):
    """
    Change image shape to CWH
    """
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.array(observation).transpose(2, 0, 1)

class Environment():
    def __init__(self, game="PongNoFrameskip-v4"):
        self.env = make_atari(game)
        self.env = wrap_deepmind(self.env, frame_stack=True)
        self.env = ImageToPyTorch(self.env)
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space

    def render(self):
        self.env.render()

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)