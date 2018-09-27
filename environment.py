import gym
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np
import time
from collections import deque

class SpeedTracker(gym.ActionWrapper):
    def __init__(self, env, time_diff=1):
        super(SpeedTracker, self).__init__(env)
        self.env = env
        self.steps = 0
        self.last_steps = 0
        self.last_time = time.time()
        self.time_diff = time_diff
        self._speed = 0
    
    def action(self, action):
        self.steps += 1
        now = time.time()

        diff = now - self.last_time
        if diff > self.time_diff:
            self._speed = round((self.steps / diff), 4)
            self.steps = 0
            self.last_time = now
        
        return action
    
    @property
    def speed(self):
        return self._speed
    
    @property
    def mean_reward(self):
        return self.env.mean_reward
        
class RewardTracker(gym.RewardWrapper):
    def __init__(self, env, time_diff=1):
        super(RewardTracker, self).__init__(env)
        self.env = env
        self.episodes_rewards = deque(maxlen=100)
        self.current_rewards = []

    def step(self, action):
        state, reward, done, info = super().step(action)
        if done:
            self.episodes_rewards.append(sum(self.current_rewards))
            self.current_rewards.clear()
        
        return state, reward, done, info
    
    @property
    def mean_reward(self):
        if len(self.episodes_rewards) > 0:
            return np.round(np.mean(list(self.episodes_rewards)), 3)
        else:
            return 0


    def reward(self, reward):
        self.current_rewards.append(reward)
        return reward


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
        self.env = RewardTracker(self.env)
        self.env = SpeedTracker(self.env)
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def mean_reward(self):
        return self.env.mean_reward
    
    @property
    def speed(self):
        return self.env.speed

    def render(self):
        self.env.render()

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)