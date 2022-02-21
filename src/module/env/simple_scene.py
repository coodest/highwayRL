import gym

from gym import spaces
from gym.envs.registration import EnvSpec
from src.module.context import Profile as P

from src.util.tools import *
import random


class SimpleScene:
    @staticmethod
    def make_env(render=False):
        env = SimpleSceneEnv()

        return env


class SimpleSceneEnv(gym.Env):
    def __init__(self, max_episode_steps=P.seq_len, seed=0) -> None:
        super().__init__()

        self.spec = EnvSpec("SimpleScene-v0", max_episode_steps=max_episode_steps)
        self.action_space = spaces.Discrete(2)
        self.seed(seed)
        self.action_space.seed(seed)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(max_episode_steps,), dtype=np.int32
        )
        self.sequence = []
        self.passing_reward = 0.1
        self.end_reward = 1
        for _ in range(max_episode_steps):
            self.sequence.append(self.action_space.sample())
    
    def seed(self, seed=None) -> None:
        np.random.seed(seed)

    def reset(self):
        self.done = False
        self.reward = 0
        self.info = dict()
        self.current_step = 0
        self.obs = np.zeros(len(self.sequence)) - 1

        return self.obs

    def step(self, action):
        if self.done is True:
            raise RuntimeError("Episode is done.")

        if action == self.sequence[self.current_step]:
            self.reward = self.passing_reward
        else:
            self.reward = self.end_reward
            self.done = True
        
        self.obs[self.current_step] = action
        self.current_step += 1
        if self.current_step == len(self.sequence):
            self.done = True

        return self.obs, self.reward, self.done, self.info



    
