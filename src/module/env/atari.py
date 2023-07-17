import gym
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.wrappers import TimeLimit, RecordVideo
from src.util.imports.numpy import np
from src.module.context import Profile as P
from src.util.tools import Funcs, Logger, IO
import cv2
import time

# import atari_py
from collections import deque
import random


class Atari:
    @staticmethod
    def make_env(render=False, is_head=False, use_projected_env=False):
        if P.sticky_action:
            repeat_action_probability = 0.25
            ver = "v0"
        else:  # Deterministic
            repeat_action_probability = 0.0
            ver = "v4"
        env = gym.make(
            f"{P.env_name}NoFrameskip-{ver}", 
            frameskip=1,
            repeat_action_probability=repeat_action_probability,
            # full_action_space=True,
            # render_mode='human',
        )

        env.seed(1)

        if is_head:
            env = TimeLimit(env.env, max_episode_steps=P.max_eval_episode_steps)
        else:
            env = TimeLimit(env.env, max_episode_steps=P.max_train_episode_steps)

        if render:
            env = RecordVideo(env, f"{P.video_dir}{P.env_name}/", episode_trigger=lambda episode_id: episode_id % P.render_every == 0)  # output every episode

        env = AtariPreprocessing(env)

        return env


class AtariPreprocessing(object):
    def __init__(
        self,
        environment,
        terminal_on_life_loss=False,
    ):
        self.environment = environment
        self.environment.action_space.dtype = np.int32
        self.terminal_on_life_loss = terminal_on_life_loss
        self.frame_skip = P.num_action_repeats
        self.screen_size = P.screen_size
        obs_dims = self.environment.observation_space
        self.buffer = np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)

        self.lives = 0
        self.life_termination = False

        self.observation = None

    @property
    def observation_space(self):
        return Box(
            low=0,
            high=1,
            shape=(P.stack_frames, P.screen_size, P.screen_size,),
            dtype=np.float32,
        )

    def seed(self, seed):
        self.environment.seed(seed)

    @property
    def action_space(self):
        return self.environment.action_space
    
    def sample_action(self):
        return self.environment.action_space.sample()

    @property
    def reward_range(self):
        return self.environment.reward_range

    @property
    def metadata(self):
        return self.environment.metadata

    def close(self):
        return self.environment.close()

    def update_observation(self, obs):
        if self.observation is None:
            self.observation = np.zeros(shape=(P.stack_frames, P.screen_size, P.screen_size), dtype=np.uint8)

        self.observation = np.vstack([
            self.observation[1:, :, :], np.expand_dims(obs, axis=0)
        ])

    def max_pooled_observation(self):
        return self.observation

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.environment.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self.observation = None
            self.environment.reset()
            self.update_observation(self.fetch_grayscale_frame())
        self.lives = self.environment.ale.lives()

        return self.observation

    def render(self, mode):
        """Renders the current screen, before preprocessing.
          if mode='rgb_array': numpy array, the most recent screen.
          if mode='human': bool, whether the rendering was successful.
        """
        return self.environment.render(mode)

    def step(self, action):
        accumulated_reward = 0.0
        is_terminal = False
        info = None
        frame_buffer = np.zeros([2, P.screen_size, P.screen_size])

        for t in range(self.frame_skip):
            _, reward, is_terminal, info = self.environment.step(action)
            accumulated_reward += reward
            if t == 2:
                frame_buffer[0] = self.fetch_grayscale_frame()
            elif t == 3:
                frame_buffer[1] = self.fetch_grayscale_frame()
            if is_terminal:
                break
        observation = frame_buffer.max(0)  # max pool over last two frames
        self.update_observation(observation)
        if self.terminal_on_life_loss:
            lives = self.environment.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not is_terminal  # Only set flag when not truly done
                is_terminal = True
            self.lives = lives

        return self.observation, accumulated_reward, is_terminal, info

    def fetch_grayscale_frame(self):
        self.environment.ale.getScreenGrayscale(self.buffer)
        frame = cv2.resize(self.buffer, (P.screen_size, P.screen_size), interpolation=cv2.INTER_LINEAR)
        return frame.copy() / 255  # pixel normalization
