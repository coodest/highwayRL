import gym
from gym.spaces.box import Box
from gym.wrappers import TimeLimit, RecordVideo
from src.util.imports.numpy import np
from src.module.context import Profile as P
from src.util.tools import Funcs, Logger, IO
import cv2
import time


class Atari:
    @staticmethod
    def make_env(render=False, is_head=False):
        env_path = f"{P.env_dir}{P.env_name}.pkl"
        if is_head:
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
                full_action_space=True,
                # render_mode='human',
            )

            env.seed(2022)
            IO.write_disk_dump(env_path, env)
        else:
            while True:
                try:
                    env = IO.read_disk_dump(env_path)
                    break
                except Exception:
                    time.sleep(0.1)

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

        self.observation = None

    @property
    def observation_space(self):
        return Box(
            low=0,
            high=255,
            shape=(P.screen_size, P.screen_size),
            dtype=np.uint8,
        )

    @property
    def action_space(self):
        return self.environment.action_space

    @property
    def reward_range(self):
        return self.environment.reward_range

    @property
    def metadata(self):
        return self.environment.metadata

    def close(self):
        return self.environment.close()

    def update_observation(self):
        if self.observation is None:
            self.observation = np.zeros(shape=(P.stack_frames, P.screen_size * P.screen_size), dtype=np.uint8)

        self.observation = np.vstack([
            self.observation[1:, :], self.fetch_grayscale_frame()
        ])

    def reset(self):
        self.observation = None

        self.environment.reset()

        self.update_observation()

        return np.ndarray.flatten(self.observation)

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

        for _ in range(self.frame_skip):
            _, reward, is_terminal, info = self.environment.step(action)
            accumulated_reward += reward
            self.update_observation()

            if is_terminal:
                break

        return np.ndarray.flatten(self.observation), accumulated_reward, is_terminal, info

    def fetch_grayscale_frame(self, pixel_normalization=False):
        self.environment.ale.getScreenGrayscale(self.buffer)
        frame = cv2.resize(self.buffer, (P.screen_size, P.screen_size), interpolation=cv2.INTER_LINEAR)
        if pixel_normalization:
            return np.ndarray.flatten(frame.copy()) / 255  # pixel normalization
        else:
            return np.ndarray.flatten(frame.copy())
