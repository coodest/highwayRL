import gym
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.wrappers import TimeLimit, RecordVideo
from src.util.imports.numpy import np
from src.module.context import Profile as P
from src.util.tools import Funcs, Logger, IO
import cv2
import time
# import d4rl  # to load Mujoco envs


class Mujoco:
    @staticmethod
    def make_env(render=False, is_head=False, use_discrete_env=False):
        env_path = f"{P.env_dir}{P.env_name}.pkl"
        if is_head and not IO.file_exist(env_path):
            env = gym.make(
                f"{P.env_name}"
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
        
        if use_discrete_env:
            return DiscreteEnv(env)
        return env


class DiscreteEnv():
    def __init__(
        self,
        env,
        decimals=1,
    ):
        self.env = env
        self.decimals = decimals
        self.action_space = self.env.action_space
        self.frame_skip = P.stack_frames

    def reset(self):
        self.obs = list()
        state = np.around(self.env.reset(), decimals=self.decimals)
        for _ in range(self.frame_skip):
            self.obs.append(state)
        return np.ndarray.flatten(np.array(self.obs))

    def step(self, action):
        accumulated_reward = 0.0
        is_terminal = False

        for _ in range(self.frame_skip):
            state, reward, is_terminal, info = self.env.step(np.around(action, decimals=self.decimals))
            accumulated_reward += reward
            self.obs = self.obs[1:]
            self.obs.append(state)

            if is_terminal:
                break
        return np.ndarray.flatten(np.array(self.obs)), accumulated_reward, is_terminal, info
