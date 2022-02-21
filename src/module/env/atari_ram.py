import gym
from gym.spaces.box import Box
from gym.wrappers import TimeLimit, Monitor
from src.util.imports.numpy import np
import cv2
from src.module.context import Profile as P


class AtariRam:
    @staticmethod
    def make_env(render=False):
        # env = gym.make("{}Deterministic-v4".format(P.env_name), full_action_space=True)
        # env = gym.make("{}NoFrameskip-v4".format(P.env_name), full_action_space=True)
        env = gym.make("{}-ram-v4".format(P.env_name), full_action_space=True)
        env.seed(2022)

        env = TimeLimit(env.env, max_episode_steps=P.max_episode_steps)

        if render:
            env = Monitor(env, P.video_dir, force=True, video_callable=lambda episode_id: episode_id % P.render_every == 0)  # output every episode

        return env
