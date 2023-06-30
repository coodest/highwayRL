import gym
from gym.spaces.box import Box
from gym.wrappers import TimeLimit, RecordVideo
from src.util.imports.numpy import np
from src.module.context import Profile as P
from src.util.tools import Funcs, Logger, IO
import time


class Box2D:
    @staticmethod
    def make_env(render=False, is_head=False):
        env_path = P.env_dir + "env.pkl"
        env = None
        while True:
            try:
                time.sleep(0.01)
                env = IO.read_disk_dump(env_path)
                break
            except Exception:
                if is_head:
                    Logger.log("head_actor create the env")
                    env = gym.make(P.env_name)
                    # env.reset(seed=2022)
                    IO.write_disk_dump(env_path, env)

        # if is_head:
        #     env = TimeLimit(env.env, max_episode_steps=P.max_eval_episode_steps)
        # else:
        #     env = TimeLimit(env.env, max_episode_steps=P.max_train_episode_steps)

        return env
