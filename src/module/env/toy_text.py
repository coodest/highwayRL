import gym
from gym.spaces.box import Box
from gym.wrappers import TimeLimit, RecordVideo
from src.util.imports.numpy import np
from src.module.context import Profile as P
from src.util.tools import Funcs, Logger, IO
import time
import types


class ToyText:
    @staticmethod
    def make_env(render=False, is_head=False):
        env_path = f"{P.env_dir}{P.env_name}.pkl"
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
                    env = TimeLimit(env.env, max_episode_steps=P.max_episode_steps)
                    env = TTEnv(env)
                    env.reset(seed=123)
                    IO.write_disk_dump(env_path, env)
                    break

        if render:
            env = RecordVideo(env, f"{P.video_dir}{P.env_name}/", episode_trigger=lambda episode_id: episode_id % P.render_every == 0)  # output every episode

        # add function dynamically
        def sample_action(self):
            return self.action_space.sample()
        env.sample_action = types.MethodType(sample_action, env)

        return env
    
class TTEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, info = super().step(action)
        observation = np.array(observation)
        return observation, reward, terminated, info
    
    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        observation = np.array(observation)
        return observation
