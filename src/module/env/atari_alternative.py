import gym
from gym.spaces.box import Box
from gym.wrappers import TimeLimit, Monitor
from src.util.imports.numpy import np
import cv2
from src.module.context import Profile as P
from src.util.tools import Funcs


class Atari:
    @staticmethod
    def make_env(render=False):
        env = gym.make("{}Deterministic-v4".format(P.env_name), full_action_space=True)
        # env = gym.make("{}NoFrameskip-v4".format(P.env_name), full_action_space=True)
        env.seed(2022)

        env = TimeLimit(env.env, max_episode_steps=P.max_episode_steps)

        if render:
            env = Monitor(env, P.video_dir, force=True, video_callable=lambda episode_id: episode_id % P.render_every == 0)  # output every episode

        env = VanillaEnv(
            env,
            frame_skip=P.num_action_repeats,
            max_random_noops=P.max_random_noops
        )  # TODO: make code connection

        return env


class VanillaEnv():
    def __init__(self, args):
        self.args = args
        if args.sticky:
            # frameskip is deterministic
            self.env = gym.make(args.env + 'Deterministic-v0').env
        else:
            self.env = gym.make(args.env + 'Deterministic-v4').env

        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(84, 84, args.frames), dtype=np.float32)
        assert type(self.action_space) is gym.spaces.discrete.Discrete
        self.acts_dims = [self.action_space.n]
        self.obs_dims = list(self.observation_space.shape)

        self.render = self.env.render

        self.reset()
        self.env_info = {
            'Steps': self.process_info_steps,  # episode steps
            'Rewards@green': self.process_info_rewards  # episode cumulative rewards
        }

    def get_new_frame(self):
        # standard wrapper for atari
        frame = self.env._get_obs().astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        self.last_frame = frame.copy()
        return frame

    def get_obs(self):
        return self.last_obs.copy()

    def get_frame(self):
        return self.last_frame.copy()

    def process_info_steps(self, obs, reward, info):
        self.steps += 1
        return self.steps

    def process_info_rewards(self, obs, reward, info):
        self.rewards += reward
        return self.rewards

    def process_info(self, obs, reward, info):
        return {
            self.remove_color(key): value_func(obs, reward, info)
            for key, value_func in self.env_info.items()
        }

    def remove_color(key):
        for i in range(len(key)):
            if key[i] == '@':
                return key[:i]
        return key

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info = self.process_info(obs, reward, info)
        self.frames_stack = self.frames_stack[1:] + [self.get_new_frame()]
        self.last_obs = np.stack(self.frames_stack, axis=-1)
        if self.steps == self.args.test_timesteps: done = True
        return self.last_obs.copy(), reward, done, info

    def reset_ep(self):
        self.steps = 0
        self.rewards = 0.0

    def reset(self):
        self.reset_ep()
        while True:
            flag = True
            self.env.reset()
            for _ in range(max(self.args.noop - self.args.frames, 0)):
                _, _, done, _ = self.env.step(0)
                if done:
                    flag = False
                    break
            if flag: break

        self.frames_stack = []
        for _ in range(self.args.frames):
            self.env.step(0)
            self.frames_stack.append(self.get_new_frame())

        self.last_obs = np.stack(self.frames_stack, axis=-1)
        return self.last_obs.copy()
