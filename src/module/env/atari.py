import gym
# from gym.spaces.discrete import Discrete
from gym.wrappers import TimeLimit, RecordVideo
from src.util.imports.numpy import np
from src.module.context import Profile as P
from src.util.tools import Funcs, Logger, IO
import cv2

# import atari_py
from collections import deque
# import random

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


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

        # env = AtariPreprocessing(env)
        env = wrap_dqn(env)

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
        self.stack_frames = P.stack_frames
        self.deterministic = P.deterministic
        obs_dims = self.environment.observation_space
        self.buffer = np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)

        self.lives = 0
        self.life_termination = False

        self.observation = None
        self.seed_value = 0

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.stack_frames, self.screen_size, self.screen_size,),
            dtype=np.float32,
        )

    def seed(self, seed):
        self.environment.seed(seed)
        self.seed_value = seed

    @property
    def action_space(self):
        return self.environment.action_space
    
    def sample_action(self):
        if self.deterministic:
            return np.random.randint(0, self.environment.action_space.n)
        else:
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
            self.observation = np.zeros(shape=(self.stack_frames, self.screen_size, self.screen_size), dtype=np.uint8)

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
            self.environment.reset(seed=self.seed_value)
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
        frame_buffer = np.zeros([2, self.screen_size, self.screen_size])

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
        frame = cv2.resize(self.buffer, (self.screen_size, self.screen_size), interpolation=cv2.INTER_LINEAR)
        return frame.copy() / 255  # pixel normalization


def wrap_dqn(env):
    """Apply a common set of wrappers for Atari games."""
    # env = EpisodicLifeEnv(env)
    # env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = FrameStack(env, 4)
    env = Sampler(env, deterministic=P.deterministic)
    
    return env


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True
        self.was_real_reset = False

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
            self.was_real_reset = True
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.was_real_reset = False
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer)[-2:], axis=0)

        # max_frame = np.stack(self._obs_buffer)[-1]

        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1))

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class Sampler(gym.Wrapper):
    def __init__(self, env, deterministic=True):
        gym.Wrapper.__init__(self, env)
        self.deterministic = deterministic

    def sample_action(self):
        if self.deterministic:
            return np.random.randint(0, self.env.action_space.n)
        else:
            return self.env.action_space.sample()


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k))

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        out = np.concatenate(self.frames, axis=2)
        return out


class ScaledFloatFrame(gym.ObservationWrapper):
    def _observation(self, obs):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(obs).astype(np.float32) / 255.0
