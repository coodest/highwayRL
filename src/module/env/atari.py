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

        env = AtariPreprocessing(env, terminal_on_life_loss=not is_head)

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


class Env():
    def __init__(self, is_head):
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', 123)
        if is_head:
            self.ale.setInt('max_num_frames_per_episode', P.max_eval_episode_steps)
        else:
            self.ale.setInt('max_num_frames_per_episode', P.max_train_episode_steps)
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(str(P.env_name).lower()))  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = P.stack_frames  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=P.stack_frames)
        self.training = True  # Consistent with model training mode

    def sample_action(self):
        return self.actions[random.randrange(len(self.actions))]

    def _get_state(self):
        frame = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return frame / 255

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(np.zeros([84, 84]))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return np.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = np.zeros([2, 84, 84])
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        return np.stack(list(self.state_buffer), 0), reward, done, {}

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
