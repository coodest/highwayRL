import gym
from gym.spaces.box import Box
from gym.wrappers import TimeLimit, RecordVideo
from src.util.imports.numpy import np
from src.module.context import Profile as P
from src.util.tools import Funcs, Logger
import cv2


class Atari:
    @staticmethod
    def make_env(render=False, obs_type=None, is_head=False):
        if P.sticky_action:
            repeat_action_probability = 0.25
            ver = "v0"
        else:  # Deterministic
            repeat_action_probability = 0.0
            ver = "v4"
        if obs_type in ["classic", "historical_action"]:
            # env = gym.make(f"{P.env_name}Deterministic-{ver}", full_action_space=True)
            env = gym.make(
                f"{P.env_name}NoFrameskip-{ver}", 
                frameskip=1,
                repeat_action_probability=repeat_action_probability,
                full_action_space=True
            )
        if obs_type == "ram":
            env = gym.make(
                f"{P.env_name}-ramNoframeskip-{ver}", 
                frameskip=1,
                repeat_action_probability=repeat_action_probability,
                full_action_space=True
            )

        env.seed(2022)

        env = TimeLimit(env.env, max_episode_steps=P.max_episode_steps)

        if render:
            env = RecordVideo(env, f"{P.video_dir}{P.env_name}/", episode_trigger=lambda episode_id: episode_id % P.render_every == 0)  # output every episode

        env = AtariPreprocessing(
            env,
            obs_type=obs_type,
            frame_skip=P.num_action_repeats,
            max_random_noops=P.max_random_noops,
        )

        return env


class AtariPreprocessing(object):
    def __init__(
        self,
        environment,
        obs_type,
        frame_skip=4,
        terminal_on_life_loss=False,
        screen_size=P.screen_size,
        max_random_noops=0,
    ):
        """Constructor for an Atari 2600 preprocessor.

        Args:
          environment: Gym environment whose observations are preprocessed.
          frame_skip: int, the frequency at which the agent experiences the game.
          terminal_on_life_loss: bool, If True, the step() method returns
            is_terminal=True whenever a life is lost. See Mnih et al. 2015.
          screen_size: int, size of a resized Atari 2600 frame.
          max_random_noops: int, maximum number of no-ops to apply at the beginning
            of each episode to reduce determinism. These no-ops are applied at a
            low-level, before frame skipping.

        Raises:
          ValueError: if frame_skip or screen_size are not strictly positive.
        """
        if frame_skip <= 0:
            raise ValueError(
                "Frame skip should be strictly positive, got {}".format(frame_skip)
            )
        if screen_size <= 0:
            raise ValueError(
                "Target screen size should be strictly positive, got {}".format(
                    screen_size
                )
            )

        self.environment = environment
        self.terminal_on_life_loss = terminal_on_life_loss
        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.max_random_noops = max_random_noops
        self.environment.action_space.dtype = np.int32

        obs_dims = self.environment.observation_space
        # Stores temporary observations used for pooling over two successive
        # frames.
        self.obs_type = obs_type
        if obs_type == "classic":
            self.screen_buffer = [
                np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
                np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
            ]
        if obs_type == "historical_action":
            self.action_list = []
        if obs_type == "ram":
            self.ram_buffer = []

        self.game_over = False
        self.lives = 0  # Will need to be set by reset().

    @property
    def observation_space(self):
        # Return the observation space adjusted to match the shape of the processed
        # observations.
        if self.obs_type == "classic":
            return Box(
                low=0,
                high=255,
                shape=(128,),
                dtype=np.uint8,
            )
        if self.obs_type == "historical_action":
            return None
        if self.obs_type == "ram":
            return None

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

    def apply_random_noops(self):
        """Steps self.environment with random no-ops."""
        if self.max_random_noops <= 0:
            return self.environment.reset()
        # Other no-ops implementations actually always do at least 1 no-op. We
        # follow them.
        no_ops = self.environment.np_random.integers(1, self.max_random_noops + 1)
        for _ in range(no_ops):
            obs, _, game_over, _ = self.environment.step(0)
            if game_over:
                obs = self.environment.reset()
        return obs

    def reset(self):
        """Resets the environment.

        Returns:
          observation: numpy array, the initial observation emitted by the
            environment.
        """
        self.environment.reset()
        obs = self.apply_random_noops()

        self.lives = self.environment.ale.lives()

        if self.obs_type == "classic":
            self._fetch_grayscale_observation(self.screen_buffer[0])
            self.screen_buffer[1].fill(0)
            return self._pool_and_resize()
        if self.obs_type == "historical_action":
            self.action_list = []
            return self.action_list.copy()
        if self.obs_type == "ram":
            self.ram_buffer = []
            self.ram_buffer.append(obs)
            return self._compute()

    def render(self, mode):
        """Renders the current screen, before preprocessing.

        This calls the Gym API's render() method.

        Args:
          mode: Mode argument for the environment's render() method.
            Valid values (str) are:
              'rgb_array': returns the raw ALE image.
              'human': renders to display via the Gym renderer.

        Returns:
          if mode='rgb_array': numpy array, the most recent screen.
          if mode='human': bool, whether the rendering was successful.
        """
        return self.environment.render(mode)

    def step(self, action):
        """Applies the given action in the environment.

        Remarks:

          * If a terminal state (from life loss or episode end) is reached, this may
            execute fewer than self.frame_skip steps in the environment.
          * Furthermore, in this case the returned observation may not contain valid
            image data and should be ignored.

        Args:
          action: The action to be executed.

        Returns:
          observation: numpy array, the observation following the action.
          reward: float, the reward following the action.
          is_terminal: bool, whether the environment has reached a terminal state.
            This is true when a life is lost and terminal_on_life_loss, or when the
            episode is over.
          info: Gym API's info data structure.
        """
        accumulated_reward = 0.0
        is_terminal = False
        info = None
        game_over = None

        for time_step in range(self.frame_skip):
            # We bypass the Gym observation altogether and directly fetch the
            # grayscale image from the ALE. This is a little faster.
            snapshot, reward, game_over, info = self.environment.step(action)
            accumulated_reward += reward

            if self.terminal_on_life_loss:
                new_lives = self.environment.ale.lives()
                is_terminal = game_over or new_lives < self.lives
                self.lives = new_lives
            else:
                is_terminal = game_over

            if is_terminal:
                break
            # We max-pool over the last two frames, in grayscale.
            elif time_step >= self.frame_skip - 2:
                if self.obs_type == "classic":
                    t = time_step - (self.frame_skip - 2)
                    self._fetch_grayscale_observation(self.screen_buffer[t])
                if self.obs_type == "ram":
                    if time_step >= self.frame_skip - 1:
                        self.ram_buffer.append(snapshot)

        # Pool the last two observations.
        if self.obs_type == "classic":
            observation = self._pool_and_resize()
        if self.obs_type == "historical_action":
            self.action_list.append(action)
            observation = self.action_list.copy()
        if self.obs_type == "ram":
            observation = self._compute()

        self.game_over = game_over
        return observation, accumulated_reward, is_terminal, info

    def _compute(self):
        # return Funcs.matrix_hashing(self.ram_buffer)
        return np.array(self.ram_buffer[-1])  # shape (128,)

    def _fetch_grayscale_observation(self, output):
        """Returns the current observation in grayscale.

        The returned observation is stored in 'output'.

        Args:
          output: numpy array, screen buffer to hold the returned observation.

        Returns:
          observation: numpy array, the current observation in grayscale.
        """
        self.environment.ale.getScreenGrayscale(output)
        return output

    def _pool_and_resize(self):
        """Transforms two frames into a Nature DQN observation.

        For efficiency, the transformation is done in-place in self.screen_buffer.

        Returns:
          transformed_screen: numpy array, pooled, resized screen.
        """
        # Pool if there are enough screens to do so.
        if self.frame_skip > 1:
            np.maximum(
                self.screen_buffer[0], self.screen_buffer[1], out=self.screen_buffer[0]
            )

        transformed_image = cv2.resize(
            self.screen_buffer[0],
            (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_LINEAR,
        )

        int_image = np.asarray(transformed_image, dtype=np.uint8)
        # return np.expand_dims(int_image, axis=2)
        return np.ndarray.flatten(int_image)
