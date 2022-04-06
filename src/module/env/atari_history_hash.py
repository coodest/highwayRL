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

        env = AtariHistoryHashPreprocessing(
            env,
            frame_skip=P.num_action_repeats,
            max_random_noops=P.max_random_noops
        )

        return env


class AtariHistoryHashPreprocessing(object):
    """A class implementing image preprocessing for Atari 2600 agents.

    Specifically, this provides the following subset from the JAIR paper
    (Bellemare et al., 2013) and Nature DQN paper (Mnih et al., 2015):

      * Frame skipping (defaults to 4).
      * Terminal signal when a life is lost (off by default).
      * Grayscale and max-pooling of the last two frames.
      * Downsample the screen to a square image (defaults to 84x84).

    More generally, this class follows the preprocessing guidelines set down in
    Machado et al. (2018), "Revisiting the Arcade Learning Environment:
    Evaluation Protocols and Open Problems for General Agents".

    It also provides random starting no-ops, which are used in the Rainbow, Apex
    and R2D2 papers.
    """

    def __init__(
        self,
        environment,
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

        self.obs_dims = self.environment.observation_space
        # Stores temporary observations used for pooling over two successive
        # frames.
        self.screen_buffer = list()

        self.game_over = False
        self.lives = 0  # Will need to be set by reset().

    @property
    def observation_space(self):
        # Return the observation space adjusted to match the shape of the processed
        # observations.
        return Box(
            low=0,
            high=255,
            shape=(self.screen_size, self.screen_size, 1),
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

    def apply_random_noops(self):
        """Steps self.environment with random no-ops."""
        if self.max_random_noops <= 0:
            return
        # Other no-ops implementations actually always do at least 1 no-op. We
        # follow them.
        no_ops = self.environment.np_random.randint(1, self.max_random_noops + 1)
        for _ in range(no_ops):
            _, _, game_over, _ = self.environment.step(0)
            if game_over:
                self.environment.reset()

    def reset(self):
        """Resets the environment.

        Returns:
          observation: numpy array, the initial observation emitted by the
            environment.
        """
        self.environment.reset()
        self.apply_random_noops()

        self.lives = self.environment.ale.lives()
        self.screen_buffer = list()
        self._fetch_grayscale_observation(self.screen_buffer)
        return self._pool_and_resize()

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
            screen_snapshot, reward, game_over, info = self.environment.step(action)
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
            elif time_step >= self.frame_skip - 1:
                self._fetch_grayscale_observation(self.screen_buffer)

        # Pool the last two observations.
        observation = self._pool_and_resize()

        self.game_over = game_over
        return observation, accumulated_reward, is_terminal, info

    def _fetch_grayscale_observation(self, output):
        """Returns the current observation in grayscale.

        The returned observation is stored in 'output'.

        Args:
          output: numpy array, screen buffer to hold the returned observation.

        Returns:
          observation: numpy array, the current observation in grayscale.
        """
        f = np.empty((self.obs_dims.shape[0], self.obs_dims.shape[1]), dtype=np.uint8)
        self.environment.ale.getScreenGrayscale(f)
        hash_str = Funcs.matrix_hashing(f)
        output.append(hash_str)

    def _pool_and_resize(self):
        return Funcs.matrix_hashing(self.screen_buffer)
