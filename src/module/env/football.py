from src.util.tools import Logger, IO
from src.module.context import Profile as P
import time
import types
import gym
from src.util.imports.numpy import np
from gfootball.env import *
from gfootball.env.scenario_builder import *
from gfootball.env.config import *
from gfootball.scenarios import *
from absl import logging


class Football:
    @staticmethod
    def make_env(render=False, is_head=False, init_points=2 if not P.render else 1):
        logging.set_verbosity("error")
        assert init_points > 0, "init_points should be more than zero"
        envs = list()
        for env_id in range(init_points):
            env = create_environment(
                env_name=P.env_name,
                representation="extracted",
                channel_dimensions={
                    "default": (96, 72),
                    "medium": (120, 90),
                    "large": (144, 108),
                }["default"],
                stacked=False,
                rewards=P.reward_type,
                logdir=f"{P.video_dir}",
                write_goal_dumps=False,
                write_full_episode_dumps=render and is_head,
                write_video=render and is_head,
                render=render and is_head,
                dump_frequency=P.render_every,
                number_of_left_players_agent_controls={
                    "custom_3_vs_2": 1,
                    "custom_5_vs_5": 1,
                    "custom_11_vs_11": 1,
                }[P.env_name],  # must equal to num of controllable players, or randomly selected from them
                number_of_right_players_agent_controls=0,
                other_config_options={
                    'video_quality_level': 0,  # 0 - low, 1 - medium, 2 - high
                    "video_format": "avi",  
                    "display_game_stats": True,  # in the video
                }
            )
            envs.append(env)

        env = SyncEnv(env, envs, is_head=is_head, init_points=init_points)

        return env
    

class SyncEnv(gym.Wrapper):
    """
    sync env between head and other actors
    """

    def __init__(self, env, envs, is_head, init_points):
        gym.Wrapper.__init__(self, env)
        self.init_points = init_points
        self.env = env
        self.init_obss = list()
        self.constant_internal_states = list()

        env_dump_path = f"{P.env_dir}{P.env_name}.pkl"
        while True:
            try:
                time.sleep(0.01)
                self.constant_internal_states, self.init_obss = IO.read_disk_dump(
                    env_dump_path
                )
                break
            except Exception:
                if is_head:
                    for env_id in range(self.init_points):
                        self.init_obss.append(envs[env_id].reset())
                        self.constant_internal_states.append(envs[env_id].get_state())
                    IO.write_disk_dump(
                        env_dump_path, (self.constant_internal_states, self.init_obss)
                    )
                    break

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        env_id = np.random.choice(self.init_points)
        self.env.reset()
        self.set_state(self.constant_internal_states[env_id])
        return self.init_obss[env_id]
    
    def sample_action(self):
        return self.env.action_space.sample()


def create_environment(
    env_name="",
    stacked=False,
    representation="extracted",
    rewards="scoring",
    write_goal_dumps=False,
    write_full_episode_dumps=False,
    render=False,
    write_video=False,
    dump_frequency=1,
    logdir="",
    extra_players=None,
    number_of_left_players_agent_controls=1,
    number_of_right_players_agent_controls=0,
    channel_dimensions=(
        observation_preprocessing.SMM_WIDTH,
        observation_preprocessing.SMM_HEIGHT,
    ),
    other_config_options={},
):
    """Creates a Google Research Football environment.

    Args:
      env_name: a name of a scenario to run, e.g. "11_vs_11_stochastic".
        The list of scenarios can be found in directory "scenarios".
      stacked: If True, stack 4 observations, otherwise, only the last
        observation is returned by the environment.
        Stacking is only possible when representation is one of the following:
        "pixels", "pixels_gray" or "extracted".
        In that case, the stacking is done along the last (i.e. channel)
        dimension.
      representation: String to define the representation used to build
        the observation. It can be one of the following:
        'pixels': the observation is the rendered view of the football field
          downsampled to 'channel_dimensions'. The observation size is:
          'channel_dimensions'x3 (or 'channel_dimensions'x12 when "stacked" is
          True).
        'pixels_gray': the observation is the rendered view of the football field
          in gray scale and downsampled to 'channel_dimensions'. The observation
          size is 'channel_dimensions'x1 (or 'channel_dimensions'x4 when stacked
          is True).
        'extracted': also referred to as super minimap. The observation is
          composed of 4 planes of size 'channel_dimensions'.
          Its size is then 'channel_dimensions'x4 (or 'channel_dimensions'x16 when
          stacked is True).
          The first plane P holds the position of players on the left
          team, P[y,x] is 255 if there is a player at position (x,y), otherwise,
          its value is 0.
          The second plane holds in the same way the position of players
          on the right team.
          The third plane holds the position of the ball.
          The last plane holds the active player.
        'simple115'/'simple115v2': the observation is a vector of size 115.
          It holds:
           - the ball_position and the ball_direction as (x,y,z)
           - one hot encoding of who controls the ball.
             [1, 0, 0]: nobody, [0, 1, 0]: left team, [0, 0, 1]: right team.
           - one hot encoding of size 11 to indicate who is the active player
             in the left team.
           - 11 (x,y) positions for each player of the left team.
           - 11 (x,y) motion vectors for each player of the left team.
           - 11 (x,y) positions for each player of the right team.
           - 11 (x,y) motion vectors for each player of the right team.
           - one hot encoding of the game mode. Vector of size 7 with the
             following meaning:
             {NormalMode, KickOffMode, GoalKickMode, FreeKickMode,
              CornerMode, ThrowInMode, PenaltyMode}.
           Can only be used when the scenario is a flavor of normal game
           (i.e. 11 versus 11 players).
      rewards: Comma separated list of rewards to be added.
         Currently supported rewards are 'scoring' and 'checkpoints'.
      write_goal_dumps: whether to dump traces up to 200 frames before goals.
      write_full_episode_dumps: whether to dump traces for every episode.
      render: whether to render game frames.
         Must be enable when rendering videos or when using pixels
         representation.
      write_video: whether to dump videos when a trace is dumped.
      dump_frequency: how often to write dumps/videos (in terms of # of episodes)
        Sub-sample the episodes for which we dump videos to save some disk space.
      logdir: directory holding the logs.
      extra_players: A list of extra players to use in the environment.
          Each player is defined by a string like:
          "$player_name:left_players=?,right_players=?,$param1=?,$param2=?...."
      number_of_left_players_agent_controls: Number of left players an agent
          controls.
      number_of_right_players_agent_controls: Number of right players an agent
          controls.
      channel_dimensions: (width, height) tuple that represents the dimensions of
         SMM or pixels representation.
      other_config_options: dict that allows directly setting other options in
         the Config
    Returns:
      Google Research Football environment.
    """
    assert env_name

    scenario_config = Config({"level": env_name}).ScenarioConfig()
    players = [
        (
            "agent:left_players=%d,right_players=%d"
            % (
                number_of_left_players_agent_controls,
                number_of_right_players_agent_controls,
            )
        )
    ]

    # Enable MultiAgentToSingleAgent wrapper?
    multiagent_to_singleagent = False
    if scenario_config.control_all_players:
        if number_of_left_players_agent_controls in [
            0,
            1,
        ] and number_of_right_players_agent_controls in [0, 1]:
            multiagent_to_singleagent = True
            players = [
                (
                    "agent:left_players=%d,right_players=%d"
                    % (
                        scenario_config.controllable_left_players
                        if number_of_left_players_agent_controls
                        else 0,
                        scenario_config.controllable_right_players
                        if number_of_right_players_agent_controls
                        else 0,
                    )
                )
            ]

    if extra_players is not None:
        players.extend(extra_players)
    config_values = {
        "dump_full_episodes": write_full_episode_dumps,
        "dump_scores": write_goal_dumps,
        "players": players,
        "level": env_name,
        "tracesdir": logdir,
        "write_video": write_video,
    }
    config_values.update(other_config_options)
    c = Config(config_values)

    env = football_env.FootballEnv(c)
    if multiagent_to_singleagent:
        env = wrappers.MultiAgentToSingleAgent(
            env,
            number_of_left_players_agent_controls,
            number_of_right_players_agent_controls,
        )
    if dump_frequency > 1:
        env = PeriodicDumpWriter(env, dump_frequency, render)
    elif render:
        env.render()
    env = _apply_output_wrappers(
        env,
        rewards,
        representation,
        channel_dimensions,
        (
            number_of_left_players_agent_controls
            + number_of_right_players_agent_controls
            == 1
        ),
        stacked,
    )
    return env


class PeriodicDumpWriter(gym.Wrapper):
    """A wrapper that only dumps traces/videos periodically."""

    def __init__(self, env, dump_frequency, render=False):
        gym.Wrapper.__init__(self, env)
        self._dump_frequency = dump_frequency
        self._render = render
        self._original_dump_config = {
            'write_video': env._config['write_video'],
            'dump_full_episodes': env._config['dump_full_episodes'],
            'dump_scores': env._config['dump_scores'],
        }
        self._current_episode_number = 0

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        if (
            self._dump_frequency > 0 and
            (self._current_episode_number % self._dump_frequency == 0)
        ):
            self.env._config.update(self._original_dump_config)
            if self._render:
                self.env.render()
        else:
            self.env._config.update({
                'write_video': False,
                'dump_full_episodes': False,
                'dump_scores': False
            })
            # if self._render:
            #     self.env.disable_render()
        self._current_episode_number += 1
        return self.env.reset()


def _process_reward_wrappers(env, rewards):
    assert 'scoring' in rewards.split(',')
    if 'checkpoints' in rewards.split(','):
        env = wrappers.CheckpointRewardWrapper(env)
    return env


def _process_representation_wrappers(env, representation, channel_dimensions):
    """Wraps with necessary representation wrappers.

    Args:
        env: A GFootball gym environment.
        representation: See create_environment.representation comment.
        channel_dimensions: (width, height) tuple that represents the dimensions of
        SMM or pixels representation.
    Returns:
        Google Research Football environment.
    """
    if representation.startswith('pixels'):
        env = wrappers.PixelsStateWrapper(env, 'gray' in representation, channel_dimensions)
    elif representation == 'simple115':
        env = wrappers.Simple115StateWrapper(env)
    elif representation == 'simple115v2':
        env = wrappers.Simple115StateWrapper(env, True)
    elif representation == 'extracted':
        env = wrappers.SMMWrapper(env, channel_dimensions)
    elif representation == 'raw':
        pass
    else:
        raise ValueError('Unsupported representation: {}'.format(representation))
    return env


def _apply_output_wrappers(env, rewards, representation, channel_dimensions,
                           apply_single_agent_wrappers, stacked):
    """Wraps with necessary wrappers modifying the output of the environment.

    Args:
        env: A GFootball gym environment.
        rewards: What rewards to apply.
        representation: See create_environment.representation comment.
        channel_dimensions: (width, height) tuple that represents the dimensions of
        SMM or pixels representation.
        apply_single_agent_wrappers: Whether to reduce output to single agent case.
        stacked: Should observations be stacked.
    Returns:
        Google Research Football environment.
    """
    env = _process_reward_wrappers(env, rewards)
    env = _process_representation_wrappers(env, representation, channel_dimensions)
    if apply_single_agent_wrappers:
        if representation != 'raw':
            env = wrappers.SingleAgentObservationWrapper(env)
        env = wrappers.SingleAgentRewardWrapper(env)
    if stacked:
        env = wrappers.FrameStack(env, 4)
    env = wrappers.GetStateWrapper(env)
    return env


class Config(object):
    def __init__(self, values=None):
        self._values = {
            "action_set": "default",
            "custom_display_stats": None,
            "display_game_stats": True,
            "dump_full_episodes": False,
            "dump_scores": False,
            "players": ["agent:left_players=1"],
            "level": "11_vs_11_stochastic",
            "physics_steps_per_frame": 10,
            "render_resolution_x": 1280,
            "real_time": False,
            "tracesdir": os.path.join(tempfile.gettempdir(), "dumps"),
            "video_format": "avi",
            "video_quality_level": 0,  # 0 - low, 1 - medium, 2 - high
            "write_video": False,
        }
        self._values["render_resolution_y"] = int(
            0.5625 * self._values["render_resolution_x"]
        )
        if values:
            self._values.update(values)
        self.NewScenario()

    def number_of_left_players(self):
        return sum([count_left_players(player) for player in self._values["players"]])

    def number_of_right_players(self):
        return sum([count_right_players(player) for player in self._values["players"]])

    def number_of_players_agent_controls(self):
        return get_agent_number_of_players(self._values["players"])

    def __eq__(self, other):
        assert isinstance(other, self.__class__)
        return (
            self._values == other._values
            and self._scenario_values == other._scenario_values
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getitem__(self, key):
        if key in self._scenario_values:
            return self._scenario_values[key]
        return self._values[key]

    def __setitem__(self, key, value):
        self._values[key] = value

    def __contains__(self, key):
        return key in self._scenario_values or key in self._values

    def get_dictionary(self):
        cfg = copy.deepcopy(self._values)
        cfg.update(self._scenario_values)
        return cfg

    def set_scenario_value(self, key, value):
        """Override value of specific config key for a single episode."""
        self._scenario_values[key] = value

    def serialize(self):
        return self._values

    def update(self, config):
        self._values.update(config)

    def ScenarioConfig(self):
        return self._scenario_cfg

    def NewScenario(self, inc=1):
        if "episode_number" not in self._values:
            self._values["episode_number"] = 0
        self._values["episode_number"] += inc
        self._scenario_values = {}
        self._scenario_cfg = Scenario(self).ScenarioConfig()


class Scenario(object):
    def __init__(self, config):
        # Game config controls C++ engine and is derived from the main config.
        self._scenario_cfg = libgame.ScenarioConfig.make()
        self._config = config
        self._active_team = Team.e_Left

        self.build_scenario()
        self.SetTeam(libgame.e_Team.e_Left)
        self._FakePlayersForEmptyTeam(self._scenario_cfg.left_team)
        self.SetTeam(libgame.e_Team.e_Right)
        self._FakePlayersForEmptyTeam(self._scenario_cfg.right_team)
        self._BuildScenarioConfig()

    def _FakePlayersForEmptyTeam(self, team):
        if len(team) == 0:
            self.AddPlayer(
                -1.000000, 0.420000, libgame.e_PlayerRole.e_PlayerRole_GK, True
            )

    def _BuildScenarioConfig(self):
        """Builds scenario config from gfootball.environment config."""
        self._scenario_cfg.real_time = self._config["real_time"]
        self._scenario_cfg.left_agents = self._config.number_of_left_players()
        self._scenario_cfg.right_agents = self._config.number_of_right_players()
        # This is needed to record 'game_engine_random_seed' in the dump.
        if "game_engine_random_seed" not in self._config._values:
            self._config.set_scenario_value(
                "game_engine_random_seed", random.randint(0, 2000000000)
            )
        if not self._scenario_cfg.deterministic:
            self._scenario_cfg.game_engine_random_seed = self._config[
                "game_engine_random_seed"
            ]
            if "reverse_team_processing" not in self._config:
                self._config["reverse_team_processing"] = bool(
                    self._config["game_engine_random_seed"] % 2
                )
        if "reverse_team_processing" in self._config:
            self._scenario_cfg.reverse_team_processing = self._config[
                "reverse_team_processing"
            ]

    def config(self):
        return self._scenario_cfg

    def SetTeam(self, team):
        self._active_team = team

    def AddPlayer(self, x, y, role, lazy=False, controllable=True):
        """Build player for the current scenario.

        Args:
          x: x coordinate of the player in the range [-1, 1].
          y: y coordinate of the player in the range [-0.42, 0.42].
          role: Player's role in the game (goal keeper etc.).
          lazy: Computer doesn't perform any automatic actions for lazy player.
          controllable: Whether player can be controlled.
        """
        player = Player(x, y, role, lazy, controllable)
        if self._active_team == Team.e_Left:
            self._scenario_cfg.left_team.append(player)
        else:
            self._scenario_cfg.right_team.append(player)

    def SetBallPosition(self, ball_x, ball_y):
        self._scenario_cfg.ball_position[0] = ball_x
        self._scenario_cfg.ball_position[1] = ball_y

    def EpisodeNumber(self):
        return self._config["episode_number"]

    def ScenarioConfig(self):
        return self._scenario_cfg
    
    def build_scenario(self):
        """
        custom your task.

        Note by Zidu:
        1. agent can control every players on the ground with lazy enabled,
        but some time can't if 'lazy' is disabled:
            a. when the player is receiving the coming ball, they can not be controlled
            b. if auto-defense mechanism had been activated, players will go to defense,
            so they can not be controlled
        2. players to be controlled are selected by top-n from some ordered list of the team players.
        The order list can be changed by some condition (such as unselected players got the ball,
        and then he becomes the first of this ordered list). Thus selected players will be changed too.
        3. titled players are controlled by agent, blink titled players are auto controlled.
        title color stands for id, name is random assigned in each episode
        4. random seed is crucial, same seeds generate same episode with limited changes.
        """        
        # example settings
        # initial_pos = self.get_initial_pos()
        # pos = initial_pos[0]
        # self.SetBallPosition(pos[0], pos[1])
        # self.SetTeam(Team.e_Left)  # play ground rotate to 0 degree, team 1 is left team
        # for num in range(24, 47):
        #     if num in initial_pos:
        #         pos = initial_pos[num]
        #         self.AddPlayer(pos[0], pos[1], num2role[num])
        # self.SetTeam(Team.e_Right)  # play ground rotate to 180 degree clockwise, team 2 is right team
        # for num in range(1, 24):
        #     if num in initial_pos:
        #         pos = initial_pos[num]
        #         self.AddPlayer(-pos[0], -pos[1], num2role[num])
        # self.config().right_team_difficulty = 0.8
        # self.config().left_team_difficulty = 1.0

        # self.config().offsides = False
        # self.config().end_episode_on_score = True
        # self.config().end_episode_on_out_of_play = True
        # self.config().end_episode_on_possession_change = False

        self.config().game_duration = 500
        # self.config().second_half = 1500
        if P.deterministic:
            self.config().deterministic = True
        else:
            self.config().deterministic = False
        self.config().offsides = True
        self.config().end_episode_on_score = True
        self.config().end_episode_on_out_of_play = True
        self.config().end_episode_on_possession_change = True

        random_offset = np.random.random() * 0.05

        # range of x: [-1, 1], y: [-0.44， 0.44]
        if P.env_name == "custom_3_vs_2":
            self.SetBallPosition(0.7, -0.28)

            self.SetTeam(Team.e_Left)
            self.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)  # keeper
            self.AddPlayer(0.7, -0.3, e_PlayerRole_CB)  # player sender
            self.AddPlayer(0.65 + random_offset, 0.0, e_PlayerRole_CB)  # player receive

            self.SetTeam(Team.e_Right)
            self.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)  # keeper
            self.AddPlayer(-0.7, 0.05, e_PlayerRole_CB, controllable=False)  # defender

        if P.env_name == "custom_5_vs_5":
            self.SetBallPosition(0.7, -0.28)

            self.SetTeam(Team.e_Left)
            self.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)  # keeper
            self.AddPlayer(0.7, -0.3, e_PlayerRole_CB)  # offender
            self.AddPlayer(0.65 + random_offset, 0.0, e_PlayerRole_CB)
            self.AddPlayer(0.69, 0.0, e_PlayerRole_CB)
            self.AddPlayer(0.68, 0.0, e_PlayerRole_CB)

            self.SetTeam(Team.e_Right)
            self.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)  # keeper
            self.AddPlayer(-0.7, 0.05, e_PlayerRole_CB, controllable=False)  # defender
            self.AddPlayer(-0.71, 0.05, e_PlayerRole_CB, controllable=False)
            self.AddPlayer(-0.69, 0.05, e_PlayerRole_CB, controllable=False)
            self.AddPlayer(-0.68, 0.05, e_PlayerRole_CB, controllable=False)

        if P.env_name == "custom_11_vs_11":  # control the whole team except keeper
            self.SetBallPosition(-0.48, -0.06356)  # ball with left team
            # self.SetBallPosition(0.184212, 0.10568)  # ball with right team

            self.SetTeam(Team.e_Left)
            self.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, controllable=False)
            self.AddPlayer(0.000000, 0.020000, e_PlayerRole_RM)
            self.AddPlayer(0.000000, -0.020000, e_PlayerRole_CF)
            self.AddPlayer(-0.422000, -0.19576, e_PlayerRole_LB)
            self.AddPlayer(-0.450000 + random_offset, -0.06356, e_PlayerRole_CB)
            self.AddPlayer(-0.500000, 0.063559, e_PlayerRole_CB)
            self.AddPlayer(-0.422000, 0.195760, e_PlayerRole_RB)
            self.AddPlayer(-0.184212, -0.10568, e_PlayerRole_CM)
            self.AddPlayer(-0.267574, 0.000000, e_PlayerRole_CM)
            self.AddPlayer(-0.184212, 0.105680, e_PlayerRole_CM)
            self.AddPlayer(-0.010000, -0.21610, e_PlayerRole_LM)

            self.SetTeam(Team.e_Right)
            self.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, controllable=False)
            self.AddPlayer(-0.050000, 0.000000, e_PlayerRole_RM, controllable=False)
            self.AddPlayer(-0.010000, 0.216102, e_PlayerRole_CF, controllable=False)
            self.AddPlayer(-0.422000, -0.19576, e_PlayerRole_LB, controllable=False)
            self.AddPlayer(-0.500000, -0.06356, e_PlayerRole_CB, controllable=False)
            self.AddPlayer(-0.500000, 0.063559, e_PlayerRole_CB, controllable=False)
            self.AddPlayer(-0.422000, 0.195760, e_PlayerRole_RB, controllable=False)
            self.AddPlayer(-0.184212, -0.10568, e_PlayerRole_CM, controllable=False)
            self.AddPlayer(-0.267574, 0.000000, e_PlayerRole_CM, controllable=False)
            self.AddPlayer(-0.184212, 0.105680, e_PlayerRole_CM, controllable=False)
            self.AddPlayer(-0.010000, -0.21610, e_PlayerRole_LM, controllable=False)
