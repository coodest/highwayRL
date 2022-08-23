# coding=utf-8
# Copyright 2019 The SEED Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Football env factory."""
import gym
from gfootball.env import config
from gfootball.env import football_env
from gfootball.env import observation_preprocessing
from gfootball.env import wrappers
from gfootball.env import scenario_builder
from gfootball.scenarios import *
import gfootball_engine as libgame
import numpy as np
from src.module.context import Profile as P


# sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip
# pip install gfootball
# conda install -c conda-forge gcc
# pushd /home/zidu/miniconda3/envs/rl/lib/python3.9/site-packages/gfootball_engine && cmake . && make -j `nproc` && popd
# pushd /home/zidu/miniconda3/envs/rl/lib/python3.9/site-packages/gfootball_engine && ln -s libgame.so _gameplayfootball.so && popd

# sudo apt-get install python3-pygame


# conda install -c conda-forge sdl_image sdl_ttf sdl_mixer


class PackedBitsObservation(gym.ObservationWrapper):
    """Wrapper that encodes a frame as packed bits instead of booleans.

    8x less to be transferred across the wire (16 booleans stored as uint16
    instead of 16 uint8) and 8x less to be transferred from CPU to TPU (16
    booleans stored as uint32 instead of 16 bfloat16).

    """

    def __init__(self, env):
        super(PackedBitsObservation, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=np.iinfo(np.uint16).max,
            shape=env.observation_space.shape[:-1] + ((env.observation_space.shape[-1] + 15) // 16,),
            dtype=np.uint16
        )

    def observation(self, observation):
        data = np.packbits(observation, axis=-1)  # This packs to uint8
        # Now we want to pack pairs of uint8 into uint16's.
        # We first need to ensure that the last dimension has even size.
        if data.shape[-1] % 2 == 1:
            data = np.pad(data, [(0, 0)] * (data.ndim - 1) + [(0, 1)], 'constant')
        return data.view(np.uint16)



Player = libgame.FormationEntry
Role = libgame.e_PlayerRole
Team = libgame.e_Team
max_episode_steps = 500
random_env = False
controlled_left_player = 2
controlled_right_player = 0
use_packed_bits = True

num2role = {
    1: e_PlayerRole_GK,    2: e_PlayerRole_RB,    3: e_PlayerRole_LB,    4: e_PlayerRole_CM,
    5: e_PlayerRole_DM,    6: e_PlayerRole_DM,    7: e_PlayerRole_AM,    8: e_PlayerRole_CM,
    9: e_PlayerRole_AM,    10: e_PlayerRole_CM,    11: e_PlayerRole_CF,    12: e_PlayerRole_CM,
    13: e_PlayerRole_CM,    14: e_PlayerRole_AM,    15: e_PlayerRole_CM,    16: e_PlayerRole_AM,
    17: e_PlayerRole_CM,    18: e_PlayerRole_AM,    19: e_PlayerRole_CM,    20: e_PlayerRole_CM,
    21: e_PlayerRole_CM,    22: e_PlayerRole_GK,    23: e_PlayerRole_GK,    24: e_PlayerRole_GK,
    25: e_PlayerRole_RB,    26: e_PlayerRole_LB,    27: e_PlayerRole_CM,    28: e_PlayerRole_DM,
    29: e_PlayerRole_DM,    30: e_PlayerRole_AM,    31: e_PlayerRole_CM,    32: e_PlayerRole_AM,
    33: e_PlayerRole_CM,    34: e_PlayerRole_CF,    35: e_PlayerRole_CM,    36: e_PlayerRole_CM,
    37: e_PlayerRole_AM,    38: e_PlayerRole_CM,    39: e_PlayerRole_AM,    40: e_PlayerRole_CM,
    41: e_PlayerRole_AM,    42: e_PlayerRole_CM,    43: e_PlayerRole_CM,    44: e_PlayerRole_CM,
    45: e_PlayerRole_GK,    46: e_PlayerRole_GK
}


class Football:
    @staticmethod
    def make_env(render=False, is_head=False):
        smm_size = 'default'
        channel_dimensions = {
            'default': (96, 72),
            'medium': (120, 90),
            'large': (144, 108),
        }[smm_size]

        initial_pos = None
        env = create_custom_environment(
            "academy_custom_scenario",
            logdir=P.result_dir + "video/",
            channel_dimensions=channel_dimensions,
            write_full_episode_dumps=render,
            write_video=render,
            render=render,
            number_of_left_players_agent_controls=controlled_left_player,  # must equal to num of controllable players, or randomly selected from them
            number_of_right_players_agent_controls=controlled_right_player,
            other_config_options={
                'video_quality_level': 0,
                'initial_pos': initial_pos
            }
        )
        
        env.seed(100)

        # action space of google research football
        # 0:idle 1:right 2:bottom_right 3:bottom 4:bottom_left 5:left 6:top_left 7:top 8:top_right
        # 9:long_pass 10:high_pass 11:short_pass 12:shot 13:sprint 14:release_direction 15:release_sprint
        # 16:sliding 17:dribble 18:release_dribble

        # network reused by multi-agent
        env.action_space.dtype = np.int32
        if controlled_left_player + controlled_right_player > 1:
            env.observation_space.shape = env.observation_space.shape[1:]
            env.action_space.shape = ()
        if use_packed_bits:
            return PackedBitsObservation(env)
        else:
            return env


def create_custom_environment(
    env_name='',
    stacked=False,
    representation='extracted',
    rewards='scoring',
    write_goal_dumps=False,
    write_full_episode_dumps=False,
    render=False,
    write_video=False,
    dump_frequency=1,
    logdir='',
    extra_players=None,
    number_of_left_players_agent_controls=1,
    number_of_right_players_agent_controls=0,
    channel_dimensions=(
            observation_preprocessing.SMM_WIDTH,
            observation_preprocessing.SMM_HEIGHT),
    other_config_options={}
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
    players = [('agent:left_players=%d,right_players=%d' % (
        number_of_left_players_agent_controls,
        number_of_right_players_agent_controls))]
    if extra_players is not None:
        players.extend(extra_players)
    config_values = {
        'dump_full_episodes': write_full_episode_dumps,
        'dump_scores': write_goal_dumps,
        'players': players,
        'level': env_name,
        'tracesdir': logdir,
        'write_video': write_video,
    }
    config_values.update(other_config_options)
    c = CustomConfig(config_values)
    env = football_env.FootballEnv(c)
    if render:
        env.render()
    if dump_frequency > 1:
        env = wrappers.PeriodicDumpWriter(env, dump_frequency)
    env = _apply_output_wrappers(
        env, rewards, representation, channel_dimensions,
        (number_of_left_players_agent_controls +
         number_of_right_players_agent_controls == 1), stacked)
    return env


class CustomConfig(config.Config):
    def __init__(self, values=None):
        self._game_config = libgame.GameConfig()
        self._values = {
            'action_set': 'default',
            'custom_display_stats': None,
            'display_game_stats': True,
            'dump_full_episodes': False,
            'dump_scores': False,
            'players': ['agent:left_players=1'],
            'level': '11_vs_11_stochastic',
            'physics_steps_per_frame': 10,
            'real_time': False,
            'tracesdir': '/tmp/dumps',
            'video_format': 'avi',
            'video_quality_level': 0,  # 0 - low, 1 - medium, 2 - high
            'write_video': False
        }
        if values:
            self._values.update(values)
        self.NewScenario()

    def NewScenario(self, inc=1):
        if 'episode_number' not in self._values:
            self._values['episode_number'] = 0
        self._values['episode_number'] += inc
        self._scenario_values = {}
        self._scenario_cfg = CustomScenario(self).ScenarioConfig()


class CustomScenario(scenario_builder.Scenario):
    def __init__(self, config):
        # Game config controls C++ engine and is derived from the main config.
        self._scenario_cfg = libgame.ScenarioConfig.make()
        self._config = config
        self._active_team = Team.e_Left
        build_scenario(self)
        self.SetTeam(libgame.e_Team.e_Left)
        self._FakePlayersForEmptyTeam(self._scenario_cfg.left_team)
        self.SetTeam(libgame.e_Team.e_Right)
        self._FakePlayersForEmptyTeam(self._scenario_cfg.right_team)
        self._BuildScenarioConfig()

    def get_initial_pos(self):
        return self._config['initial_pos']


def build_scenario(builder):
    """
    customize your task.

    Note by Zidu:
    1. agent can control every players on the ground with lazy enabled,
    but some time can't if 'lazy' disabled:
        a. when the player is receiving the coming ball, they can not be controlled
        b. if auto-defense mechanism had been activated, players will go to defense,
        so they can not be controlled
    2. players to be controlled are selected by top-n from some ordered list of the team players.
    The order list can be changed by some condition (such as unselected players got the ball,
    and then he becomes the first of this ordered list). Thus selected players will be changed too.
    3. titled players are controlled by agent, blink titled players are auto controlled.
    title color stands for id, name is random assigned in each episode
    4. random seed is crucial, same seeds generate same episode with limited changes.
    :param builder:
    :return:
    """
    # initial_pos = builder.get_initial_pos()
    # builder.config().game_duration = 3000
    # builder.config().right_team_difficulty = 0.05
    # builder.config().deterministic = False
    #
    # pos = initial_pos[0]
    # builder.SetBallPosition(pos[0], pos[1])
    # builder.SetTeam(Team.e_Left)  # play ground rotate to 0 degree, team 1 is left team
    # for num in range(24, 47):
    #     if num in initial_pos:
    #         pos = initial_pos[num]
    #         builder.AddPlayer(pos[0], pos[1], num2role[num])
    # builder.SetTeam(Team.e_Right)  # play ground rotate to 180 degree clockwise, team 2 is right team
    # for num in range(1, 24):
    #     if num in initial_pos:
    #         pos = initial_pos[num]
    #         builder.AddPlayer(-pos[0], -pos[1], num2role[num])

    # pass and shot with keeper
    builder.config().game_duration = max_episode_steps
    if random_env:
        builder.config().deterministic = False
    else:
        builder.config().deterministic = True
    builder.config().offsides = True
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    # builder.config().game_duration = 3000
    # builder.config().second_half = 1500
    # builder.config().right_team_difficulty = 0.8
    # builder.config().left_team_difficulty = 1.0

    # builder.config().game_duration = P.max_episode_steps
    # builder.config().deterministic = False
    # builder.config().offsides = False
    # builder.config().end_episode_on_score = True
    # builder.config().end_episode_on_out_of_play = True
    # builder.config().end_episode_on_possession_change = False

    # x: [-1, 1], y: [-0.44， 0.44]
    if controlled_left_player == 1:
        builder.SetBallPosition(0.7, -0.28)

        builder.SetTeam(Team.e_Left)
        builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)  # keeper
        builder.AddPlayer(0.7, -0.3, e_PlayerRole_CB, lazy=True, controllable=True)  # player sender
        builder.AddPlayer(0.7, 0.0, e_PlayerRole_CB, controllable=False)  # player receive

        builder.SetTeam(Team.e_Right)
        builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)  # keeper
        builder.AddPlayer(-0.7, 0.05, e_PlayerRole_CB, controllable=False)  # defender

    if controlled_left_player == 2:
        builder.SetBallPosition(0.7, -0.28)

        builder.SetTeam(Team.e_Left)
        builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)  # keeper
        builder.AddPlayer(0.7, -0.3, e_PlayerRole_CB, lazy=True, controllable=True)  # player sender
        builder.AddPlayer(0.7, 0.0, e_PlayerRole_CB, lazy=True, controllable=True)  # player receive

        builder.SetTeam(Team.e_Right)
        builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)  # keeper
        builder.AddPlayer(-0.7, 0.05, e_PlayerRole_CB, controllable=False)  # defender

    if controlled_left_player == 4:
        builder.SetBallPosition(0.7, -0.28)

        builder.SetTeam(Team.e_Left)
        builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)  # keeper
        builder.AddPlayer(0.7, -0.3, e_PlayerRole_CB, lazy=True, controllable=True)  # player sender
        builder.AddPlayer(0.7, 0.0, e_PlayerRole_CB, lazy=True, controllable=True)  # player receive
        builder.AddPlayer(0.69, 0.0, e_PlayerRole_CB, lazy=True, controllable=True)
        builder.AddPlayer(0.68, 0.0, e_PlayerRole_CB, lazy=True, controllable=True)

        builder.SetTeam(Team.e_Right)
        builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)  # keeper
        builder.AddPlayer(-0.7, 0.05, e_PlayerRole_CB, controllable=False)  # defender
        builder.AddPlayer(-0.71, 0.05, e_PlayerRole_CB, controllable=False)
        builder.AddPlayer(-0.69, 0.05, e_PlayerRole_CB, controllable=False)
        builder.AddPlayer(-0.68, 0.05, e_PlayerRole_CB, controllable=False)

    if controlled_left_player == 10:  # control the whole team except keeper
        # builder.SetBallPosition(-0.48, -0.06356)  # ball with left team
        builder.SetBallPosition(0.184212, 0.10568)  # ball with right team

        builder.SetTeam(Team.e_Left)
        builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, controllable=False)
        builder.AddPlayer(0.000000, 0.020000, e_PlayerRole_RM, lazy=True, controllable=True)
        builder.AddPlayer(0.000000, -0.020000, e_PlayerRole_CF, lazy=True, controllable=True)
        builder.AddPlayer(-0.422000, -0.19576, e_PlayerRole_LB, lazy=True, controllable=True)
        builder.AddPlayer(-0.500000, -0.06356, e_PlayerRole_CB, lazy=True, controllable=True)
        builder.AddPlayer(-0.500000, 0.063559, e_PlayerRole_CB, lazy=True, controllable=True)
        builder.AddPlayer(-0.422000, 0.195760, e_PlayerRole_RB, lazy=True, controllable=True)
        builder.AddPlayer(-0.184212, -0.10568, e_PlayerRole_CM, lazy=True, controllable=True)
        builder.AddPlayer(-0.267574, 0.000000, e_PlayerRole_CM, lazy=True, controllable=True)
        builder.AddPlayer(-0.184212, 0.105680, e_PlayerRole_CM, lazy=True, controllable=True)
        builder.AddPlayer(-0.010000, -0.21610, e_PlayerRole_LM, lazy=True, controllable=True)

        builder.SetTeam(Team.e_Right)
        builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, controllable=False)
        builder.AddPlayer(-0.050000, 0.000000, e_PlayerRole_RM, controllable=False)
        builder.AddPlayer(-0.010000, 0.216102, e_PlayerRole_CF, controllable=False)
        builder.AddPlayer(-0.422000, -0.19576, e_PlayerRole_LB, controllable=False)
        builder.AddPlayer(-0.500000, -0.06356, e_PlayerRole_CB, controllable=False)
        builder.AddPlayer(-0.500000, 0.063559, e_PlayerRole_CB, controllable=False)
        builder.AddPlayer(-0.422000, 0.195760, e_PlayerRole_RB, controllable=False)
        builder.AddPlayer(-0.184212, -0.10568, e_PlayerRole_CM, controllable=False)
        builder.AddPlayer(-0.267574, 0.000000, e_PlayerRole_CM, controllable=False)
        builder.AddPlayer(-0.184212, 0.105680, e_PlayerRole_CM, controllable=False)
        builder.AddPlayer(-0.010000, -0.21610, e_PlayerRole_LM, controllable=False)


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
    env = _process_representation_wrappers(env, representation,
                                           channel_dimensions)
    if apply_single_agent_wrappers:
        if representation != 'raw':
            env = wrappers.SingleAgentObservationWrapper(env)
        env = wrappers.SingleAgentRewardWrapper(env)
    if stacked:
        env = wrappers.FrameStack(env, 4)
    env = wrappers.GetStateWrapper(env)
    return env


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
        env = wrappers.PixelsStateWrapper(env, 'gray' in representation,
                                          channel_dimensions)
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

