import gym
from gym.spaces.box import Box
from gym import spaces
from gym.wrappers import TimeLimit, RecordVideo
from src.util.imports.numpy import np
from src.module.context import Profile as P
from src.util.tools import Funcs, Logger, IO
import time
import gym_sokoban
import copy
import random
from gym_sokoban.envs.boxoban_env import BoxobanEnv
from gym_sokoban.envs.sokoban_env import SokobanEnv
from gym_sokoban.envs.render_utils import room_to_rgb
import os
from os import listdir
from os.path import isfile, join
import requests
import zipfile
from tqdm import tqdm
import random
import numpy as np


class Sokoban:
    @staticmethod
    def make_env(render=False, is_head=False):
        # Logger.log(f"env_name: {P.env_name}")
        # env = TinySokoban(P.env_name)
        env = OptimizedBoxobanEnv(max_steps=100)

        return env


class OptimizedBoxobanEnv(gym.Env):
    def __init__(self, max_steps=120):
        """
        each call randomly generate n levels to train the agent.
        """
        self.env = FilteredTinyBoxobanEnv(max_steps=max_steps)
        self.action_space = spaces.Discrete(4)
        self.reward_range = self.env.reward_range
        self.observation_space = self.env.observation_space
        self.spec = self.env.spec
        self.metadata = self.env.metadata

    def step(self, action):
        state = self.env.room_state
        coord = np.where(state == 5)
        player = [coord[0][0], coord[1][0]]

        action += 5

        if state[player[0]][player[1] - 1] == 4:  # player at the right of the box
            if action == 7:
                action = 3
        if state[player[0]][player[1] + 1] == 4:  # player at the left of the box
            if action == 8:
                action = 4
        if state[player[0] + 1][player[1]] == 4:  # player at the top of the box
            if action == 6:
                action = 2
        if state[player[0] - 1][player[1]] == 4:  # player at the bottom of the box
            if action == 5:
                action = 1

        obs, reward, done, info = self.env.step(action)

        obs = np.array2string(obs).replace(" ", "").replace("[", "").replace("]", "").replace("\n", "")

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = np.array2string(obs).replace(" ", "").replace("[", "").replace("]", "").replace("\n", "")
        return obs

    def render(self, mode):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()


class FilteredTinyBoxobanEnv(SokobanEnv):
    num_boxes = 4
    dim_room = (10, 10)

    def __init__(
        self,
        max_steps=120,
        difficulty='unfiltered', 
        split='train',
        max_level=1,
        max_file=1,
    ):
        self.difficulty = difficulty
        self.split = split
        self.verbose = False
        self.max_file = max_file
        self.max_level = max_level
        super().__init__(self.dim_room, max_steps, self.num_boxes, None)
    
    def step(self, action):
        _, reward, done, info = super().step(action)
        obs = self.room_state
        return obs, reward, done, info

    def reset(self):
        self.cache_path = '.sokoban_cache'
        self.train_data_dir = os.path.join(self.cache_path, 'boxoban-levels-master', self.difficulty, self.split)

        if not os.path.exists(self.cache_path):
           
            url = "https://github.com/deepmind/boxoban-levels/archive/master.zip"
            
            if self.verbose:
                print('Boxoban: Pregenerated levels not downloaded.')
                print('Starting download from "{}"'.format(url))

            response = requests.get(url, stream=True)

            if response.status_code != 200:
                raise "Could not download levels from {}. If this problem occurs consistantly please report the bug under https://github.com/mpSchrader/gym-sokoban/issues. ".format(url)

            os.makedirs(self.cache_path)
            path_to_zip_file = os.path.join(self.cache_path, 'boxoban_levels-master.zip')
            with open(path_to_zip_file, 'wb') as handle:
                for data in tqdm(response.iter_content()):
                    handle.write(data)

            zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
            zip_ref.extractall(self.cache_path)
            zip_ref.close()
        
        self.select_room()

        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0

        # starting_observation = room_to_rgb(self.room_state, self.room_fixed)

        return self.room_state

    def select_room(self):
        
        generated_files = [f for f in listdir(self.train_data_dir) if isfile(join(self.train_data_dir, f))]
        source_file = join(self.train_data_dir, random.choice(generated_files[: self.max_file]))

        maps = []
        current_map = []
        
        with open(source_file, 'r') as sf:
            for line in sf.readlines():
                if ';' in line and current_map:
                    maps.append(current_map)
                    current_map = []
                if '#' == line[0]:
                    current_map.append(line.strip())
        
        maps.append(current_map)

        selected_map = random.choice(maps[: self.max_level])

        if self.verbose:
            print('Selected Level from File "{}"'.format(source_file))

        self.room_fixed, self.room_state, self.box_mapping = self.generate_room(selected_map)


    def generate_room(self, select_map):
        room_fixed = []
        room_state = []

        targets = []
        boxes = []
        for row in select_map:
            room_f = []
            room_s = []

            for e in row:
                if e == '#':
                    room_f.append(0)
                    room_s.append(0)

                elif e == '@':
                    self.player_position = np.array([len(room_fixed), len(room_f)])
                    room_f.append(1)
                    room_s.append(5)


                elif e == '$':
                    boxes.append((len(room_fixed), len(room_f)))
                    room_f.append(1)
                    room_s.append(4)

                elif e == '.':
                    targets.append((len(room_fixed), len(room_f)))
                    room_f.append(2)
                    room_s.append(2)

                else:
                    room_f.append(1)
                    room_s.append(1)

            room_fixed.append(room_f)
            room_state.append(room_s)


        # used for replay in room generation, unused here because pre-generated levels
        box_mapping = {}

        return np.array(room_fixed), np.array(room_state), box_mapping


# class TinySokoban(gym.Env):
#     def __init__(self, env_name, level_pool_size=2):
#         """
#         each call randomly generate n levels to train the agent.
#         """
#         self.env = gym.make(env_name)
#         self.action_space = self.env.action_space
#         self.reward_range = self.env.reward_range
#         self.observation_space = self.env.observation_space
#         self.spec = self.env.spec
#         self.metadata = self.env.metadata
#         self.level_pool = list()
#         self.env.seed(seed=0)
#         for _ in range(level_pool_size):
#             self.env.reset()
#             self.level_pool.append(copy.deepcopy(self.env))

#     def step(self, action):
#         _, reward, done, info = self.env.step(action)
#         obs = self.env.env.room_state
#         return obs, reward, done, info

#     def reset(self):
#         self.env = copy.deepcopy(random.choice(self.level_pool))
#         obs = self.env.env.room_state
#         return obs

#     def render(self, mode):
#         return self.env.render(mode=mode)

#     def close(self):
#         return self.env.close()
