import time

from src.util.tools import IO, Logger
from src.module.context import Profile as P
from src.module.agent.memory.storage import TransitionStorage
from src.module.agent.memory.storage import OptimalStorage
import numpy as np


class Graph:
    def __init__(self, id, is_head) -> None:
        self.id = id
        self.is_head = is_head
        self.last_sync = time.time()
        self.main = self.make_new_store()
        self.increments = self.make_new_store()

    def make_new_store(self):
        return None

    def get_action(self, obs):
        pass

    def store_increments(self, trajectory, total_reward):
        pass

    def sync(self):
        if not self.is_head:
            # write increments (not head)
            IO.write_disk_dump(P.result_dir + f'{self.id}.pkl', self.increments)
            self.increments = self.make_new_store()
            IO.write_disk_dump(P.result_dir + f'{self.id}.ready', ["ready"])
            IO.stick_read_disk_dump(P.result_dir + 'target.ok')
            self.main = IO.read_disk_dump(P.result_dir + 'target.pkl')
            IO.write_disk_dump(P.result_dir + f'{self.id}.finish', ["finish"])
        else:
            # make sure writes are complete
            for i in range(P.num_actor):
                if self.id == i:
                    continue
                IO.stick_read_disk_dump(P.result_dir + f'{i}.ready')
            
            # read increments (head)
            for i in range(P.num_actor):
                if self.id == i:
                    continue  # head has no increments stored
                inc = IO.stick_read_disk_dump(P.result_dir + f'{i}.pkl')
                self.merge_inc(inc)
            
            # write target (head)
            IO.write_disk_dump(P.result_dir + 'target.pkl', self.main)
            IO.write_disk_dump(P.result_dir + 'target.ok', ['ok'])

            # remove increments (head)
            for i in range(P.num_actor):
                if self.id == i:
                    continue
                IO.stick_read_disk_dump(P.result_dir + f'{i}.finish')

            IO.delete_file(P.optimal_graph_path)
            IO.move_file(P.result_dir + 'target.pkl', P.optimal_graph_path)
            IO.renew_dir(P.result_dir)

    def merge_inc(self, inc):
        pass


class OptimalGraph(Graph):
    def __init__(self, id, is_head) -> None:
        super().__init__(id, is_head)

    def make_new_store(self):
        return OptimalStorage()

    def get_action(self, obs):
        if obs in self.main:
            return self.main[obs][0]
        else:
            return None

    def store_increments(self, trajectory, total_reward):
        for last_obs, pre_action, obs, reward in trajectory:
            info = [pre_action, total_reward]

            add = False
            if last_obs not in self.main:
                add = True
            elif self.main[last_obs][1] < total_reward:
                add = True

            if add:
                if P.sync_mode == P.sync_modes[0]:
                    if last_obs not in self.increments:
                        self.increments[last_obs] = info
                        self.increments.update_max(total_reward, last_obs)
                    elif self.increments[last_obs][1] < total_reward:
                        self.increments[last_obs] = info
                        self.increments.update_max(total_reward, last_obs)
                if P.sync_mode == P.sync_modes[1]:
                    self.increments[last_obs] = info
                    self.increments.update_max(total_reward, last_obs)
                if P.sync_mode == P.sync_modes[2]:
                    if last_obs not in self.increments:
                        self.increments[last_obs] = info
                        self.increments.update_max(total_reward, last_obs)
                    elif self.increments.max_value * P.sync_tolerance < total_reward:
                        self.increments[last_obs] = info
                        self.increments.update_max(total_reward, last_obs)

    def merge_inc(self, inc):
        for last_obs in inc:
            if last_obs not in self.main:
                self.main[last_obs] = inc[last_obs]
                self.main.update_max(inc[last_obs][1], last_obs)
            elif self.main[last_obs][1] < inc[last_obs][1]:
                self.main[last_obs] = inc[last_obs]
                self.main.update_max(inc[last_obs][1], last_obs)


class TransitionGraph:
    def __init__(self, id, is_head) -> None:
        super().__init__(id, is_head)

    def make_new_store(self):
        return TransitionStorage()

    def get_action(self, obs):
        if obs in self.main:
            return np.argmax(self.main[obs]['action'])
        else:
            return None

    def store_increments(self, trajectory, total_reward):
        pt = []
        last_value = 0
        for last_obs, pre_action, obs, reward in trajectory[::-1]:
            last_value = last_value * P.gamma + reward
            pt.append(last_obs, pre_action, obs, last_value)
        for last_obs, pre_action, obs, value in pt[::-1]:
            if obs not in self.increments:
                self.increments[obs] = {}
                self.increments[obs]['action'] = [0] * P.num_action
                self.increments[obs]['parents'] = {}
            self.increments[obs]['parents'][last_obs] = True
            if last_obs not in self.increments:
                info = dict()
                info['action'] = [0] * P.num_action
            info['parents'] = []
            if P.add_obs:
                info = [pre_action, total_reward, obs]
                
            else:
                info = [pre_action, total_reward]

            if P.sync_mode == P.sync_modes[0]:
                if last_obs not in self.increments:
                    self.increments[last_obs] = info
                    self.increments.update_max(total_reward)
                elif self.increments.max_value < total_reward:
                    self.increments[last_obs] = info
                    self.increments.update_max(total_reward)
            if P.sync_mode == P.sync_modes[1]:
                self.increments[last_obs] = info
                self.increments.update_max(total_reward)
            if P.sync_mode == P.sync_modes[2]:
                if last_obs not in self.increments:
                    self.increments[last_obs] = info
                    self.increments.update_max(total_reward)
                elif self.increments.max_value * P.sync_tolerance < total_reward:
                    self.increments[last_obs] = info
                    self.increments.update_max(total_reward)
    
    def merge_inc(self, inc):
        pass
