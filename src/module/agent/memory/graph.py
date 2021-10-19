import time

from src.util.tools import IO
from src.module.context import Profile as P
from src.module.agent.memory.storage import TransitionStorage
from src.module.agent.memory.storage import OptimalStorage
from src.module.agent.memory.storage import OptimalStorageCell, TransitionStorageCell
import numpy as np
from sklearn.neighbors import KDTree
from src.module.agent.memory.indexer import Indexer


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
        """
        Synconization that independent to the inner structure and content of increments 
        and main storage.
        """
        if not self.is_head:
            # write increments (not head)
            IO.write_disk_dump(P.result_dir + f"{self.id}.pkl", self.increments)
            self.increments = self.make_new_store()
            IO.write_disk_dump(P.result_dir + f"{self.id}.ready", ["ready"])
            IO.stick_read_disk_dump(P.result_dir + "target.ok")
            self.main = IO.read_disk_dump(P.result_dir + "target.pkl")
            IO.write_disk_dump(P.result_dir + f"{self.id}.finish", ["finish"])
        else:
            # make sure writes are complete
            for i in range(P.num_actor):
                if self.id == i:
                    continue
                IO.stick_read_disk_dump(P.result_dir + f"{i}.ready")

            # read increments (head)
            for i in range(P.num_actor):
                if self.id == i:
                    continue  # head has no increments stored
                inc = IO.stick_read_disk_dump(P.result_dir + f"{i}.pkl")
                self.merge_inc(inc)
            self.post_process()

            # write target (head)
            IO.write_disk_dump(P.result_dir + "target.pkl", self.main)
            IO.write_disk_dump(P.result_dir + "target.ok", ["ok"])

            # remove increments (head)
            for i in range(P.num_actor):
                if self.id == i:
                    continue
                IO.stick_read_disk_dump(P.result_dir + f"{i}.finish")

            IO.delete_file(P.optimal_graph_path)
            IO.move_file(P.result_dir + "target.pkl", P.optimal_graph_path)
            IO.renew_dir(P.result_dir)

    def merge_inc(self, inc):
        pass

    def post_process(self):
        pass


class OptimalGraph(Graph):
    def __init__(self, id, is_head) -> None:
        super().__init__(id, is_head)

    def make_new_store(self):
        return OptimalStorage()

    def get_action(self, obs):
        if obs in self.main:
            return self.main[obs].best_action
        else:
            return None

    def store_increments(self, trajectory, total_reward):  # by the non-head
        for last_obs, pre_action, obs, reward in trajectory:
            # check the importance of the traj to determine whether to add it
            if last_obs not in self.main:
                add = True
            elif self.main[last_obs].total_reward < total_reward:
                add = True
            else:
                add = False

            if add:
                if P.statistic_crossing_obs:
                    # check crossing obs
                    if last_obs not in self.main and last_obs not in self.increments:
                        last_obs_existence = 0  # check the existence in the graph
                    else:
                        last_obs_existence = 1
                    if obs not in self.main and obs not in self.increments:
                        obs_existence = 0
                    else:
                        obs_existence = 1
                    if last_obs_existence + obs_existence == 1:
                        if last_obs_existence == 1:
                            self.increments.crossing_obs.add(last_obs)
                        if obs_existence == 1:
                            self.increments.crossing_obs.add(obs)

                # add last_oobs to the storage
                info = OptimalStorageCell(action=pre_action, total_action=total_reward)
                if P.sync_mode == P.sync_modes[0]:
                    if last_obs not in self.increments:
                        self.increments[last_obs] = info
                        self.increments.update_max(total_reward, last_obs)
                    elif self.increments[last_obs].total_reward < total_reward:
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

    def merge_inc(self, inc):  # by the head
        for last_obs in inc:
            if last_obs not in self.main:
                self.main[last_obs] = inc[last_obs]
                self.main.update_max(inc[last_obs].total_reward, last_obs)
            elif self.main[last_obs].total_reward < inc[last_obs].total_reward:
                self.main[last_obs] = inc[last_obs]
                self.main.update_max(inc[last_obs].total_reward, last_obs)
        self.main.crossing_obs = self.main.crossing_obs.union(inc.crossing_obs)


class TransitionGraph(Graph):
    def __init__(self, id, is_head) -> None:
        super().__init__(id, is_head)
        self.stack = list()
        self.known_states = []

    def make_new_store(self):
        return TransitionStorage()

    def get_action(self, obs, raw_obs):
        if obs in self.main:
            return np.argmax(self.main[obs].action)
        else:
            if raw_obs not in self.known_states:
                self.known_states.append(raw_obs)
            if len(self.known_states) > P.k_nearest_neighbor:
                kdtree = KDTree(self.known_states)
                # return index of the closest neighbors in self.known_states
                neighbors = kdtree.query([raw_obs], P.k_nearest_neighbor)[1][0]
                values = np.zeros([P.num_action])
                num_validate_nb = 0
                for neighbor in neighbors:
                    ind = Indexer.get_ind(self.known_states[neighbor])
                    if ind in self.main:
                        values += self.main[ind].action
                        num_validate_nb += 1
                if num_validate_nb > 0:
                    values /= num_validate_nb
                    best_actions = np.argwhere(values == np.max(values)).flatten()
                    action = np.random.choice(best_actions)
                    return action

            return None

    def store_increments(self, trajectory, total_reward):
        # filtering
        if P.sync_mode == P.sync_modes[0]:
            if self.increments.max_value >= total_reward:
                return
        if P.sync_mode == P.sync_modes[2]:
            if self.increments.max_value * P.sync_tolerance >= total_reward:
                return

        # update increment
        self.increments.add_end(trajectory[-1][2])
        self.increments.update_max(total_reward)
        current_value = 0
        for last_obs, pre_action, obs, reward in trajectory[::-1]:
            if P.statistic_crossing_obs:
                # check crossing obs
                if last_obs not in self.main and last_obs not in self.increments:
                    last_obs_existence = 0  # check the existence in the graph
                else:
                    last_obs_existence = 1
                if obs not in self.main and obs not in self.increments:
                    obs_existence = 0
                else:
                    obs_existence = 1
                if last_obs_existence + obs_existence == 1:
                    if last_obs_existence == 1:
                        self.increments.crossing_obs.add(last_obs)
                    if obs_existence == 1:
                        self.increments.crossing_obs.add(obs)

            current_value = current_value * P.gamma + reward
            if obs not in self.increments:
                self.increments[obs] = TransitionStorageCell(action=np.zeros([P.num_action]), parents=dict(), reward=0)
            self.increments[obs].parents[last_obs] = pre_action
            self.increments[obs].reward = reward
            if last_obs not in self.increments:
                self.increments[last_obs] = TransitionStorageCell(action=np.zeros([P.num_action]), parents=dict(), reward=0)

            if self.increments[last_obs].action[pre_action] < current_value:
                self.increments[last_obs].action[
                    pre_action
                ] = current_value  # may over estimate the value
            # self.increments[last_obs]['action'][pre_action] = 0.8 * self.increments[last_obs]['action'][pre_action] + 0.2 * current_value

    def merge_inc(self, inc):
        for last_obs in inc:
            if last_obs not in self.main:
                self.main[last_obs] = inc[last_obs]
            else:
                for p in inc[last_obs].parents:
                    self.main[last_obs].parents[p] = inc[last_obs].parents[p]

                if not P.soft_merge:
                    # hard merge
                    self.main[last_obs].action = np.maximum(
                        self.main[last_obs].action, inc[last_obs].action
                    )

                    self.main[last_obs].reward = max(
                        self.main[last_obs].reward, inc[last_obs].reward
                    )
                else:
                    # soft merge
                    self.main[last_obs].action = (
                        self.main[last_obs].action * 0.8
                        + inc[last_obs].action * 0.2
                    )

                    self.main[last_obs].reward = (
                        self.main[last_obs].reward * 0.8
                        + inc[last_obs].reward * 0.2
                    )

        self.main.update_max(inc.max_value)
        for e in inc.ends:
            self.main.ends.add(e)

    def post_process(self):
        for _ in range(P.num_bp):
            for e in self.main.ends:
                self.stack.append(e)
            self.back_propagation()

    def back_propagation(self):
        while len(self.stack) > 0:
            cur_obs = self.stack.pop()
            max_value = max(self.main[cur_obs].action)
            for par_obs in self.main[cur_obs].parents:
                action = self.main[cur_obs].parents[par_obs]
                new_value = max_value * P.gamma + self.main[cur_obs].reward
                if self.main[par_obs].action[action] < new_value:
                    self.main[par_obs].action[action] = new_value
                self.stack.append(par_obs)
