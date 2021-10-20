import time

from src.util.tools import IO, Logger
from src.module.context import Profile as P
from src.module.agent.memory.storage import TransitionStorage
from src.module.agent.memory.storage import OptimalStorage
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

    def store_increments(self, trajectory, value):
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
            return self.main[obs][OptimalStorage.best_action]
        else:
            return None

    def store_increments(self, trajectory, value):  # by the non-head
        # 1. store trajector info
        trajectory_ind = Indexer.get_trajectory_ind(trajectory)
        if trajectory_ind not in self.increments.trajectory_infos:
            self.increments.trajectory_infos[trajectory_ind] = [
                value,
                trajectory[0][0],
                list(),
                np.zeros(len(trajectory))
            ]

        # 2. store obs info
        for step, [last_obs, pre_action, obs, reward] in enumerate(trajectory):
            # 2.1 add traj info
            self.increments.trajectory_infos[trajectory_ind][OptimalStorage.action].append(pre_action)
            self.increments.trajectory_infos[trajectory_ind][OptimalStorage.reward][step] = reward

            # 2.2 statistic crossing obs
            if P.statistic_crossing_obs:
                if last_obs not in self.main and last_obs not in self.increments:
                    last_obs_existence = 0  # check the existence in the graph
                else:
                    last_obs_existence = 1
                if obs not in self.main and obs not in self.increments:
                    obs_existence = 0
                else:
                    obs_existence = 1
                if last_obs_existence + obs_existence == 1:  # corssing
                    if last_obs_existence == 1:
                        self.increments.crossing_obs.add(last_obs)
                    if obs_existence == 1:
                        self.increments.crossing_obs.add(obs)

            # 2.3 add last_oobs to the storage
            if last_obs not in self.increments:
                self.increments[last_obs] = [None, dict()]
            self.increments[last_obs][OptimalStorage.best_action] = pre_action
            self.increments[last_obs][OptimalStorage.afiliated_trajectories][trajectory_ind] = step

    def merge_inc(self, inc):  # by the head
        for last_obs in inc:
            if last_obs not in self.main:
                self.main[last_obs] = inc[last_obs]
            else:
                self.main[last_obs][OptimalStorage.afiliated_trajectories].update(
                    inc[last_obs][OptimalStorage.afiliated_trajectories]
                )
        for trajectory_ind in inc.trajectory_infos:
            self.main.update_max(
                inc.trajectory_infos[trajectory_ind][OptimalStorage.value], 
                inc.trajectory_infos[trajectory_ind][OptimalStorage.init_obs]
            )
        self.main.crossing_obs = self.main.crossing_obs.union(inc.crossing_obs)

    def post_process(self):
        pass


class TransitionGraph(Graph):
    def __init__(self, id, is_head) -> None:
        super().__init__(id, is_head)
        self.stack = list()
        self.known_states = []

    def make_new_store(self):
        return TransitionStorage()

    def get_action(self, obs, raw_obs):
        if obs in self.main:
            return np.argmax(self.main[obs][TransitionStorage.action])
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
                        values += self.main[ind][TransitionStorage.action]
                        num_validate_nb += 1
                if num_validate_nb > 0:
                    values /= num_validate_nb
                    best_actions = np.argwhere(values == np.max(values)).flatten()
                    action = np.random.choice(best_actions)
                    return action

            return None

    def store_increments(self, trajectory, value):
        # filtering
        if P.sync_mode == P.sync_modes[0]:
            if self.increments.max_value >= value:
                return
        if P.sync_mode == P.sync_modes[2]:
            if self.increments.max_value * P.sync_tolerance >= value:
                return

        # update increment
        self.increments.add_end(trajectory[-1][2])
        self.increments.update_max(value)
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
                self.increments[obs] = [
                    np.zeros([P.num_action]),
                    dict(),
                    0
                ]
            self.increments[obs][TransitionStorage.parents][last_obs] = pre_action
            self.increments[obs][TransitionStorage.reward] = reward
            if last_obs not in self.increments:
                self.increments[last_obs] = [
                    np.zeros([P.num_action]),
                    dict(),
                    0
                ]
            if self.increments[last_obs][TransitionStorage.action][pre_action] < current_value:
                self.increments[last_obs][TransitionStorage.action][
                    pre_action
                ] = current_value  # may over estimate the value

    def merge_inc(self, inc):
        for last_obs in inc:
            if last_obs not in self.main:
                self.main[last_obs] = inc[last_obs]
            else:
                for p in inc[last_obs][TransitionStorage.parents]:
                    self.main[last_obs][TransitionStorage.parents][p] = inc[last_obs][TransitionStorage.parents][p]

                if not P.soft_merge:
                    # hard merge
                    self.main[last_obs][TransitionStorage.action] = np.maximum(
                        self.main[last_obs][TransitionStorage.action], inc[last_obs][TransitionStorage.action]
                    )

                    self.main[last_obs][TransitionStorage.reward] = max(
                        self.main[last_obs][TransitionStorage.reward], inc[last_obs][TransitionStorage.reward]
                    )
                else:
                    # soft merge
                    self.main[last_obs][TransitionStorage.action] = (
                        self.main[last_obs][TransitionStorage.action] * 0.8
                        + inc[last_obs][TransitionStorage.action] * 0.2
                    )

                    self.main[last_obs][TransitionStorage.reward] = (
                        self.main[last_obs][TransitionStorage.reward] * 0.8
                        + inc[last_obs][TransitionStorage.reward] * 0.2
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
            max_value = max(self.main[cur_obs][TransitionStorage.action])
            for par_obs in self.main[cur_obs][TransitionStorage.parents]:
                action = self.main[cur_obs][TransitionStorage.parents][par_obs]
                new_value = max_value * P.gamma + self.main[cur_obs][TransitionStorage.reward]
                if self.main[par_obs][TransitionStorage.action][action] < new_value:
                    self.main[par_obs][TransitionStorage.action][action] = new_value
                self.stack.append(par_obs)
