# from types import prepare_class
# from typing import Sized
from src.util.tools import IO, Logger
from src.module.context import Profile as P
from src.module.agent.memory.graph import Graph
import operator
from collections import defaultdict
import networkx as nx
# import matplotlib as mpl
import matplotlib.pyplot as plt
# import time
from src.util.imports.numpy import np


class Memory:
    """
    memory for the highway graph

    Assumptions:

    1. one state can not be both a terminate state and a middle state
    2. one state can have different obs
    3. from the historical obs, algorithms have the chance to restore the current state
    """
    def __init__(self, id, is_head) -> None:
        self.id = id
        self.is_head = is_head
        self.main = Graph()  # main storage
        self.inc = Graph()  # incremental storage

    def sync_by_pipe_disk(self, head_slave_queues, slave_head_queues, sync):
        if not self.is_head:
            # write increments (slave)
            slave_head_queues[self.id].put(self.inc)
            self.inc = Graph()
            ready = head_slave_queues[self.id].get()
            self.main = IO.read_disk_dump(P.sync_dir + "target.pkl")
            slave_head_queues[self.id].put(["finish"])
        else:
            # read increments (head)
            for i in range(P.num_actor):
                if self.id == i:
                    continue  # head has no increments stored
                inc = slave_head_queues[i].get()
                self.merge_inc(inc)
            self.post_process()

            # write target (head)
            IO.renew_dir(P.sync_dir)
            IO.write_disk_dump(P.sync_dir + "target.pkl", self.main)
            sync.value = False
            for i in range(P.num_actor):
                if self.id == i:
                    continue
                head_slave_queues[i].put(["ready"])

            # wait for all slave finished (head)
            for i in range(P.num_actor):
                if self.id == i:
                    continue
                finished = slave_head_queues[i].get()
                assert finished == ["finish"], "sync error"

    def info(self):
        return "G/C: {}/{}({:.1f}%) V: {:.1f}/{}".format(
            len(self.main.obs_node),
            self.main.crossing_node_size() if P.statistic_crossing_obs else "-",
            100 * (self.main.crossing_node_size() / (len(self.main.obs_node) + 1e-8)) if P.statistic_crossing_obs else "-",
            self.main.general_info["max_total_reward"],
            str(self.main.general_info["max_total_reward_init_obs"])[-4:],
        )

    def save(self):
        IO.write_disk_dump(P.optimal_graph_path, self.main)
        Logger.log("memory saved")

    def sanity_check(self):
        pass

    def get_action(self, obs):
        if obs in self.main.obs_best_action:
            return self.main.obs_best_action[obs]
        return None

    def store_inc(self, trajectory, total_reward):
        """
        amend and store the trajectory by the non-head process.
        trajectory: o0, a0, o1, r1 --> o1, a1, o2, r2 --> ... --> on-1, an-1, on, rn
        amend_traj: o0, a0, o1, r0 --> o1, a1, o2, r1 --> ... --> on-1, an-1, on, rn-1 --> on, None, on, rn
        """
        amend_traj = list()
        last_reward = 0
        final_obs = None
        final_reward = None
        for last_obs, prev_action, obs, reward in trajectory:
            amend_traj.append([last_obs, prev_action, obs, last_reward])
            last_reward = reward
            final_obs = obs
            final_reward = reward
        amend_traj.append([final_obs, None, final_obs, final_reward])  # last transition

        self.inc.trajs.append(amend_traj)

        # last_obs, prev_action, obs, reward (form obs) = trajectory item
        self.inc.general_info_update({
            "max_total_reward": total_reward, 
            "max_total_reward_init_obs": trajectory[0][0],
        })
    
    def merge_inc(self, inc: Graph): 
        """
        merger increments to main store by the head process
        """

        def node_split(cross_node_ind, cross_obs):
            obs_list = self.main.node_obs[cross_node_ind]
            if len(obs_list) == 1:
                return cross_node_ind
            cross_ind = obs_list.index(cross_obs)
            if cross_ind == 0:
                self.main.node_obs[cross_node_ind] = [obs_list[cross_ind]]
                self.main.node_obs[self.main.next_node_ind()] = obs_list[cross_ind + 1:]
                return cross_node_ind
            elif cross_ind == len(obs_list) - 1:
                self.main.node_obs[cross_node_ind] = obs_list[:cross_ind]
                new_cross_node_ind = self.main.next_node_ind()
                self.main.node_obs[new_cross_node_ind] = [obs_list[cross_ind]]
                return new_cross_node_ind
            else:
                self.main.node_obs[cross_node_ind] = obs_list[:cross_ind]
                new_cross_node_ind = self.main.next_node_ind()
                self.main.node_obs[new_cross_node_ind] = [obs_list[cross_ind]]
                self.main.node_obs[self.main.next_node_ind()] = obs_list[cross_ind + 1:]
                return new_cross_node_ind

        # 1. build graph structure
        for traj_ind, traj in enumerate(inc.trajs):
            last_node_ind = None
            last_crossing_node_ind = None
            for last_obs, prev_action, obs, last_reward in traj:
                # 1.1 add transitions to obs_next, obs_reward
                self.main.obs_reward[last_obs] = last_reward
                if prev_action is not None:
                    self.main.obs_next[last_obs][prev_action] = obs
                # 1.2 add node--obs bimap
                if last_obs not in self.main.obs_node:
                    if last_node_ind is None:
                        last_node_ind = self.main.next_node_ind()
                    self.main.obs_node[last_obs] = last_node_ind
                    self.main.node_obs[last_node_ind].append(last_obs)
                    if last_crossing_node_ind is not None:
                        self.main.node_next[last_crossing_node_ind][prev_action] = last_node_ind
                        last_crossing_node_ind = None
                else:
                    if len(self.main.obs_next[last_obs]) > 1:
                        # meet existing intersection
                        cross_node_ind = self.main.obs_node[last_obs]
                        if last_node_ind is not None:
                            if prev_action in self.main.node_next[last_node_ind]:
                                assert self.main.node_next[last_node_ind][prev_action] == cross_node_ind, "env is stochastic."
                            else:
                                self.main.node_next[last_node_ind][prev_action] = cross_node_ind
                        if last_crossing_node_ind is not None:
                            self.main.node_next[last_crossing_node_ind][prev_action] = cross_node_ind
                        last_crossing_node_ind = cross_node_ind
                        last_node_ind = None 
                    else:
                        # meet existing highway
                        if last_node_ind is None:  # hit the highway at the beginning
                            last_node_ind = self.main.obs_node[last_obs]
                            cross_node_ind = last_node_ind
                        else:
                            current_node_ind = self.main.obs_node[last_obs]
                            if last_node_ind != current_node_ind:
                                # enter the highway from other node: split and connect
                                cross_node_ind = node_split(current_node_ind, last_obs)
                                last_crossing_node_ind = cross_node_ind
                            else:
                                cross_node_ind = last_node_ind
                        if prev_action not in self.main.obs_next[last_obs]:
                            # leave highway: split and connect
                            cross_node_ind = node_split(current_node_ind, last_obs)
                            last_crossing_node_ind = cross_node_ind

                        if last_crossing_node_ind is not None:
                            self.main.node_next[last_crossing_node_ind][prev_action] = cross_node_ind
                            last_crossing_node_ind = None

        # 2. total reward update
        self.main.general_info_update(inc.general_info)

    def post_process(self):
        # 1. value iteration
        self.main.node_value_iteration()
        
        # 2. update action of crossing obs
        self.main.best_action_update()

        # 3. draw graph (optinal)
        if P.draw_graph:
            self.main.draw_graph()

        if P.graph_sanity_check:
            self.sanity_check()

 
