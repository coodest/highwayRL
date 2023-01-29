# from types import prepare_class
# from typing import Sized
from src.util.tools import IO, Logger
from src.module.context import Profile as P
from src.module.agent.policy.graph import Graph
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
        return "G/C: {}/{}({:.1f}%) V: {:.2f}/{}".format(
            len(self.main.obs_node),
            len(self.main.intersections) if P.statistic_crossing_obs else "-",
            100 * (len(self.main.intersections) / (len(self.main.obs_node) + 1e-8)) if P.statistic_crossing_obs else "-",
            self.main.general_info["max_total_reward"],
            str(self.main.general_info["max_total_reward_init_obs"])[-4:],
        )

    def save(self):
        IO.write_disk_dump(P.optimal_policy_path, self.main)
        Logger.log("memory saved")

    def get_action(self, obs):
        if obs in self.main.obs_best_action:
            action = self.main.obs_best_action[obs]
            value = self.main.get_obs_value(obs)
            return action, value
        return None, None

    def store_inc(self, trajectory, total_reward):
        """
        amend and store the trajectory by the non-head process.
        trajectory: o0, a0, o1, r1 --> o1, a1, o2, r2 --> ... --> on-1, an-1, on, rn
        amend_traj: o0, a0, o1, r0 --> o1, a1, o2, r1 --> ... --> on-1, an-1, on, rn-1 --> on, None, on, rn
        """
        amend_traj = list()
        last_reward = 0.0
        final_obs = None
        final_reward = None
        for last_obs, prev_action, obs, reward in trajectory:
            amend_traj.append([last_obs, prev_action, obs, last_reward])
            last_reward = reward
            final_obs = obs
            final_reward = reward
        amend_traj.append([final_obs, None, final_obs, final_reward])  # last transition

        self.inc.trajs.append(amend_traj)

    def merge_inc(self, inc: Graph): 
        """
        merger increments to main store by the head worker of learner
        """
        # 1. add transitions to obs_next, obs_reward
        self.main.add_trajs(inc.trajs)
        
        # 2. update general info
        self.main.general_info_update(inc.general_info)

    def post_process(self):
        # 1. graph reconstruction
        self.main.graph_construction()

        # 2. check the vlidity of the graph
        if P.graph_sanity_check:
            self.main.sanity_check()

        # 3. value iteration
        self.main.node_value_iteration()
        
        # 4. update action of crossing obs
        self.main.best_action_update()

        # 5. draw graph (optinal)
        if P.draw_graph:
            self.main.draw_graph()
        

 
