from src.util.tools import IO, Logger
from src.module.context import Profile as P
from src.module.agent.memory.storage import Storage
from src.module.agent.memory.indexer import Indexer   
from src.util.imports.numpy import np


class Graph:
    """
    observatin graph
    """
    def __init__(self, id, is_head) -> None:
        self.id = id
        self.is_head = is_head
        self.main = Storage()  # main storage
        self.inc = Storage()  # incremental storage
        if self.is_head:
            self.shrinked_trajs = dict()
            self.processed_crossing_obs = set()

    def sync(self):
        """
        Synconization that independent to the inner structure and content of increments 
        and main storage.
        """
        if not self.is_head:
            # write increments (not head)
            IO.write_disk_dump(P.result_dir + f"{self.id}.pkl", self.inc)
            self.inc = Storage()
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

    def get_action(self, obs):
        if self.main.obs_exist(obs):
            return self.main.obs_best_action(obs)
        else:
            return None

    def store_inc(self, trajectory, total_reward):  # by the non-head
        # 1. store trajector info
        traj_ind = Indexer.get_traj_ind(trajectory)
        if not self.inc.traj_exist(traj_ind):
            self.inc.traj_add(
                traj_ind=traj_ind,
                total_reward=total_reward,
                init_obs=trajectory[0][0],
                traj_len=len(trajectory)
            )

        # 2. store obs info
        for step, [last_obs, pre_action, obs, reward] in enumerate(trajectory):
            # 2.1 add traj info
            self.inc.traj_add_action(traj_ind, pre_action)
            self.inc.traj_add_reward(traj_ind, step, reward)

            # 2.2 statistic crossing obs
            if P.statistic_crossing_obs:
                if not self.main.obs_exist(last_obs) and not self.inc.obs_exist(last_obs):
                    last_obs_existence = 0  # check the existence in the graph
                else:
                    last_obs_existence = 1
                if not self.main.obs_exist(obs) and not self.inc.obs_exist(obs):
                    obs_existence = 0
                else:
                    obs_existence = 1
                if last_obs_existence + obs_existence == 1:  # corssing
                    if last_obs_existence == 1:
                        self.inc.crossing_obs_add(last_obs)
                    if obs_existence == 1:
                        self.inc.crossing_obs_add(obs)

            # 2.3 add last_oobs to the storage
            if not self.inc.obs_exist(last_obs):
                self.inc.obs_add(last_obs)
            self.inc.obs_update(
                obs=last_obs, 
                action=pre_action, 
                afiliated_traj=traj_ind, 
                step=step
            )

    def merge_inc(self, inc: Storage):  # by the head
        for last_obs in inc.obs_dict():
            if not self.main.obs_exist(last_obs):
                self.main.obs_dict()[last_obs] = inc.obs_dict()[last_obs]
            else:
                self.main.obs_afiliated_traj(last_obs).update(
                    inc.obs_afiliated_traj(last_obs)
                )
        for traj_ind in inc.traj_dict():
            self.main.total_reward_update(
                total_reward=inc.traj_total_reward(traj_ind),
                init_obs=inc.traj_init_obs(traj_ind)
            )
        self.main.crossing_obs_union(inc.crossing_obs_set())

    def post_process(self):
        # 1. insert new crossing obs and value propagation
        for cb in self.main.crossing_obs_set():
            if cb in self.processed_crossing_obs:
                continue

            # for each new cb, find related traj and shrinked node to insert
            self.shrinked_graph.processed_crossing_obs.add(cb)
            for at in self.main[cb][Storage.afiliated_trajectories]:
                cb_step = self.main[cb][Storage.afiliated_trajectories][at]
                if at in self.shrinked_graph.shrinked_trajs:  # find existing traj on the shrinked graph
                    for shrink_node in self.shrinked_graph.shrinked_trajs[at]:
                        if shrink_node["range"][0] < cb_step and shrink_node["range"][1] > cb_step:
                            self.shrinked_graph.crossing_nodes.add(cb)
                            shrink_node_1 = dict()
                            shrink_node_2 = dict()
                            
                            break

                else:  # create a new traj on the shrinked graph
                    self.shrinked_graph.shrinked_trajs[at] = list()
                    shrink_node_1 = dict()
                    crossing_node = dict()
                    shrink_node_2 = dict()

                    start = 0
                    end = cb_step - 1
                    shrink_node_1["type"] = "shrink"
                    shrink_node_1["range"] = [start, end]
                    shrink_node_1["total_reward"] = np.sum(self.main.trajectory_infos[at][Storage.reward][start : end])
                    shrink_node_1["next"] = crossing_node
                    shrink_node_1["prev"] = None

                    crossing_node["type"] = "crossing"
                    crossing_node["total_reward"] = self.main.trajectory_infos[at][Storage.reward][cb_step]
                    crossing_node["next"] = shrink_node_2
                    crossing_node["prev"] = shrink_node_1

                    start = cb_step + 1
                    end = len(self.main.trajectory_infos[at][Storage.action]) - 1
                    shrink_node_2["type"] = "shrink"
                    shrink_node_2["range"] = [start, end]
                    shrink_node_2["total_reward"] = np.sum(self.main.trajectory_infos[at][Storage.reward][start : end])
                    shrink_node_2["next"] = None
                    shrink_node_2["prev"] = crossing_node

                    self.shrinked_graph.shrinked_trajs[at].append(shrink_node_1)
                    self.shrinked_graph.shrinked_trajs[at].append(crossing_node)
                    self.shrinked_graph.shrinked_trajs[at].append(shrink_node_2)

            # 2. update action of crossing obs
