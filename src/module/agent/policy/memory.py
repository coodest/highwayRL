# from types import prepare_class
# from typing import Sized
from src.util.tools import IO, Logger
from src.module.context import Profile as P
from src.module.agent.policy.graph import Graph
from tqdm import tqdm
# from src.util.imports.numpy import np  # influence the hit rate of actor


class Memory:
    """
    memory, the RL model, consists of the highway graph

    Assumptions:

    1. one state can not be both a terminate state and a middle state
    2. one state can have different obs
    3. from the historical obs, algorithms have the chance to restore the current state (fully observable)
    """
    def __init__(self, id, is_head) -> None:
        self.id = id
        self.is_head = is_head
        self.graph_path = P.model_dir + "graph.pkl"
        if P.load_graph:
            self.graph = IO.read_disk_dump(self.graph_path)
        else:
            self.graph = Graph()
        self.new_trajs = list()

    def sync_by_pipe_disk(self, head_slave_queues, slave_head_queues, sync, update):
        if not self.is_head:
            # write trajs (slave)
            slave_head_queues[self.id].put(self.new_trajs)
            del self.new_trajs
            self.new_trajs = list()
            ready = head_slave_queues[self.id].get()
            # read latest graph (slave)
            self.graph = IO.read_disk_dump(self.graph_path)
            slave_head_queues[self.id].put(["finish"])
        else:
            # read trajs (head)
            num_skip_trans = 0
            num_total_trans = 0
            num_all_traj = 0
            for i in range(P.num_actor):
                if self.id == i:
                    continue  # head has no trajs stored
                new_trajs = slave_head_queues[i].get()
                skip_trans, total_trans, all_traj = self.merge_new_trajs(new_trajs)
                del new_trajs
                num_skip_trans += skip_trans
                num_total_trans += total_trans
                num_all_traj += all_traj

            Logger.log(f"skip trans. / all trans. / all traj.: {num_skip_trans} / {num_total_trans} / {num_all_traj}", color="blue")
            
            if update.value:
                self.graph.update_graph()

            # write latest graph (head)
            IO.write_disk_dump(self.graph_path, self.graph)
            with sync.get_lock():
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

    def get_graph(self):
        return self.graph

    def store_new_trajs(self, trajectory):
        """
        store the trajectory.
        trajectory: o0, a0, o1, r(o0, a0) --> o1, a1, o2, r(o1, a1) --> ... --> on-1, an-1, on, r(on-1, an-1)
        """
        total_reward = 0
        for last_obs, prev_action, obs, last_reward in trajectory:
            total_reward += last_reward
        
        # filter trajs by min reward
        if P.min_traj_reward is not None:
            if total_reward < P.min_traj_reward:
                return
            
        if P.reward_filter_ratio is not None:
            if total_reward < P.reward_filter_ratio * self.graph.general_info["max_total_reward"]:
                return
        self.new_trajs.append(trajectory)

    def merge_new_trajs(self, new_trajs): 
        """
        merger increments to graph(s) by the head worker of learner
        """
        # add transitions to obs_next, obs_reward
        return self.graph.add_trajs(new_trajs)
