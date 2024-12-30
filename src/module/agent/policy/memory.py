from src.util.tools import IO, Logger
from src.module.context import Profile as P
from src.module.agent.policy.graph import Graph


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
            num_switched_trans = 0
            num_total_trans = 0
            num_all_traj = 0
            for i in range(P.num_actor):
                if self.id == i:
                    continue  # head has no trajs stored
                new_trajs = slave_head_queues[i].get()
                if update.value:
                    switched_trans, total_trans, all_traj = self.merge_new_trajs(new_trajs)
                    num_switched_trans += switched_trans
                    num_total_trans += total_trans
                    num_all_traj += all_traj
                del new_trajs

            Logger.log(
                f"switched trans. / all trans. / all traj.: {num_switched_trans} / {num_total_trans} / {num_all_traj}",
                color="blue",
            )

            if update.value:
                self.graph.update_graph()
            else:
                Logger.log("graph update skipped")

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
        if len(trajectory) == 0:
            return
        
        if P.negative_reward_filter:
            # trunc traj before negative reward
            for ind, (last_obs, prev_action, obs, last_reward) in enumerate(trajectory):
                if last_reward < 0:
                    trajectory = trajectory[:ind]
        
        total_reward = 0
        expected_return = 0.0
        
        for last_obs, prev_action, obs, last_reward in trajectory[::-1]:
            total_reward += last_reward
            expected_return = expected_return * P.gamma + last_reward
            if last_obs is None or obs is None:
                Logger.log(
                    f"None in transition: {[last_obs, prev_action, obs, last_reward]}"
                )
                return

        # filter trajs by min reward
        if P.min_traj_reward is not None:
            if total_reward < P.min_traj_reward:
                return

        if P.reward_filter_ratio is not None:
            if total_reward < P.reward_filter_ratio * self.graph.general_info["max_income"]:
                return

        self.new_trajs.append([trajectory, total_reward, expected_return])

    def merge_new_trajs(self, new_trajs):
        """
        merger increments to graph(s) by the head worker of learner
        """
        # add transitions to obs_next, obs_reward
        return self.graph.add_trajs(new_trajs)
