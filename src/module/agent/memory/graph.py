from src.util.tools import IO, Logger
from src.module.context import Profile as P
from src.module.agent.memory.storage import Storage


class Graph:
    """
    normal and shrunk observatin graphs
    """
    def __init__(self, id, is_head) -> None:
        self.id = id
        self.is_head = is_head
        self.main = Storage()  # main storage
        self.inc = Storage()  # incremental storage

    def sync(self):
        """
        Synconize the mian and incremental stores that independent to their inner structure 
        and content.
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
            return self.main.obs_action(obs)
        else:
            return None

    def store_inc(self, trajectory, total_reward):
        """
        store the trajectory by the non-head process.
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
        amend_traj.append([final_obs, None, final_obs, final_reward])

        self.inc.trajs_add(amend_traj)

        # last_obs, prev_action, obs, reward (form obs) = trajectory item
        self.inc.max_total_reward_update(total_reward, trajectory[0][0])

    def get_traj_frag(self, traj, start, end):
        """
        get fragment [start, end) of trajectory
        """
        o = list()
        a = list()
        r = list()
        for last_obs, prev_action, obs, reward in traj[start: end]:
            o.append(last_obs)
            a.append([prev_action])
            r.append(reward)
        return o, a, r
    
    def merge_inc(self, inc: Storage): 
        """
        merger increments to main store by the head process
        """
        # build graph structure
        for traj in inc.trajs():
            # find all crossing obs in current traj
            crossing_obs = dict()
            obs_to_action = dict()
            for step, [last_obs, prev_action, obs, reward] in enumerate(traj):
                obs_to_action[last_obs] = prev_action
                if not self.main.obs_exist(last_obs):
                    last_obs_existence = 0  # check the existence in the graph
                else:
                    last_obs_existence = 1
                if not self.main.obs_exist(obs):
                    obs_existence = 0
                else:
                    obs_existence = 1
                # ad new crossing obs
                if last_obs_existence + obs_existence == 1: 
                    if last_obs_existence == 1:
                        crossing_obs[last_obs] = step
                    if obs_existence == 1:
                        crossing_obs[obs] = step + 1
                if last_obs_existence + obs_existence == 2:
                    if not self.main.node_next_contain(last_obs, obs):
                        crossing_obs[last_obs] = step
                        crossing_obs[obs] = step + 1
                # add exising crossing obs
                if self.main.obs_is_crossing(last_obs):
                    crossing_obs[last_obs] = step
                if self.main.obs_is_crossing(obs):
                    crossing_obs[obs] = step + 1

            # add node and build interralation
            last_crossing_node_id = None
            last_action = None
            last_step = 0
            for co in crossing_obs:
                step = crossing_obs[co]
                # NOTE: process croossing_obs with ascending order
                # assert step >= last_step, "order wrong"
                crossing_node_ind = self.main.node_split(co)
                action = obs_to_action[co]
                o, a, r = self.get_traj_frag(traj, last_step, step)
                if len(o) > 0:
                    shrunk_node_ind = self.main.node_add(o, a, r, [crossing_node_ind])
                    if last_crossing_node_id is not None:
                        self.main.crossing_node_add_action(last_crossing_node_id, last_action, shrunk_node_ind)
                last_crossing_node_id = crossing_node_ind
                last_action = action
                last_step = step + 1
            # fragment after alst crossing obs or the traj without crossing obs
            o, a, r = self.get_traj_frag(traj, last_step, len(traj))
            if len(o) > 0:
                shrunk_node_ind = self.main.node_add(o, a, r, [None])
                if last_crossing_node_id is not None:
                    self.main.crossing_node_add_action(last_crossing_node_id, last_action, shrunk_node_ind)

        # total reward update
        self.main.max_total_reward_update(
            total_reward=inc.max_total_reward(),
            init_obs=inc.max_total_reward_init_obs()
        )

    def post_process(self):
        # 1. value propagation
        self.main.node_value_propagate()
        
        # 2. update action of crossing obs
        self.main.crossing_node_action_update()
