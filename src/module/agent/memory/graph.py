from types import prepare_class
from src.util.tools import IO, Logger
from src.module.context import Profile as P
from src.module.agent.memory.storage import Storage
import operator


class Graph:
    """
    normal and shrunk observatin graphs
    """
    def __init__(self, id, is_head) -> None:
        self.id = id
        self.is_head = is_head
        self.main = Storage(self.id)  # main storage
        self.inc = Storage(self.id)  # incremental storage

    def sync(self):
        """
        Synconize the mian and incremental stores that independent to their inner structure 
        and content.
        """
        if not self.is_head:
            # write increments (not head)
            IO.write_disk_dump(P.result_dir + f"{self.id}.pkl", self.inc)
            self.inc = Storage(self.id)
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
            past_obs = []
            for step, [last_obs, prev_action, obs, reward] in enumerate(traj):
                # find corssing obs
                obs_to_action[last_obs] = prev_action
                if not self.main.obs_exist(last_obs):
                    last_obs_existence = 0  # check the existence in the graph
                else:
                    last_obs_existence = 1
                if not self.main.obs_exist(obs):
                    obs_existence = 0
                else:
                    obs_existence = 1
                # add new crossing obs
                if last_obs_existence + obs_existence == 1:  # leave or enter a traj.
                    if last_obs_existence == 1:
                        crossing_obs[last_obs] = step
                    if obs_existence == 1:
                        crossing_obs[obs] = step + 1
                if last_obs_existence + obs_existence == 2:  # between two existing traj.
                    from_node = self.main.obs_node_ind(last_obs)
                    to_node = self.main.obs_node_ind(obs)
                    if from_node != to_node and to_node in self.main.node_next_accessable(from_node):
                        crossing_obs[last_obs] = step
                        crossing_obs[obs] = step + 1
                # add exising crossing obs
                last_obs_is_crossing = False
                if self.main.obs_is_crossing(last_obs):
                    last_obs_is_crossing = True
                    crossing_obs[last_obs] = step

                # check self-loop in the traj.
                past_obs.append(last_obs)
                if obs in past_obs:
                    # NOTE: we assum RNN projector do not produce self-loop in a traj.
                    Logger.log("self-loop detected in the traj.")
                    pass

                # update the obs-action prob
                if step < len(traj) - 1:
                    if last_obs_is_crossing and obs_existence == 1:
                        self.main.node_next_update_visit(last_obs, prev_action, obs)  # last_obs cover all obs in the traj.

            # add node and build interralation
            last_crossing_node_id = None
            last_action = None
            last_step = 0
            sorted_crossing_obs = dict(sorted(crossing_obs.items(), key=operator.itemgetter(1)))
            for co in sorted_crossing_obs:
                step = sorted_crossing_obs[co]
                # NOTE: process croossing_obs with ascending order
                assert step >= last_step, f"order wrong, last_step: {last_step}, step: {step}"
                crossing_node_ind = self.main.node_split(co)
                action = obs_to_action[co]
                o, a, r = self.get_traj_frag(traj, last_step, step)
                if len(o) > 0:
                    shrunk_node_ind = self.main.node_add(o, a, r, [{crossing_node_ind: 1}])
                    if last_crossing_node_id is not None:
                        self.main.crossing_node_add_action(last_crossing_node_id, last_action, shrunk_node_ind)
                last_crossing_node_id = crossing_node_ind
                last_action = action
                last_step = step + 1  # step is the crossing node, thus let it as step + 1
            # fragment after alst crossing obs or the traj without crossing obs
            o, a, r = self.get_traj_frag(traj, last_step, len(traj))
            if len(o) > 0:
                shrunk_node_ind = self.main.node_add(o, a, r, [{None: 1}])
                if last_crossing_node_id is not None:
                    self.main.crossing_node_add_action(last_crossing_node_id, last_action, shrunk_node_ind)

            for ind, [last_obs, prev_action, obs, reward] in enumerate(traj[:-1]):
                assert self.main.obs_exist(last_obs) and self.main.obs_exist(obs), "add traj. error"

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
