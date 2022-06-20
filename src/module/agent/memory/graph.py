# from types import prepare_class
# from typing import Sized
from src.util.tools import IO, Logger
from src.module.context import Profile as P
from src.module.agent.memory.storage import Storage
import operator
from collections import defaultdict
import networkx as nx
# import matplotlib as mpl
import matplotlib.pyplot as plt
# import time
from src.util.imports.numpy import np


class Graph:
    """
    normal and shrunk observatin graphs

    Assumptions:

    1. one state can not be both a terminate state and a middle state
    2. one state can have different obs
    3. from the historical obs, algorithms have the chance to restore the current state
    """
    def __init__(self, id, is_head) -> None:
        self.id = id
        self.is_head = is_head
        self.main = Storage(self.id)  # main storage
        self.inc = Storage(self.id)  # incremental storage

    def sync_by_pipe_disk(self, head_slave_queues, slave_head_queues, sync):
        if not self.is_head:
            # write increments (slave)
            slave_head_queues[self.id].put(self.inc)
            self.inc = Storage(self.id)
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

    def sync_by_pipe(self, head_slave_queues, slave_head_queues, sync):
        if not self.is_head:
            # write increments (slave)
            slave_head_queues[self.id].put(self.inc)
            self.inc = Storage(self.id)
            self.main = head_slave_queues[self.id].get()
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
            sync.value = False
            for i in range(P.num_actor):
                if self.id == i:
                    continue
                head_slave_queues[i].put(self.main)

            # wait for all slave finished (head)
            for i in range(P.num_actor):
                if self.id == i:
                    continue
                finished = slave_head_queues[i].get()
                assert finished == ["finish"], "sync error"

    def sync_by_file(self, sync):
        """
        Synconize the mian and incremental stores that independent to their inner structure 
        and content. Only support small 'sync_every' (like 1)
        """
        if not self.is_head:
            # write increments (not head)
            IO.write_disk_dump(P.sync_dir + f"{self.id}.pkl", self.inc)
            self.inc = Storage(self.id)
            IO.write_disk_dump(P.sync_dir + f"{self.id}.ready", ["ready"])
            IO.stick_read_disk_dump(P.sync_dir + "target.ok")
            self.main = IO.read_disk_dump(P.sync_dir + "target.pkl")
            IO.write_disk_dump(P.sync_dir + f"{self.id}.finish", ["finish"])
        else:
            # make sure writes are complete
            for i in range(P.num_actor):
                if self.id == i:
                    continue
                IO.stick_read_disk_dump(P.sync_dir + f"{i}.ready")

            # read increments (head)
            for i in range(P.num_actor):
                if self.id == i:
                    continue  # head has no increments stored
                inc = IO.stick_read_disk_dump(P.sync_dir + f"{i}.pkl")
                self.merge_inc(inc)
            self.post_process()

            # write target (head)
            sync.value = False
            IO.write_disk_dump(P.sync_dir + "target.pkl", self.main)
            IO.write_disk_dump(P.sync_dir + "target.ok", ["ok"])

            # remove increments (head)
            for i in range(P.num_actor):
                if self.id == i:
                    continue
                IO.stick_read_disk_dump(P.sync_dir + f"{i}.finish")

            IO.delete_file(P.optimal_graph_path)
            IO.move_file(P.sync_dir + "target.pkl", P.optimal_graph_path)
            IO.renew_dir(P.sync_dir)

    def draw_graph(self):
        fig_path = f"{P.result_dir}graph"

        # 1. get all the node from the graph
        connection_dict = self.main.node_connection_dict()

        # 2. build networkx DiGraph
        dg = nx.DiGraph()
        node_to_ind = dict()
        node_size = []
        node_color = []
        edge_color = []
        edge_list = []
        pos = []
        node_label = dict()
        crossing_node_size = 100
        normal_node_size = 10
        for ind, node in enumerate(connection_dict.keys()):
            color_weight = self.main.node_value(node)
            if node in self.main.crossing_nodes():
                dg.add_node(ind, label=node, color=color_weight, size=crossing_node_size)
                node_size.append(crossing_node_size)
                node_color.append(color_weight)
            else:
                dg.add_node(ind, label=node, color=color_weight, size=normal_node_size)
                node_size.append(normal_node_size)
                node_color.append(color_weight)
            pos.append(np.random.rand(2,))
            node_to_ind[node] = ind
            node_label[ind] = node

        for ind, from_node in enumerate(connection_dict.keys()):
            for to_node in connection_dict[from_node]:
                if to_node is None:
                    continue
                dg.add_edge(node_to_ind[from_node], node_to_ind[to_node])
                edge_list.append((node_to_ind[from_node], node_to_ind[to_node]))
                color_weight = (self.main.node_value(from_node) + self.main.node_value(to_node)) / 2.0
                edge_color.append(color_weight)

        # 3. plot the graph with matplotlab
        # save as GEXF file
        # nx.write_gexf(dg, fig_path + ".gexf")
        # save as graphML file
        # nx.write_graphml(dg, fig_path + ".graphml")

        # save as PDF file
        if len(node_color) > 0:
            vmax = max(node_color)
            vmin = min(node_color)
        else:
            vmax = 1
            vmin = 0
        cmap = plt.cm.cividis
        nx.draw_networkx_labels(
            dg, 
            pos, 
            labels=node_label,
            font_color="grey",
            font_size=6, 
            font_family='sans-serif',
            verticalalignment="bottom"
        )
        nx.draw_networkx_nodes(
            dg, 
            pos, 
            node_size=node_size, 
            node_color=node_color,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=0.9,
        )
        nx.draw_networkx_edges(
            dg, 
            pos, 
            edgelist=edge_list,
            edge_color=edge_color,
            edge_cmap=cmap,
            edge_vmin=vmin,
            edge_vmax=vmax,
            node_size=node_size, 
            arrowstyle='->',
            arrowsize=5, 
            width=0.5,
            style="solid",
            connectionstyle="arc3,rad=0.1",
            alpha=0.5
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        cb = plt.colorbar(sm)
        cb.set_label('Node Value')
        plt.savefig(fig_path + ".pdf", format="PDF")
        plt.clf()

    def save_graph(self):
        IO.write_disk_dump(P.optimal_graph_path, self.main)

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
        # 1. build graph structure
        for traj in inc.trajs():
            # 1.1 find all crossing obs in current traj
            crossing_steps = dict()
            obs_to_steps = defaultdict(list)
            past_obs = []
            for step, [last_obs, prev_action, obs, reward] in enumerate(traj[:-1]):
                # 1.1.1 find corssing obs for existing graph
                if not self.main.obs_exist(last_obs):
                    last_obs_in_graph = 0  # check the existence in the graph
                else:
                    last_obs_in_graph = 1
                if not self.main.obs_exist(obs):
                    obs_in_graph = 0
                else:
                    obs_in_graph = 1
                # add new crossing obs
                if last_obs_in_graph + obs_in_graph == 1:  # leave or enter a traj.
                    if last_obs_in_graph == 1:
                        crossing_steps[step] = last_obs
                    if obs_in_graph == 1:
                        crossing_steps[step + 1] = obs
                if last_obs_in_graph + obs_in_graph == 2:  # between two existing traj.
                    from_node = self.main.obs_node_ind(last_obs)
                    to_node = self.main.obs_node_ind(obs)
                    if from_node != to_node:
                        crossing_steps[step] = last_obs
                        crossing_steps[step + 1] = obs
                
                # 1.2.2 add exising crossing obs, new traj. need to consider previously detected crossing_obs
                if self.main.obs_is_crossing(last_obs):
                    crossing_steps[step] = last_obs
                if self.main.obs_is_crossing(obs):
                    crossing_steps[step + 1] = obs

                # 1.1.3 find crossing obs for self loops in current traj.
                # map obs to list of steps
                obs_to_steps[last_obs].append(step)

                if last_obs not in past_obs:
                    last_obs_in_past = 0
                else:
                    last_obs_in_past = 1
                if obs not in past_obs:
                    obs_in_past = 0
                else:
                    obs_in_past = 1
                
                # add new crossing obs
                crossing = []
                if last_obs_in_past + obs_in_past == 1:  # leave or enter the past traj.
                    if last_obs_in_past == 1:
                        crossing.append(last_obs)
                    if obs_in_past == 1:
                        crossing.append(obs)
                if last_obs_in_past + obs_in_past == 2:  # between two past fragments.
                    from_step = obs_to_steps[last_obs][0]
                    to_step = obs_to_steps[obs][0]
                    if from_step != to_step - 1:
                        crossing.append(last_obs)
                        crossing.append(obs)

                for o in crossing:
                    for cs in obs_to_steps[o]:
                        if cs not in crossing_steps:
                            crossing_steps[cs] = o

                past_obs.append(last_obs)

            # 1.2 add node and build interralation to existing graph
            last_crossing_node_id = None
            last_action = None
            last_step = 0
            sorted_crossing_steps = list(crossing_steps.keys())
            list.sort(sorted_crossing_steps)
            for step in sorted_crossing_steps:  # process croossing_obs with ascending order
                assert step >= last_step, f"order wrong, last_step: {last_step}, step: {step}"
                crossing_obs = crossing_steps[step]
                crossing_node_ind = self.main.node_split(crossing_obs, reward=traj[step][3])
                o, a, r = self.get_traj_frag(traj, last_step, step)
                if len(o) > 0:
                    shrunk_or_crossing_node_ind = self.main.node_add(o, a, r, [{crossing_node_ind: 1}])
                else:
                    shrunk_or_crossing_node_ind = crossing_node_ind
                if last_crossing_node_id is not None:
                    self.main.crossing_node_add_action(last_crossing_node_id, last_action, shrunk_or_crossing_node_ind)
                last_crossing_node_id = crossing_node_ind
                last_action = traj[step][1]
                last_step = step + 1  # step is for the crossing node, thus let it as step + 1
            # fragment after last crossing obs or the traj without crossing obs
            o, a, r = self.get_traj_frag(traj, last_step, len(traj))
            if len(o) > 0:
                shrunk_or_crossing_node_ind = self.main.node_add(o, a, r, [{None: 1}])
            else:
                shrunk_or_crossing_node_ind = None
            if last_crossing_node_id is not None:
                self.main.crossing_node_add_action(last_crossing_node_id, last_action, shrunk_or_crossing_node_ind)

            # 1.3 check all obs-action-reward triples are added.
            for ind, [last_obs, prev_action, obs, reward] in enumerate(traj[:-1]):
                assert self.main.obs_exist(last_obs) and self.main.obs_exist(obs), "add traj. error"

        # 2. total reward update
        self.main.max_total_reward_update(
            total_reward=inc.max_total_reward(),
            init_obs=inc.max_total_reward_init_obs()
        )

    def post_process(self):
        # 1. value propagation
        self.main.node_value_propagate()
        
        # 2. update action of crossing obs
        self.main.crossing_node_action_update()

        # 3. draw graph (optinal)
        if P.draw_graph:
            self.draw_graph()
