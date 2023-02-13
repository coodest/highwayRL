from collections import defaultdict
from distutils.command.config import dump_file
# from types import prepare_class
# from typing import Sized
from src.util.tools import IO, Logger
from src.module.context import Profile as P
from collections import defaultdict
import networkx as nx
# import matplotlib as mpl
import matplotlib.pyplot as plt
# import time
from src.util.imports.numpy import np
from src.util.tools import LinkedListElement
from src.module.agent.policy.iterator import Iterator
from tqdm import tqdm



class Graph:
    def __init__(self):
        self.obs_reward = dict()
        self.obs_next = defaultdict(dict)
        self.obs_prev = defaultdict(dict)
        self.obs_best_action = dict()
        
        self.general_info = dict()
        self.general_info["max_total_reward"] = - float("inf")
        self.general_info["max_total_reward_init_obs"] = None
        self.general_info["max_total_reward_traj"] = None

        self.iterator = Iterator()

        self.reset_node()

    def reset_node(self):
        self.obs_node = dict()
        self.node_obs = defaultdict(list)
        self.node_value = dict()
        self.node_next = defaultdict(dict)
        self.intersections = set()

    def next_node_ind(self):
        return len(self.node_obs)

    def add_trajs(self, trajs):
        num_skip_traj = 0
        for traj_ind, traj in enumerate(trajs):
            # filter randomness
            skip_traj = False
            for last_obs, prev_action, obs, last_reward in traj:
                if prev_action is not None:
                    if prev_action in self.obs_next[last_obs]:
                        exist_obs = list(self.obs_next[last_obs][prev_action].keys())[0]
                        if obs != exist_obs:
                            skip_traj = True
            if skip_traj:
                num_skip_traj += 1
                continue

            # add transition
            total_reward = 0
            for ind, [last_obs, prev_action, obs, last_reward] in enumerate(traj):
                total_reward += last_reward
                if last_obs in self.obs_reward:
                    self.obs_reward[last_obs] = max(self.obs_reward[last_obs], float(last_reward))
                else:
                    self.obs_reward[last_obs] = float(last_reward)
                if prev_action is not None:
                    if prev_action not in self.obs_next[last_obs]:
                        self.obs_next[last_obs][prev_action] = defaultdict(int)
                    self.obs_next[last_obs][prev_action][obs] += 1

                    if prev_action not in self.obs_prev[obs]:
                        self.obs_prev[obs][prev_action] = defaultdict(int)
                    self.obs_prev[obs][prev_action][last_obs] += 1

            # last_obs, prev_action, obs, reward (form obs) = trajectory item
            self.general_info_update({
                "max_total_reward": total_reward, 
                "max_total_reward_init_obs": traj[0][0],
                "max_total_reward_traj": traj,
            })
        return skip_traj, len(trajs)

    def general_info_update(self, gi):
        if gi["max_total_reward"] > self.general_info["max_total_reward"]:
            self.general_info["max_total_reward"] = gi["max_total_reward"]
            self.general_info["max_total_reward_init_obs"] = gi["max_total_reward_init_obs"]
            self.general_info["max_total_reward_traj"] = gi["max_total_reward_traj"]

    def is_intersection(self, obs):
        intersection = False
        if obs in self.obs_prev and len(self.obs_prev[obs]) > 1:
            intersection = True
        if obs in self.obs_prev and len(self.obs_prev[obs]) == 1 and len(list(self.obs_prev[obs].values())[0]) > 1:
            intersection = True
        if obs in self.obs_next and len(self.obs_next[obs]) > 1:
            intersection = True
        if obs in self.obs_next and len(self.obs_next[obs]) == 1 and len(list(self.obs_next[obs].values())[0]) > 1:
            intersection = True
        return intersection

    def graph_construction(self):
        self.reset_node()

        highway_ele = dict()
        for obs in self.obs_reward:
            if self.is_intersection(obs):
                intersection_node_ind = self.next_node_ind()
                self.intersections.add(intersection_node_ind)
                self.obs_node[obs] = intersection_node_ind
                self.node_obs[intersection_node_ind].append(obs)
                self.node_value[intersection_node_ind] = self.obs_reward[obs]
            else:
                if obs not in highway_ele:
                    highway_ele[obs] = LinkedListElement(obs)
                if obs in self.obs_next:
                    for action in self.obs_next[obs]:
                        next_obs = list(self.obs_next[obs][action].keys())[0]
                    if not self.is_intersection(next_obs):
                        if next_obs not in highway_ele:
                            highway_ele[next_obs] = LinkedListElement(next_obs)
                        highway_ele[obs].next = highway_ele[next_obs]
                        highway_ele[next_obs].prev = highway_ele[obs]
                if obs in self.obs_prev:
                    for action in self.obs_prev[obs]:
                        prev_obs = list(self.obs_prev[obs][action].keys())[0]
                    if not self.is_intersection(prev_obs):
                        if prev_obs not in highway_ele:
                            highway_ele[prev_obs] = LinkedListElement(prev_obs)
                        highway_ele[obs].prev = highway_ele[prev_obs]
                        highway_ele[prev_obs].next = highway_ele[obs]

        visited_obs = dict()
        fragments = defaultdict(list)
        total_len = 0
        for obs in highway_ele:
            if obs in visited_obs:
                continue
            fragment = highway_ele[obs].get_entire_list()
            dup_obs = dict()
            for ind, f in enumerate(fragment):
                if f not in visited_obs:
                    visited_obs[f] = fragment
                else:
                    dup_obs[f] = ind
            fragments[fragment[0]] = fragment
            total_len += len(fragment)
            if total_len != len(visited_obs) and len(dup_obs) > 0:
                Logger.log(f"total_len: {total_len}, visited_obs: {len(visited_obs)}, duplicated obs: {len(dup_obs)}", color="red")

        for first_obs in fragments:
            highway_node_ind = self.next_node_ind()
            obs_list = fragments[first_obs]
            reward_list = [self.obs_reward[obs] for obs in obs_list]
            discount_list = np.power(P.gamma, list(range(len(obs_list))))
            value = np.sum(np.multiply(reward_list, discount_list))
            self.node_obs[highway_node_ind] = obs_list
            self.node_value[highway_node_ind] = value
            for obs in obs_list:
                self.obs_node[obs] = highway_node_ind

            if first_obs in self.obs_prev:
                prev_action = list(self.obs_prev[first_obs].keys())[0]
                prev_obs = list(self.obs_prev[first_obs][prev_action].keys())[0]
                if prev_action not in self.node_next[self.obs_node[prev_obs]]:
                    self.node_next[self.obs_node[prev_obs]][prev_action] = defaultdict(int)
                self.node_next[self.obs_node[prev_obs]][prev_action][highway_node_ind] += 1
            last_obs = fragments[first_obs][-1]
            if last_obs in self.obs_next:
                next_action = list(self.obs_next[last_obs].keys())[0]
                next_obs = list(self.obs_next[last_obs][next_action].keys())[0]
                if next_action not in self.node_next[highway_node_ind]:
                    self.node_next[highway_node_ind][next_action] = defaultdict(int)
                self.node_next[highway_node_ind][next_action][self.obs_node[next_obs]] += 1

        # last, connect intersection node
        for node in self.intersections:
            obs = self.node_obs[node][0]
            if obs in self.obs_next:
                for action in self.obs_next[obs]:
                    for next_obs in self.obs_next[obs][action]:
                        next_node = self.obs_node[next_obs]
                        if action not in self.node_next[node]:
                            self.node_next[node][action] = defaultdict(int)
                        self.node_next[node][action][next_node] += 1

    def sanity_check(self):
        for node in self.intersections:
            intersection_obs = self.node_obs[node][0]
            if intersection_obs in self.obs_next and node in self.node_next:
                assert len(self.obs_next[intersection_obs]) == len(self.node_next[node]), f"incomplete intersection: {intersection_obs}"

        merging = 0
        for obs in self.obs_prev:
            prev_actions = list(self.obs_prev[obs].keys())
            if len(prev_actions) > 1:
                merging += 1
        Logger.log(f"graph is not a tree structure, {merging} mergings found", color="yellow")

    def node_value_iteration(self):
        """
        GNN-based value propagation
        """
        total_nodes = len(self.node_obs)
        if total_nodes == 0:
            return
        adj = np.zeros([total_nodes, total_nodes], dtype=np.float32)
        rew = np.zeros([total_nodes], dtype=np.float32)
        gamma = np.zeros([total_nodes], dtype=np.float32)
        val_0 = np.zeros([total_nodes], dtype=np.float32)
        colume_sum = np.zeros([total_nodes], dtype=np.int8)
        for node in self.node_obs:
            rew[node] = self.node_value[node]
            gamma[node] = np.power(P.gamma, len(self.node_obs[node]))
            val_0[node] = rew[node]
            if node in self.node_next:
                for action in self.node_next[node]:
                    for next_node in self.node_next[node][action]:
                        adj[node][next_node] = 1
                        colume_sum[next_node] += 1
        m1 = (np.sum(np.where(colume_sum > 1, 1, 0)) / total_nodes) * 100
        m5 = (np.sum(np.where(colume_sum > 5, 1, 0)) / total_nodes) * 100
        m10 = (np.sum(np.where(colume_sum > 10, 1, 0)) / total_nodes) * 100
        # to dertermine the graph memory is more like a graph or more like a tree
        # all 0% means it is a tree
        Logger.log(f"{m1:>4.1f}%|{m5:>4.1f}%|{m10:>4.1f}% nodes with >1|>5|>10 merging trails", color="yellow")  
        
        # value propagation
        if P.build_dag:
            adj = adj - self.iterator.build_dag(adj)
        val_n, iters, divider = self.iterator.iterate(adj, rew, gamma, val_0)
        Logger.log(f"learner value propagation: {iters} iters * {divider} batch", color="yellow")
        for ind, val in enumerate(val_n):
            self.node_value[ind] = val

    def best_action_update(self):
        for obs in self.obs_node:
            if obs in self.obs_next:
                if len(self.obs_next[obs]) > 1:
                    intersection_node = self.obs_node[obs]
                    max_value = - float("inf")
                    for action in self.node_next[intersection_node]:
                        for next_node in self.node_next[intersection_node][action]:
                            next_node_value = self.node_value[next_node]
                            if next_node_value >= max_value:
                                max_value = next_node_value
                                best_action = action
                    self.obs_best_action[obs] = best_action
                else:
                    best_action = list(self.obs_next[obs].keys())[0]
                    self.obs_best_action[obs] = best_action

    def get_obs_value(self, obs):
        node = self.obs_node[obs]
        value = self.node_value[node]
        for o in self.node_obs[node]:
            value -= self.obs_reward[o]
            if o == obs:
                break
        return value

    def draw_graph(self):
        fig_path = f"{P.result_dir}graph"

        # 1. build networkx DiGraph
        dg = nx.DiGraph()
        node_size = []
        node_color = []
        edge_color = []
        edge_list = []
        pos = []
        node_label = dict()
        crossing_node_size = 100
        normal_node_size = 10
        for ind, node in enumerate(self.node_value):
            color_weight = self.node_value[node]
            if node in self.intersections:
                dg.add_node(node, color=color_weight, size=crossing_node_size)
                node_size.append(crossing_node_size)
                node_color.append(color_weight)
            else:
                dg.add_node(node, color=color_weight, size=normal_node_size)
                node_size.append(normal_node_size)
                node_color.append(color_weight)

            if P.env_type == P.env_types[5]:
                coord = self.node_obs[node][0]
                pos.append([coord[0], - coord[1]])
            else:
                pos.append(np.random.rand(2,))
            node_label[ind] = node

        for ind, from_node in enumerate(self.node_next):
            for action in self.node_next[from_node]:
                for to_node in self.node_next[from_node][action]:
                    dg.add_edge(from_node, to_node, weight=action)
                    edge_list.append([from_node, to_node])
                    color_weight = (self.node_value[from_node] + self.node_value[to_node]) / 2.0
                    edge_color.append(color_weight)

        # 2. plot the graph with matplotlab
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
        plt.savefig(fig_path + ".pdf", format="PDF")  # save as PDF file
        plt.clf()

    
