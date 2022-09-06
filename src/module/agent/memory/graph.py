from collections import defaultdict
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


class Graph:
    def __init__(self):
        self.obs_reward = dict()
        self.obs_next = defaultdict(dict)
        self.obs_best_action = dict()
        self.obs_node = dict()
        self.node_obs = defaultdict(list)
        self.node_value = dict()
        self.node_next = defaultdict(dict)

        self.trajs = list()

        self.general_info = dict()
        self.general_info["max_total_reward"] = - float("inf")
        self.general_info["max_total_reward_init_obs"] = None

    def next_node_ind(self):
        return len(self.node_obs)

    def best_action_update(self):
        for obs in self.obs_node:
            best_action = None
            node = self.obs_node[obs]
            if len(self.node_next[node]) > 1:
                max_value = - float("inf")
                for action in self.node_next[node]:
                    next_node_value = self.node_value[self.node_next[node][action]]
                    if next_node_value > max_value:
                        max_value = next_node_value
                        best_action = action
            else:
                best_action = self.obs_next[obs].keys()[0]
            self.obs_best_action[obs] = best_action

    def node_value_iteration(self):
        """
        GNN-based value propagation
        """
        total_nodes = len(self.node_obs)
        if total_nodes == 0:
            return
        adj = np.zeros([total_nodes, total_nodes], dtype=np.int8)
        rew = np.zeros([total_nodes], dtype=np.float32)
        val_0 = np.zeros([total_nodes], dtype=np.float32)
        colume_sum = np.zeros([total_nodes], dtype=np.int8)
        for node in self.node_obs:
            obs_list = self.node_obs[node]
            rew[node] = sum([self.obs_reward[o] for o in obs_list])
            val_0[node] = rew[node]
            for action in self.node_next[node]:
                n = self.node_next[node][action]
                adj[node][n] = 1
                colume_sum[n] += 1
        m1 = (np.sum(np.where(colume_sum > 1, 1, 0)) / total_nodes) * 100
        m5 = (np.sum(np.where(colume_sum > 5, 1, 0)) / total_nodes) * 100
        m10 = (np.sum(np.where(colume_sum > 10, 1, 0)) / total_nodes) * 100
        # to dertermine the graph memory is more like a graph or more like a tree
        # all 0% means it is a tree
        Logger.log(f"{m1:>4.1f}%|{m5:>4.1f}%|{m10:>4.1f}% nodes with >1|>5|>10 merging trails", color="yellow")  
        
        # value propagation
        if P.build_dag:
            adj = adj - self.iterator.build_dag(adj)
        val_n, iters, divider = self.iterator.iterate(adj, rew, val_0)
        Logger.log(f"learner value propagation: {iters} iters * {divider} batch", color="yellow")
        for ind, val in enumerate(val_n):
            self.node_value[ind] = val

    def draw_graph(self):
        fig_path = f"{P.result_dir}graph"

        # 1. get all the node from the graph
        connection_dict = defaultdict(set)
        for from_node in self.node_next:
            for action in self.node_next[from_node]:
                to_node = self.node_next[from_node][action]
                connection_dict[from_node].add(to_node)

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
            color_weight = self.node_value[node]
            if len(self.node_next[node]) > 1:
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
                color_weight = (self.node_value[from_node] + self.node_value[to_node]) / 2.0
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

    def general_info_update(self, gi):
        if gi["max_total_reward"] > self.general_info["max_total_reward"]:
            self.general_info["max_total_reward"] = gi["max_total_reward"]
            self.general_info["max_total_reward_init_obs"] = gi["max_total_reward_init_obs"]