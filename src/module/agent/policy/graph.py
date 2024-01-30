from src.util.tools import IO, Logger
from src.module.context import Profile as P
import networkx as nx
import matplotlib.pyplot as plt
from src.util.imports.numpy import np
from src.module.agent.policy.iterator import Iterator


class Graph:
    """
    node: intersection obss
    edge: highway with 1 to n transitions
    """
    def __init__(self):
        self.reset_trajs()

        self.reset_graph()

        # the value iteration operator
        self.iterator = Iterator()

    def reset_trajs(self):
        """
        clean all the transition data stored for highway graph construction
        """
        # reward: [o][a]=r(o, a)
        self.obs_action_reward = dict()
        # possible next obss of a obs: [o][a]=o'
        self.obs_next = dict()
        # possible prev obss of a obs: [o'][a]={o1, ...,on}
        self.obs_prev = dict()
        # {obs}
        self.starting_obs = set()
        # {obs}
        self.terminal_obs = set()

        # information about the graph for debugging
        self.general_info = dict()
        self.general_info["max_total_reward"] = - float("inf")
        self.general_info["max_total_reward_init_obs"] = None
        self.general_info["max_total_reward_traj"] = None

    def add_trajs(self, trajs, edge_spliting_support=True):
        num_skip_trans = 0
        total_trans = 0
        for traj in trajs:
            # add transition
            total_reward = 0.0
            self.starting_obs.add(traj[0][0])
            self.terminal_obs.add(traj[-1][2])
            current_value = 0.0
            for last_obs, prev_action, obs, last_reward in traj[::-1]:
                total_trans += 1
                total_reward += last_reward
                current_value = current_value * P.gamma + last_reward

                switch_to_new_trans = False
                first_time_transition = True
                if last_obs not in self.obs_next:
                    self.obs_next[last_obs] = dict()
                if prev_action in self.obs_next[last_obs]:
                    first_time_transition = False
                    if edge_spliting_support:  # if randomness occured, switch to obs with lower value
                        old_obs = self.obs_next[last_obs][prev_action]
                        if old_obs != obs:
                            # switch
                            if last_obs in self.Q:
                                if prev_action in self.Q[last_obs]:
                                    if self.Q[last_obs][prev_action] > current_value:
                                        switch_to_new_trans = True
                                        num_skip_trans += 1

                                        if old_obs in self.obs_prev:
                                            if prev_action in self.obs_prev[old_obs]:
                                                if last_obs in self.obs_prev[old_obs][prev_action]:
                                                    self.obs_prev[old_obs][prev_action].remove(last_obs)
                                                    if len(self.obs_prev[old_obs][prev_action]) == 0:
                                                        self.obs_prev[old_obs].pop(prev_action)
                                        self.obs_next[last_obs][prev_action] = obs
                else:
                    self.obs_next[last_obs][prev_action] = obs

                if first_time_transition or switch_to_new_trans:
                    if last_obs not in self.obs_action_reward:
                        self.obs_action_reward[last_obs] = dict()
                    self.obs_action_reward[last_obs][prev_action] = float(last_reward)

                    if obs not in self.obs_prev:
                        self.obs_prev[obs] = dict()
                    if prev_action not in self.obs_prev[obs]:
                        self.obs_prev[obs][prev_action] = set()
                    self.obs_prev[obs][prev_action].add(last_obs)

            # update general info
            if total_reward > self.general_info["max_total_reward"]:
                self.general_info["max_total_reward"] = total_reward
                self.general_info["max_total_reward_init_obs"] = traj[0][0]
                self.general_info["max_total_reward_traj"] = traj

        return num_skip_trans, total_trans, len(trajs)

    def reset_graph(self):
        """
        clean the current highway graph
        """
        # {node_id,}
        self.node = set()
        # [node_id] = value
        self.node_value = dict()
        # [node_id] = obs
        self.node_obs = dict()

        # [(from_node_id, action, to_node_id)] = length
        self.edge_length = dict()
        # [(from_node_id, action, to_node_id)] = discounted_reward
        self.edge_discounted_reward = dict()
        # [(from_node_id, action, to_node_id)] = value
        self.edge_value = dict()

        # {obs}
        self.all_obs = set()
        # bridge s-t graph and highway graph: [obs] = (from_node_id, action, to_node_id) or (intersection_node_id)
        self.obs_belonging = dict()
        
        # Q table on highway graph as a standard RL interface: [obs][action] = value
        # we use sparse Q and the value of unknow (o, a) should be -inf
        self.Q = dict()

    def next_node_ind(self):
        return len(self.node)
    
    def is_intersection(self, obs):
        is_intersection = False

        if obs not in self.obs_prev:
            is_intersection = True
        else:
            prev_obs_size = 0
            for prev_action in self.obs_prev[obs]:
                prev_obs_size += len(self.obs_prev[obs][prev_action])
            if prev_obs_size != 1:
                is_intersection = True

        if obs not in self.obs_next:
            is_intersection = True
        elif len(self.obs_next[obs].keys()) != 1:
            is_intersection = True

        if obs in self.starting_obs or obs in self.terminal_obs:
            is_intersection = True

        return is_intersection

    def graph_construction(self):
        self.reset_graph()

        # find all nodes of highway graph
        self.all_obs = set(self.obs_next.keys()).union(set(self.obs_prev.keys()))
        for obs in self.all_obs:
            if self.is_intersection(obs):
                node_id = self.next_node_ind()
                self.node.add(node_id)
                self.obs_belonging[obs] = (node_id,)
                self.node_obs[node_id] = obs

        # find all edges by nodes
        for from_node_obs in list(self.node_obs.values()):
            if from_node_obs not in self.obs_next:
                continue
            for action in self.obs_next[from_node_obs]:
                value = 0
                length = 0
                current_obs = from_node_obs
                current_action = action
                visited_obs = list()
                while True:
                    value += np.power(P.gamma, length) * self.obs_action_reward[current_obs][current_action]
                    length += 1
                    current_obs = self.obs_next[current_obs][current_action]
                    if self.is_intersection(current_obs):
                        to_node_id = self.obs_belonging[current_obs][0]
                        reach_end = True
                    else:
                        current_action = list(self.obs_next[current_obs].keys())[0]
                        visited_obs.append(current_obs)
                        reach_end = False
                    
                    if reach_end:
                        from_node_id = self.obs_belonging[from_node_obs][0]
                        self.edge_length[(from_node_id, action, to_node_id)] = length
                        self.edge_discounted_reward[(from_node_id, action, to_node_id)] = value
                        for obs in visited_obs:
                            self.obs_belonging[obs] = (from_node_id, action, to_node_id)
                        break

    def edge_value_iteration(self):
        """
        value propagation on edges
        """
        # init value iteration 
        total_edges = len(self.edge_length)
        if total_edges == 0:
            return
        rew = np.zeros([total_edges], dtype=np.float32)
        gamma = np.zeros([total_edges], dtype=np.float32)
        val_0 = np.zeros([total_edges], dtype=np.float32)
        adj = np.zeros([total_edges, total_edges], dtype=np.int32)
        ind_to_edge = dict()
        for ind, (f, a, t) in enumerate(self.edge_discounted_reward):
            ind_to_edge[ind] = (f, a, t)

            rew[ind] = self.edge_discounted_reward[(f, a, t)]
            gamma[ind] = np.power(P.gamma, self.edge_length[(f, a, t)])
            val_0[ind] = rew[ind]
        for from_ind, (_, _, t) in enumerate(self.edge_discounted_reward):
            for to_ind, (f, _, _) in enumerate(self.edge_discounted_reward):
                if f == t:
                    adj[from_ind, to_ind] = 1

        # value propagation
        val_n, n_iters, divider = self.iterator.iterate(rew, gamma, val_0, adj)
        Logger.log(f"learner value propagation: {n_iters} iters * {divider} batch", color="yellow")
        
        # update node and edge value
        for ind, val in enumerate(val_n):
            (f, a, t) = ind_to_edge[ind]
            self.edge_value[(f, a, t)] = val
            if f in self.node_value:
                if val > self.node_value[f]:
                    self.node_value[f] = val
            else:
                self.node_value[f] = val
        for node in self.node:
            if node not in self.node_value:  # termination nodes
                self.node_value[node] = 0.0

    def update_Q(self):
        for obs in self.all_obs:
            if len(self.obs_belonging[obs]) == 1:  # obs is a node
                node_id = self.obs_belonging[obs][0]
                for (f, a, t) in self.edge_value:
                    if f == node_id:
                        if obs not in self.Q:
                            self.Q[obs] = dict()
                        self.Q[obs][a] = self.edge_value[(f, a, t)]  # termination node will not in Q
            if len(self.obs_belonging[obs]) == 3:  # obs on an edge
                next_obs = list(self.obs_next[obs].values())[0]
                if len(self.obs_belonging[next_obs]) == 1:
                    current_obs = obs
                    node_id = self.obs_belonging[next_obs][0]
                    current_value = - float("inf")
                    for (f, a, t) in self.edge_value:
                        if f == node_id:
                            if current_value < self.edge_value[(f, a, t)]:
                                current_value = self.edge_value[(f, a, t)]
                    if current_value == - float("inf"):  # for termination node
                        current_value = 0.0
                    while True:
                        current_action = list(self.obs_next[current_obs].keys())[0]
                        current_reward = self.obs_action_reward[current_obs][current_action]
                        current_value = P.gamma * current_value + current_reward
                        if current_obs not in self.Q:
                            self.Q[current_obs] = dict()
                        self.Q[current_obs][current_action] = current_value
                        current_obs = list(self.obs_prev[current_obs].values())[0].copy().pop()
                        if len(self.obs_belonging[current_obs]) == 1:  # reach the from_node
                            break

        Logger.log("learner update action ready", color="yellow")

    def draw_graph(self, add_label=["id+value", "none"][0], self_loop=True):
        if len(self.node) > P.max_node_draw:
            Logger.log("graph is too large to draw")
            return
        
        fig_path = f"{P.result_dir}graph.pdf"
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)

        # 1. build networkx DiGraph
        dg = nx.MultiDiGraph()
        node_color = np.zeros([len(self.node_obs)])  # order must same to adding sequence of node
        edge_color = []
        edge_list = []
        pos = dict()
        node_label = dict()
        node_size = 100
        for ind, node in enumerate(self.node_value):
            color_weight = self.node_value[node]
            dg.add_node(node)
            node_color[ind] = color_weight

            if P.env_type == "maze":
                coord = self.node_obs[node]
                pos[node] = [coord[0], - coord[1]]
            elif P.env_name == "CliffWalking-v0":
                coord = self.node_obs[node]
                pos[node] = [coord % 12, -int(coord / 12)]
            else:
                pos[node] = np.random.rand(2,)

            if add_label == "id+value":
                node_label[node] = f"{node}\n{float(self.node_value[node]):.2f}"
            else:
                node_label[node] = ""

        for (f, a, t) in self.edge_discounted_reward:
            if not self_loop and f == t:
                continue
            dg.add_edge(f, t, weight=a)
            edge_list.append([f, t])
            color_weight = (self.node_value[f] + self.node_value[t]) / 2.0
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
            font_color="red",
            font_size=3, 
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
            alpha=1,
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
            alpha=0.5,
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        cb = plt.colorbar(sm)
        cb.set_label('Node Value')

        ax.set_axis_off()

        plt.savefig(fig_path, format="PDF")  # save as PDF file
        plt.clf()

    def update_graph(self):
        # 1. graph reconstruction
        self.graph_construction()

        # 2. value iteration
        self.edge_value_iteration()
        
        # 3. update Q
        self.update_Q()

        # 4. draw graph
        self.draw_graph()

    def get_action(self, obs):
        if self.general_info["max_total_reward_traj"] is not None:
            steps = len(self.general_info["max_total_reward_traj"])
        else:
            steps = 0

        if obs in self.Q:
            if len(list(self.Q[obs].keys())) > 0:
                # if multiple max values, return the first occurence
                action = max(self.Q[obs], key=self.Q[obs].get)
                value = self.Q[obs][action]
                return action, value, steps
        return None, None, steps

    def info(self):
        return "ST/HW: {}/{}({:.1f}%) MaxEpiRwd: {:.2f}/{}".format(
            len(self.all_obs),
            len(self.node),
            100 * (len(self.node) / (len(self.all_obs) + 1e-8)),
            self.general_info["max_total_reward"],
            str(self.general_info["max_total_reward_init_obs"])[-4:],
        )
