from src.module.agent.transition.prob_tgn import ProbTGN
from src.module.agent.memory.graph import Graph
from src.module.context import Profile as P
from src.module.agent.transition.utils.data_processing import Data
from src.util.tools import *

import math


class Policy:
    def __init__(self):
        # make episodic memory
        self.graph = Graph()
        # make transition function
        self.prob_func = ProbTGN()
        self.MCTS_n = None

    def get_action(self, last_obs, pre_action, obs, reward, add=False):
        # 1. add transition to graph memory
        if add:
            self.graph.add(last_obs, pre_action, obs, reward)

        # 2. find current/root node
        root = self.graph.get_node_id(obs)
        if root not in self.graph.his_edges:  # can not give action for previously never interacted obs
            return None

        # 3. use UCB1 formula to propagate value
        self.update_children(root)

        # 4. for training actor: select the child with max UCB1 and
        # return corresponding action;
        # for testing actor: select the child with max value and
        # return corresponding action
        child_id, action = self.get_max_child(root, value_type="value")

        return action

    def update_children(self, root):
        """
        build or update the UCB1 profile of root and its children
        1. avoid loop error
        2. avoid intersection ucb1 compute error
        """
        self.MCTS_n = dict()
        for t in range(P.propagations):
            # simulate
            total_reward = 0
            visit_list = []
            current_node = root  # node of root obs
            simulate_steps = P.simulate_steps
            while current_node is not None:
                simulate_steps -= 1
                if simulate_steps <= 0:
                    total_reward += self.graph.node_value[current_node]  # only obs node has reward
                    visit_list.append(current_node)
                    break
                else:
                    total_reward += self.graph.node_reward[current_node]  # only obs node has reward
                    visit_list.append(current_node)
                action_node, _ = self.get_max_child(current_node)  # action node
                visit_list.append(action_node)
                current_node, _ = self.get_max_child(action_node)  # node of next obs where needs to be visit most
                if current_node in visit_list:  # avoid loop
                    current_node = None

            # back propagation and expand UCB1 profiles
            last_node = root
            for node in visit_list:
                if last_node is not root:
                    if last_node in self.MCTS_n:
                        if self.MCTS_n[last_node] == 1:  # if UCB1 profiles just expand to last node
                            break
                self.graph.node_value[node] += total_reward
                if node not in self.MCTS_n:
                    self.MCTS_n[node] = 1
                else:
                    self.MCTS_n[node] += 1
                last_node = node

        # update node value
        for node in self.MCTS_n:
            self.graph.node_value[node] /= self.MCTS_n[node]

    def get_max_child(self, root, value_type="ucb1"):
        max_value = - float("inf")
        child_index = None
        child_id = None
        if root in self.graph.his_edges:
            # trans_prob = self.get_transition_prob(root, self.graph.his_edges[root])
            # trans_prob = trans_prob.squeeze(-1).cpu().detach().numpy().tolist()
            for a in range(len(self.graph.his_edges[root])):
                if value_type == "ucb1":
                    # value = trans_prob[a] * self.get_ucb1(root, self.graph.his_edges[root][a])  # todo
                    value = self.get_ucb1(root, self.graph.his_edges[root][a])
                else:
                    # value = trans_prob[a] * self.get_avg_value(self.graph.his_edges[root][a])
                    value = self.get_avg_value(self.graph.his_edges[root][a])
                if value > max_value:
                    max_value = value
                    child_index = a
                    child_id = self.graph.his_edges[root][a]
                elif value == max_value:  # random selection among save-value nodes
                    if Funcs.rand_prob() > 0.5:
                        max_value = value
                        child_index = a
                        child_id = self.graph.his_edges[root][a]
        return child_id, child_index

    def get_ucb1(self, root, child):
        if child not in self.MCTS_n:  # n == 0
            return float("inf")
        n = self.MCTS_n[child]
        v = self.graph.node_value[child] / n
        if root not in self.MCTS_n:  # N == 0, such as loop structure in the state graph
            N = 1  # make sure ucb1 >= 0
        else:
            N = self.MCTS_n[root]
        ln_N = math.log(N)
        return v + P.ucb1_c * ((ln_N / n) ** 0.5)

    def get_avg_value(self, root):
        return self.graph.node_value[root]  # return the reward form the env

    def update_prob_function(self):
        # data = self.graph.get_data()
        # self.prob_func.train(data)
        pass  # todo

    def update(self):
        while True:
            self.update_prob_function()
            if self.graph.frames > P.total_frames:
                return

    def get_transition_prob(self, src, dsts):
        test_data = Data(np.array([src] * len(dsts)), np.array(dsts), np.array([0] * len(dsts)), np.array([0] * len(dsts)), np.array([0] * len(dsts)))
        return self.prob_func.test(test_data)
