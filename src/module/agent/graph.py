from src.util.tools import *
import numpy as np
from src.module.agent.transition.utils.data_processing import Data


class Graph:
    def __init__(self, num_action):
        self.node_hash_to_id = IndexedDict()
        self.edge_counter = Counter()
        self.num_action = num_action

        self.node_feats = []
        self.node_reward = []
        self.edge_feats = []

        self.src = []
        self.dst = []
        self.ts = []
        self.idx = []
        self.label = []

    def get_node_id(self, obs, reward=0, action=None, not_add=False):
        if action:
            node_hash = Funcs.matrix_hashing(obs) + "".join([str(int(i)) for i in Funcs.one_hot(action, self.num_action)])
        else:
            node_hash = Funcs.matrix_hashing(obs) + "".join([str(int(i)) for i in np.zeros(self.num_action)])
        node_id, add_node = self.node_hash_to_id.get_index(node_hash)

        if add_node:  # if it is a new node
            node_feat = np.array([ord(i) for i in node_hash], dtype=np.int8)
            self.node_feats.append(node_feat)
            self.node_reward.append(reward)  # if from node is new, it will be the first node with 0 reward

        if not_add:
            assert add_node is False
        return node_id

    def add_edge(self, from_node_id, to_node_id):
        edge_id = self.edge_counter.get_index()
        action_edge_feat = np.zeros(self.num_action)
        self.edge_feats.append(action_edge_feat)

        self.src.append(from_node_id)
        self.dst.append(to_node_id)
        self.ts.append(edge_id)
        self.idx.append(edge_id)
        self.label.append(0)  # all edges are label 0

    def add(self, last_obs, obs, action, reward):
        # 1. convert src obs into id (index) and features (hashing)
        from_node_id = self.get_node_id(last_obs)
        for a in range(self.num_action):  # add action nodes from src node
            action_node_id = self.get_node_id(last_obs, action=a)
            self.add_edge(from_node_id, action_node_id)

        # 2. convert dst obs into id (index) and features (hashing)
        to_node_id = self.get_node_id(obs, reward=reward)
        # add action nodes from dst node
        for a in range(self.num_action):
            action_node_id = self.get_node_id(obs, action=a)
            self.add_edge(to_node_id, action_node_id)

        # 3. add edge of transition
        current_action_node_id = self.get_node_id(last_obs, action=action, not_add=True)
        self.add_edge(current_action_node_id, to_node_id)

    def get_data(self):
        nodes = np.unique(self.src + self.dst)
        assert np.max(nodes) < len(self.node_feats)
        # node_embedding = np.array([self.node_feats[i] for i in nodes])
        # edge_embedding = np.array([self.edge_feats[i] for i in self.idx])
        node_embedding = np.array(self.node_feats)
        edge_embedding = np.array(self.edge_feats)
        train_data = Data(np.array(self.src), np.array(self.dst), np.array(self.ts), np.array(self.idx), np.array(self.label))

        self.src, self.dst, self.ts, self.idx, self.label = [], [], [], [], []

        return node_embedding, edge_embedding, train_data
