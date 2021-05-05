from src.util.tools import IndexedDict, Counter
from src.util.tools import Funcs as F
import numpy as np


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

        self.last_action_node_hash = None

    def add_node(self, obs, action, reward):
        # 1. convert frame into state id (index) and state features (hashing)
        node_hash = F.matrix_hashing(obs) + "".join([str(int(i)) for i in np.zeros(self.num_action)])
        node_id = self.node_hash_to_id.get_index(node_hash)
        node_feat = np.array([ord(i) for i in node_hash], dtype=np.int8)
        self.node_feats.append(node_feat)
        self.node_reward.append(reward)

        # 2. add edge to current node, find corresponding actions as edge (start state id to end state id),
        # and convert action into edges with time (index) and features (action one-hot)
        if self.last_action_node_hash is not None:
            edge_feat = np.zeros(self.num_action)
            self.edge_feats.append(edge_feat)

            from_id = self.node_hash_to_id.get_index(self.last_action_node_hash)
            to_id = node_id
            edge_id = self.edge_counter.get_index()
            self.src.append(from_id)
            self.dst.append(to_id)
            self.ts.append(edge_id)
            self.idx.append(edge_id)
            self.label.append(0)  # all edges are label 0

        # 3. add action nodes from current node
        for a in range(self.num_action):
            action_node_hash = F.matrix_hashing(obs) + "".join([str(int(i)) for i in F.one_hot(a, self.num_action)])
            action_node_id = self.node_hash_to_id.get_index(action_node_hash)
            action_node_feat = np.array([ord(i) for i in action_node_hash], dtype=np.int8)
            self.node_feats.append(action_node_feat)
            self.node_reward.append(0)  # action node has no reward
            if a == action:
                self.last_action_node_hash = action_node_id

            action_edge_feat = np.zeros(self.num_action)
            self.edge_feats.append(action_edge_feat)

            from_id = node_id
            to_id = action_node_id
            edge_id = self.edge_counter.get_index()
            self.src.append(from_id)
            self.dst.append(to_id)
            self.ts.append(edge_id)
            self.idx.append(edge_id)
            self.label.append(0)  # all edges are label 0
