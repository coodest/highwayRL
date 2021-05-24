from src.util.tools import *
from src.module.context import Profile as P
import numpy as np
from src.module.agent.transition.utils.data_processing import Data
from src.module.agent.memory.projector import Projector
from src.module.agent.memory.observation_indexer import ObservationIndexer


class Graph:
    def __init__(self, num_action):
        self.projector = Projector()
        self.frames = 0

        self.node_feat_to_id = ObservationIndexer()
        self.edge_counter = Counter()
        self.num_action = num_action

        self.node_feats = []
        self.node_reward = []
        self.node_visit = []  # for LRU
        self.node_value = []
        self.edge_feats = []

        self.src = []
        self.dst = []
        self.ts = []
        self.idx = []
        self.label = []
        self.his_edges = dict()

    def project(self, obs):
        return self.projector.project(obs)

    def get_node_id(self, obs, reward=0, action=None, not_add=False):
        if action:
            node_feat = np.concatenate([
                self.project(obs),
                100 * P.obs_min_dis * Funcs.one_hot(action, self.num_action)
            ], axis=0)
        else:
            node_feat = np.concatenate([
                self.project(obs),
                np.zeros(self.num_action)
            ], axis=0)
        node_id, add_node = self.node_feat_to_id.get_index(node_feat)

        if add_node:  # if it is a new node
            self.node_feats.append(node_feat)
            self.node_reward.append(reward)  # if from node is new, it will be the first node with 0 reward
            self.node_visit.append(1)  # init visit state for later tree search
            self.node_value.append(0)
        else:
            self.node_visit[node_id] += 1
            self.node_reward[node_id] = (self.node_reward[node_id] * (self.node_visit[node_id] - 1) + reward) / self.node_visit[node_id]  # update reward

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
        """
        add transition to current increment
        """
        # 1. store feature of src node and query/return its id
        from_node_id = self.get_node_id(last_obs)
        for a in range(self.num_action):  # add action nodes from src node
            action_node_id = self.get_node_id(last_obs, action=a)
            self.add_edge(from_node_id, action_node_id)

        # 2. store feature of dst node and query/return its id
        to_node_id = self.get_node_id(obs, reward=reward)
        # add action nodes from dst node
        for a in range(self.num_action):
            action_node_id = self.get_node_id(obs, action=a)
            self.add_edge(to_node_id, action_node_id)

        # 3. add edge of the transition from src to dst
        current_action_node_id = self.get_node_id(last_obs, action=action, not_add=True)
        self.add_edge(current_action_node_id, to_node_id)

        # 4. update frame
        self.frames += 2

        # 5. check for update policy
        update = False
        if len(self.idx) >= P.tgn.bs:
            update = True

        return update

    def get_data(self):
        nodes = np.unique(self.src + self.dst)
        assert np.max(nodes) < len(self.node_feats)

        # get training data
        node_embedding = np.array(self.node_feats)
        edge_embedding = np.array(self.edge_feats)
        train_data = Data(np.array(self.src), np.array(self.dst), np.array(self.ts), np.array(self.idx), np.array(self.label))

        # put edge to history as permanent graph memory
        for s, d in zip(self.src, self.dst):
            if s not in self.his_edges:
                self.his_edges[s] = [d]
            elif d not in self.his_edges[s]:
                self.his_edges[s].append(d)

        # empty current increment
        Logger.log(f"nodes: {len(node_embedding)}, edges: {len(edge_embedding)}")
        self.src, self.dst, self.ts, self.idx, self.label = [], [], [], [], []

        return node_embedding, edge_embedding, train_data
