from src.util.tools import *
from src.module.context import Profile as P
from src.module.agent.memory.feature_indexer import FeatureIndexer
from multiprocessing import Pool, Process, Value, Queue, Lock, Manager


class Graph:
    manager = Manager()
    lock = Lock()
    node_feats = manager.list()
    node_reward = manager.list()
    node_visit = manager.list()  # for LRU
    node_value = manager.list()
    edge_feats = manager.list()
    node_type = manager.list()

    frames = Value('d', 0)

    src = manager.list()
    dst = manager.list()
    ts = manager.list()
    idx = manager.list()
    label = manager.list()
    his_edges = manager.dict()

    @staticmethod
    def get_node_id(obs, reward=0, action=None, not_add=False):
        if action is not None:
            node_feat = np.concatenate([
                obs,
                100 * P.obs_min_dis * Funcs.one_hot(action, P.num_action)
            ], axis=0)
            node_type = 1
        else:
            node_feat = np.concatenate([
                obs,
                np.zeros(P.num_action)
            ], axis=0)
            node_type = 0
        with Graph.lock:
            node_id = FeatureIndexer.get_index(node_feat)

            if not_add:
                assert node_id < len(Graph.node_feats)
        
            if node_id == len(Graph.node_feats):  # if it is a new node
                Graph.node_feats.append(node_feat)
                Graph.node_reward.append(reward)  # if from node is new, it will be the first node with 0 reward
                Graph.node_visit.append(1)  # init visit state for later tree search
                Graph.node_value.append(0)
                Graph.node_type.append(node_type)
            else:
                Graph.node_visit[node_id] += 1
                Graph.node_reward[node_id] = (Graph.node_reward[node_id] * (Graph.node_visit[node_id] - 1) + reward) / Graph.node_visit[node_id]  # update reward

        return node_id

    @staticmethod
    def add_edge(from_node_id, to_node_id):
        edge_id = len(Graph.src)
        action_edge_feat = np.zeros(P.num_action)
        with Graph.lock:
            Graph.edge_feats.append(action_edge_feat)

            if from_node_id not in Graph.his_edges.keys():
                Graph.his_edges[from_node_id] = [to_node_id]
            elif to_node_id not in Graph.his_edges[from_node_id]:
                Graph.his_edges[from_node_id].append(to_node_id)


            Graph.src.append(from_node_id)
            Graph.dst.append(to_node_id)
            Graph.ts.append(edge_id)
            Graph.idx.append(edge_id)
            Graph.label.append(0)  # all edges are label 0

    @staticmethod
    def add(last_obs, action, obs, reward):
        """
        add transition to current increment
        """
        # 1. store feature of src node and query/return its id
        from_node_id = Graph.get_node_id(last_obs)
        for a in range(P.num_action):  # add action nodes from src node
            action_node_id = Graph.get_node_id(last_obs, action=a)
            Graph.add_edge(from_node_id, action_node_id)

        # 2. store feature of dst node and query/return its id
        to_node_id = Graph.get_node_id(obs, reward=reward)
        # add action nodes from dst node
        for a in range(P.num_action):
            action_node_id = Graph.get_node_id(obs, action=a)
            Graph.add_edge(to_node_id, action_node_id)

        # 3. add edge of the transition from src to dst
        current_action_node_id = Graph.get_node_id(last_obs, action=action, not_add=True)
        Graph.add_edge(current_action_node_id, to_node_id)

        # 4. update frame
        with Graph.lock:
            Graph.frames.value += P.num_action_repeats

    @staticmethod
    def get_data():
        nodes = np.unique(Graph.src + Graph.dst)
        assert np.max(nodes) < len(Graph.node_feats)

        src, dst, ts, idx, label, node_feats, edge_feats = Graph.src, Graph.dst, Graph.ts, Graph.idx, Graph.label, Graph.node_feats, Graph.edge_feats

        # empty current increment
        with Graph.lock:
            Graph.src, Graph.dst, Graph.ts, Graph.idx, Graph.label = Graph.manager.list(), Graph.manager.list(), Graph.manager.list(), Graph.manager.list(), Graph.manager.list()

        return src, dst, ts, idx, label, node_feats, edge_feats
