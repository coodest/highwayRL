from src.util.tools import *
from src.module.context import Profile as P
from multiprocessing import Pool, Process, Value, Queue, Lock, Manager


class Graph:
    manager = Manager()

    node_feats = manager.dict()
    node_feats_lock = Lock()

    node_reward = manager.dict()
    node_reward_lock = Lock()

    node_value = manager.dict()
    node_value_lock = Lock()

    frames = Value('d', 0)

    his_edges = manager.dict()
    his_edges_lock = Lock()

    cell_to_id = manager.dict()
    cell_to_id_lock = Lock()

    @staticmethod
    def get_node_id(obs, reward=0):
        cell_num = [int(i / P.obs_min_dis) for i in obs]
        cell_hash = Funcs.matrix_hashing(cell_num)

        with Graph.cell_to_id_lock:
            if cell_hash not in Graph.cell_to_id.keys():
                Graph.cell_to_id[cell_hash] = len(Graph.cell_to_id.keys())
        node_id = Graph.cell_to_id[cell_hash]

        with Graph.node_feats_lock:
            if node_id not in Graph.node_feats.keys():
                Graph.node_feats[node_id] = obs

        with Graph.node_reward_lock:
            if node_id not in Graph.node_reward.keys():
                Graph.node_reward[node_id] = reward 
        
        with Graph.node_value_lock:
            if node_id not in Graph.node_value.keys():
                Graph.node_value[node_id] = reward

        with Graph.his_edges_lock:
            if node_id not in Graph.his_edges.keys():
                Graph.his_edges[node_id] = Graph.manager.dict()
                for a in range(P.num_action):
                    Graph.his_edges[node_id][a] = Graph.manager.list()

        return node_id

    @staticmethod
    def add(last_obs, action, obs, reward):
        """
        add transition to current increment
        """
        # 1. store feature of src node and query/return its id
        from_node_id = Graph.get_node_id(last_obs)
        to_node_id = Graph.get_node_id(obs, reward=reward)

        assert len(Graph.his_edges[from_node_id]) == P.num_action

        with Graph.his_edges_lock:
            if to_node_id not in Graph.his_edges[from_node_id][action]:
                Graph.his_edges[from_node_id][action].append(to_node_id)

        with Graph.frames.get_lock():
            Graph.frames.value += P.num_action_repeats
