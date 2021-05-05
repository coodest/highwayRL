import time
import pandas as pd
import gym
import hashlib
import cv2
import numpy as np
import pickle
from src.util.tools import Tools as T


class IndexedDict:
    def __init__(self):
        self.index = 0
        self.dict = dict()

    def get_index(self, key):
        if key not in self.dict:
            self.dict[key] = self.index
            self.index += 1
        return self.dict[key]


class Counter:
    def __init__(self):
        self.index = -1

    def get_index(self):
        self.index += 1
        return self.index


# env = gym.make('SpaceInvaders-v0')
env = gym.make('Pong-v0')
num_action = 18
obs_dims = env.observation_space
last_action_node_hash = None
indexed_dict = IndexedDict()
edge_counter = Counter()

node_feats = []
edge_feats = []

u = []
i = []
ts = []
idx = []
label = []

for episode in range(20):
    if episode % 10 == 0:
        print(f"current episode: {episode}")
    env.reset()
    for step in range(5000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        # 1. get (grey) image of each frame
        # env.render()
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(f"ls/{episode}-{step}.jpeg", obs)

        # 2. convert frame into state id (index) and state features (hashing)
        node_hash = T.matrix_hashing(obs) + "".join([str(int(i)) for i in np.zeros(num_action)])
        node_id = indexed_dict.get_index(node_hash)
        node_feat = np.array([ord(i) for i in node_hash], dtype=np.int8)
        node_feats.append(node_feat)

        # 3. add edge to current node, find corresponding actions as edge (start state id to end state id),
        # and convert action into edges with time (index) and features (action one-hot)
        if last_action_node_hash is not None:
            edge_feat = np.zeros(num_action)
            edge_feats.append(edge_feat)

            from_id = indexed_dict.get_index(last_action_node_hash)
            to_id = node_id
            edge_id = edge_counter.get_index()
            u.append(from_id)
            i.append(to_id)
            ts.append(edge_id)
            idx.append(edge_id)
            label.append(0)  # all edges are label 0

        # 4. add action nodes from current node
        for a in range(num_action):
            action_node_hash = T.matrix_hashing(obs) + "".join([str(int(i)) for i in T.one_hot(a, num_action)])
            action_node_id = indexed_dict.get_index(action_node_hash)
            action_node_feat = np.array([ord(i) for i in action_node_hash], dtype=np.int8)
            node_feats.append(action_node_feat)
            if a == action:
                last_action_node_hash = action_node_id

            action_edge_feat = np.zeros(num_action)
            edge_feats.append(action_edge_feat)

            from_id = node_id
            to_id = action_node_id
            edge_id = edge_counter.get_index()
            u.append(from_id)
            i.append(to_id)
            ts.append(edge_id)
            idx.append(edge_id)
            label.append(0)  # all edges are label 0

        if done:
            break
    env.close()

graph = pd.DataFrame({
    'u': u,
    'i': i,
    'idx': idx,
    'ts': ts,
    'label': label
})

print(f"nodes: {len(node_feats)}, edges: {len(edge_feats)}")
T.write_disk_dump("data/atari.pkl", [graph, np.array(node_feats), np.array(edge_feats)])
