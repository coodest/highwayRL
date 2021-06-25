from src.util.imports.num import np
from src.module.agent.memory.optimal_graph import OptimalGraph
from multiprocessing import Pool, Process, Value, Queue, Lock, Manager

manager = Manager()
center_oa = dict()


def expand_graph(trajectory, total_reward):
    for last_obs, pre_action in trajectory:
        if last_obs not in center_oa:
            center_oa[last_obs] = [pre_action, total_reward]
        elif center_oa[last_obs][1] < total_reward:
            center_oa[last_obs] = [pre_action, total_reward]


t = []
for i in range(int(8000000)):
    t.append([i, i + 1])
r = 100

expand_graph(t, r)

print(len(center_oa.keys()))
