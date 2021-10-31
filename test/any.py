# from ctypes import sizeof
# from src.util.imports.numpy import np
# from src.module.agent.memory.indexer import Indexer
# from multiprocessing import Pool, Process, Value, Queue, Lock, Manager
# from src.module.env.atari import Atari
# from src.util.tools import *
# from src.module.context import Profile as P
# from src.module.agent.memory.projector import RandomProjector

# import numpy as np
# from sklearn.neighbors import KDTree


# r = RandomProjector(0)
# a = np.random.random([1, 7084])

# b = r.batch_project([a])

# print(b)


# print([1, 2, 3, 4, 9, 4].index(0))





# -------------------------------------------------------------------
# import gym
# from gym import envs

# # print(envs.registry.all())

# gym.make("StarGunner-v0")

# -------------------------------------------------------------------

# from src.module.agent.memory.projector import CNNProjector
# import torch


# a = CNNProjector(0)
# b = torch.rand(2, 84 * 84)
# c = a.batch_project(b)
# print(c)

# -------------------------------------------------------------------

# import sys
# from src.util.tools import Funcs
# import ctypes

# class Storeage(dict):
#     def __init__(self) -> None:
#         self.ggg = None


# class Game:
#     def __init__(self) -> None:
#         pass


# a = dict()
# b = Storeage()
# c = list()
# d = list()


# print(sys.getsizeof(a), "B")
# print(sys.getsizeof(b), "B")
# print(sys.getsizeof(c), "B")
# print(sys.getsizeof(d), "B")

# a = dict()
# s = Funcs.matrix_hashing([1, 2, 3, 4, 5, 6, 7])
# a[s] = [90000, 5698]
# c = id(a[s])
# a[s] = [90000, 6698]

# print(sys.getsizeof(c), "B")
# print(sys.getsizeof(s), "B")

# print(a[s])
# value = ctypes.cast(c, ctypes.py_object).value
# print(value)

# a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# step = 3

# b = a[:step]
# print(b)
# c = [a[step]]
# print(c)
# d = a[step + 1:]
# print(d)

# -------------------------------------------------------------------

from src.module.agent.memory.graph import Graph
from src.module.agent.memory.storage import Storage

a = Graph(0, True)


a.store_inc(
    [
        ["a0", 1, "a1", 5],
        ["a1", 1, "a2", 0],
        ["a2", 1, "a3", 0],
        ["a3", 1, "a4", 0],
        ["a4", 1, "a5", 3],
        ["a5", 1, "a6", 2],
    ], 10
)
a.store_inc(
    [
        ["b0", 2, "b1", 0],
        ["b1", 2, "a2", 0],
        ["a2", 2, "b3", 0],
        ["b3", 2, "b4", 1],
        ["b4", 2, "b5", 0],
    ], 1
)
a.store_inc(
    [
        ["c0", 3, "c1", 0],
        ["c1", 3, "b4", 1],
        ["b4", 3, "c3", 0],
        ["c3", 3, "c4", 0],
        ["c4", 3, "a4", 0],
        ["a4", 3, "c6", 0],
        ["c6", 3, "c7", 1],
    ], 2
)

a.merge_inc(a.inc)

nodes = a.main.node()

for i in nodes:
    ind = i
    obs, action, reward, next, value = nodes[i]
    print(f"node:{ind} obs:{obs} action:{action} reward:{reward} next:{next} value:{value}")








# -------------------------------------------------------------------

# a = Memory()

# a['a'] = [0, 100]
# a['b'] = [0, 200]


# if 'c' not in a:
#     a['c'] = [1, 300]

# print('for')
# for key in a:
#     print(key)

# print(f"a[\'a\'] {a['a']}")

# print(len(a))

# print(a.max_value)
# print(a.max_value_init_obs)
# print(dict(a))
# print(a)




# -------------------------------------------------------------------

# a = dict()
# sample = set()
# env = Atari.make_env(False)
# for _ in range(5000):
#     obs = env.reset()  # random_noops = 15
#     # obs = env.reset()  # random_noops = 30
#     # obs = env.reset()  # random_noops = 11
#     for i in range(1):
#         obs, reward, done, info = env.step(i)
#     # h = Indexer.get_ind(obs)
#     h = Funcs.matrix_hashing(obs)
#     if h not in a:
#         a[h] = 1
#     else:
#         a[h] += 1

# # print(a)
# print(len(a))

# -------------------------------------------------------------------

# o = IO.read_disk_dump(P.optimal_graph_path)
# m = max([i[1] for i in o.values()])
# print(f'max value: {m}')
# print(f'o len: {len(o)}')


# one = set()
# for key in o:
#     if o[key][1] == m:
#         one.add(key)
# print(f'num max point {len(one)}')

# two = set()
# for key in o:
#     if o[key][1] == m:
#         two.add(o[key][2])

# two_out = set()
# two_end = set()
# for e in two:
#     if e in o:
#         if o[e][1] != m:
#             two_out.add(e)
#     else:
#         two_end.add(e)
# print(f'two_out {len(two_out)}')
# print(f'two_end {len(two_end)}')



# xx = []
# for e in ml:
#     if o[e][1] != m:
#         xx.append(e)
# print(f'num ending obs from last max value point: {len(xx)}')

# m2 = set()
# for key in o:
#     if o[key][1] == m:
#         m2.add(key)

# for e in m2:
#     if o[e][1] == m:
#         if e not in o:
#             print(f'pure ending point with max value: {x}')


# env = Atari.make_env(False)
# random_projector = RandomProjector(0)
# obs = env.reset()
# total_reward = 0
# while True:
#     obs = random_projector.batch_project([obs])[0]
#     obs = Indexer.get_ind(obs)
#     a = o[obs]
#     obs, reward, done, info = env.step(0)

#     total_reward += reward

#     if done:
#         print(f'R: {total_reward}')
#         break
# print('fin')



# -------------------------------------------------------------------
# a = Queue()
# a.put(1)
# a.put(2)
# a.put(3)
# print(a.qsize())


# print(a.get())
# print(a.get())
# print(a.get())
# -------------------------------------------------------------------

# a =[0,0,0,1]

# print(np.argmin(a))
