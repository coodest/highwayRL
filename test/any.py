from ctypes import sizeof
from src.util.imports.num import np
from src.module.agent.memory.optimal_graph import OptimalGraph, Memory
from src.module.agent.memory.indexer import Indexer
from multiprocessing import Pool, Process, Value, Queue, Lock, Manager
from src.module.env.atari import Atari
from src.util.tools import *
from src.module.context import Profile as P
from src.module.agent.memory.projector import RandomProjector


# r = RandomProjector(0)
# a = np.random.random([1, 7084])

# b = r.batch_project([a])

# print(b)


# print([1, 2, 3, 4, 9, 4].index(0))





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

o = IO.read_disk_dump(P.model_dir + 'optimal.pkl')
m = max([i[1] for i in o.values()])
print(f'max value: {m}')
print(f'm len: {len(o)}')

print("starting point with max value")
ml = set()
for key in o:
    if o[key][1] == m:
        ml.add(o[key][2])
xx = []
for key in o:
    if o[key][1] == m:
        if key not in ml:
            xx.append(key)
print(xx)
print(len(xx))

print("end point with max value")
ml = set()
for key in o:
    if o[key][1] == m:
        ml.add(o[key][2])
xx = []
for e in ml:
    if o[e][1] != m:
        print(o[e][1])
        xx.append(e)
print(xx)
print(len(xx))


env = Atari.make_env(False)
random_projector = RandomProjector(0)
obs = env.reset()
# for _ in range(500):
obs = random_projector.batch_project([obs])[0]
obs = Indexer.get_ind(obs)
print(obs)
#     if obs in ml:
#         print(1)
#     obs, reward, done, info = env.step(0)

#     if done:
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
