from src.util.imports.num import np
from src.module.agent.memory.optimal_graph import OptimalGraph
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



print([1,2,3,4,9,4].index(0))


#-------------------------------------------------------------------

# a = dict()
# sample = set()
# for _ in range(50):
#     env = Atari.make_env(False)
#     obs = env.reset()  # random_noops = 15
#     obs = env.reset()  # random_noops = 30
#     obs = env.reset()  # random_noops = 11
#     for i in range(10):
#         obs, reward, done, info = env.step(i)
#     # h = Indexer.get_ind(obs)
#     h = Funcs.matrix_hashing(obs)
#     if h not in a:
#         a[h] = 1
#     else:
#         a[h] += 1

# print(a)
# print(len(a))

#-------------------------------------------------------------------

# o = IO.read_disk_dump(P.model_dir + 'optimal.pkl')
# env = Atari.make_env(False)
# obs = env.reset()
# random_projector = RandomProjector(0)
# obs = random_projector.batch_project([obs])[0]
# obs = Indexer.get_ind(obs)

# print(o[obs])
# print(max([i[1] for i in o.values()]))


#-------------------------------------------------------------------
# a = Queue()
# a.put(1)
# a.put(2)
# a.put(3)
# print(a.qsize())


# print(a.get())
# print(a.get())
# print(a.get())






