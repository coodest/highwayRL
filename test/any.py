from src.util.imports.num import np
from src.module.agent.memory.optimal_graph import OptimalGraph
from src.module.agent.memory.indexer import Indexer
from multiprocessing import Pool, Process, Value, Queue, Lock, Manager
from src.module.env.atari import Atari
from src.util.tools import *



# from src.module.agent.memory.projector import RandomProjector


# r = RandomProjector(0)
# a = np.random.random([1, 7084])

# b = r.batch_project([a])

# print(b)






#-------------------------------------------------------------------

a = dict()
sample = set()
for _ in range(50):
    env = Atari.make_env(False)
    obs = env.reset()
    for i in range(10):
        obs, reward, done, info = env.step(i)
    # h = Indexer.get_ind(obs)
    h = Funcs.matrix_hashing(obs)
    if h not in a:
        a[h] = 1
    else:
        a[h] += 1

print(a)
print(len(a))

#-------------------------------------------------------------------
