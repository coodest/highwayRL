from src.module.agent.memory.projector import RandomProjector
import numpy as np


a = np.random.random(size=[3])
print(a)
b = 0
c = np.random.random(size=[3])
print(c)
print(RandomProjector.random_matrix.weight)
print(RandomProjector.random_matrix.bias)
d = RandomProjector.batch_project([[a, b, c]])[0]

print(d[0])
print(d[2])
# breakpoint()

x1 = 0.60597828 * -0.1249 + 0.73336936 * 0.7314 + 0.13894716 * 1.0424
x1 += 0.2493

x2 = 0.60597828 * -0.4702 + 0.73336936 * 0.8090 + 0.13894716 * 0.2913
x2 += 0.5344
print(x2)