from src.module.context import Profile as P
import numpy as np

if P.deterministic:
    np.random.seed(int(P.run))
