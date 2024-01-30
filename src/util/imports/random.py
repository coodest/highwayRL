import random
from src.module.context import Profile as P

if P.deterministic:
    random.seed(int(P.run))
