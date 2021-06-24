from src.util.tools import Funcs
from src.module.context import Profile as P
from multiprocessing import Pool, Process, Value, Queue, Lock, Manager


class OptimalGraph:
    manager = Manager()

    # dict of observation to action with value [action, reward]
    oa = manager.dict()
    oa_lock = Lock()

    @staticmethod
    def get_action(obs):
        if obs in OptimalGraph.oa.keys():
            action = OptimalGraph.oa[obs][0]
        else:
            action = None
        return action

    @staticmethod
    def expand_graph(trajectory):
        pass
