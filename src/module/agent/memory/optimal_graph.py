from src.util.tools import Funcs
from src.module.context import Profile as P
from multiprocessing import Pool, Process, Value, Queue, Lock, Manager


class OptimalGraph:
    manager = Manager()

    oa = manager.dict()
    oa_lock = Lock()

    @staticmethod
    def get_action(obs):
        return OptimalGraph.oa[obs][0]

    @staticmethod
    def expand_graph():
        pass
