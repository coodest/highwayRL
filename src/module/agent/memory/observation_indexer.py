from src.module.context import Profile as P
from src.util.tools import *
from scipy.spatial import distance
from multiprocessing import Pool, Process, Value, Queue, Lock, Manager


class ObservationIndexer:
    manager = Manager()
    cell_to_cluster = manager.dict()

    @staticmethod
    def get_index(key):
        # 1. find cell
        cell_num = [int(i / P.cell_size) for i in key]
        cell_hash = Funcs.matrix_hashing(cell_num)

        # 2. find cluster(s)
        if cell_hash not in ObservationIndexer.cell_to_cluster.keys():
            ObservationIndexer.cell_to_cluster[cell_hash] = len(ObservationIndexer.cell_to_cluster.keys())
        cluster_id = ObservationIndexer.cell_to_cluster[cell_hash]

        return cluster_id
