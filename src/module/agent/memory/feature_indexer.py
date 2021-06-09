from src.module.context import Profile as P
from src.util.tools import *
from scipy.spatial import distance
from multiprocessing import Pool, Process, Value, Queue, Lock, Manager


class FeatureIndexer:
    lock = Lock()
    manager = Manager()
    cell_to_cluster = manager.dict()

    @staticmethod
    def get_index(key):
        # 1. find cell
        cell_num = [int(i / P.obs_min_dis) for i in key]
        cell_hash = Funcs.matrix_hashing(cell_num)

        # 2. find cluster(s)
        if cell_hash not in FeatureIndexer.cell_to_cluster.keys():
            with FeatureIndexer.lock:
                FeatureIndexer.cell_to_cluster[cell_hash] = len(FeatureIndexer.cell_to_cluster.keys())
        cluster_id = FeatureIndexer.cell_to_cluster[cell_hash]

        return cluster_id
