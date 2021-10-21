from src.module.context import Profile as P
from src.util.tools import Funcs


class Indexer:
    @staticmethod
    def get_ind(obs):
        cell_num = [int(i / P.obs_min_dis) for i in obs]
        return Funcs.matrix_hashing(cell_num)

    @staticmethod
    def get_traj_ind(trajectory):
        obs_list = list()
        for t in trajectory:
            obs_list.append(t[0])
        return Funcs.matrix_hashing(obs_list)

    @staticmethod
    def batch_get_ind(obs_list):
        inds = []
        for obs in obs_list:
            inds.append(Indexer.get_ind(obs))
        return inds
