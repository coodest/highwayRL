from src.module.context import Profile as P
from src.util.tools import *
from scipy.spatial import distance


class Cluster:
    def __init__(self, coordinates, cluster_id):
        self.coordinates = coordinates
        self.num_nodes = 1
        self.cluster_id = cluster_id


class ObservationIndexer:
    def __init__(self):
        self.ind = 0
        self.id_counter = Counter()
        self.cell_to_cluster = dict()

    def get_index(self, key):
        # 1. find cell
        cell_num = [int(i / P.cell_size) for i in key]
        cell_hash = Funcs.matrix_hashing(cell_num)

        # 2. find cluster(s)
        if cell_hash not in self.cell_to_cluster:
            self.cell_to_cluster[cell_hash] = []  # init a new cell
        clusters = self.cell_to_cluster[cell_hash]

        # 3. find the closest cluster or create one
        min_dis = float("inf")
        min_ind = None
        add_node = False
        for index in range(len(clusters)):
            dis = distance.euclidean(key, clusters[index].coordinates)
            if dis < min_dis:  # smallest index strategy
                min_dis = dis
                min_ind = index
        if min_dis > P.obs_min_dis:  # include len(clusters) == 0
            # not found suitable cluster, create one
            add_node = True
            cluster_id = self.id_counter.get_index()
            clusters.append(Cluster(key, cluster_id))
        else:
            # update suitable cluster
            cluster_id = clusters[min_ind].cluster_id
            temp_coord = clusters[min_ind].coordinates * clusters[min_ind].num_nodes
            clusters[min_ind].num_nodes += 1
            clusters[min_ind].coordinates = (temp_coord + key) / clusters[min_ind].num_nodes

        return cluster_id, add_node
