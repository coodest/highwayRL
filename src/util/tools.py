import hashlib
import os
import pickle

import numpy as np


class Tools:
    @staticmethod
    def one_hot(indices, max_value):
        if indices >= max_value:
            return Tools.one_hot(max_value - 1, max_value)
        one_hot = np.zeros(max_value)
        one_hot[indices] = 1
        return one_hot

    @staticmethod
    def matrix_hashing(obs):
        """
        hashing function for matrix without collision
        :param obs: the sparse matrix
        :return: the hashed str
        """
        return hashlib.sha256(obs).hexdigest()

    @staticmethod
    def write_disk_dump(path, target_object):
        file_name = str(path).split("/")[-1]
        directory = str(path)[0:(len(path) - len(file_name))]
        Tools.make_non_exist_dir(directory)
        with open(path, "wb") as object_persistent:
            pickle.dump(target_object, object_persistent)

    @staticmethod
    def read_disk_dump(path):
        with open(path, "rb") as object_persistent:
            restore = pickle.load(object_persistent)
        return restore

    @staticmethod
    def make_non_exist_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)