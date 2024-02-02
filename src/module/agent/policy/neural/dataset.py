from src.module.context import Profile as P
from src.util.tools import Logger, Funcs, IO
from src.util.imports.numpy import np
from src.util.imports.torch import torch
from torch.utils.data import Dataset


class OfflineDataset(Dataset):
    def __init__(self, store_path=""):
        self.store_path = f"{P.dataset_dir}{store_path}.pkl"

        self.obss = dict()
        self.Q = dict()

    def save(self):
        IO.write_disk_dump(self.store_path, self.obss)

    def load_all(self, paths):
        for path in paths:
            self.load(path)
        
    def load(self, path):
        o = IO.read_disk_dump(path)
        self.obss.update(o)

    def add(self, obs, proj_obs):
        if P.save_transition:
            self.obss[proj_obs] = obs

    def make(self, Q):
        for proj_obs in self.obss:
            if proj_obs in Q:
                for action in Q[proj_obs]:
                    self.Q[(proj_obs, action)] = Q[proj_obs][action]
        self.keys = list(self.Q.keys())
        self.values = list(self.Q.values())

    def __len__(self):
        return len(self.Q)

    def __getitem__(self, idx):
        proj_obs, action = self.keys[idx]
        action = torch.tensor(np.array([action]), dtype=torch.long)

        value = self.values[idx]
        value = torch.tensor(np.array([value]), dtype=torch.float32)

        if P.dnn == "dqn":
            obs = self.obss[proj_obs]
            obs = torch.tensor(np.array(obs).transpose(2, 1, 0), dtype=torch.float32)
        if P.dnn == "dqn-q":
            obs = proj_obs
            obs = torch.tensor(np.array(obs), dtype=torch.float32)
        
        return obs, action, value
