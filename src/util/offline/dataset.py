from src.module.context import Profile as P
from src.util.tools import Logger, Funcs, IO
from src.util.imports.numpy import np
from src.util.imports.torch import torch
from torch.utils.data import Dataset


class OfflineDataset(Dataset):

    def __init__(self, store_path="", block_size=30 * 3):
        self.store_path = f"{P.dataset_dir}{store_path}.pkl"

        self.obss = list()
        self.actions = list()
        self.rewards = list()
        self.dones = list()
        self.next_reward = 0.0
        self.last_done_idx = 0

        self.done_idxs = list()
        self.rtgs = list()
        self.timesteps = list()

        self.block_size = block_size
        self.vocab_size = None

    def save(self):
        IO.write_disk_dump(self.store_path, [
            self.obss[:self.last_done_idx], 
            self.actions[:self.last_done_idx], 
            self.rewards[:self.last_done_idx], 
            self.dones[:self.last_done_idx], 
        ])

    def load_all(self, paths):
        for path in paths:
            self.load(path)
        
    def load(self, path):
        o, a, r, d = IO.read_disk_dump(path)
        self.obss += o
        self.actions += a
        self.rewards += r
        self.dones += d

    def add(self, obs, action, reward, done):
        if P.env_type == "maze":
            self.obss.append(obs)
        if P.env_type == "toy_text":
            self.obss.append(obs)
        if P.env_type == "football":
            self.obss.append(obs)
        if P.env_type == "atari":
            self.obss.append(np.array(obs, dtype=np.uint8).transpose(2, 1, 0))
        self.actions.append(action)
        self.rewards.append(self.next_reward)
        if done:
            self.next_reward = 0.0
            self.last_done_idx = len(self.actions)
        else:
            self.next_reward = reward
        self.dones.append(done)

    def make(self, gamma=1.0):
        self.vocab_size = max(self.actions) + 1

        self.actions = np.array(self.actions, dtype=np.int32)

        self.rtgs = np.zeros(self.actions.shape, dtype=np.float32)
        index = len(self.rewards) - 1
        for reward, done in zip(self.rewards[::-1], self.dones[::-1]):
            if done:
                self.rtgs[index] = reward
            else:
                self.rtgs[index] = self.rtgs[index + 1] * gamma + reward / 100.0
            index -= 1

        offset = 0
        for idx, done in enumerate(self.dones):  
            self.timesteps.append(idx - offset)
            if done:
                self.done_idxs.append(idx + 1)
                offset = idx + 1
        self.done_idxs = np.array(self.done_idxs, dtype=np.int64)
        self.timesteps = np.array(self.timesteps, dtype=np.int64)

    def get_max_timestep(self):
        return max(self.timesteps)
    
    def __len__(self):
        return len(self.obss) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        end_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx:  # first done_idx greater than idx
                end_idx = min(int(i), end_idx)
                break
        idx = end_idx - block_size
        states = torch.tensor(np.array(self.obss[idx:end_idx]), dtype=torch.float32).reshape(block_size, -1)  # (block_size, 4*84*84)
        states = states / 255.
        actions = torch.tensor(self.actions[idx:end_idx], dtype=torch.long).unsqueeze(1)  # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:end_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx + 1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps
