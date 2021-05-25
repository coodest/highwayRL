import torch
from src.module.context import Profile as P
import numpy as np


class Projector:
    def __init__(self):
        self.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

    def project(self, obs):
        pass


class RandomProjector(Projector):
    def __init__(self):
        super().__init__()
        self.random_matrix = None
        if P.env_type == "atari":
            self.random_matrix = torch.nn.Linear(in_features=84 * 84, out_features=P.tgn.memory_dim)

    def project(self, obs):
        input = None
        if P.env_type == "atari":
            input = torch.tensor(obs, dtype=torch.float, requires_grad=False).to(self.device)
            input = input.squeeze(-1)
            input = torch.flatten(input)
            input = input.unsqueeze(0)
        output = self.random_matrix(input)
        output = output.squeeze(0)
        output = output.cpu().detach().numpy()  # detach to remove grad_fn

        return output


class CNNProjector(Projector):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, (3, 3), stride=(2, 2)).to(self.device)
        self.conv2 = torch.nn.Conv2d(1, 1, (3, 3), stride=(2, 2)).to(self.device)
        self.conv3 = torch.nn.Conv2d(1, 1, (3, 3), stride=(2, 2)).to(self.device)

    def project(self, obs):
        input = None
        if P.env_type == "atari":
            input = torch.tensor(obs, dtype=torch.float, requires_grad=False).to(self.device)
            input = input.squeeze(-1)
            input = input.unsqueeze(0).unsqueeze(0)
        input = self.conv1(input)
        input = self.conv2(input)
        output = self.conv3(input)
        output = torch.flatten(output)
        output = output.cpu().detach().numpy()  # detach to remove grad_fn

        return output
