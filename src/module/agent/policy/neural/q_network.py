from src.util.imports.torch import torch
from src.module.context import Profile as P
from src.util.tools import IO, Logger


class QNetwork(torch.nn.Module):
    def __init__(self, env, encode=True):
        super().__init__()
        self.encode = encode
        if self.encode:
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(4, 32, 8, stride=4),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(32, 64, 4, stride=2),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(64, 64, 3, stride=1),
                torch.nn.LeakyReLU(),
                torch.nn.Flatten(),
            )

        if P.env_type == "maze":
            input_dim = 2
            self.Q = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 512),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(512, 512),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(512, env.action_space.n),
            )
        if P.env_type == "toy_text":
            if P.env_name == "Blackjack-v1":
                input_dim = 3
            else:
                input_dim = 1
            self.Q = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 512),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(512, 512),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(512, env.action_space.n),
            )
        if P.env_type == "football":
            input_dim = P.projected_dim + 1
            self.Q = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 512),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(512, 512),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(512, env.action_space.n),
            )
        if P.env_type == "atari":
            if self.encode:
                input_dim = 3136
            else:
                input_dim = P.projected_dim + 1  # + 1 is for step
        
            self.Q = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 512),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(512, 512),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(512, env.action_space.n),
            )

    def forward(self, x):
        if self.encode:
            x /= 255.0
            x = self.encoder(x)
        return self.Q(x)