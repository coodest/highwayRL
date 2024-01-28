from src.util.imports.torch import torch
from src.module.context import Profile as P


class QNetwork(torch.nn.Module):
    def __init__(self, env, encode=True):
        super().__init__()
        self.encode = encode
        input_dim = P.projected_dim + 1  # + 1 is for step
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
            input_dim = 3136
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