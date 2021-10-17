from src.util.imports.torch import torch
from src.util.imports.num import np
from src.module.context import Profile as P
from src.util.tools import Logger


class RandomMatrix(torch.nn.Module):
    def __init__(self, inf, outf):
        super().__init__()
        self.layer = torch.nn.Linear(in_features=inf, out_features=outf)
        torch.nn.init.xavier_uniform_(self.layer.weight)

    def forward(self, input):
        return self.layer(input)


class Projector:
    def __init__(self, id) -> None:
        ind = P.prio_gpu
        if P.num_gpu > 1:
            ind = id % P.num_gpu
        self.device = torch.device(f"cuda:{ind}" if torch.cuda.is_available() else "cpu")

    def project(self, obs):
        pass

    def batch_project(self, infos):
        pass


class RandomProjector(Projector):
    def __init__(self, id) -> None:
        super().__init__(id)
        self.random_matrix = None
        if P.env_type == "atari":
            self.random_matrix = RandomMatrix(84 * 84, P.projected_dim).to(self.device)

    def batch_project(self, obs_list):     
        batch = np.vstack(obs_list)  
        input = None
        if P.env_type == "atari":
            # [2, 84 * 84]
            input = torch.tensor(batch, dtype=torch.float).to(self.device)

        with torch.no_grad():  # no grad calculation
            output = self.random_matrix(input)

        return output.cpu().detach().numpy().tolist()


class CNNProjector(Projector):
    def __init__(self, id) -> None:
        super().__init__(id)

        self.conv1 = torch.nn.Conv2d(1, 1, kernel_size=(6, 6), stride=(5, 5), dilation=(2, 2)).to(self.device)
        self.conv2 = torch.nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(5, 5), dilation=(2, 2)).to(self.device)

    def batch_project(self, obs_list):
        # [2, 84 * 84]
        batch = np.vstack(obs_list)  
        input = torch.tensor(batch, dtype=torch.float, requires_grad=False).to(self.device)
        input = input.unsqueeze(1)
        # [2, 1, 84, 84]
        input = input.reshape(input.shape[0], input.shape[1], 84, -1)

        with torch.no_grad():  # no grad calculation
            # [2, 1, 15, 15]
            output = self.conv1(input)
            # [2, 1, 3, 3]
            output = self.conv2(output)
            # [2, 3, 3]
            output = output.squeeze(1)
            # [2, 9]
            output = torch.flatten(output, start_dim=1)

        return output.cpu().detach().numpy().tolist()  # detach to remove grad_fn
