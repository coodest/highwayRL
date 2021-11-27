from src.util.imports.torch import torch
from src.util.imports.numpy import np
from src.module.context import Profile as P
from src.util.tools import Logger, Funcs


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

    def reset(self):
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


class RNNProjector(Projector):
    def __init__(self, id) -> None:
        super().__init__(id)
        self.random_matrix = None
        self.hidden_0 = torch.rand(P.projected_hidden_dim).to(self.device)
        self.hidden = None
        if P.env_type == "atari":
            self.random_matrix = RandomMatrix(84 * 84 + P.projected_hidden_dim, P.projected_dim + P.projected_hidden_dim).to(self.device)
        self.reset()

    def reset(self):
        if P.env_type == "atari":
            self.hidden = self.hidden_0

    def project(self, obs):
        batch = obs
        input = None
        if P.env_type == "atari":
            # [84 * 84]
            input = torch.tensor(batch, dtype=torch.float).to(self.device)
            # [84 * 84 + hidden_dim]
            input = torch.cat([input, self.hidden], dim=0)
            # [1, 84 * 84 + hidden_dim]
            input = input.unsqueeze(0)

        with torch.no_grad():  # no grad calculation
            # [1, projected_dim + hidden_dim]
            output = self.random_matrix(input)
            self.hidden = output[0, P.projected_dim:]
            output = output[0, :P.projected_dim]

        return output.cpu().detach().numpy().tolist()

    def batch_project(self, obs_list):   
        results = []
        for obs in obs_list:
            results.append(self.project(obs))
        return results


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
