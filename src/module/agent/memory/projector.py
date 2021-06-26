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
            input = torch.tensor(batch, dtype=torch.float, requires_grad=False).to(self.device)
        output = self.random_matrix(input)

        return output.cpu().detach().numpy().tolist()


# class CNNProjector(Projector):
#     conv1 = torch.nn.Conv2d(1, 1, (3, 3), stride=(2, 2)).to(Projector.device)
#     conv2 = torch.nn.Conv2d(1, 1, (3, 3), stride=(2, 2)).to(Projector.device)
#     conv3 = torch.nn.Conv2d(1, 1, (3, 3), stride=(2, 2)).to(Projector.device)

#     @staticmethod
#     def project(obs):
#         input = None
#         if P.env_type == "atari":
#             input = torch.tensor(obs, dtype=torch.float, requires_grad=False).to(Projector.device)
#             input = input.squeeze(-1)
#             input = input.unsqueeze(0).unsqueeze(0)
#         input = CNNProjector.conv1(input)
#         input = CNNProjector.conv2(input)
#         output = CNNProjector.conv3(input)
#         output = torch.flatten(output)
#         output = output.cpu().detach().numpy()  # detach to remove grad_fn

#         return output
