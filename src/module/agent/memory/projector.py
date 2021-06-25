from src.util.imports.torch import torch
from src.util.imports.num import np
from src.module.context import Profile as P
from src.util.tools import Logger


class Projector:
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def project(obs):
        pass

    @staticmethod
    def batch_project(infos):
        pass


class RandomProjector(Projector):
    random_matrix = None
    if P.env_type == "atari":
        random_matrix = torch.nn.Linear(in_features=84 * 84, out_features=P.projected_dim).to(Projector.device)
        torch.nn.init.xavier_uniform_(random_matrix.weight)

    @staticmethod
    def batch_project(obs_list):     
        batch = np.vstack(obs_list)  
        input = None
        if P.env_type == "atari":
            # [2, 84 * 84]
            input = torch.tensor(batch, dtype=torch.float, requires_grad=False).to(Projector.device)
        output = RandomProjector.random_matrix(input)

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
