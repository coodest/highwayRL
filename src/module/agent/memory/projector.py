import torch
from src.module.context import Profile as P


class Projector:
    def __init__(self):
        self.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
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
