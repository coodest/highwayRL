from src.util.imports.torch import torch
from src.util.imports.numpy import np
from src.module.context import Profile as P
from src.util.tools import Logger, Funcs


class RandomMatrixLayer(torch.nn.Module):
    def __init__(self, inf, outf):
        super().__init__()
        with torch.no_grad():
            self.layer = torch.nn.Linear(in_features=inf, out_features=outf)
            torch.nn.init.xavier_uniform_(self.layer.weight)

    def forward(self, input):
        with torch.no_grad():
            return self.layer(input)


class Projector:
    def __init__(self, id) -> None:
        self.id = id
        ind = P.prio_gpu
        if len(P.gpus) > 1:
            ind = self.id % len(P.gpus)
        if torch.cuda.is_available():
            torch.cuda.set_device(int(ind))
            self.device = torch.device(f"cuda:{ind}")
        else:
            self.device = torch.device("cpu")

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
        if P.env_type == "atari_classic":
            self.random_matrix = RandomMatrixLayer(P.screen_size * P.screen_size, P.projected_dim).to(self.device)
        if P.env_type == "atari_ram":
            self.random_matrix = RandomMatrixLayer(128, P.projected_dim).to(self.device)
        if P.env_type == "atari_alternative":
            self.random_matrix = RandomMatrixLayer(P.screen_size * P.screen_size * P.stack_frames, P.projected_dim).to(self.device)
        if P.env_type == "simple_scene":
            self.random_matrix = RandomMatrixLayer(P.seq_len, P.projected_dim).to(self.device)

    def batch_project(self, obs_list):     
        batch = np.vstack(obs_list)  
        input = None
        with torch.no_grad():
            # [2, P.screen_size * P.screen_size]
            input = torch.tensor(batch, dtype=torch.float, requires_grad=False).to(self.device)

            output = self.random_matrix(input)
            result = output.cpu().detach().numpy().tolist()

        return result


class RNNProjector(Projector):
    def __init__(self, id) -> None:
        super().__init__(id)
        self.random_matrix = None
        with torch.no_grad():
            self.hidden_0 = torch.rand(P.projected_hidden_dim, requires_grad=False).to(self.device)
            self.hidden = None
            self.reset()
        if P.env_type == "atari_classic":
            self.random_matrix = RandomMatrixLayer(P.screen_size * P.screen_size + P.projected_hidden_dim, P.projected_dim + P.projected_hidden_dim).to(self.device)
        if P.env_type == "atari_ram":
            self.random_matrix = RandomMatrixLayer(128 + P.projected_hidden_dim, P.projected_dim + P.projected_hidden_dim).to(self.device)
        if P.env_type == "atari_alternative":
            self.random_matrix = RandomMatrixLayer(P.screen_size * P.screen_size * P.stack_frames + P.projected_hidden_dim, P.projected_dim + P.projected_hidden_dim).to(self.device)
        if P.env_type == "simple_scene":
            self.random_matrix = RandomMatrixLayer(P.seq_len + P.projected_hidden_dim, P.projected_dim + P.projected_hidden_dim).to(self.device)
        self.last_result = np.zeros(P.projected_dim)

    def reset(self):
        if P.env_type == "atari_classic":
            self.hidden = self.hidden_0
        if P.env_type == "atari_ram":
            self.hidden = self.hidden_0
        if P.env_type == "atari_alternative":
            self.hidden = self.hidden_0
        if P.env_type == "simple_scene":
            self.hidden = self.hidden_0

    def project(self, obs):
        batch = obs
        input = None
        with torch.no_grad():  # no grad calculation
            # [P.screen_size * P.screen_size]
            input = torch.tensor(batch, dtype=torch.float, requires_grad=False).to(self.device)
            # [P.screen_size * P.screen_size + hidden_dim]
            input = torch.cat([input, self.hidden], dim=0)
            # [1, P.screen_size * P.screen_size + hidden_dim]
            input = input.unsqueeze(0)

            # [1, projected_dim + hidden_dim]
            output = self.random_matrix(input)
            self.hidden = output[0, P.projected_dim:]
            output = output[0, :P.projected_dim]

            result = output.cpu().detach().numpy().tolist()
        self.last_result = result.copy()

        return result

    def batch_project(self, obs_list):   
        results = []
        results.append(self.last_result)
        results.append(self.project(obs_list[-1]))
        return results


class NRNNProjector(Projector):
    def __init__(self, id, steps=5) -> None:
        super().__init__(id)
        self.steps = steps
        with torch.no_grad():
            self.hidden_share = torch.rand(P.projected_hidden_dim, requires_grad=False).to(self.device)
            self.hidden = None
            self.reset()
        self.random_matrix = None
        if P.env_type == "atari_classic":
            self.unit_weight = torch.rand([
                P.projected_dim + P.projected_hidden_dim,  # number nerons (output dim)
                P.screen_size * P.screen_size * P.stack_frames + P.projected_hidden_dim  # weight of neurons (input dim)
            ], requires_grad=False)
            self.random_matrix = torch.tile(self.unit_weight, (self.steps, 1, 1)).to(self.device)
                
        self.step_status = torch.arange(steps).to(self.device)  # indicate current step
        self.last_result = np.zeros(P.projected_dim)

    def reset(self):
        self.hidden = torch.tile(self.hidden_share, (self.steps, 1))

    def project(self, obs):
        with torch.no_grad():  # no grad calculation
            # [P.screen_size * P.screen_size [* P.stack_frames]]
            raw = torch.tensor(obs, dtype=torch.float32, requires_grad=False).to(self.device)
            # [self.steps, P.screen_size * P.screen_size [* P.stack_frames]]
            input = torch.tile(raw, (self.steps, 1))
            # [self.steps, P.screen_size * P.screen_size [* P.stack_frames] + P.projected_hidden_dimP.projected_hidden_dim]
            input = torch.cat([input, self.hidden], dim=1)
            # [self.steps, 1, P.screen_size * P.screen_size [* P.stack_frames] + P.projected_hidden_dimP.projected_hidden_dim]
            input = input.unsqueeze(dim=1)

            output = torch.mul(self.random_matrix, input) 
            output = torch.sum(output, dim=2)
            self.hidden = output[:, P.projected_dim:]
            output = output[:, :P.projected_dim]

            # find the index of full steps result
            self.step_status += 1
            self.step_status %= self.steps
            output_index = torch.argmin(self.step_status)
            
            # select the result with full steps and reset its hidden
            output = output[output_index]
            self.hidden[output_index] = self.hidden_share

            result = output.cpu().detach().numpy().tolist()
        
        self.last_result = result.copy()
        return result

    def batch_project(self, obs_list):   
        results = []
        results.append(self.last_result)
        results.append(self.project(obs_list[-1]))
        return results


class CNNProjector(Projector):
    def __init__(self, id) -> None:
        super().__init__(id)

        if P.env_type == "atari_classic":
            self.conv1 = torch.nn.Conv2d(1, 1, kernel_size=(6, 6), stride=(5, 5), dilation=(2, 2)).to(self.device)
            self.conv2 = torch.nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(5, 5), dilation=(2, 2)).to(self.device)
        if P.env_type == "atari_ram":
            self.conv1 = torch.nn.Conv2d(1, 1, kernel_size=(6, 6), stride=(5, 5), dilation=(2, 2)).to(self.device)
            self.conv2 = torch.nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(5, 5), dilation=(2, 2)).to(self.device)
        if P.env_type == "atari_alternative":
            self.conv1 = torch.nn.Conv2d(1, 1, kernel_size=(6, 6), stride=(5, 5), dilation=(2, 2)).to(self.device)
            self.conv2 = torch.nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(5, 5), dilation=(2, 2)).to(self.device)
        if P.env_type == "simple_scene":
            self.conv1 = torch.nn.Conv2d(1, 1, kernel_size=(6, 6), stride=(5, 5), dilation=(2, 2)).to(self.device)
            self.conv2 = torch.nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(5, 5), dilation=(2, 2)).to(self.device)

    def batch_project(self, obs_list):
        with torch.no_grad():  # no grad calculation
            # [2, P.screen_size * P.screen_size]
            batch = np.vstack(obs_list)  
            input = torch.tensor(batch, dtype=torch.float, requires_grad=False).to(self.device)
            input = input.unsqueeze(1)
            if P.env_type == "atari_classic":
                # [2, 1, P.screen_size, P.screen_size]
                input = input.reshape(input.shape[0], input.shape[1], P.screen_size, -1)
            if P.env_type == "atari_alternative":
                # [2, 1, P.screen_size, P.screen_size]
                input = input.reshape(input.shape[0], input.shape[1], P.screen_size, -1)

            # [2, 1, 15, 15]
            output = self.conv1(input)
            # [2, 1, 3, 3]
            output = self.conv2(output)
            # [2, 3, 3]
            output = output.squeeze(1)
            # [2, 9]
            output = torch.flatten(output, start_dim=1)
            result = output.cpu().detach().numpy().tolist()  # detach to remove grad_fn

        return result
