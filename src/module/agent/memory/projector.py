from src.util.imports.torch import torch
from src.util.imports.numpy import np
from src.module.context import Profile as P
from src.util.tools import Logger, Funcs, IO
import time



class Projector:
    def __init__(self, id, is_head):
        if P.projector == "raw":
            self.projector = RawProjector(id)
        if P.projector == "random":
            self.projector = RandomProjector(id)
        if P.projector == "rnn":
            self.projector = RNNProjector(id, is_head)
        if P.projector == "n-rnn":
            self.projector = NRNNProjector(id)
        if P.projector == "sha256_hash":
            self.projector = HashProjector(id)
        if P.projector == "multiple_hash":
            self.projector = HashProjector(id)

    def batch_project(self, inputs, first_frame=False):
        last_obs, obs = inputs

        if self.projector is not None:
            last_obs, obs = self.projector.batch_project([last_obs, obs])

        if first_frame:
            self.reset()
            obs = last_obs

        return last_obs, obs

    def reset(self):
        if self.projector is not None:
            self.projector.reset()


class RandomMatrixLayer(torch.nn.Module):
    def __init__(self, inf, outf):
        super().__init__()
        with torch.no_grad():
            self.layer = torch.nn.Linear(in_features=inf, out_features=outf)
            torch.nn.init.xavier_uniform_(self.layer.weight)

    def forward(self, input):
        with torch.no_grad():
            return self.layer(input)


class RawProjector:
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

    def project(self, input):
        pass

    def batch_project(self, inputs):
        return tuple(inputs[0]), tuple(inputs[1])

    def reset(self):
        pass


class HashProjector(RawProjector):
    def __init__(self, id) -> None:
        self.hist_input = np.expand_dims(np.array([]), axis=0)
        self.hist_hash = ""

    def project(self, input):
        # input = input.tobytes()
        if P.projector == "sha256_hash":
            return Funcs.matrix_hashing(input)
        if P.projector == "multiple_hash":
            res = ""
            res += Funcs.matrix_hashing(input, type="sha256")
            res += Funcs.matrix_hashing(input, type="md5")
            res += Funcs.matrix_hashing(input, type="shake_256")
            return res

    def batch_project(self, inputs):
        last_obs, obs = inputs
        if len(self.hist_hash) == 0:
            self.hist_input = np.array([last_obs])
            a = self.project(self.hist_input)
            self.hist_input = np.append(self.hist_input, [np.array(obs)], axis=0)
            b = self.project(self.hist_input)
            self.hist_hash = b
            return a, b
        else:
            a = self.hist_hash
            self.hist_input = np.append(self.hist_input, [np.array(obs)], axis=0)
            b = self.project(self.hist_input)
            self.hist_hash = b
            return a, b

    def reset(self):
        self.hist_input = np.expand_dims(np.array([]), axis=0)
        self.hist_hash = ""


class RandomProjector(RawProjector):
    def __init__(self, id):
        super().__init__(id)
        obs_dim = P.screen_size * P.screen_size * P.stack_frames
        self.random_matrix = RandomMatrixLayer(obs_dim, P.projected_dim).to(self.device)

    def batch_project(self, obs_list):     
        batch = np.vstack(obs_list)  
        input = None
        with torch.no_grad():
            # [2, obs_dim]
            input = torch.tensor(batch, dtype=torch.float, requires_grad=False).to(self.device)

            output = self.random_matrix(input)
            results = output.cpu().detach().numpy().tolist()

        return Funcs.matrix_hashing(results[0]), Funcs.matrix_hashing(results[1])


class RNNProjector(RawProjector):
    def __init__(self, id, is_head):
        super().__init__(id)
        projector_path = f"{P.model_dir}{P.env_name}-projector.pkl"
        if is_head:
            obs_dim = P.screen_size * P.screen_size * P.stack_frames
            self.hidden_0 = torch.rand(P.projected_hidden_dim, requires_grad=False)
            self.random_matrix = RandomMatrixLayer(obs_dim + P.projected_hidden_dim, P.projected_dim + P.projected_hidden_dim)
            IO.write_disk_dump(projector_path, [self.hidden_0, self.random_matrix])
        else:
            while True:
                try:
                    self.hidden_0, self.random_matrix = IO.read_disk_dump(projector_path)
                    break
                except Exception:
                    time.sleep(0.1)

        self.random_matrix = self.random_matrix.to(self.device)
        self.hidden_0 = self.hidden_0.to(self.device)
        self.last_result = ""
        self.hidden = None
        self.step = 0
        self.reset()

    def reset(self):
        self.hidden = self.hidden_0
        self.step = 0

    def project(self, obs):
        batch = obs
        input = None
        with torch.no_grad():  # no grad calculation
            # [obs_dim]
            input = torch.tensor(batch, dtype=torch.float, requires_grad=False).to(self.device)
            # [obs_dim + hidden_dim]
            input = torch.cat([input, self.hidden], dim=0)
            # [1, obs_dim + hidden_dim]
            input = input.unsqueeze(0)

            # [1, projected_dim + hidden_dim]
            output = self.random_matrix(input)
            self.hidden = output[0, P.projected_dim:]
            output = output[0, :P.projected_dim]

            result = output.cpu().detach().numpy().tolist()
            result = np.concatenate([result, [self.step]])
            result = Funcs.matrix_hashing(result)
        self.last_result = result
        self.step += 1

        return result

    def batch_project(self, obs_list):   
        results = []
        results.append(self.last_result)
        results.append(self.project(obs_list[-1]))
        return results


class NRNNProjector(RawProjector):
    def __init__(self, id, steps=5):
        super().__init__(id)
        self.steps = steps
        with torch.no_grad():
            self.hidden_share = torch.rand(P.projected_hidden_dim, requires_grad=False).to(self.device)
            self.hidden = None
            self.reset()
        self.random_matrix = None

        obs_dim = P.screen_size * P.screen_size * P.stack_frames
        self.unit_weight = torch.rand([
            P.projected_dim + P.projected_hidden_dim,  # number nerons (output dim)
            obs_dim + P.projected_hidden_dim  # weight of neurons (input dim)
        ], requires_grad=False)
        self.random_matrix = torch.tile(self.unit_weight, (self.steps, 1, 1)).to(self.device)
                
        self.step_status = torch.arange(steps).to(self.device)  # indicate current step
        # self.last_result = np.zeros(P.projected_dim)
        self.last_result = ""

    def reset(self):
        self.hidden = torch.tile(self.hidden_share, (self.steps, 1))

    def project(self, obs):
        with torch.no_grad():  # no grad calculation
            # [obs_dim]
            raw = torch.tensor(obs, dtype=torch.float32, requires_grad=False).to(self.device)
            # [self.steps, obs_dim]
            input = torch.tile(raw, (self.steps, 1))
            # [self.steps, obs_dim + P.projected_hidden_dimP.projected_hidden_dim]
            input = torch.cat([input, self.hidden], dim=1)
            # [self.steps, 1, obs_dim + P.projected_hidden_dimP.projected_hidden_dim]
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
        
            result = Funcs.matrix_hashing(result)
        self.last_result = result
        return result

    def batch_project(self, obs_list):   
        results = []
        results.append(self.last_result)
        results.append(self.project(obs_list[-1]))
        return results
