from src.util.imports.torch import torch
from src.util.imports.numpy import np
from src.module.context import Profile as P
from src.util.tools import Logger, Funcs, IO
import time



class Projector:
    def __init__(self, id, is_head):
        if P.projector == "raw":
            self.projector = RawProjector(id)
        if P.projector == "random_rnn":
            self.projector = RNNProjector(id, is_head)
        if P.projector == "historical_hash":
            self.projector = HashProjector(id)
        if P.projector == "ae":
            self.projector = AEProjector(id, is_head)
        if P.projector == "seq":
            self.projector = SeqProjector(id)
        if P.projector == "linear":
            self.projector = LinearProjector(id, is_head)

    def batch_project(self, transition):
        last_obs, obs = self.projector.batch_project(transition)

        return last_obs, obs

    def reset(self):
        self.projector.reset()


class RandomMatrixLayer(torch.nn.Module):
    def __init__(self, inf, outf):
        super().__init__()
        with torch.no_grad():
            self.layer = torch.nn.Linear(in_features=inf, out_features=outf)
            torch.nn.init.xavier_uniform_(self.layer.weight)
            # should be c7da for atari games
            Logger.log(f"random matrix weights hash: {Funcs.matrix_hashing(self.layer.weight)[-4:]}")

    def forward(self, input):
        with torch.no_grad():
            return self.layer(input)
        

class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.enc1 = torch.nn.Linear(in_features=3136, out_features=784)  # Input image (28*28 = 784)
        self.enc2 = torch.nn.Linear(in_features=784, out_features=256)  # Input image (28*28 = 784)
        self.enc3 = torch.nn.Linear(in_features=256, out_features=128)
        self.enc4 = torch.nn.Linear(in_features=128, out_features=64)
        self.enc5 = torch.nn.Linear(in_features=64, out_features=32)
        self.enc6 = torch.nn.Linear(in_features=32, out_features=16)

        self.dec1 = torch.nn.Linear(in_features=16, out_features=32)
        self.dec2 = torch.nn.Linear(in_features=32, out_features=64)
        self.dec3 = torch.nn.Linear(in_features=64, out_features=128)
        self.dec4 = torch.nn.Linear(in_features=128, out_features=256)
        self.dec5 = torch.nn.Linear(in_features=256, out_features=784)  # Output image (28*28 = 784)
        self.dec6 = torch.nn.Linear(in_features=784, out_features=3136)  # Output image (28*28 = 784)

    def forward(self, x, preprocess=True):
        input = x
        x = torch.nn.functional.relu(self.enc1(x))
        x = torch.nn.functional.relu(self.enc2(x))
        x = torch.nn.functional.relu(self.enc3(x))
        x = torch.nn.functional.relu(self.enc4(x))
        x = torch.nn.functional.relu(self.enc5(x))

        rep = self.enc6(x)
        x = torch.nn.functional.relu(rep)

        x = torch.nn.functional.relu(self.dec1(x))
        x = torch.nn.functional.relu(self.dec2(x))
        x = torch.nn.functional.relu(self.dec3(x))
        x = torch.nn.functional.relu(self.dec4(x))
        x = torch.nn.functional.relu(self.dec5(x))
        x = torch.nn.functional.relu(self.dec6(x))

        return input, rep, x


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
        if isinstance(input, np.ndarray):
            input = np.ndarray.flatten(input).tolist()
        if isinstance(input, int):
            return input
    
        if P.hashing:
            rep = Funcs.matrix_hashing(input)
        else:
            rep = tuple(input)

        return rep

    def batch_project(self, transition):
        last_obs, pre_action, obs, reward, done = transition    
        return self.project(last_obs), self.project(obs)

    def reset(self):
        pass


class LinearProjector(RawProjector):
    def __init__(self, id, is_head, projector_path=f"{P.model_dir}{P.env_name}-projector.pkl"):
        super().__init__(id)
        if is_head:
            obs_dim = 72 * 96 * 4
            self.random_matrix = RandomMatrixLayer(obs_dim, P.projected_dim)
            IO.write_disk_dump(projector_path, self.random_matrix)
        else:
            while True:
                try:
                    self.random_matrix = IO.read_disk_dump(projector_path)
                    break
                except Exception:
                    time.sleep(0.1)

        self.random_matrix = self.random_matrix.to(self.device)
        self.reset()

    def reset(self):
        self.last_result = ""
        self.step = 0

    def project(self, obs, hashing=P.hashing):
        batch = np.ndarray.flatten(obs)
        with torch.no_grad():  # no grad calculation
            # [obs_dim]
            input = torch.tensor(batch, dtype=torch.float, requires_grad=False).to(self.device)
            # [1, obs_dim]
            input = input.unsqueeze(0)

            # [1, projected_dim]
            output = self.random_matrix(input)

            result = output.cpu().detach().numpy().tolist()[0]
            result = tuple(np.concatenate([result, [self.step]]))
            if hashing:
                result = Funcs.matrix_hashing(result)
        self.last_result = result
        self.step += 1

        return result

    def batch_project(self, transition):
        last_obs, pre_action, obs, reward, done = transition   
        results = []
        results.append(self.last_result)
        results.append(self.project(obs))
        return results


class HashProjector(RawProjector):
    def __init__(self, id) -> None:
        self.reset()

    def project(self, input):
        return Funcs.matrix_hashing(input)

    def batch_project(self, transition):
        last_obs, pre_action, obs, reward, done = transition    
        if self.last_result == "":
            self.hist_input = np.array([obs])
        else:
            self.hist_input = np.append(self.hist_input, np.array([obs]), axis=0)
        a = self.last_result
        b = self.project(self.hist_input)
        self.last_result = b
        return a, b

    def reset(self):
        self.hist_input = np.expand_dims(np.array([]), axis=0)
        self.last_result = ""


class RNNProjector(RawProjector):
    def __init__(self, id, is_head, projector_path=f"{P.model_dir}{P.env_name}-projector.pkl"):
        super().__init__(id)
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
        self.reset()

    def reset(self):
        self.last_result = ""
        self.hidden = self.hidden_0
        self.step = 0

    def project(self, obs, hashing=P.hashing):
        batch = np.ndarray.flatten(obs)
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
            result = tuple(np.concatenate([result, [self.step]]))
            if hashing:
                result = Funcs.matrix_hashing(result)
        self.last_result = result
        self.step += 1

        return result

    def batch_project(self, transition):
        last_obs, pre_action, obs, reward, done = transition   
        results = []
        results.append(self.last_result)
        results.append(self.project(obs))
        return results


class AEProjector(RawProjector):
    def __init__(self, id, is_head):
        super().__init__(id)
        self.model = Autoencoder().to(self.device)
        self.model.load_state_dict(torch.load(f"{P.asset_dir}/pretrained_projectors/ae/atari_stargunner.pt"))
        self.last_result = None

    def project(self, obs):
        batch = torch.from_numpy(np.array([obs], dtype=np.float32)).to(self.device)
        self.model.eval()
        input, rep, output = self.model(batch)
        rep = rep.squeeze(0).detach().cpu().numpy()

        self.last_result = tuple(rep)
        return tuple(rep)

    def batch_project(self, transition):
        last_obs, pre_action, obs, reward, done = transition     
        results = []
        results.append(self.last_result)
        results.append(self.project(obs))
        return results


class SeqProjector(RawProjector):
    def __init__(self, id):
        super().__init__(id)
        from src.util.mingpt.model_atari import GPT, GPTConfig

        self.max_timestep = 3884
        self.model = GPT(GPTConfig(
            18, 
            90,
            n_layer=6, 
            n_head=8, 
            n_embd=128, 
            model_type="reward_conditioned", 
            max_timestep=self.max_timestep
        ))
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        self.model.load_state_dict(torch.load(f"{P.asset_dir}pretrained_projectors/transformer/{P.env_type}_{str(P.env_name).lower()}.pt"))
        self.model = self.model.module
        self.model.eval()
        self.reset()

    def reset(self):
        self.actions = None
        self.all_obss = None
        self.last_result = None
        self.rtgs = [10000]
        self.step = -1

    def project(self, pre_action, obs, reward):
        obs = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).to(self.device)

        self.step += 1

        if self.all_obss is None:
            self.all_obss = obs
        else:
            if self.actions is None:
                self.actions = []
            self.actions += [pre_action]
            self.all_obss = torch.cat([self.all_obss, obs], dim=1)
            self.rtgs += [self.rtgs[-1] - reward]

        rep = self.get_rep(
            x=self.all_obss, 
            actions=None if self.actions is None else torch.tensor(self.actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0), 
            rtgs=torch.tensor(self.rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
            timesteps=(min(self.step, self.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device))
        )  
        self.last_result = rep
        return rep

    def get_rep(self, x, actions=None, rtgs=None, timesteps=None, hashing=P.hashing):
        block_size = self.model.get_block_size()
        # x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        x_cond = x if x.size(1) <= block_size // 3 else x[:, -block_size // 3:]  # crop context if needed
        if actions is not None:
            actions = actions if actions.size(1) <= block_size // 3 else actions[:, -block_size // 3:]  # crop context if needed
        rtgs = rtgs if rtgs.size(1) <= block_size // 3 else rtgs[:, -block_size // 3:]  # crop context if needed
        rep, logits, _ = self.model(x_cond, actions=actions, targets=None, rtgs=rtgs, timesteps=timesteps)
        rep = rep[0, -1]
        rep = rep.detach().cpu().numpy()
        if hashing:
            rep = Funcs.matrix_hashing(rep)
        else:
            rep = tuple(rep)

        return rep

    def batch_project(self, transition):
        last_obs, pre_action, obs, reward, done = transition

        results = []
        results.append(self.last_result)
        results.append(self.project(pre_action, obs, reward))
        return results
