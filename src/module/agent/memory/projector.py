from src.util.torch_util import *
from src.module.context import Profile as P


class Projector:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def project(obs):
        pass

    @staticmethod
    def batch_project(infos):
        pass


class RandomProjector(Projector):
    random_matrix = None
    if P.env_type == "atari":
        random_matrix = torch.nn.Linear(in_features=84 * 84, out_features=P.tgn.memory_dim).to(Projector.device)
        torch.nn.init.xavier_uniform_(random_matrix.weight)


    @staticmethod
    def project(obs):
        input = None
        if P.env_type == "atari":
            input = torch.tensor(obs, dtype=torch.float, requires_grad=False).to(Projector.device)
            input = input.unsqueeze(0)
        output = RandomProjector.random_matrix(input)
        output = output.squeeze(0)
        output = output.cpu().detach().numpy()  # detach to remove grad_fn

        return output

    @staticmethod
    def batch_project(infos):
        # info is [actor_id, last_obs, pre_action, obs, reward, add] 
        batch_last_obs = [x[1] for x in infos]
        batch_obs = [x[3] for x in infos]
        batch = np.concatenate([batch_last_obs, batch_obs], axis=0)
        
        input = None
        if P.env_type == "atari":
            input = torch.tensor(batch, dtype=torch.float, requires_grad=False).to(Projector.device)
        output = RandomProjector.random_matrix(input)
        output = output.cpu().detach().numpy().tolist()

        for id in range(len(infos)):
            infos[id][1] = output[id]
            infos[id][3] = output[len(infos) + id]

        print(infos[0])
        exit(0)

        return infos

        


class CNNProjector(Projector):
    conv1 = torch.nn.Conv2d(1, 1, (3, 3), stride=(2, 2)).to(Projector.device)
    conv2 = torch.nn.Conv2d(1, 1, (3, 3), stride=(2, 2)).to(Projector.device)
    conv3 = torch.nn.Conv2d(1, 1, (3, 3), stride=(2, 2)).to(Projector.device)

    @staticmethod
    def project(obs):
        input = None
        if P.env_type == "atari":
            input = torch.tensor(obs, dtype=torch.float, requires_grad=False).to(Projector.device)
            input = input.squeeze(-1)
            input = input.unsqueeze(0).unsqueeze(0)
        input = CNNProjector.conv1(input)
        input = CNNProjector.conv2(input)
        output = CNNProjector.conv3(input)
        output = torch.flatten(output)
        output = output.cpu().detach().numpy()  # detach to remove grad_fn

        return output
