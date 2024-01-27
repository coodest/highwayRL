from src.util.tools import Logger, Funcs, IO
from src.module.agent.policy.neural.dataset import OfflineDataset
from src.module.agent.policy.projector import Projector
from src.module.context import Profile as P
from torch.utils.data import DataLoader
from src.util.imports.torch import torch
from src.module.env.atari import Atari
import numpy as np


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


class Parameterizer:
    def __init__(self) -> None:
        self.dnn_model = None

    def utilize_graph_data(self):
        if P.dnn == "dqn":
            train_dataset = OfflineDataset()
            train_dataset.load_all([f"{P.dataset_dir}{i}" for i in IO.list_dir(f"{P.dataset_dir}")])
            graph = IO.read_disk_dump(f"{P.model_dir}graph.pkl")
            train_dataset.make(graph.Q)
            batch_size = 256
            dataloader = DataLoader(train_dataset, batch_size, num_workers=4, pin_memory=True)
            device = torch.device("cuda" if torch.cuda.is_available() and f"cuda:{P.prio_gpu}" else "cpu")
            env = Atari.make_env()
            self.dnn_model = QNetwork(env).to(device)
            learning_rate = 1e-4
            optimizer = torch.optim.Adam(self.dnn_model.parameters(), lr=learning_rate)
            epochs = 200

            for _ in range(epochs):
                for data in dataloader:
                    obs, action, value = data
                    obs = obs.to(device, non_blocking=True)
                    action = action.to(device, non_blocking=True)
                    value = value.to(device, non_blocking=True)
                    pred_value = self.dnn_model(obs).gather(1, action)
                    loss = torch.nn.functional.mse_loss(value, pred_value)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                done = False
                obs = env.reset()
                total_reward = 0
                step = 0
                while not done:
                    step += 1
                    obs = torch.tensor(np.array(obs).transpose(2, 1, 0), dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
                    q_values = self.dnn_model(obs)
                    action = torch.argmax(q_values, dim=1).cpu().numpy()[0]
                    obs, reward, done, info = env.step(action)
                    total_reward += reward

                Logger.log(f"loss: {loss.item():6.2f} R: {total_reward:6.2f} S: {step}")
        if P.dnn == "random_rnn_dqn":
            train_dataset = OfflineDataset()
            train_dataset.load_all([f"{P.dataset_dir}{i}" for i in IO.list_dir(f"{P.dataset_dir}")])
            graph = IO.read_disk_dump(f"{P.model_dir}graph.pkl")
            projector = Projector(0, False)
            train_dataset.make(graph.Q)
            batch_size = 2560
            dataloader = DataLoader(train_dataset, batch_size, num_workers=4, pin_memory=True)
            device = torch.device("cuda" if torch.cuda.is_available() and f"cuda:{P.prio_gpu}" else "cpu")
            env = Atari.make_env()
            self.dnn_model = QNetwork(env, encode=False).to(device)
            learning_rate = 1e-4
            optimizer = torch.optim.Adam(self.dnn_model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 500, 0.8)
            epochs = 1000
            highscore = -float("inf")

            for e in range(epochs):
                for data in dataloader:
                    obs, action, value = data
                    obs = obs.to(device, non_blocking=True)
                    action = action.to(device, non_blocking=True)
                    value = value.to(device, non_blocking=True)
                    pred_value = self.dnn_model(obs).gather(1, action)
                    loss = torch.nn.functional.mse_loss(value, pred_value)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                scheduler.step()

                # eval on current dnn
                done = False
                obs = env.reset()
                total_reward = 0
                expected_return = 0
                step = 0
                projector.reset()
                while not done:
                    with torch.no_grad():
                        _, projected_obs = projector.batch_project([None, None, obs, None, None])
                        projected_obs = torch.tensor(projected_obs, dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
                        q_values = self.dnn_model(projected_obs)
                        action = torch.argmax(q_values, dim=1).cpu().numpy()[0]
                    obs, reward, done, info = env.step(action)
                    total_reward += reward
                    expected_return += reward * np.power(P.gamma, step)
                    step += 1
                Logger.log(f"#{e} lr: {scheduler.get_lr()[0]:.2e} loss: {loss.item():6.2f} R: {total_reward:6.2f} RTN:{expected_return:6.2f} S: {step}")

                if total_reward > highscore:
                    highscore = total_reward
                    self.save_model()

            # eval on current best dnn
            total_rewards = []
            expected_returns = []
            self.load_model()
            for _ in range(10):
                done = False
                obs = env.reset()
                total_reward = 0
                expected_return = 0
                step = 0
                projector.reset()
                while not done:
                    with torch.no_grad():
                        _, projected_obs = projector.batch_project([None, None, obs, None, None])
                        projected_obs = torch.tensor(projected_obs, dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
                        q_values = self.dnn_model(projected_obs)
                        action = torch.argmax(q_values, dim=1).cpu().numpy()[0]
                    obs, reward, done, info = env.step(action)
                    total_reward += reward
                    expected_return += reward * np.power(P.gamma, step)
                    step += 1
                total_rewards.append(total_reward)
                expected_returns.append(expected_return)
            Logger.log(f"AR:{np.average(total_rewards):6.2f} ARTN: {np.average(expected_returns):6.2f}")
                
    def load_model(self):
        self.dnn_model.load_state_dict(torch.load(f"{P.model_dir}dnn_model.pt"))
        Logger.log("best dnn model loaded")

    def save_model(self):
        torch.save(self.dnn_model.state_dict(), f"{P.model_dir}dnn_model.pt")
        Logger.log("better dnn model saved")
