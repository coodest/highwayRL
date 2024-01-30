from src.module.context import Profile as P
from multiprocessing import Process, Value, Manager
from src.util.imports.numpy import np


class Parameterizer:
    def __init__(self) -> None:
        self.highscore = Value("f", -float("inf"))

    def utilize_graph_data(self, slaves=P.num_actor, total_epoch=1000):
        master_slave_queues = list()
        for _ in range(slaves):
            master_slave_queues.append(Manager().Queue())
        slave_master_queues = list()
        for _ in range(slaves):
            slave_master_queues.append(Manager().Queue())
        
        processes = []
        for id in range(slaves):
            p = Process(target=Parameterizer.parameterize, args=(
                id,
                master_slave_queues[id],
                slave_master_queues[id],
                self.highscore, 
                int(total_epoch/slaves),
            ))
            p.start()
            processes.append(p)

        finish = 0
        while True:
            state_dicts = dict()
            for id in range(slaves):
                result = slave_master_queues[id].get()
                if result is None:
                    finish += 1
                    continue
                for key in result:
                    if key not in state_dicts:
                        state_dicts[key] = list()
                    state_dicts[key].append(result[key])
            if finish == slaves:
                break
            new_state_dict = dict()
            for key in state_dicts:
                value = np.array(state_dicts[key], dtype=state_dicts[key][0].dtype)
                avg = np.average(value, axis=0)
                new_state_dict[key] = avg
            for id in range(slaves):
                master_slave_queues[id].put(new_state_dict)

        self.eval_dnn_model()

    def eval_dnn_model(self, evals=10):
        from src.util.tools import Logger, Funcs, IO
        from src.module.agent.policy.projector import Projector
        from src.module.context import Profile as P
        from src.util.imports.torch import torch
        from src.module.env.atari import Atari
        from src.module.agent.policy.neural.q_network import QNetwork
        from src.util.imports.numpy import np

        device = torch.device("cuda" if torch.cuda.is_available() and f"cuda:{P.prio_gpu}" else "cpu")
        env = Atari.make_env()
        if P.dnn == "random_rnn_dqn":
            projector = Projector(0, False)
            dnn_model = QNetwork(env, encode=False).to(device)
        if P.dnn == "dqn":
            dnn_model = QNetwork(env).to(device)
        dnn_model.load_state_dict(torch.load(f"{P.model_dir}dnn_model.pt"))
        Logger.log("best dnn model loaded")

        total_rewards = []
        expected_returns = []
        for _ in range(evals):
            done = False
            obs = env.reset()
            total_reward = 0
            expected_return = 0
            step = 0
            if P.dnn == "random_rnn_dqn":
                projector.reset()
            while not done:
                with torch.no_grad():
                    if P.dnn == "random_rnn_dqn":
                        _, projected_obs = projector.batch_project([None, None, obs, None, None])
                        projected_obs = torch.tensor(projected_obs, dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
                        q_values = dnn_model(projected_obs)
                    if P.dnn == "dqn":
                        obs = torch.tensor(np.array(obs).transpose(2, 1, 0), dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
                        q_values = dnn_model(obs)
                    action = torch.argmax(q_values, dim=1).cpu().numpy()[0]
                obs, reward, done, info = env.step(action)
                total_reward += reward
                expected_return += reward * np.power(P.gamma, step)
                step += 1
            total_rewards.append(total_reward)
            expected_returns.append(expected_return)
        Logger.log(f"{evals} evals AR:{np.average(total_rewards):6.2f} ARTN: {np.average(expected_returns):6.2f}", color="green")

    @staticmethod
    def parameterize(id, master_slave_queue, slave_master_queue, highscore, epochs, batch_size=2560, learning_rate=1e-4):
        from src.util.tools import Logger, Funcs, IO
        from src.module.agent.policy.neural.dataset import OfflineDataset
        from src.module.agent.policy.projector import Projector
        from src.module.context import Profile as P
        from torch.utils.data import DataLoader
        from src.util.imports.torch import torch
        from src.module.agent.policy.neural.q_network import QNetwork
        from src.module.env.atari import Atari
        from src.util.imports.numpy import np
        from src.util.imports.random import random
        import copy

        train_dataset = OfflineDataset()
        train_dataset.load_all([f"{P.dataset_dir}{i}" for i in IO.list_dir(f"{P.dataset_dir}")])
        graph = IO.read_disk_dump(f"{P.model_dir}graph.pkl")
        train_dataset.make(graph.Q)
        if P.deterministic:
            # remove dataloader randomness
            seed = (int(P.run) + 1) * id
            Logger.log(f"slave-{id} seed: {seed}")
            def worker_init_fn(worker_id):
                random.seed(seed + worker_id)
            g = torch.Generator()
            g.manual_seed(seed)
            dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn, generator=g, pin_memory=True)
        else:
            dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
        env = Atari.make_env()
        device = torch.device("cuda" if torch.cuda.is_available() and f"cuda:{P.prio_gpu}" else "cpu")
        if P.dnn == "random_rnn_dqn":
            projector = Projector(0, False)
            dnn_model = QNetwork(env, encode=False).to(device)
        if P.dnn == "dqn":
            dnn_model = QNetwork(env).to(device)
        optimizer = torch.optim.Adam(dnn_model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(epochs/4), 0.8)

        for e in range(epochs):
            for data in dataloader:
                obs, action, value = data
                obs = obs.to(device, non_blocking=True)
                action = action.to(device, non_blocking=True)
                value = value.to(device, non_blocking=True)
                pred_value = dnn_model(obs).gather(1, action)
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
            if P.dnn == "random_rnn_dqn":
                projector.reset()
            while not done:
                with torch.no_grad():
                    if P.dnn == "random_rnn_dqn":
                        _, projected_obs = projector.batch_project([None, None, obs, None, None])
                        projected_obs = torch.tensor(projected_obs, dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
                        q_values = dnn_model(projected_obs)
                    if P.dnn == "dqn":
                        obs = torch.tensor(np.array(obs).transpose(2, 1, 0), dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
                        q_values = dnn_model(obs)
                    action = torch.argmax(q_values, dim=1).cpu().numpy()[0]
                obs, reward, done, info = env.step(action)
                total_reward += reward
                expected_return += reward * np.power(P.gamma, step)
                step += 1

            with highscore.get_lock():
                if total_reward > highscore.value:
                    highscore.value = total_reward
                    torch.save(dnn_model.state_dict(), f"{P.model_dir}dnn_model.pt")

            Logger.log(f"slave-{id}#{e} lr: {scheduler.get_last_lr()[0]:.2e} loss: {loss.item():6.2f} R: {total_reward:6.2f} RTN:{expected_return:6.2f} S: {step} highscore: {highscore.value}")

            new_dict = dict()
            sd = dnn_model.state_dict()
            for key in sd:
                value = sd[key]
                new_dict[key] = copy.deepcopy(value).cpu().detach().numpy()
            slave_master_queue.put(new_dict)

            new_dict = master_slave_queue.get()
            update_dict = dict()
            for key in new_dict:
                update_dict[key] = torch.from_numpy(new_dict[key]).to(device)
            dnn_model.load_state_dict(update_dict)
        slave_master_queue.put(None)
