from src.util.imports.torch import torch
from src.util.imports.numpy import np
from src.module.context import Profile as P
from src.util.tools import Logger


class Iterator:
    def __init__(self) -> None:
        ind = P.prio_gpu
        self.device = torch.device(f"cuda:{ind}" if torch.cuda.is_available() else "cpu")

    def iterate(self, np_adj, np_rew, np_val_0):
        adj = torch.from_numpy(np_adj).to(self.device)
        rew = torch.from_numpy(np_rew).to(self.device)
        val = torch.from_numpy(np_val_0).to(self.device)

        iters = 0
        while True:
            iters += 1
            last_val = val
            val = torch.max(adj * val, dim=1).values + rew
            if torch.sum(last_val - val) == 0:
                break
        Logger.log(f"iters: {iters}", color="yellow")
        return val.cpu().detach().numpy().tolist()
