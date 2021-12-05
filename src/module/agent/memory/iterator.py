from src.util.imports.torch import torch
from src.util.imports.numpy import np
from src.module.context import Profile as P
from src.util.tools import Logger


class Iterator:
    def __init__(self, id) -> None:
        self.id = id
        ind = P.prio_gpu
        if len(P.gpus) > 1:
            ind = self.id % len(P.gpus)
        torch.cuda.set_device(int(ind))
        self.device = torch.device(f"cuda:{ind}" if torch.cuda.is_available() else "cpu")

    def iterate(self, np_adj, np_rew, np_val_0):
        with torch.no_grad():
            adj = torch.from_numpy(np_adj).to(self.device)
            rew = torch.from_numpy(np_rew).to(self.device)
            val = torch.from_numpy(np_val_0).to(self.device)

            iters = 0
            while iters < P.max_vp_iter:
                iters += 1
                last_val = val
                val = torch.max(adj * val, dim=1).values * P.gamma + rew
                if torch.sum(last_val - val) == 0:
                    break
            Logger.log(f"iters: {iters}", color="yellow")
            result = val.cpu().detach().numpy().tolist()

            # release resorces
            del adj, rew, val, last_val
            torch.cuda.empty_cache()

        return result
