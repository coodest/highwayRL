from src.util.imports.torch import torch
from src.module.context import Profile as P


class Iterator:
    """
    value iterator over highway graph
    """
    
    def __init__(self, id=0) -> None:
        self.id = id
        ind = P.prio_gpu
        if len(P.gpus) > 1:
            ind = self.id % len(P.gpus)
        if torch.cuda.is_available():
            torch.cuda.set_device(int(ind))
            self.device = torch.device(f"cuda:{ind}")
        else:
            self.device = torch.device("cpu")
    
    def iterate(self, np_rew, np_gamma, np_val_0, np_adj):
        with torch.no_grad():
            rew = torch.from_numpy(np_rew).to(self.device)
            gamma = torch.from_numpy(np_gamma).to(self.device)
            val = torch.from_numpy(np_val_0).to(self.device)
            adj = torch.from_numpy(np_adj).to(self.device)

            n_iters = 0
            while n_iters < P.max_vp_iter:
                n_iters += 1
                last_val = val
                divider = 1
                
                max_inherit_value = torch.max((adj - 1) * 1e31 + torch.mul(adj, val) * gamma.view([-1, 1]), dim=1).values
                val = torch.where(max_inherit_value == -1e31, 0.0, max_inherit_value) + rew

                if torch.abs(torch.sum(last_val - val)) < 1e-4:
                    break
            val_n = val.cpu().detach().numpy().tolist()

            # release resorces
            del rew, gamma, val, adj, last_val
            torch.cuda.empty_cache()

        return val_n, n_iters, divider
