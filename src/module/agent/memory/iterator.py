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

    def build_dag(self, np_adj):
        with torch.no_grad():
            adj0 = torch.from_numpy(np_adj).to(self.device)
            edge_to_remove = None
            adj = adj0

            for i in range(2, P.max_vp_iter):
                print(i)
                adj_new = torch.mm(adj, adj0)
                diag = torch.diagonal(adj_new)
                prev_rows = diag * torch.transpose(adj, 0, 1)
                prev_cols = diag * adj0
                change = prev_rows * prev_cols
                if edge_to_remove is not None:
                    if torch.sum(change) == 0 and torch.sum(adj_new - adj) == 0:
                        break
                    edge_to_remove += change.cpu().detach().numpy()
                else:
                    edge_to_remove = change.cpu().detach().numpy()
                adj = adj_new
                adj0 = adj0 - change
                
            # release resorces
            del adj, adj_new, diag, prev_rows, prev_cols
            torch.cuda.empty_cache()

        return edge_to_remove

    
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
