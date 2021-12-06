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
            adj0 = torch.tensor(np_adj, dtype=torch.float32).to(self.device)
            edge_to_remove = None

            if P.dag_stategy == 0:  # remove edge and continue adj mul
                adj = adj0
                for i in range(2, P.max_vp_iter):
                    adj_new = torch.mm(adj, adj0)
                    diag = torch.diagonal(adj_new)
                    adj_new = torch.where(adj_new > 0, 1.0, 0.0)
                    diag_bool = torch.where(diag > 0.5, 1.0, 0.0)
                    prev_rows_t = diag_bool * torch.transpose(adj, 0, 1)
                    prev_cols = diag_bool * adj0
                    change = prev_rows_t * prev_cols

                    abs_dist = torch.where((adj_new - adj) != 0, 1.0, 0.0)
                    if torch.sum(change) == 0 and torch.sum(abs_dist) == 0:
                        break

                    if edge_to_remove is not None:
                        edge_to_remove += change.cpu().detach().numpy()
                    else:
                        edge_to_remove = change.cpu().detach().numpy()

                    if torch.sum(change) == 0:
                        adj = adj_new
                    else:
                        adj0 = adj0 - change

            if P.dag_stategy == 1:  # remove edge and restart adj mul
                fin = False
                while fin:
                    adj = adj0
                    for i in range(2, P.max_vp_iter):
                        adj_new = torch.mm(adj, adj0)
                        diag = torch.diagonal(adj_new)
                        adj_new = torch.where(adj_new > 0, 1, 0)
                        diag_bool = torch.where(diag > 0.5, 1, 0)
                        prev_rows_t = diag_bool * torch.transpose(adj, 0, 1)
                        prev_cols = diag_bool * adj0
                        change = prev_rows_t * prev_cols


                        abs_dist = torch.where((adj_new - adj) != 0, 1, 0)
                        if torch.sum(change) == 0 and torch.sum(abs_dist) == 0:
                            fin = True
                            break

                        if edge_to_remove is not None:
                            edge_to_remove += change.cpu().detach().numpy()
                        else:
                            edge_to_remove = change.cpu().detach().numpy()

                        if torch.sum(change) == 0:
                            adj = adj_new
                        else:
                            adj0 = adj0 - change
                            break
                
            # release resorces
            del adj0, adj, adj_new, diag, diag_bool, prev_rows_t, prev_cols, change
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
