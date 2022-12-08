from src.util.imports.torch import torch
from src.util.imports.numpy import np
from src.module.context import Profile as P
from src.util.tools import Logger


class Iterator:
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

    def build_dag(self, np_adj):
        with torch.no_grad():
            adj0 = torch.tensor(np_adj, dtype=torch.float32).to(self.device)
            edge_to_remove = None

            fin = False
            while not fin:
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
                        # nodes in adj can go no where, and no loop currently found
                        # that is: adj_new = adj = a zero mat
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
                        if P.start_over:
                            break  # start over of adj mat mul
                
            # release resorces
            del adj0, adj, adj_new, diag, diag_bool, prev_rows_t, prev_cols, change
            torch.cuda.empty_cache()

        return edge_to_remove

    
    def iterate(self, np_adj, np_rew, np_gamma, np_val_0):
        with torch.no_grad():
            # debug: limit the VRam size
            # torch.cuda.set_per_process_memory_fraction(0.5)

            adj = torch.from_numpy(np_adj).to(self.device)
            rew = torch.from_numpy(np_rew).to(self.device)
            gamma = torch.from_numpy(np_gamma).to(self.device)
            val = torch.from_numpy(np_val_0).to(self.device)

            iters = 0
            while iters < P.max_vp_iter:
                iters += 1
                last_val = val

                mul = None
                pro = None
                divider = 1
                while True:
                    try:
                        last_position = 0
                        divided_len = int(len(adj) / divider)
                        mul = torch.tensor([], dtype=val.dtype, device=val.device)
                        while True:
                            pro = torch.max(adj[last_position:last_position + divided_len] * val, dim=1).values
                            mul = torch.concat([mul, pro])
                            last_position += divided_len
                            if last_position + divided_len > len(adj):
                                if last_position < len(adj):
                                    pro = torch.max(adj[last_position:] * val, dim=1).values
                                    mul = torch.concat([mul, pro])
                                break
                        break
                    except RuntimeError:
                        mul = None
                        pro = None
                        divider *= 2

                val = mul * gamma + rew
                if torch.sum(last_val - val) == 0:
                    break
            result = val.cpu().detach().numpy().tolist()

            # release resorces
            del adj, rew, val, last_val, mul
            torch.cuda.empty_cache()

        return result, iters, divider
