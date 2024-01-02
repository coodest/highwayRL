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

                # mul = None
                # pro = None
                divider = 1
                # while True:
                #     try:
                #         last_position = 0
                #         divided_len = int(len(adj) / divider)
                #         mul = torch.tensor([], dtype=val.dtype, device=val.device)
                #         while True:
                #             pro = torch.max(adj[last_position:last_position + divided_len] * val, dim=1).values
                #             mul = torch.concat([mul, pro])
                #             last_position += divided_len
                #             if last_position + divided_len > len(adj):
                #                 if last_position < len(adj):
                #                     pro = torch.max(adj[last_position:] * val, dim=1).values
                #                     mul = torch.concat([mul, pro])
                #                 break
                #         break
                #     except RuntimeError:
                #         mul = None
                #         pro = None
                #         divider *= 2

                # val = mul * gamma + rew
                
                val = torch.max((adj - 1) * 1e31 + torch.mul(adj, val * gamma), dim=1).values + rew

                if torch.sum(last_val - val) == 0:
                    break
            result = val.cpu().detach().numpy().tolist()

            # release resorces
            del adj, rew, val, last_val
            torch.cuda.empty_cache()

        return result, iters, divider
