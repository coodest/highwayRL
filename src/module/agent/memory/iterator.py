from src.util.imports.torch import torch
from src.util.imports.numpy import np
from src.module.context import Profile as P
from src.util.tools import Logger


class Iterator:
    def __init__(self) -> None:
        ind = P.prio_gpu
        self.device = torch.device(f"cuda:{ind}" if torch.cuda.is_available() else "cpu")
        self.adj = None
        self.rew = None
        self.val = None

    def init(self, adj, rew, val_0):
        self.adj = torch.from_numpy(adj).to(self.device)
        self.rew = torch.from_numpy(rew).to(self.device)
        self.val = torch.from_numpy(val_0).to(self.device)

    def iterate(self):
        last_changed_node = None
        max_iter = 1e5
        while True:
            max_iter -= 1
            if max_iter <= 0:
                break
            
            old_val = self.val
            self.val = torch.max(self.adj * self.val, dim=1).values + self.rew
            
            change = self.val - old_val
            if torch.sum(change) == 0:
                break

            changed_node = change > 0
            changed_node = changed_node.long()
            if last_changed_node is not None:
                if torch.sum(torch.abs(last_changed_node - changed_node)) == 0:
                    # contains loop
                    break
            last_changed_node = changed_node
        return self.val.cpu().detach().numpy().tolist()
