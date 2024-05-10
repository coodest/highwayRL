import torch
from src.module.context import Profile as P

if P.deterministic:
    seed = int(P.run)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # remove dataloader randomness
    # def worker_init_fn(worker_id):
    #     random.seed(SEED + worker_id)
    # g = torch.Generator()
    # g.manual_seed(SEED)
    # DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     worker_init_fn=worker_init_fn
    #     generator=g,
    # )
