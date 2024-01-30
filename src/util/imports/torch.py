import torch
from src.module.context import Profile as P

if P.deterministic:
    seed = int(P.run)

    torch.manual_seed(seed)  # cpu种子
    torch.cuda.manual_seed(seed)  # 当前GPU的种子
    torch.cuda.manual_seed_all(seed)  # 所有可用GPU的种子
    torch.backends.cudnn.deterministic = True  # 默认为False
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
