from src.util.tools import Logger, Funcs, IO
from src.util.offline.dataset import OfflineDataset
from src.util.offline.model_atari import GPT, GPTConfig
from src.util.offline.trainer_atari import Trainer, TrainerConfig
from src.module.context import Profile as P



class Model:
    def __init__(self) -> None:
        pass

    def utilize_graph_data(self, context_length=30):
        train_dataset = OfflineDataset(block_size=context_length * 3)
        train_dataset.load_all([f"{P.dataset_dir}{i}" for i in IO.list_dir(f"{P.dataset_dir}")])
        # train_dataset.make(P.gamma)
        train_dataset.make()

        Logger.log("dataset loaded")
        # breakpoint()

        mconf = GPTConfig(
            train_dataset.vocab_size, 
            train_dataset.block_size,
            n_layer=6, 
            n_head=8, 
            n_embd=128, 
            model_type="reward_conditioned", 
            max_timestep=train_dataset.get_max_timestep()
        )
        model = GPT(mconf)  

        # initialize a trainer instance and kick off training
        tconf = TrainerConfig(
            max_epochs=50, 
            batch_size=512, 
            learning_rate=6e-4,
            lr_decay=True, 
            warmup_tokens=512 * 20, 
            final_tokens=2 * len(train_dataset) * context_length * 3,
            num_workers=4, 
            seed=123, 
            model_type="reward_conditioned", 
            game=P.env_name, 
            max_timestep=train_dataset.get_max_timestep(), 
            load_model=P.load_model
        )
        trainer = Trainer(model, train_dataset, None, tconf)

        trainer.train(estimated_reward=2000)

    def save(self):
        Logger.log("dnn model saved")
