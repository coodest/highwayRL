from src.module.context import Profile as P
from src.util.tools import Logger, Funcs, IO
from multiprocessing import Process, Value, Queue


class MemRL:
    @staticmethod
    def start():
        # 1. init
        Logger.path = P.log_dir + Logger.get_date() + ".log"
        IO.make_dir(P.out_dir)
        IO.make_dir(P.log_dir)
        IO.make_dir(P.model_dir)
        IO.make_dir(P.result_dir)
        IO.make_dir(P.video_dir)
        if P.clean:
            IO.renew_dir(P.log_dir)
            IO.renew_dir(P.model_dir)
            IO.renew_dir(P.result_dir)
            IO.renew_dir(P.video_dir)

        # 2. show args
        Funcs.print_obj(P)

        # 3. prepare queues for learner and actor
        finish = Value("b", False)
        actor_learner_queues = list()
        for _ in range(P.num_actor):
            actor_learner_queues.append(Queue())
        learner_actor_queues = list()
        for _ in range(P.num_actor):
            learner_actor_queues.append(Queue())

        # 4. launch
        Process(target=MemRL.learner_run, args=(
            actor_learner_queues,
            learner_actor_queues,
            finish
        )).start()
        for id in range(P.num_actor):
            Process(target=MemRL.actor_run, args=(
                id,
                actor_learner_queues[id],
                learner_actor_queues[id],
                finish
            )).start()

    @staticmethod
    def learner_run(actor_learner_queues, learner_actor_queues, finish):
        # 1. make model-based agent
        from src.module.agent.policy import Policy

        # 2. train
        policy = Policy(actor_learner_queues, learner_actor_queues)
        try:
            optimal_data = policy.train()  # tain the policy
        except Exception:
            Funcs.trace_exception()
        Logger.log("training finished")
        finish.value = True

        # 3. parameterization
        try:
            # TODO: Q-table parameterization
            Logger.log(len(optimal_data))  # convert policy into dnn
        except Exception:
            Funcs.trace_exception()
        Logger.log("dnn model saved")
        
        Logger.log("learner exit")
        
    @staticmethod
    def actor_run(id, actor_learner_queues, learner_actor_queues, finish):
        # 1. make env and actor
        from src.module.env.actor import Actor

        actor = Actor(id, MemRL.create_env, actor_learner_queues, learner_actor_queues)

        # 2. start interaction loop (actor loop)
        while True:
            try:
                actor.interact()
            except Exception:
                if finish.value:
                    Logger.log(f"actor{id} exit")
                    return
                else:
                    Funcs.trace_exception()

    @staticmethod
    def create_env(render=False):
        if P.env_type == "atari":
            from src.module.env.atari import Atari
            return Atari.make_env(render)


if __name__ == "__main__":
    MemRL.start()
