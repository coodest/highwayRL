from src.module.context import Profile as P
from src.util.tools import *
from multiprocessing import Process, Value, Queue


class MemRL:
    @staticmethod
    def start():
        # 1. init
        Logger.path = P.log_dir + Logger.get_date() + ".log"
        IO.make_dir(P.work_dir)
        IO.make_dir(P.log_dir)
        IO.make_dir(P.model_dir)
        IO.make_dir(P.result_dir)
        if P.clean:
            IO.renew_dir(P.log_dir)
            IO.renew_dir(P.model_dir)
            IO.renew_dir(P.result_dir)

        # 2. show args
        Funcs.print_obj(P)

        # 3. prepare learner and actor
        finish = Value("b", False)
        inference_queue = Queue()
        actor_queues = list()
        for _ in range(P.num_actor):
            actor_queues.append(Queue())

        Process(
            target=MemRL.policy_run,
            args=(inference_queue, actor_queues, finish),
        ).start()
        for id in range(P.num_actor):
            Process(
                target=MemRL.actor_run,
                args=(id, inference_queue, actor_queues[id], finish),
            ).start()

    @staticmethod
    def policy_run(inference_queue, actor_queues, finish):
        # 1. make model-based agent
        from src.module.agent.policy import Policy
        policy = Policy(inference_queue, actor_queues, finish)

        # 2. start inference loop
        policy.inference()  # one inference iteration 
        Logger.log("policy exit.")

    @staticmethod
    def actor_run(id, inference_queue, actor_queue, finish):
        # 1. make env and actor
        from src.module.agent.actor import Actor
        env = MemRL.create_env()
        actor = Actor(id, env, inference_queue, actor_queue)

        # 2. start interaction loop (actor loop)
        while True:
            try:
                actor.interact()  
            except Exception:
                if finish.value:
                    Logger.log(f"actor{id} exit.")
                    return

    @staticmethod
    def create_env():
        if P.env_type == "atari":
            from src.module.env.atari import Atari
            return Atari.make_env()


if __name__ == "__main__":
    MemRL.start()
