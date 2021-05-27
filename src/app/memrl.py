from src.module.context import Profile as P
from src.util.tools import *
from multiprocessing import Pool, Process, Value
import os


class MemRL:
    finish = Value("b", False)

    @staticmethod
    def start():
        # 1. init
        Logger.path = P.log_dir + Logger.get_date() + ".log"
        Logger.log("----- init -----")
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

        # 3. prepare learner
        Process(target=MemRL.learner_run, args=()).start()

        # 4. prepare actor
        with Pool(processes=os.cpu_count()) as pool:
            pool.map(func=MemRL.actor_run, iterable=list(range(P.num_actor)))

        Logger.log("--- finished ---")

    @ staticmethod
    def learner_run():
        # 1. import
        from src.module.agent.policy import Policy
        from src.util.grpc.communication import Server

        # 2. make model-based agent
        learner = Server(P.server_address)
        learner.start()
        policy = Policy()

        # 3. start update loop (learner loop)
        policy.update()

        # 4. set flag if learner finished
        with MemRL.finish.get_lock():
            MemRL.finish.value = True

    @staticmethod
    def actor_run(id):
        # 1. import
        from src.module.agent.actor import Actor

        # 2. make env
        env = MemRL.create_env()

        # 3. start interaction loop (actor loop)
        actor = Actor(id, env)
        while True:
            try:
                actor.interact()
            except Exception:
                if MemRL.finish.value:
                    return

    @staticmethod
    def create_env():
        if P.env_type == "atari":
            from src.module.env.atari import Atari
            return Atari.make_env()


if __name__ == "__main__":
    MemRL.start()
