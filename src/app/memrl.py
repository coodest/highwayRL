from src.module.context import Profile as P
from src.util.tools import *


class MemRL:
    @staticmethod
    def start():
        # 0. init
        Logger.path = P.log_dir + Logger.get_date() + ".log"
        Logger.log("init")
        IO.make_dir(P.work_dir)
        IO.make_dir(P.log_dir)
        IO.make_dir(P.model_dir)
        IO.make_dir(P.result_dir)
        if P.clean:
            IO.renew_dir(P.log_dir)
            IO.renew_dir(P.model_dir)
            IO.renew_dir(P.result_dir)
        # show args
        Funcs.print_obj(P)
        # import
        from src.module.env.atari import Atari
        from src.module.agent.actor import Actor
        from src.module.agent.memory.graph import Graph
        from src.module.agent.policy import Policy
        from src.module.agent.transition.prob_tgn import ProbTGN

        # 1. make env
        Logger.log("make env")
        env = None
        num_action = None
        if P.env_type == "atari":
            env, num_action = Atari.make_env()

        # 2. make model-based agent
        Logger.log("make model-based agent")
        # 2.1 make episodic memory
        graph = Graph(num_action)
        # 2.2 make transition function
        prob_func = ProbTGN()
        # 2.3 make policy
        policy = Policy(graph, prob_func)

        # 3. interact with env using env model, and update policy
        for a in range(P.num_actor):
            actor = Actor(a)
            actor.interact(env, policy)


if __name__ == "__main__":
    MemRL.start()
