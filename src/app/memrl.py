from src.module.env.atari import Atari
from src.module.agent.actor import Actor
from src.module.agent.graph import Graph
from src.module.agent.policy import Policy
from src.module.context import Profile as P
from src.util.tools import *


class MemRL:
    @staticmethod
    def start():
        # 0. init
        IO.make_non_exist_dir(P.work_dir)
        IO.make_non_exist_dir(P.log_dir)
        Funcs.print_obj(P)

        # 1. make env
        env = None
        num_action = None
        if P.env_type == "atari":
            env, num_action = Atari.make_env()

        # 2. make model-based agent
        # 2.1 make episodic memory
        graph = Graph(num_action)
        # 2.2 make transition function
        prob_func = None
        # 2.3 make policy
        policy = Policy(graph, prob_func)

        while P.num_episodes > 0:
            P.num_episodes -= 1

            # 3. interact with env using env model
            Actor.interact(env, policy)

            # 4. use interaction data to update env model
            policy.update_prob_function()


if __name__ == "__main__":
    MemRL.start()
