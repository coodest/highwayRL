from src.module.agent.transition.prob_tgn import ProbTGN
from src.util.tools import *


class Policy:
    def __init__(self, graph, prob_func: ProbTGN):
        self.graph = graph
        self.prob_func = prob_func

    def get_action(self, obs):
        pass

    def update_prob_function(self):
        self.prob_func.train(self.graph.get_data())
