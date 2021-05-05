from src.module.agent.tgn.train_self_supervised import *

class Policy:
    def __init__(self, graph, prob_func):
        self.graph = graph
        self.prob_func = prob_func

    def get_action(self, obs):
        pass

    def update_prob_function(self):
        # test
        main_func(self.graph)
