from src.module.context import Profile as P
from src.util.tools import *
from src.module.agent.transition.prob_tgn import ProbTGN
from src.module.agent.memory.graph import Graph
from multiprocessing import Pool, Process, Value, Queue, Lock, Manager

import math


class Policy:
    graph = Graph()
    prob_func = ProbTGN()
    manager = Manager()
    MCTS_n = manager.list()

    def __init__(self, inference_queue, actor_queues, finish):
        self.inference_queue = inference_queue
        self.actor_queues = actor_queues
        self.finish = finish

    def inference(self):
        last_record = time.time()
        while True:
            actor_ids = list()
            infos = list()
            for _ in range(P.num_actor):
                info = self.inference_queue.get()
                actor_ids.append(info[0])
                infos.append(info[1:])
            with Pool(os.cpu_count()) as pool:
                actions = pool.map(Policy.get_action, infos)
            for actor_id in actor_ids:
                self.actor_queues[actor_id].put(actions[actor_id])
            
            # if time.time() - last_record > 10:
            #     Logger.log("frames: {} {} {}".format(
            #         self.graph.frames, len(self.graph.node_feats),
            #         len(self.graph.edge_feats)
            #     ))
            #     last_record = time.time()
            
            if self.graph.frames > P.total_frames:
                with self.finish.get_lock():
                    self.finish.value = True
                break
            else:
                pass
                # Logger.log("skip train")
                # data = self.graph.get_data()
                # self.prob_func.train(data)

    @staticmethod
    def get_transition_prob(src, dsts):
        info = [src, dsts]
        return Policy.prob_func.test(info)

    @staticmethod
    def get_action(info):
        last_obs, pre_action, obs, reward, add = info
        # 1. add transition to graph memory
        if add:
            Policy.graph.add(last_obs, pre_action, obs, reward)

        # 2. find current/root node
        root = Policy.graph.get_node_id(obs)
        if root not in Policy.graph.his_edges:  # can not give action for previously never interacted obs
            return None

        # 3. use UCB1 formula to propagate value
        Policy.update_children(root)

        # 4. for training actor: select the child with max UCB1 and
        # return corresponding action;
        # for testing actor: select the child with max value and
        # return corresponding action
        child_id, action = Policy.get_max_child(root, value_type="value")

        return action

    @staticmethod
    def update_children(root):
        """
        build or update the UCB1 profile of root and its children
        1. avoid loop error
        2. avoid intersection ucb1 compute error
        """
        for t in range(P.propagations):
            # simulate
            total_reward = 0
            visit_list = []
            current_node = root  # node of root obs
            simulate_steps = P.simulate_steps
            while current_node is not None:
                simulate_steps -= 1
                if simulate_steps <= 0:
                    total_reward += Policy.graph.node_value[current_node]  # only obs node has reward
                    visit_list.append(current_node)
                    break
                else:
                    total_reward += Policy.graph.node_reward[current_node]  # only obs node has reward
                    visit_list.append(current_node)
                action_node, _ = Policy.get_max_child(current_node)  # action node
                visit_list.append(action_node)
                current_node, _ = Policy.get_max_child(action_node)  # node of next obs where needs to be visit most
                if current_node in visit_list:  # avoid loop
                    current_node = None

            # back propagation and expand UCB1 profiles
            last_node = root
            for node in visit_list:
                if last_node is not root:
                    if last_node in Policy.MCTS_n:
                        if Policy.MCTS_n[last_node] == 1:  # if UCB1 profiles just expand to last node
                            break
                Policy.graph.node_value[node] += total_reward
                if node not in Policy.MCTS_n:
                    Policy.MCTS_n[node] = 1
                else:
                    Policy.MCTS_n[node] += 1
                last_node = node

        # update node value
        for node in Policy.MCTS_n:
            Policy.graph.node_value[node] /= Policy.MCTS_n[node]

    @staticmethod
    def get_max_child(root, value_type="ucb1"):
        max_value = - float("inf")
        child_index = None
        child_id = None
        if root in Policy.graph.his_edges:
            # trans_prob = Policy.get_transition_prob(root, Policy.graph.his_edges[root])
            # trans_prob = trans_prob.squeeze(-1).cpu().detach().numpy().tolist()
            for a in range(len(Policy.graph.his_edges[root])):
                if value_type == "ucb1":
                    # value = trans_prob[a] * Policy.get_ucb1(root, Policy.graph.his_edges[root][a])  # todo
                    value = Policy.get_ucb1(root, Policy.graph.his_edges[root][a])
                else:
                    # value = trans_prob[a] * self.get_avg_value(self.graph.his_edges[root][a])
                    value = Policy.get_avg_value(Policy.graph.his_edges[root][a])
                if value > max_value:
                    max_value = value
                    child_index = a
                    child_id = Policy.graph.his_edges[root][a]
                elif value == max_value:  # random selection among save-value nodes
                    if Funcs.rand_prob() > 0.5:
                        max_value = value
                        child_index = a
                        child_id = Policy.graph.his_edges[root][a]
        return child_id, child_index

    @staticmethod
    def get_ucb1(root, child):
        if child not in Policy.MCTS_n:  # n == 0
            return float("inf")
        n = Policy.MCTS_n[child]
        v = Policy.graph.node_value[child] / n
        if root not in Policy.MCTS_n:  # N == 0, such as loop structure in the state graph
            N = 1  # make sure ucb1 >= 0
        else:
            N = Policy.MCTS_n[root]
        ln_N = math.log(N)
        return v + P.ucb1_c * ((ln_N / n) ** 0.5)

    @staticmethod
    def get_avg_value(root):
        return Policy.graph.node_value[root]  # return the reward form the env

    
