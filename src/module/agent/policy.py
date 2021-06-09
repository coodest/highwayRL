from src.module.context import Profile as P
from src.util.tools import *
from src.module.agent.transition.prob_tgn import ProbTGN
from src.module.agent.memory.graph import Graph
from src.module.agent.memory.projector import RandomProjector
from multiprocessing import Pool, Process, Value, Queue, Lock, Manager

import math


class Policy:
    graph_lock = Lock()
    prob_func = ProbTGN()

    manager = Manager()
    MCTS_n = manager.dict()
    mcts_lock = Lock()

    actor_learner_queues = None
    learner_actor_queues = None
    finish = None

    @staticmethod
    def inference(actor_learner_queues, learner_actor_queues, finish):
        Policy.actor_learner_queues = actor_learner_queues
        Policy.learner_actor_queues = learner_actor_queues
        Policy.finish = finish

        for id in range(P.num_actor):
            Process(
                target=Policy.get_action,
                args=[id],
            ).start()

        while True:
            time.sleep(1)
            # data = Graph.get_data()
            # self.prob_func.train(data)
            if Policy.finish.value:
                break
                
    @staticmethod
    def get_transition_prob(src, dsts):
        info = [src, dsts]
        return Policy.prob_func.test(info)

    @staticmethod
    def get_action(index):
        last_time = time.time()
        last_frames = Graph.frames.value
        while True:
            if index == 0 and time.time() - last_time > 5:
                Logger.log(f"|learner| fps: {(Graph.frames.value - last_frames) / (time.time() - last_time)} nodes: {len(Graph.node_feats)}")
                last_time = time.time()
                last_frames = Graph.frames.value

            if Graph.frames.value > P.total_frames:
                if Policy.finish.value == False:
                    with Policy.finish.get_lock():
                        Policy.finish.value = True
                break
            
            try:
                info = Policy.actor_learner_queues[index].get()
                info = RandomProjector.batch_project([info])[0]
                last_obs, pre_action, obs, reward, add = info
                
                # 1. add transition to graph memory
                if add:
                    Graph.add(last_obs, pre_action, obs, reward)
                
                # 2. find current/root node
                root = Graph.get_node_id(obs)
                if root not in Graph.his_edges.keys():  # can not give action for previously never interacted obs
                    Policy.learner_actor_queues[index].put(None)
                    continue

                # 3. use UCB1 formula to propagate value
                Policy.update_children(root)
                
                # 4. for training actor: select the child with max UCB1 and
                # return corresponding action;
                # for testing actor: select the child with max value and
                # return corresponding action
                child_id, action = Policy.get_max_child(root, value_type="value")
                
                Policy.learner_actor_queues[index].put(action)
            except Exception:
                Funcs.trace_exception()
                Policy.learner_actor_queues[index].put(None)

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
                    total_reward += Graph.node_value[current_node]  # only obs node has reward
                    visit_list.append(current_node)
                    break
                else:
                    total_reward += Graph.node_reward[current_node]  # only obs node has reward
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
                with Graph.lock:
                    Graph.node_value[node] += total_reward
                with Policy.mcts_lock:
                    if node not in Policy.MCTS_n:
                        Policy.MCTS_n[node] = 1
                    else:
                        Policy.MCTS_n[node] += 1
                last_node = node

        # update node value
        for node in Policy.MCTS_n.keys():
            with Graph.lock:
                Graph.node_value[node] /= Policy.MCTS_n[node]

    @staticmethod
    def get_max_child(root, value_type="ucb1"):
        max_value = - float("inf")
        child_index = None
        child_id = None
        if root in Graph.his_edges.keys():
            # trans_prob = Policy.get_transition_prob(root, Graph.his_edges[root])
            # trans_prob = trans_prob.squeeze(-1).cpu().detach().numpy().tolist()
            for a in range(len(Graph.his_edges[root])):
                if value_type == "ucb1":
                    # value = trans_prob[a] * Policy.get_ucb1(root, Graph.his_edges[root][a])  # todo
                    value = Policy.get_ucb1(root, Graph.his_edges[root][a])
                else:
                    # value = trans_prob[a] * self.get_avg_value(Graph.his_edges[root][a])
                    value = Policy.get_avg_value(Graph.his_edges[root][a])
                if value > max_value:
                    max_value = value
                    child_index = a
                    child_id = Graph.his_edges[root][a]
                elif value == max_value:  # random selection among save-value nodes
                    if Funcs.rand_prob() > 0.5:
                        max_value = value
                        child_index = a
                        child_id = Graph.his_edges[root][a]
        return child_id, child_index

    @staticmethod
    def get_ucb1(root, child):
        if child not in Policy.MCTS_n:  # n == 0
            return float("inf")
        n = Policy.MCTS_n[child]
        v = Graph.node_value[child] / n
        if root not in Policy.MCTS_n:  # N == 0, such as loop structure in the state graph
            N = 1  # make sure ucb1 >= 0
        else:
            N = Policy.MCTS_n[root]
        ln_N = math.log(N)
        return v + P.ucb1_c * ((ln_N / n) ** 0.5)

    @staticmethod
    def get_avg_value(root):
        return Graph.node_value[root]  # return the reward form the env

    
