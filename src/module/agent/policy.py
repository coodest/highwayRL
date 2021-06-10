from src.module.context import Profile as P
from src.util.tools import *
# from src.module.agent.transition.prob_tgn import ProbTGN
from src.module.agent.memory.graph import Graph
from multiprocessing import Pool, Process, Value, Queue, Lock, Manager

import math


class Policy:
    # prob_func = ProbTGN()

    manager = Manager()

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
        from src.module.agent.memory.projector import RandomProjector
        last_time = time.time()
        last_frames = Graph.frames.value
        while True:
            if index == 0 and time.time() - last_time > 5:
                Logger.log("|learner| fps: {:3.1f} frame: {:4.3f}M nodes: {}".format(
                    (Graph.frames.value - last_frames) / (time.time() - last_time),
                    Graph.frames.value / 1e6,
                    len(Graph.node_feats),
                ))
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
                child_id, action = Policy.get_max_child(root)
                
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
        for _ in range(P.propagations):
            # simulate
            total_reward = 0
            final_value = 0
            visit_list = []
            current_node = root  # node of root obs
            simulate_steps = P.simulate_steps
            while current_node is not None:
                simulate_steps -= 1
                if simulate_steps <= 0:
                    break
                total_reward += Graph.node_reward[current_node]
                trend_value = total_reward + Graph.node_value[current_node] - Graph.node_reward[current_node] # only obs node has reward
                current_value = max(trend_value, total_reward)
                final_value = max(final_value, current_value)
                visit_list.append(current_node)

                action = Policy.sample_node(list(range(P.num_action)))
                current_node = Policy.sample_node(Graph.his_edges[current_node][action])
                
                if current_node in visit_list:  # avoid loop
                    current_node = None

            # back propagation and expand UCB1 profiles
            with Graph.node_value_lock:
                Graph.node_value[root] = final_value

    @staticmethod
    def get_max_child(root):
        max_value = - float("inf")
        child_index = None
        child_id = None
        if root in Graph.his_edges.keys():
            # trans_prob = Policy.get_transition_prob(root, Graph.his_edges[root])
            # trans_prob = trans_prob.squeeze(-1).cpu().detach().numpy().tolist()
            for a in Graph.his_edges[root].keys():
                # value = trans_prob[a] * self.get_avg_value(Graph.his_edges[root][a])
                for node in Graph.his_edges[root][a]:
                    value = Graph.node_value[node]
                    if value > max_value:
                        max_value = value
                        child_index = a
                        child_id = node
                    elif value == max_value:  # random selection among save-value nodes
                        if Funcs.rand_prob() > 0.5:
                            max_value = value
                            child_index = a
                            child_id = node
        return child_id, child_index

    @staticmethod
    def sample_node(root):
        if len(root) == 0:
            return None
        else:
            return random.choice(root)

    
