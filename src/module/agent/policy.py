from src.module.context import Profile as P
from src.util.tools import Logger, Funcs
import time
import random
from src.module.agent.memory.optimal_graph import OptimalGraph
from multiprocessing import Pool, Process, Value, Queue, Lock, Manager


class Policy:
    manager = Manager()

    actor_learner_queues = None
    learner_actor_queues = None
    frames = Value('d', 0)

    def __init__(self, actor_learner_queues, learner_actor_queues):
        Policy.actor_learner_queues = actor_learner_queues
        Policy.learner_actor_queues = learner_actor_queues

    def train(self):
        processes = []
        for id in range(P.num_actor):
            p = Process(target=Policy.response_action, args=[id])
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        
        return OptimalGraph

    @staticmethod
    def response_action(index):
        from src.module.agent.memory.projector import RandomProjector
        last_report = time.time()
        last_frame = Policy.frames.value
        while True:
            trajectory = []
            while True:
                try:
                    if Policy.frames.value > P.total_frames:
                        return
                    if index == 0:  # process 0 report infomation
                        now = time.time()
                        cur_frame = Policy.frames.value
                        if now - last_report > P.log_every:
                            Logger.log(f'learner fps: {(cur_frame - last_frame) / (now - last_report)}')
                            last_report = now
                            last_frame = cur_frame
                        
                    info = Policy.actor_learner_queues[index].get()
                    info = RandomProjector.batch_project([info])[0]
                    last_obs, pre_action, obs, reward, done, add = info

                    if add:
                        trajectory.append([last_obs, pre_action, obs, reward])
                    if done:
                        with Policy.frames.get_lock():
                            Policy.frames.value += len(trajectory) * P.num_action_repeats
                        OptimalGraph.expand_graph(trajectory)
                        Policy.learner_actor_queues[index].put(None)  # last action is not been used
                        break
                    else:
                        action = OptimalGraph.get_action(obs)
                        Policy.learner_actor_queues[index].put(action)
                except Exception:
                    Funcs.trace_exception()
                    return
