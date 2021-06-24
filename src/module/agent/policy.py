from src.module.context import Profile as P
from src.util.tools import Logger, Funcs
import time
import random
from src.module.agent.memory.optimal_graph import OptimalGraph
from multiprocessing import Pool, Process, Value, Queue, Lock, Manager
from src.module.agent.memory.projector import RandomProjector


class Policy:
    manager = Manager()

    actor_learner_queues = None
    learner_actor_queues = None
    finish = None
    frames = Value('d', 0)

    @staticmethod
    def inference(actor_learner_queues, learner_actor_queues, finish):
        Policy.actor_learner_queues = actor_learner_queues
        Policy.learner_actor_queues = learner_actor_queues
        Policy.finish = finish

        for id in range(P.num_actor):
            Process(target=Policy.response_action, args=[id]).start()

        while True:
            time.sleep(1)
            if Policy.finish.value:
                # TODO: Q-table parameterization
                break

    @staticmethod
    def response_action(index):
        trajectory = []
        while True:
            try:
                info = Policy.actor_learner_queues[index].get()
                info = RandomProjector.batch_project([info])[0]
                last_obs, pre_action, obs, reward, done, add = info

                trajectory.append([last_obs, pre_action, obs, reward])
                if done:
                    with Policy.frames.get_lock():
                        Policy.frames += len(trajectory) * P.

                action = OptimalGraph.get_action(obs)
                Policy.learner_actor_queues[index].put(action)
            except Exception:
                Funcs.trace_exception()
                Policy.learner_actor_queues[index].put(None)
