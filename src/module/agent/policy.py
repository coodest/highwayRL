from src.module.context import Profile as P
from src.util.tools import Logger, Funcs, IO
import time
from multiprocessing import Process, Value
import os


class Policy:
    def __init__(self, actor_learner_queues, learner_actor_queues):
        self.frames = Value("d", 0)
        self.actor_learner_queues = actor_learner_queues
        self.learner_actor_queues = learner_actor_queues

    def train(self):
        processes = []
        for id in range(P.num_actor):
            p = Process(
                target=Policy.response_action,
                args=[
                    id,
                    self.actor_learner_queues[id],
                    self.learner_actor_queues[id],
                    self.frames
                ],
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        return IO.read_disk_dump(P.optimal_graph_path)

    @staticmethod
    def is_head(index):
        return index == P.head_actor

    @staticmethod
    def response_action(id, actor_learner_queue, learner_actor_queue, frames):
        from src.module.agent.memory.indexer import Indexer
        from src.module.agent.memory.graph import Graph

        last_report = time.time()
        last_frame = frames.value
        last_sync = time.time()
        graph = Graph(id, Policy.is_head(id))

        if P.projector == P.projector_types[1]:
            from src.module.agent.memory.projector import RandomProjector
            projector = RandomProjector(id)
        if P.projector == P.projector_types[2]:
            from src.module.agent.memory.projector import CNNProjector
            projector = CNNProjector(id)
        if P.projector == P.projector_types[3]:
            from src.module.agent.memory.projector import RNNProjector
            projector = RNNProjector(id)

        while True:
            trajectory = []
            total_reward = 0
            proj_index_init_obs = None
            while True:
                try:
                    # check to stop
                    if frames.value > P.total_frames:
                        return
                    # sync graph
                    if time.time() - last_sync > P.sync_every:
                        graph.sync()
                        last_sync = time.time()
                    # logging info
                    if Policy.is_head(id):
                        cur_frame = frames.value
                        now = time.time()
                        if now - last_report > P.log_every:
                            Logger.log("learner frames: {:4.1f}M fps: {:6.1f} G/C: {}/{}({:.1f}%) V: {}/{}".format(
                                cur_frame / 1e6,
                                (cur_frame - last_frame) / (now - last_report),
                                graph.main.obs_size(),
                                graph.main.crossing_node_size() if P.statistic_crossing_obs else "-",
                                100 * (graph.main.crossing_node_size() / (graph.main.obs_size() + 1e-8)) if P.statistic_crossing_obs else "-",
                                graph.main.max_total_reward(),
                                str(graph.main.max_total_reward_init_obs())[-4:],
                            ), color="yellow")
                            last_report = now
                            last_frame = cur_frame
                    
                    info = actor_learner_queue.get()
                    last_obs, pre_action, obs, reward, done, add = info
                    if P.projector is not None:
                        last_obs, obs = projector.batch_project([last_obs, obs])
                    last_obs, obs = Indexer.batch_get_ind([last_obs, obs])

                    if proj_index_init_obs is None:
                        proj_index_init_obs = obs

                    if add:  # head for testing actor does not add traj
                        trajectory.append([last_obs, pre_action, obs, reward])
                        total_reward += reward
                    if done:
                        if add:
                            with frames.get_lock():
                                frames.value += (len(trajectory) * P.num_action_repeats)
                            graph.store_inc(trajectory, total_reward)
                        learner_actor_queue.put(proj_index_init_obs)
                        if P.projector is not None:
                            projector.reset()
                        break
                    else:
                        action = graph.get_action(obs)
                        learner_actor_queue.put(action)
                except Exception:
                    Funcs.trace_exception()
                    return
