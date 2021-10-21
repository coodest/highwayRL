from src.module.context import Profile as P
from src.util.tools import Logger, Funcs, IO
import time
from multiprocessing import Process, Value


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

        optimal_graph = IO.read_disk_dump(P.optimal_graph_path)

        Logger.log("stored corssing obs and obs are: {} / {}: {}%".format(
            len(optimal_graph.crossing_obs) if P.statistic_crossing_obs else "-",
            len(optimal_graph),
            ((100 * len(optimal_graph.crossing_obs)) / len(optimal_graph)) if P.statistic_crossing_obs else "-",
        ))

        return optimal_graph

    @staticmethod
    def is_head(index):
        return index == P.num_actor - 1

    @staticmethod
    def response_action(id, actor_learner_queue, learner_actor_queue, frames):
        from src.module.agent.memory.projector import RandomProjector
        from src.module.agent.memory.projector import CNNProjector
        from src.module.agent.memory.indexer import Indexer
        from src.module.agent.memory.graph import Graph

        last_report = time.time()
        last_frame = frames.value
        last_sync = time.time()
        graph = Graph(id, Policy.is_head(id))
        
        if P.projector == P.projector_types[0]:
            projector = RandomProjector(id)
        if P.projector == P.projector_types[1]:
            projector = CNNProjector(id)

        while True:
            trajectory = []
            total_reward = 0
            init_obs = None
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
                            Logger.log("learner frames: {:4.1f}M fps: {:6.1f} G/C: {}/{} V: {}/{}".format(
                                cur_frame / 1e6,
                                (cur_frame - last_frame) / (now - last_report),
                                len(graph.main.obs_dict()),
                                len(graph.main.crossing_obs_set()) if P.statistic_crossing_obs else "-",
                                graph.main.get_max_total_reward(),
                                str(graph.main.get_max_total_reward_init_obs())[-4:],
                            ))
                            last_report = now
                            last_frame = cur_frame
                    
                    info = actor_learner_queue.get()
                    last_obs, pre_action, obs, reward, done, add = info
                    last_obs, obs = projector.batch_project([last_obs, obs])
                    last_obs, obs = Indexer.batch_get_ind([last_obs, obs])

                    if init_obs is None:
                        init_obs = obs

                    if add:  # head for testing actor does not add traj
                        trajectory.append([last_obs, pre_action, obs, reward])
                        total_reward += reward
                    if done:
                        if add:
                            with frames.get_lock():
                                frames.value += (len(trajectory) * P.num_action_repeats)
                            graph.store_inc(trajectory, total_reward)
                        learner_actor_queue.put(init_obs)
                        break
                    else:
                        action = graph.get_action(obs)
                        learner_actor_queue.put(action)
                except Exception:
                    Funcs.trace_exception()
                    return
