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

        return IO.read_disk_dump(P.model_dir + 'optimal.pkl')


    @staticmethod
    def is_head(index):
        return index == P.num_actor - 1

    @staticmethod
    def response_action(id, actor_learner_queue, learner_actor_queue, frames):
        from src.module.agent.memory.projector import RandomProjector
        from src.module.agent.memory.indexer import Indexer
        from src.module.agent.memory.optimal_graph import OptimalGraph

        last_report = time.time()
        last_frame = frames.value
        last_sync = time.time()
        optimal_graph = OptimalGraph(id, Policy.is_head(id))
        random_projector = RandomProjector(id)
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
                    now = time.time()
                    if now - last_sync > P.sync_every:
                        optimal_graph.sync()
                        last_sync = now
                    # logging info
                    if Policy.is_head(id):
                        cur_frame = frames.value
                        if now - last_report > P.log_every:
                            Logger.log("learner frames: {:4.1f}M fps: {:6.1f} G: {} V: {}/{}".format(
                                cur_frame / 1e6,
                                (cur_frame - last_frame) / (now - last_report),
                                len(optimal_graph.main.keys()),
                                optimal_graph.main.max_value,
                                str(optimal_graph.main.max_value_init_obs)[-4:]
                            ))
                            last_report = now
                            last_frame = cur_frame
                    

                    info = actor_learner_queue.get()
                    last_obs, pre_action, obs, reward, done, add = info
                    last_obs, obs = random_projector.batch_project([last_obs, obs])
                    last_obs, obs = Indexer.batch_get_ind([last_obs, obs])

                    if init_obs is None:
                        init_obs = obs

                    if add:
                        trajectory.append([last_obs, pre_action, obs])
                        total_reward += reward
                    if done:
                        if add:
                            with frames.get_lock():
                                frames.value += (
                                    len(trajectory) * P.num_action_repeats
                                )
                            optimal_graph.store_increments(trajectory, total_reward)
                        learner_actor_queue.put(
                            init_obs
                        )
                        break
                    else:
                        action = optimal_graph.get_action(obs)
                        learner_actor_queue.put(action)
                except Exception:
                    Funcs.trace_exception()
                    return
