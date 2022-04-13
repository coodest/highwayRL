from src.module.context import Profile as P
from src.util.tools import Logger, Funcs, IO
import time
import os
from multiprocessing import Process, Value, Queue


class Policy:
    def __init__(self, actor_learner_queues, learner_actor_queues, finish):
        self.frames = Value("d", 0)
        self.sync = Value("b", False)
        self.actor_learner_queues = actor_learner_queues
        self.learner_actor_queues = learner_actor_queues

        self.head_slave_queues = list()
        for _ in range(P.num_actor):
            self.head_slave_queues.append(Queue())
        self.slave_head_queues = list()
        for _ in range(P.num_actor):
            self.slave_head_queues.append(Queue())

        self.processes = []
        self.finish = finish

    def wait_to_finish(self):
        Logger.log("leaner master wait for worker (head and slaves) to join")
        title_out = False
        for ind, p in enumerate(self.processes):
            p.join()
            if not title_out:
                Logger.log("learner worker ", new_line=False)
                title_out = True
            Logger.log(f"{ind} ", new_line=False, make_title=False)
        Logger.log("joined", make_title=False)

    def train(self):
        for id in range(P.num_actor):
            p = Process(
                target=Policy.response_action,
                args=[
                    id,
                    self.actor_learner_queues[id],
                    self.learner_actor_queues[id],
                    self.head_slave_queues,
                    self.slave_head_queues,
                    self.frames,
                    self.sync,
                    self.finish,
                ],
            )
            p.start()
            self.processes.append(p)

        self.wait_to_finish()
        
        optimal_graph = IO.read_disk_dump(P.optimal_graph_path)

        return optimal_graph

    @staticmethod
    def is_head(index):
        return index == P.head_actor

    @staticmethod
    def response_action(
        id, 
        actor_learner_queue, 
        learner_actor_queue, 
        head_slave_queues, 
        slave_head_queues, 
        frames, 
        sync, 
        finish
    ):
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
                try:  # sub-sub-process exception detection
                    # sync graph
                    if Policy.is_head(id) and (
                        time.time() - last_sync > P.sync_every or 
                        frames.value > P.total_frames
                    ):
                        sync.value = True
                    if sync.value:
                        if P.sync_mode == 0:
                            graph.sync_by_pipe(
                                head_slave_queues, 
                                slave_head_queues, 
                                sync
                            )
                        if P.sync_mode == 1:
                            graph.sync_by_file(sync)
                        if P.sync_mode == 2:
                            graph.sync_by_pipe_disk(
                                head_slave_queues, 
                                slave_head_queues, 
                                sync
                            )
                        last_sync = time.time()
                    # logging info
                    if Policy.is_head(id):
                        cur_frame = frames.value
                        now = time.time()
                        if (
                            now - last_report > P.log_every or 
                            frames.value > P.total_frames
                        ):
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
                    # check to stop
                    if Policy.is_head(id) and frames.value > P.total_frames:
                        graph.save_graph()
                        Logger.log("graph saved")
                        finish.value = True
                    if finish.value:
                        return
                    
                    info = actor_learner_queue.get()
                    last_obs, pre_action, obs, reward, done, add = info
                    if P.projector is not None:
                        last_obs, obs = projector.batch_project([last_obs, obs])

                    if P.indexer_enabled:
                        last_obs, obs = Indexer.batch_get_ind([last_obs, obs])

                    if proj_index_init_obs is None:
                        proj_index_init_obs = obs

                    if add:  # does not add traj from head actor, and first transition from other actors
                        trajectory.append([last_obs, pre_action, obs, reward])
                        total_reward += reward
                    if done:
                        if add:
                            with frames.get_lock():
                                frames.value += (
                                    len(trajectory) * P.num_action_repeats
                                )
                            graph.store_inc(trajectory, total_reward)
                        learner_actor_queue.put(proj_index_init_obs)
                        if P.projector is not None:
                            projector.reset()
                        break
                    else:
                        action = graph.get_action(obs)
                        learner_actor_queue.put(action)
                except KeyboardInterrupt:
                    Logger.log(f"learner worker {id} {'(head)' if Policy.is_head(id) else '(slave)'} returned with KeyboardInterrupt")
                    return
                except Exception:
                    Funcs.trace_exception()
                    return
