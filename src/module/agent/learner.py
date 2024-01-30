from src.module.context import Profile as P
from src.util.tools import Logger, Funcs, IO
import time
from multiprocessing import Process, Value, Queue


class Learner:
    def __init__(self, actor_learner_queues, learner_actor_queues, finish, frames, update):
        self.frames = frames
        self.update = update
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
        for ind, p in enumerate(self.processes):
            p.join()
        Logger.log("learner worker joined")

    def learn(self):
        for id in range(P.num_actor):
            p = Process(
                target=Learner.response_action,
                args=[
                    id,
                    self.actor_learner_queues[id],
                    self.learner_actor_queues[id],
                    self.head_slave_queues,
                    self.slave_head_queues,
                    self.frames,
                    self.sync,
                    self.finish,
                    self.update,
                ],
            )
            p.start()
            self.processes.append(p)
        self.wait_to_finish()

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
        finish,
        update,
    ):
        try:  # sub-sub-process exception
            from src.module.agent.policy.projector import Projector
            from src.module.agent.policy.memory import Memory
            from src.module.agent.policy.neural.dataset import OfflineDataset

            last_report = time.time()
            last_frame = frames.value
            sync_count_down = P.sync_every
            memory = Memory(id, Learner.is_head(id))
            projector = Projector(id, Learner.is_head(id))
            offline_dataset = OfflineDataset(f"{P.env_name}-{id}")

            while True:
                trajectory = []
                proj_index_init_obs = None
                while True:
                    if sync_count_down <= 0 or frames.value > P.total_frames:
                        if Learner.is_head(id):
                            # check to sync
                            with sync.get_lock():
                                sync.value = True
                        
                    if sync.value:
                        # sync memory
                        memory.sync_by_pipe_disk(
                            head_slave_queues, 
                            slave_head_queues, 
                            sync,
                            update,
                        )
                        if Learner.is_head(id):
                            # logging info
                            cur_frame = frames.value
                            now = time.time()
                            Logger.log("learner frames: {:4.1f}M fps: {:6.1f} {}".format(
                                cur_frame / 1e6,
                                (cur_frame - last_frame) / (now - last_report),
                                memory.get_graph().info()
                            ), color="yellow")
                            last_report = now
                            last_frame = cur_frame
                            sync_count_down = P.sync_every
                            if frames.value > P.total_frames:
                                with finish.get_lock():
                                    finish.value = True

                    if finish.value:
                        if not Learner.is_head(id):
                            offline_dataset.save()
                            Logger.log(f"dataset-{id} saved.")
                        return

                    info = actor_learner_queue.get()
                    raw_last_obs, pre_action, raw_obs, reward, done, add = info

                    # project the raw obs into obs
                    last_obs, obs = projector.batch_project([raw_last_obs, pre_action, raw_obs, reward, done])

                    if proj_index_init_obs is None:
                        # drop last_obs of the first interaction
                        proj_index_init_obs = obs

                    if add:  # does not add traj from head actor, and first transition from other actors
                        trajectory.append([last_obs, pre_action, obs, reward])
                        if not Learner.is_head(id):
                            offline_dataset.add(obs=raw_obs, proj_obs=obs)
                    if done:
                        if add:
                            with frames.get_lock():
                                frames.value += (
                                    len(trajectory) * P.num_action_repeats
                                )
                            memory.store_new_trajs(trajectory)
                        learner_actor_queue.put([proj_index_init_obs, None, 0])
                        projector.reset()
                        sync_count_down -= 1
                        break
                    else:
                        learner_actor_queue.put(memory.get_graph().get_action(obs))
        except KeyboardInterrupt:
            pass
        except Exception:
            Funcs.trace_exception(f"(learner {id})")
