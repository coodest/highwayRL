from src.module.context import Profile as P
from src.util.tools import Logger, Funcs, IO
from multiprocessing import Process, Value, Queue
import time
import os


class MemRL:
    @staticmethod
    def start():
        # 1. init
        Logger.path = f"{P.log_dir}{P.env_name}-{Logger.get_date()}.log"
        IO.make_dir(P.out_dir)
        IO.make_dir(P.log_dir)
        IO.make_dir(P.model_dir)
        IO.make_dir(P.result_dir)
        IO.make_dir(P.video_dir)
        IO.make_dir(P.sync_dir)
        if P.clean:
            IO.renew_dir(P.log_dir)
            IO.renew_dir(P.model_dir)
            IO.renew_dir(P.result_dir)
            IO.renew_dir(P.video_dir)
            IO.renew_dir(P.sync_dir)
        start_time = time.time()

        # 2. show args
        Funcs.print_obj(P.C)
        Funcs.print_obj(P)

        # 3. prepare queues for learner and actor
        finish = Value("b", False)
        actor_learner_queues = list()
        for _ in range(P.num_actor):
            actor_learner_queues.append(Queue())
        learner_actor_queues = list()
        for _ in range(P.num_actor):
            learner_actor_queues.append(Queue())

        # 4. launch
        processes = []

        learner_process = Process(target=MemRL.learner_run, args=(
            actor_learner_queues,
            learner_actor_queues,
            finish
        ))
        learner_process.start()
        processes.append(learner_process)

        for id in range(P.num_actor):
            p = Process(target=MemRL.actor_run, args=(
                id,
                actor_learner_queues[id],
                learner_actor_queues[id],
                finish
            ))
            p.start()
            processes.append(p)

        while True:
            try:  # process exception detection
                title_out = False
                for ind, p in enumerate(processes):
                    p.join()
                    if ind == 0:
                        Logger.log("learner exit")
                        continue  # skip the learner process
                    if not title_out:
                        Logger.log("actor ", new_line=False)
                        title_out = True
                    Logger.log(f"{ind - 1} ", new_line=False, make_title=False)
                Logger.log("exit", make_title=False)
                break
            except KeyboardInterrupt:
                pass
            except Exception:
                Funcs.trace_exception()
                break

        # 5. report time
        minutes = (time.time() - start_time) / 60
        Logger.log(f"up {minutes:.1f} min, exit")

    @staticmethod
    def learner_run(actor_learner_queues, learner_actor_queues, finish):
        # 1. init
        from src.module.agent.policy import Policy
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{str(P.gpus).replace(' ', '')[1:-1]}"

        # 2. train
        policy = Policy(actor_learner_queues, learner_actor_queues)
        try:  # sub-process exception detection
            optimal_graph = policy.train()  # tain the policy
        except KeyboardInterrupt:
            Logger.error("ctrl-c pressed")
            policy.wait_to_finish()
        except FileNotFoundError:
            Logger.error("optimal graph not found, training failed")
        except Exception:
            Funcs.trace_exception()
        finally:
            Logger.log("training finished")
            finish.value = True

        # 3. parameterization
        try:  # sub-process exception detection
            # TODO: Q-table parameterization
            pass
        except Exception:
            Funcs.trace_exception()
        Logger.log("dnn model saved")
        
    @staticmethod
    def actor_run(id, actor_learner_queues, learner_actor_queues, finish):
        # 1. make env and actor
        from src.module.env.actor import Actor

        actor = Actor(id, MemRL.create_env, actor_learner_queues, learner_actor_queues, finish)

        # 2. start interaction loop (actor loop)
        while True:
            try:  # sub-process exception detection
                actor.interact()
            except KeyboardInterrupt:
                pass
            except Exception:
                if finish.value:
                    break
                else:
                    Funcs.trace_exception()

    @staticmethod
    def create_env(render=False):
        if P.env_type == P.env_types[0]:
            from src.module.env.atari import Atari
            return Atari.make_env(render)
        if P.env_type == P.env_types[1]:
            from src.module.env.atari_alternative import Atari
            return Atari.make_env(render)
        if P.env_type == P.env_types[2]:
            from src.module.env.atari_history_hash import Atari
            return Atari.make_env(render)
        if P.env_type == P.env_types[3]:
            from src.module.env.atari_ram import AtariRam
            return AtariRam.make_env(render)
        if P.env_type == P.env_types[4]:
            from src.module.env.simple_scene import SimpleScene
            return SimpleScene.make_env(render)


if __name__ == "__main__":
    MemRL.start()
