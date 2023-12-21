from src.module.context import Profile as P
from src.util.tools import Logger, Funcs, IO
from multiprocessing import Process, Value, Manager

import time
import os


class HighwayRL:
    @staticmethod
    def start():
        # 1. init
        Logger.log_path = f"{P.log_dir}{P.env_name}.log"
        Logger.summary_dir = f"{P.summary_dir}{P.env_name}/"
        for dir in P.out_dirs:
            IO.make_dir(dir)
        start_time = time.time()

        # 2. show args
        Funcs.print_obj([P.C, P])

        # 3. staging
        HighwayRL.staging()

        # 4. report time
        minutes = (time.time() - start_time) / 60
        Logger.log(f"up {minutes:.1f} min, exit")

    @staticmethod
    def staging():
        # 1. learn the graph
        if P.stages[0] is True:
            finish = Value("b", False)
            frames = Value("d", 0)
            update = Value("b", True)
            actor_learner_queues = list()
            for _ in range(P.num_actor):
                actor_learner_queues.append(Manager().Queue())
            learner_actor_queues = list()
            for _ in range(P.num_actor):
                learner_actor_queues.append(Manager().Queue())

            learner_process = Process(target=HighwayRL.learner_run, args=(
                actor_learner_queues,
                learner_actor_queues,
                finish,
                frames,
                update,
            ))
            learner_process.start()

            actor_processes = []
            for id in range(P.num_actor):
                p = Process(target=HighwayRL.actor_run, args=(
                    id,
                    actor_learner_queues[id],
                    learner_actor_queues[id],
                    finish,
                    frames,
                    update,
                ))
                p.start()
                actor_processes.append(p)
            
            while True:
                try:
                    learner_process.join()
                    Logger.log("learner exit")

                    title_out = False
                    for ind, p in enumerate(actor_processes, start=0):
                        p.join()
                        if not title_out:
                            Logger.log("actor ", new_line=False)
                            title_out = True
                        Logger.log(f"{ind} ", new_line=False, make_title=False)
                    Logger.log("exit", make_title=False)
                    break
                except KeyboardInterrupt:
                    Logger.error("ctrl-c pressed")

            Logger.log("stage 1 finished")
        # 2. highway graph to offline rl
        if P.stages[1] is True:
            from src.module.agent.policy.model import Model
            model = Model()
            model.utilize_graph_data()
            model.save()
            Logger.log("stage 2 finished")
        # 3. online updating the model
        if P.stages[2] is True:
            Logger.log("stage 3 finished")

    @staticmethod
    def learner_run(actor_learner_queues, learner_actor_queues, finish, frames, update):
        try:
            from src.module.agent.learner import Learner
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{str(P.gpus).replace(' ', '')[1:-1]}"

            # Funcs.run_cmd("nvidia-cuda-mps-control -d", 2)
            learner = Learner(actor_learner_queues, learner_actor_queues, finish, frames, update)
            try:  # sub-process exception detection
                learner.learn()  # learn the graph
            except KeyboardInterrupt:
                learner.wait_to_finish()
                pass
            except Exception:
                Funcs.trace_exception()
            finally:
                with finish.get_lock():
                    finish.value = True
            # Funcs.run_cmd("echo quit | nvidia-cuda-mps-control", 2)
        except Exception:
            Funcs.trace_exception("(learner)")
        
    @staticmethod
    def actor_run(id, actor_learner_queues, learner_actor_queues, finish, frames, update):
        try:
            from src.module.agent.actor import Actor

            actor = Actor(id, actor_learner_queues, learner_actor_queues, finish, frames, update)
            while True:
                try:  # sub-process exception detection
                    actor.interact()
                except KeyboardInterrupt:
                    return
                except Exception:
                    if finish.value:
                        return
                    else:
                        Funcs.trace_exception()
        except Exception:
            Funcs.trace_exception(f"(actor {id})")


if __name__ == "__main__":
    HighwayRL.start()
