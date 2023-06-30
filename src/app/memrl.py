from src.module.context import Profile as P
from src.util.tools import Logger, Funcs, IO
from multiprocessing import Process, Value, Queue
import time
import os


class MemRL:
    @staticmethod
    def start():
        # 1. init
        Logger.log_path = f"{P.log_dir}{P.env_name}-{Logger.get_date()}.log"
        Logger.summary_dir = f"{P.summary_dir}{P.env_name}/"
        for dir in P.out_dirs:
            IO.make_dir(dir)
        start_time = time.time()

        # 2. show args
        Funcs.print_obj([P.C, P])

        # 3. staging
        MemRL.staging()

        # 4. report time
        minutes = (time.time() - start_time) / 60
        Logger.log(f"up {minutes:.1f} min, exit")

    @staticmethod
    def staging():
        # 1. learn the graph
        if P.start_stage <= 0:
            finish = Value("b", False)
            actor_learner_queues = list()
            for _ in range(P.num_actor):
                actor_learner_queues.append(Queue())
            learner_actor_queues = list()
            for _ in range(P.num_actor):
                learner_actor_queues.append(Queue())

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
            Logger.log("stage 1 finished")
        # 2. parameterize the graph
        if P.start_stage <= 1:
            from src.module.agent.policy.model import Model
            model = Model()
            model.utilize_graph()
            model.evaluate()
            model.save()
            Logger.log("stage 2 finished")
        # 3. online updating the model
        if P.start_stage <= 2:
            Logger.log("stage 3 finished")

    @staticmethod
    def learner_run(actor_learner_queues, learner_actor_queues, finish):
        try:
            from src.module.agent.learner import Learner
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{str(P.gpus).replace(' ', '')[1:-1]}"

            Funcs.run_cmd("nvidia-cuda-mps-control -d", 2)
            learner = Learner(actor_learner_queues, learner_actor_queues, finish)
            try:  # sub-process exception detection
                learner.learn()  # learn the graph
            except KeyboardInterrupt:
                Logger.error("ctrl-c pressed")
                learner.wait_to_finish()
            except FileNotFoundError:
                Logger.error("no saved graph")
            except Exception:
                Funcs.trace_exception()
            finally:
                with finish.get_lock():
                    finish.value = True
            Funcs.run_cmd("echo quit | nvidia-cuda-mps-control", 2)
        except Exception:
            Funcs.trace_exception("(learner)")
        
    @staticmethod
    def actor_run(id, actor_learner_queues, learner_actor_queues, finish):
        try:
            from src.module.agent.actor import Actor

            actor = Actor(id, actor_learner_queues, learner_actor_queues, finish)
            while True:
                try:  # sub-process exception detection
                    actor.interact()
                except KeyboardInterrupt:
                    break
                except Exception:
                    if finish.value:
                        break
                    else:
                        Funcs.trace_exception()
        except Exception:
            Funcs.trace_exception(f"(actor {id})")


if __name__ == "__main__":
    MemRL.start()
