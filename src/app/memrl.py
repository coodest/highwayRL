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

        # 2. show args
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
        Process(target=MemRL.learner_run, args=(
            actor_learner_queues,
            learner_actor_queues,
            finish
        )).start()
        for id in range(P.num_actor):
            Process(target=MemRL.actor_run, args=(
                id,
                actor_learner_queues[id],
                learner_actor_queues[id],
                finish
            )).start()

    @staticmethod
    def learner_run(actor_learner_queues, learner_actor_queues, finish):
        # 1. init
        from src.module.agent.policy import Policy
        start_time = time.time()

        # 2. train
        policy = Policy(actor_learner_queues, learner_actor_queues)
        try:
            # start CUDA multi-process server 
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{str(P.gpus).replace(' ', '')[1:-1]}"
            Logger.log("start cuda mps")
            os.popen("nvidia-cuda-mps-control -d").close()

            optimal_graph = policy.train()  # tain the policy
        except KeyboardInterrupt:
            Logger.log("ctrl-c pressed")
            exit(0)
        except Exception:
            Funcs.trace_exception()
        finally:
            # stop CUDA multi-process server 
            Logger.log("stop cuda mps")
            os.popen("echo quit | nvidia-cuda-mps-control").close()
        Logger.log("training finished")
        finish.value = True

        # 3. parameterization
        try:
            # TODO: Q-table parameterization
            pass
        except Exception:
            Funcs.trace_exception()
        Logger.log("dnn model saved")
        
        # 4. output time
        minutes = (time.time() - start_time) / 60
        Logger.log(f"learner exit, up {minutes:.1f} min")
        
    @staticmethod
    def actor_run(id, actor_learner_queues, learner_actor_queues, finish):
        # 1. make env and actor
        from src.module.env.actor import Actor

        actor = Actor(id, MemRL.create_env, actor_learner_queues, learner_actor_queues, finish)

        # 2. start interaction loop (actor loop)
        while True:
            try:
                actor.interact()
            except Exception:
                if finish.value:
                    Logger.log(f"actor{id} exit")
                    return
                else:
                    Funcs.trace_exception()

    @staticmethod
    def create_env(render=False):
        if P.env_type == "atari":
            from src.module.env.atari import Atari
            return Atari.make_env(render)
        if P.env_type == "atari_ram":
            from src.module.env.atari_ram import AtariRam
            return AtariRam.make_env(render)
        if P.env_type == "simple_scene":
            from src.module.env.simple_scene import SimpleScene
            return SimpleScene.make_env(render)


if __name__ == "__main__":
    MemRL.start()
