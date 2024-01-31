from src.module.context import Profile as P
from src.util.tools import Logger, Funcs, IO
import time
from multiprocessing import Queue
from collections import deque
from src.util.imports.random import random
from src.util.imports.numpy import np
import wandb
import os
import math
from src.module.agent.policy.projector import Projector


class Actor:
    def __init__(self, id, actor_learner_queue: Queue, learner_actor_queues: Queue, finish, frames, update):
        random.seed(id)
        np.random.seed(id)
        self.id = id  # actor identifier
        self.num_episode = 0
        self.env = self.create_env(render=self.is_head() and P.render, is_head=self.is_head())
        self.finish = finish
        self.fps = deque(maxlen=P.average_window)
        self.actor_learner_queue = actor_learner_queue
        self.learner_actor_queues = learner_actor_queues
        self.episodic_reward = list()
        self.episodic_return = list()
        self.record_time = list()
        self.max_episodic_reward = None
        self.p = (P.e_greedy[1] - P.e_greedy[0]) / (P.num_actor - 1) * self.id + P.e_greedy[0]
        self.hit = None
        self.total_reward = None
        self.frames = frames
        self.update = update
        self.loop_start_time = None
        if self.is_head() and P.wandb_enabled:
            # delete previous run(s)
            os.environ["WANDB_MODE"] = "offline"
            api = wandb.Api()
            runs = api.runs('centergoodroid/mrl')
            for run in runs:
                if run.job_type==f"{P.env_name}" and run.group=="HG" and run.name==f"run-{P.run}":
                    run.delete()
            
            # init current run
            wandb.init(
                project="mrl",
                job_type=f"{P.env_name}",
                group="HG",
                name=f"run-{P.run}",
                dir=f"{P.out_dir}",
                # config=vars(args),
            )

    def create_env(self, render=False, is_head=False):
        if P.env_type == "atari":
            from src.module.env.atari import Atari
            return Atari.make_env(render, is_head)
        if P.env_type == "maze":
            from src.module.env.maze import Maze
            return Maze.make_env(render, is_head)
        if P.env_type == "toy_text":
            from src.module.env.toy_text import ToyText
            return ToyText.make_env(render, is_head)
        if P.env_type == "box_2d":
            from src.module.env.box_2d import Box2D
            return Box2D.make_env(render, is_head)
        if P.env_type == "sokoban":
            from src.module.env.sokoban import Sokoban
            return Sokoban.make_env(render, is_head)
        if P.env_type == "football":
            from src.module.env.football import Football
            return Football.make_env(render, is_head)
        if P.env_type == "mujoco":
            from src.module.env.mujoco import Mujoco
            return Mujoco.make_env(render, is_head)

    def is_head(self):
        return self.id == P.head_actor
    
    def final_eval(self):
        projector = Projector(0, False)
        graph = IO.read_disk_dump(f"{P.model_dir}graph.pkl")
        done = False
        obs = self.env.reset()
        total_reward = 0
        expected_return = 0
        step = 0
        projector.reset()
        while not done:
            _, proj_obs = projector.batch_project([obs, None, obs, None, None])
            action, value, steps = graph.get_action(proj_obs)
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            expected_return += reward * np.power(P.gamma, step)
            step += 1

        Logger.log(f"Highway graph R: {total_reward:6.2f} RTN: {expected_return:6.2f} Step: {step}", color="green")

        if P.wandb_enabled:
            try:
                wandb.log(data={"Expected_Return": expected_return, "Total_Reward": total_reward, "Minutes": (time.time() - self.loop_start_time) / 60}, step=int(self.frames.value))
            except Exception:
                pass
            wandb.finish()

    def get_action(self, last_obs, pre_action, obs, reward, done, epi_step, receiver_init_obs=False):
        # query action from policy
        if epi_step == 1:
            self.actor_learner_queue.put([last_obs, pre_action, obs, reward, done, False])
        else:
            self.actor_learner_queue.put([last_obs, pre_action, obs, reward, done, not self.is_head()])
            
        while True:
            try:  # sub-process exception
                action, value, steps = self.learner_actor_queues.get(timeout=0.1)
                break
            except KeyboardInterrupt:
                raise KeyboardInterrupt()
            except Exception:
                if self.finish.value:
                    if self.is_head():
                        self.final_eval()
                    raise Exception()

        if receiver_init_obs:  # the projected and indexed init obs
            return action

        if action is not None:
            self.hit.append(1)
        else:
            self.hit.append(0)
        
        # if policy can not return action
        if action is None:
            action = self.env.sample_action()
        
        # epsilon-greedy
        if np.random.rand() > self.p:
            if epi_step > steps * P.stick_on_graph:
                action = self.env.sample_action()
        
        if self.random_ops > 0 and not self.is_head():
            self.random_ops -= 1
            if P.random_init_ops["ops_option"] == "all":
                action = self.env.sample_action()
            else:
                action = np.random.choice(P.random_init_ops["ops_option"])

        if isinstance(action, np.ndarray):
            return tuple(action)
        else:
            return action

    def interact(self):
        self.loop_start_time = time.time()
        while True:  # episode loop
            # 0. init episode
            obs = last_obs = self.env.reset()
            self.random_ops = int(Funcs.rand_prob() * P.random_init_ops["max_random_ops"])
            self.total_reward = 0.0
            self.expected_return = 0.0
            epi_step = 1
            pre_action = 0
            done = False
            start_time = time.time()
            reward = 0.0
            self.hit = list()
            while self.learner_actor_queues.qsize() > 0:  # empty queue before env interaction
                self.learner_actor_queues.get()
            while True:  # step loop
                # 1. get action for obs
                action = self.get_action(last_obs, pre_action, obs, reward, done, epi_step)
                last_obs = obs if isinstance(obs, str) or isinstance(obs, int) or isinstance(obs, tuple) else obs.copy()

                # 2. interact
                obs, reward, done, info = self.env.step(action)

                # 3. post ops
                pre_action = action
                self.total_reward += reward
                self.expected_return += math.pow(P.gamma, epi_step) * reward
                epi_step += 1

                # 4. done ops
                if done:
                    proj_index_init_obs = self.get_action(last_obs, pre_action, obs, reward, done, epi_step, receiver_init_obs=True)
                    self.fps.append(
                        epi_step * P.num_action_repeats / (time.time() - start_time)
                    )
                    # 4.1 reward
                    self.episodic_reward.append(self.total_reward)
                    self.episodic_return.append(self.expected_return)
                    self.record_time.append(time.time())
                    if self.max_episodic_reward is None:
                        self.max_episodic_reward = self.total_reward
                    elif self.max_episodic_reward < self.total_reward:
                        self.max_episodic_reward = self.total_reward
                    # 4.2 graph hit rate
                    hit_rate = 100 * (sum(self.hit) / len(self.hit))
                    # 4.3 hop-off point of the graph
                    if hit_rate < 100:
                        last_step_before_loss = self.hit.index(0) + 1
                    else:
                        last_step_before_loss = len(self.hit)
                    # 4.4 report
                    latest_avg_reward = np.mean(self.episodic_reward[-P.average_window:])
                    latest_avg_return = np.mean(self.episodic_return[-P.average_window:])
                    if self.is_head():
                        if P.target_total_rewrad is not None:
                            if abs(latest_avg_reward - P.target_total_rewrad) < 0.01:
                                with self.update.get_lock():
                                    self.update.value = False
                        Logger.log("evl_actor R: {:6.2f} AR: {:6.2f} MR: {:6.2f} RTN: {:6.2f} ARTN: {:6.2f} Fps: {:6.1f} H: {:4.1f}% L: {}/{} O1: {}".format(
                            self.episodic_reward[-1],
                            latest_avg_reward,
                            self.max_episodic_reward,
                            self.episodic_return[-1],
                            latest_avg_return,
                            self.fps[-1],
                            hit_rate,
                            last_step_before_loss,
                            len(self.hit),
                            str(proj_index_init_obs)[-4:]
                        ))
                        if P.wandb_enabled:
                            try:
                                wandb.log(data={"Expected_Return": self.episodic_return[-1], "Total_Reward": self.episodic_reward[-1], "Minutes": (time.time() - self.loop_start_time) / 60}, step=int(self.frames.value))
                            except Exception:
                                pass
                    Logger.write(f"Actor_{self.id}/R", self.episodic_reward[-1], self.num_episode)
                    Logger.write(f"Actor_{self.id}/Rtn", self.episodic_return[-1], self.num_episode)
                    Logger.write(f"Actor_{self.id}/AvgR", latest_avg_reward, self.num_episode)
                    Logger.write(f"Actor_{self.id}/MaxR", self.max_episodic_reward, self.num_episode)
                    Logger.write(f"Actor_{self.id}/FPS", self.fps[-1], self.num_episode)
                    Logger.write(f"Actor_{self.id}/Hit%", hit_rate, self.num_episode)
                    Logger.write(f"Actor_{self.id}/LostAt", f"{last_step_before_loss}/{len(self.hit)}", self.num_episode, type="text")
                    Logger.write(f"Actor_{self.id}/O_1", str(proj_index_init_obs)[-4:], self.num_episode, type="text")

                    break
            self.num_episode += 1
