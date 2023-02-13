from src.module.context import Profile as P
from src.util.tools import Logger, Funcs
import time
from multiprocessing import Queue
from collections import deque
from src.util.imports.random import *
from src.util.imports.numpy import *


class Actor:
    def __init__(self, id, env_func, actor_learner_queue: Queue, learner_actor_queues: Queue, finish):
        self.id = id  # actor identifier
        self.num_episode = 0
        self.env = env_func(render=self.is_testing_actor() and P.render, is_head=self.is_testing_actor())
        self.finish = finish
        self.fps = deque(maxlen=10)
        self.actor_learner_queue = actor_learner_queue
        self.learner_actor_queues = learner_actor_queues
        self.episodic_reward = deque(maxlen=10)
        self.max_episodic_reward = None
        self.p = (P.e_greedy[1] - P.e_greedy[0]) / (P.num_actor - 1) * self.id + P.e_greedy[0]
        self.hit = None
        self.epi_return_est = None
        self.total_reward = None

    def reset_random_ops(self):
        self.random_ops = int(Funcs.rand_prob() * P.random_init_ops["max_random_ops"])
            
    def is_testing_actor(self):
        """
        only last actor will be the testing actor
        """
        return self.id == P.head_actor

    def get_action(self, last_obs, pre_action, obs, reward, done, epi_step, receiver_init_obs=False):
        # query action from policy
        if epi_step == 1:
            self.actor_learner_queue.put([last_obs, pre_action, obs, reward, done, False])
        else:
            self.actor_learner_queue.put([last_obs, pre_action, obs, reward, done, not self.is_testing_actor()])

        while True:
            try:  # sub-process exception
                action, value, steps = self.learner_actor_queues.get(timeout=0.1)
                break
            except KeyboardInterrupt:
                raise KeyboardInterrupt()
            except Exception:
                if self.finish.value:
                    raise Exception()

        if receiver_init_obs:  # the projected and indexed init obs
            return action

        if action is not None:
            self.hit.append(1)
        else:
            self.hit.append(0)
        
        if value is not None and (value + self.total_reward) not in self.epi_return_est:
            self.epi_return_est[value + self.total_reward] = epi_step

        # if policy can not return action
        if action is None:
            action = self.env.action_space.sample()
        
        # epsilon-greedy
        if random.random() > self.p:
            if epi_step > steps * P.stick_on_graph:
                action = self.env.action_space.sample()
        
        if self.random_ops > 0 and not self.is_testing_actor():
            self.random_ops -= 1
            if P.random_init_ops["ops_option"] == "all":
                action = self.env.action_space.sample()
            else:
                action = random.choice(P.random_init_ops["ops_option"])

        return action

    def interact(self):
        while True:  # episode loop
            # 0. init episode
            obs = last_obs = self.env.reset()
            self.reset_random_ops()
            self.total_reward = 0.0
            epi_step = 1
            pre_action = 0
            done = False
            start_time = time.time()
            reward = 0.0
            self.hit = list()
            self.epi_return_est = {}
            while self.learner_actor_queues.qsize() > 0:  # empty queue before env interaction
                self.learner_actor_queues.get()
            while True:  # step loop
                # 1. get action
                action = self.get_action(last_obs, pre_action, obs, reward, done, epi_step)
                last_obs = obs if isinstance(obs, str) or isinstance(obs, int) or isinstance(obs, tuple) else obs.copy()

                # 2. interact
                obs, reward, done, info = self.env.step(action)

                # 3. post ops
                pre_action = action
                self.total_reward += reward
                epi_step += 1

                # 4. done ops
                if done:
                    proj_index_init_obs = self.get_action(last_obs, pre_action, obs, reward, done, epi_step, receiver_init_obs=True)
                    self.fps.append(
                        epi_step * P.num_action_repeats / (time.time() - start_time)
                    )
                    self.episodic_reward.append(self.total_reward)
                    if self.max_episodic_reward is None:
                        self.max_episodic_reward = self.total_reward
                    elif self.max_episodic_reward < self.total_reward:
                        self.max_episodic_reward = self.total_reward
                    hit_rate = 100 * (sum(self.hit) / len(self.hit))
                    if hit_rate < 100:
                        last_step_before_loss = self.hit.index(0) + 1
                    else:
                        last_step_before_loss = len(self.hit)
                    if self.is_testing_actor():
                        Logger.log("evl_actor R: {:6.2f} AR: {:6.2f} MR: {:6.2f} Fps: {:6.1f} H: {:4.1f}% L: {}/{} OFF: {} O1: {}".format(
                            self.episodic_reward[-1],
                            np.mean(self.episodic_reward),
                            self.max_episodic_reward,
                            self.fps[-1],
                            hit_rate,
                            last_step_before_loss,
                            len(self.hit),
                            self.epi_return_est,
                            str(proj_index_init_obs)[-4:]
                        ))
                    Logger.write(f"Actor_{self.id}/R", self.episodic_reward[-1], self.num_episode)
                    Logger.write(f"Actor_{self.id}/AvgR", np.mean(self.episodic_reward), self.num_episode)
                    Logger.write(f"Actor_{self.id}/MaxR", self.max_episodic_reward, self.num_episode)
                    Logger.write(f"Actor_{self.id}/FPS", self.fps[-1], self.num_episode)
                    Logger.write(f"Actor_{self.id}/Hit%", hit_rate, self.num_episode)
                    Logger.write(f"Actor_{self.id}/LostAt", f"{last_step_before_loss}/{len(self.hit)}", self.num_episode, type="text")
                    Logger.write(f"Actor_{self.id}/O_1", str(proj_index_init_obs)[-4:], self.num_episode, type="text")
                    Logger.write(f"Actor_{self.id}/return_curve", np.array([
                        [0.0] if len(list(self.epi_return_est.keys())) == 0 else list(self.epi_return_est.keys()),
                    ]), self.num_episode, type="histogram")

                    break
            self.num_episode += 1
