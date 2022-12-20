from src.module.context import Profile as P
from src.util.tools import Logger, Funcs
import time
from multiprocessing import Queue
from collections import deque
from src.util.imports.random import *
from src.util.imports.numpy import *


class Actor:
    def __init__(self, id, env_func, actor_learner_queue: Queue, learner_actor_queues: Queue, finish):
        random.seed(id)
        np.random.seed(id)
        self.id = id  # actor identifier
        self.num_episode = 0
        self.env = env_func(render=self.is_testing_actor() and P.render, is_head=self.is_testing_actor())
        self.env.seed = id
        self.finish = finish
        self.fps = deque(maxlen=10)
        self.actor_learner_queue = actor_learner_queue
        self.learner_actor_queues = learner_actor_queues
        self.episodic_reward = deque(maxlen=10)
        self.max_episodic_reward = None
        self.p = (P.e_greedy[1] - P.e_greedy[0]) / (P.num_actor - 1) * self.id + P.e_greedy[0]
        self.hit = None

    def reset_random_ops(self):
        self.random_ops = int(Funcs.rand_prob() * P.max_random_ops)
            
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
                action = self.learner_actor_queues.get(timeout=0.1)
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

        if random.random() > self.p:
            # epsilon-greedy
            action = self.env.action_space.sample()
        elif action is None:
            # if policy can not return action
            action = self.env.action_space.sample()
        
        while self.random_ops > 0 and not self.is_testing_actor():
            self.random_ops -= 1
            action = self.env.action_space.sample()
            break

        return action

    def interact(self):
        while True:  # episode loop
            # 0. init episode
            obs = last_obs = self.env.reset()
            self.reset_random_ops()
            total_reward = 0.0
            epi_step = 1
            pre_action = 0
            done = False
            start_time = time.time()
            reward = 0.0
            self.hit = list()
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
                total_reward += reward
                epi_step += 1

                # 4. done ops
                if done:
                    proj_index_init_obs = self.get_action(last_obs, pre_action, obs, reward, done, epi_step, receiver_init_obs=True)
                    self.fps.append(
                        epi_step * P.num_action_repeats / (time.time() - start_time)
                    )
                    self.episodic_reward.append(total_reward)
                    if self.max_episodic_reward is None:
                        self.max_episodic_reward = total_reward
                    elif self.max_episodic_reward < total_reward:
                        self.max_episodic_reward = total_reward
                    hit_rate = 100 * (sum(self.hit) / len(self.hit))
                    if hit_rate < 100:
                        last_step_before_loss = self.hit.index(0)
                    else:
                        last_step_before_loss = len(self.hit)
                    if self.is_testing_actor():
                        Logger.log("evl_actor R: {:6.2f} AR: {:6.2f} MR: {:6.2f} Fps: {:6.1f} H: {:4.1f}% L: {}/{} O1: {}".format(
                            self.episodic_reward[-1],
                            np.mean(self.episodic_reward),
                            self.max_episodic_reward,
                            self.fps[-1],
                            hit_rate,
                            last_step_before_loss,
                            len(self.hit),
                            str(proj_index_init_obs)[-4:]
                        ))
                    break
            self.num_episode += 1
