from src.module.context import Profile as P
from src.util.tools import Logger
import time
from multiprocessing import Queue
from collections import deque
from src.util.imports.random import random
from src.util.imports.numpy import np


class Actor:
    def __init__(self, id, env_func, actor_learner_queue: Queue, learner_actor_queues: Queue, finish):
        self.id = id  # actor identifier
        self.num_episode = 0
        self.env = env_func(render=self.is_testing_actor() and P.render)
        self.finish = finish
        self.fps = deque(maxlen=10)
        self.actor_learner_queue = actor_learner_queue
        self.learner_actor_queues = learner_actor_queues
        self.episodic_reward = deque(maxlen=10)
        self.p = (P.e_greedy[1] - P.e_greedy[0]) * (self.id + 1) / P.num_actor + P.e_greedy[0]
        self.hit = None
            
    def is_testing_actor(self):
        """
        only last actor will be the testing actor
        """
        return self.id == P.num_actor - 1

    def get_action(self, last_obs, pre_action, obs, reward, done, first_frame):
        # query action from policy
        if first_frame:
            self.actor_learner_queue.put([last_obs, pre_action, obs, reward, done, False])
        else:
            self.actor_learner_queue.put([last_obs, pre_action, obs, reward, done, not self.is_testing_actor()])
        
        while True:
            try:
                action = self.learner_actor_queues.get(timeout=0.1)
                break
            except Exception:
                if self.finish.value:
                    raise Exception()

        if type(action) is str:  # hashing 
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

        if self.is_testing_actor():
            assert not random.random() > self.p
            
        return action

    def interact(self):
        while True:  # episode loop
            # 0. init episode
            last_obs = self.env.reset()
            obs = last_obs
            total_reward = 0
            epi_step = 0
            pre_action = 0
            done = False
            start_time = time.time()
            reward = 0
            self.hit = list()
            while self.learner_actor_queues.qsize() > 0:
                self.learner_actor_queues.get()
            while True:  # step loop
                # 1. get action
                action = self.get_action(last_obs, pre_action, obs, reward, done, epi_step == 1)
                last_obs = obs

                # 2. interact
                obs, reward, done, info = self.env.step(action)

                # 3. post ops
                pre_action = action
                total_reward += reward
                epi_step += 1

                # 4. done ops
                if done:
                    init_obs = self.get_action(last_obs, pre_action, obs, reward, done, epi_step == 1)
                    self.fps.append(
                        epi_step * P.num_action_repeats / (time.time() - start_time)
                    )
                    self.episodic_reward.append(total_reward)
                    hit_rate = 100 * (sum(self.hit) / len(self.hit))
                    if hit_rate < 100:
                        lost_step = self.hit.index(0)
                    else:
                        lost_step = len(self.hit)
                    if self.is_testing_actor():
                        Logger.log("evl_actor R: {:6.2f} AR:{:6.2f} Fps: {:6.1f} H: {:4.1f}% L: {}/{} O1: {}".format(
                            self.episodic_reward[-1],
                            np.mean(self.episodic_reward),
                            self.fps[-1],
                            hit_rate,
                            lost_step,
                            epi_step,
                            str(init_obs)[-4:]
                        ))
                    break
            self.num_episode += 1
