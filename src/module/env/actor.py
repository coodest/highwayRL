from src.module.context import Profile as P
from src.util.tools import Logger
import time
from multiprocessing import Queue
from collections import deque
from src.util.imports.random import random
from gym.wrappers import Monitor


class Actor:
    def __init__(self, id, env_func, actor_learner_queue: Queue, learner_actor_queues: Queue):
        self.id = id  # actor identifier
        self.num_episode = 0
        self.env = env_func(render=self.is_testing_actor() and P.render)
        self.fps = deque(maxlen=10)
        self.actor_learner_queue = actor_learner_queue
        self.learner_actor_queues = learner_actor_queues
        self.episodic_reward = deque(maxlen=10)
        self.p = (P.e_greedy[1] - P.e_greedy[0]) * (self.id + 1) / P.num_actor + P.e_greedy[0]
        self.hit = None

        if self.is_testing_actor():
            assert self.p == 1
            
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
        action = self.learner_actor_queues.get(timeout=10)
        if action is not None:
            self.hit += 1

        if random.random() < self.p:
            # epsilon-greedy
            action = self.env.action_space.sample()
        elif action is None:
            # if policy can not return action
            action = self.env.action_space.sample()

        return action

    def interact(self):
        while True:  # episode loop
            # 0. init episode
            last_obs = self.env.reset()
            obs = last_obs
            total_reward = 0
            epi_step = 1
            pre_action = 0
            done = False
            start_time = time.time()
            reward = 0
            self.hit = 0
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
                    unused_action = self.get_action(last_obs, pre_action, obs, reward, done, epi_step == 1)
                    self.fps.append(
                        epi_step * P.num_action_repeats / (time.time() - start_time)
                    )
                    self.episodic_reward.append(total_reward)
                    if self.is_testing_actor():
                        Logger.log("evl_actor R: {:6.2f} Fps: {:6.1f} H: {:4.1f}%".format(
                            self.episodic_reward[-1],
                            self.fps[-1],
                            100 * (self.hit / epi_step)
                        ))
                    break
            self.num_episode += 1
