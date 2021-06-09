from src.module.context import Profile as P
from src.util.tools import *
from multiprocessing import Pool, Process, Value, Queue, Lock
from collections import deque


class Actor:
    def __init__(self, id, env, actor_learner_queue: Queue, learner_actor_queues: Queue):
        self.id = id  # actor identifier
        self.num_episode = 0
        self.env = env
        self.fps = deque(maxlen=10)
        self.actor_learner_queue = actor_learner_queue
        self.learner_actor_queues = learner_actor_queues
        self.episodic_reward = deque(maxlen=10)

    def is_testing_actor(self):
        """
        only last actor will be the testing actor
        """
        return self.id == P.num_actor - 1

    def get_action(self, last_obs, pre_action, obs, reward, is_first=False):
        # query action from policy
        if is_first:
            self.actor_learner_queue.put([last_obs, pre_action, obs, reward, False])
        else:
            self.actor_learner_queue.put([last_obs, pre_action, obs, reward, not self.is_testing_actor()])
        action = self.learner_actor_queues.get(timeout=10)

        if Funcs.rand_prob() - 0.5 > (self.id / (P.num_actor - 1)):
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
            start_time = time.time()
            reward = 0
            while True:  # step loop
                # 1. get action
                action = self.get_action(last_obs, pre_action, obs, reward, is_first=epi_step == 1)
                last_obs = obs

                # 2. interact
                obs, reward, done, info = self.env.step(action)

                # 3. post ops
                pre_action = action
                total_reward += reward
                epi_step += 1

                # 4. render
                if P.render_dir is not None:
                    self.env.render(mode="human")

                # 5. done ops
                if done:
                    self.fps.append( epi_step * P.num_action_repeats / (time.time() - start_time) )
                    self.episodic_reward.append(total_reward)
                    Logger.log(f"|actor| R: {self.episodic_reward[-1]}")
                    break
            self.num_episode += 1
