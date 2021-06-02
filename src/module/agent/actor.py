from src.module.context import Profile as P
from src.util.tools import *
from multiprocessing import Pool, Process, Value, Queue, Lock


class Actor:
    def __init__(self, id, env, inference_queue, actor_queue: Queue):
        self.id = id  # actor identifier
        self.num_episode = 0
        self.env = env
        self.fps = 0
        self.inference_queue = inference_queue
        self.actor_queue =actor_queue

    def is_testing_actor(self):
        """
        only last actor will be the testing actor
        """
        return self.id == P.num_actor - 1

    def get_action(self, last_obs, pre_action, obs, reward, is_first=False):
        # query action from policy
        if is_first:
            self.inference_queue.put([self.id, last_obs, pre_action, obs, reward, False])
        else:
            self.inference_queue.put([self.id, last_obs, pre_action, obs, reward, not self.is_testing_actor()])
        action = self.actor_queue.get(timeout=10)

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
                    self.fps = epi_step / (time.time() - start_time)
                    if self.is_testing_actor() and self.num_episode % P.log_every_episode == 0:
                        Logger.log(f"R: {total_reward}, fps: {self.fps}")
                    break
            self.num_episode += 1
