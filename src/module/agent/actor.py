import time

from src.module.context import Profile as P
from src.util.tools import *
from src.module.agent.policy import Policy


class Actor:
    def __init__(self, id):
        self.id = id

    def is_testing_actor(self):
        """
        only last actor will be the testing actor
        """
        return self.id == P.num_actor - 1

    def interact(self, env, policy: Policy):
        num_episode = 0
        last_obs = env.reset()
        while True:
            total_reward = 0
            epi_step = 0
            start_time = time.time()
            while True:
                action = None  # policy.get_action(last_obs)
                if action is None:
                    action = env.action_space.sample()
                if Funcs.rand_prob() - 0.5 > (self.id / (P.num_actor - 1)):  # epsilon-greedy
                    action = env.action_space.sample()
                if action >= 18:
                    action = 17
                obs, reward, done, info = env.step(action)
                total_reward += reward
                if P.render_dir is not None:
                    env.render(mode="human")

                if not self.is_testing_actor():
                    update = policy.graph.add(last_obs, obs, action, reward)
                    if update:
                        policy.update_prob_function()
                last_obs = obs

                epi_step += 1

                if policy.graph.frames > P.total_frames:
                    return

                if done:
                    Logger.log(f"id{self.id} {epi_step: 4}~{epi_step / (time.time() - start_time):<3.3}fps r{total_reward} n{len(policy.graph.node_feats)}")
                    last_obs = env.reset()
                    num_episode += 1
                    break
