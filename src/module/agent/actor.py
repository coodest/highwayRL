from src.module.context import Profile as P
from src.util.tools import *


class Actor:
    def __init__(self, id):
        self.id = id

    def is_testing_actor(self):
        """
        only last actor will be the testing actor
        """
        return self.id == P.num_actor - 1

    def interact(self, env, policy):
        last_obs = env.reset()
        num_episode = 0
        total_reward = 0
        while num_episode < P.num_episodes:
            action = policy.get_action(last_obs)
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

            if done:
                Logger.log(f"id: {self.id} Total reward: {total_reward}")
                last_obs = env.reset()
                num_episode += 1
                continue
