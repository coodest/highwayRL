from src.module.context import Profile as P
from src.util.tools import *
from src.module.agent.policy import Policy


class Actor:
    def __init__(self, id):
        self.id = id  # actor identifier

    def is_testing_actor(self):
        """
        only last actor will be the testing actor
        """
        return self.id == P.num_actor - 1

    def interact(self, env):
        num_episode = 0
        last_obs = env.reset()
        while True:
            total_reward = 0
            epi_step = 0
            start_time = time.time()
            while True:
                # 1. make action
                action = policy.get_action(last_obs)
                if action is None:
                    action = env.action_space.sample()

                # 2. epsilon-greedy
                if Funcs.rand_prob() - 0.5 > (self.id / (P.num_actor - 1)):  # epsilon-greedy
                    action = env.action_space.sample()

                # 3. interact
                obs, reward, done, info = env.step(action)

                # 4. graph memory ops
                if not self.is_testing_actor():
                    update = policy.graph.add(last_obs, obs, action, reward, env.action_space.n)
                    if update:
                        policy.update_prob_function()
                last_obs = obs

                # 5. post step
                total_reward += reward
                epi_step += 1
                if P.render_dir is not None:
                    env.render(mode="human")

                # 6. stop check
                if policy.graph.frames > P.total_frames:
                    return

                # 7. done ops
                if done:
                    Logger.log(f"id{self.id} {epi_step: 4}~{epi_step / (time.time() - start_time):<3.3}fps r{total_reward} n{len(policy.graph.node_feats)}")
                    last_obs = env.reset()
                    num_episode += 1
                    break
