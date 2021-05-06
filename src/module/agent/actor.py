from src.module.context import Profile as P


class Actor:
    @staticmethod
    def interact(env, policy):
        env.reset()
        while True:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if P.render_dir is not None:
                env.render(mode="human")

            policy.graph.add_node(obs, action, reward)

            if done:
                break
