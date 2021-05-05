class Actor:
    @staticmethod
    def interact(env, policy):
        env.reset()
        while True:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

            policy.graph.add_node(obs, action, reward)

            if done:
                break
