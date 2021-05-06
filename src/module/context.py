class Context:
    # common
    work_dir = "/mnt/shard/Builds/memrl/"
    log_dir = work_dir + "log/"
    model_dir = work_dir + "model/"
    output_dir = work_dir + "output/"

    # env
    num_episodes = 1000
    env_type = "atari"
    env_name = "Pong"
    max_episode_steps = 108000
    max_random_noops = 0
    num_action_repeats = 2
    render_dir = None


class Profile(Context):
    base = Context
    profile = ""
