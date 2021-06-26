import sys


class Context:
    # common
    out_dir = "output/"
    log_dir = out_dir + "log/"
    model_dir = out_dir + "model/"
    result_dir = out_dir + "result/"
    video_dir = out_dir + "video/"
    clean = True
    log_every = 20
    num_gpu = 2
    prio_gpu = 1

    # env
    total_frames = 0.5e7
    env_type = "atari"
    env_name = "StarGunner"  # StarGunner, Pong
    max_episode_steps = 108000
    max_random_noops = 0
    num_action_repeats = 4
    render = False  # whether test actor to render the env
    render_every = 5
    num_action = None

    # agent
    num_actor = 16
    actor_read_timeout = 20
    obs_min_dis = 1e-3
    projected_dim = 8
    sync_every = 60  # in second
    e_greedy = [0.5, 1]


class Profile(Context):
    C = Context

    profiles = dict()
    profiles[1] = "atari"

    current_profile = sys.argv[1]

    if current_profile == profiles[1]:
        C.num_action = 18
