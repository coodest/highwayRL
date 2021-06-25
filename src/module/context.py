import sys
from typing import Collection, NamedTuple


class Context:
    # common
    work_dir = "output/"
    log_dir = work_dir + "log/"
    model_dir = work_dir + "model/"
    result_dir = work_dir + "result/"
    clean = True
    log_every = 20

    # env
    total_frames = 1e9
    env_type = "atari"
    env_name = "Pong"
    max_episode_steps = 108000
    max_random_noops = 0
    num_action_repeats = 4
    render_dir = None
    num_action = None

    # agent
    num_actor = 8
    obs_min_dis = 1e-4
    projected_dim = 4
    sync_every = 60  # in second


class Profile(Context):
    C = Context

    profiles = dict()
    profiles[1] = "atari"

    current_profile = sys.argv[1]

    if current_profile == profiles[1]:
        C.num_action = 18
