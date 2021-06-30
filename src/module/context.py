import sys


class Context:
    # common
    work_dir = './'
    out_dir = work_dir + "output/"
    log_dir = out_dir + "log/"
    model_dir = out_dir + "model/"
    result_dir = out_dir + "result/"
    video_dir = out_dir + "video/"
    clean = True
    log_every = 20
    num_gpu = 1
    prio_gpu = 1

    # env
    total_frames = 1e7
    env_type = "atari"
    num_action = 18
    env_name = "Pong"  # StarGunner, Pong
    max_episode_steps = 108000
    max_random_noops = 30
    num_action_repeats = 4
    render = False  # whether test actor to render the env
    render_every = 5
    num_action = None

    # agent
    num_actor = 8
    obs_min_dis = 1e-3
    projected_dim = 8
    sync_every = 60  # in second
    sync_modes = ['highest', 'confident', 'mixed']  # highest to max R, confident to max hit rate 
    sync_tolerance = 0.8
    sync_mode = sync_modes[2]
    e_greedy = [0.5, 1]
    add_obs = True  # false: last_obs-prev_action pairs, True: last_obs-prev_action-obs triple


class Profile(Context):
    C = Context

    profiles = dict()
    profiles[1] = "1"
    profiles[2] = "2"
    profiles[3] = "3"
    profiles[4] = "4"
    profiles[5] = "5"
    profiles[6] = "6"
    profiles[7] = "7"
    profiles[8] = "8"

    current_profile = sys.argv[1]

    if current_profile == profiles[1]:
        clean = True
    if current_profile == profiles[2]:
        clean = False
    if current_profile == profiles[3]:
        clean = False
