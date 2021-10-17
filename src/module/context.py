import sys
from src.util.tools import IO


class Context:
    # common
    work_dir = './'
    out_dir = work_dir + "output/"
    asset_dir = work_dir + "assets/"
    log_dir = out_dir + "log/"
    model_dir = out_dir + "model/"
    result_dir = out_dir + "result/"
    video_dir = out_dir + "video/"
    clean = False
    log_every = 20
    num_gpu = 1
    prio_gpu = 0

    # env
    total_frames = 1e7
    env_type = "atari"
    env_name_list = IO.read_file(asset_dir + "Atari_game_list.txt")
    env_name = None
    max_episode_steps = 108000
    max_random_noops = 30
    num_action_repeats = 4
    render = False  # whether test actor to render the env
    render_every = 5
    num_action = 18

    # agent
    num_actor = num_gpu * 8
    obs_min_dis = 1e-3
    projected_dim = 4
    gamma = 0.99
    sync_every = 20  # in second
    sync_modes = ['highest', 'confident', 'mixed']  # highest to max R, confident to max hit rate 
    sync_tolerance = 0.7  # work with mixed mode
    sync_mode = sync_modes[2]
    graph_types = ['transition', 'optimal']
    graph_type = graph_types[1]
    projector_types = ["random", "cnn"]
    projector = projector_types[0]
    e_greedy = [0.1, 1]
    optimal_graph_path = None
    # transition graph
    num_bp = 0  # set 0 to disable back value propagation
    k_nearest_neighbor = 5
    soft_merge = True


class Profile(Context):
    C = Context

    profiles = dict()
    for i in range(1, 27):
        profiles[i] = str(i)

    current_profile = sys.argv[1]
    
    for i in range(1, 27):
        if current_profile == str(i):
            C.env_name = C.env_name_list[int(current_profile)]

    C.clean = True

    C.optimal_graph_path = C.model_dir + f'{C.env_name}-optimal.pkl'
