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
    total_frames = 1e7  # default 1e7
    env_type = "atari"
    env_name_list = IO.read_file(asset_dir + "Atari_game_list.txt")
    env_name = None
    max_episode_steps = 108000
    max_random_noops = 0  # 30, to control wheter the env is random initialized
    num_action_repeats = 4
    render = False  # whether test actor to render the env
    render_every = 5
    screen_size = 84

    # agent
    num_actor = num_gpu * 8
    head_actor = num_actor - 1  # num_actor - 1, last actor
    obs_min_dis = 0  # 0: turn  off associative memory, 1e-3: distance
    projected_dim = 8
    projected_hidden_dim = 32
    gamma = 0.99
    sync_every = 20  # in second
    projector_types = [None, "random", "cnn", "rnn"]
    projector = projector_types[3]  # select None to disable random projection
    e_greedy = [0.1, 1]
    optimal_graph_path = None
    statistic_crossing_obs = True
    max_vp_iter = 5000  # num or float("inf")


class Profile(Context):
    C = Context

    profiles = dict()
    for i in range(1, 27):
        profiles[i] = str(i)
    
    current_profile = sys.argv[1] if len(sys.argv) > 1 else "1"
    
    for i in range(1, 27):
        if current_profile == str(i):
            C.env_name = C.env_name_list[int(current_profile)]

    C.clean = False

    C.optimal_graph_path = C.model_dir + f'{C.env_name}-optimal.pkl'
