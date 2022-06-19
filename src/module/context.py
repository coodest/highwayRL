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
    sync_dir = out_dir + "sync/"
    clean = False
    log_every = 10
    gpus = [0]  # [0, 1]
    prio_gpu = gpus[0]  # first device in gpu list

    # env
    total_frames = 1e7  # default 1e7
    env_types = [
        "atari_classic", 
        "atari_historical_action",  # not support projectors
        "atari_ram", 
        "atari_alternative", 
        "simple_scene", 
        "maze"
    ]
    env_type = env_types[2]
    render = False  # whether test actor to render the env
    render_every = 5
    # atari
    env_name_list = IO.read_file(asset_dir + "Atari_game_list.txt")
    env_name = None
    max_episode_steps = 108000
    max_random_noops = 0  # 30, to control wheter the env is random initialized
    num_action_repeats = 4
    screen_size = 84
    # simple_scene
    seq_len = 500

    # agent
    num_actor = len(gpus) * 8
    head_actor = num_actor - 1  # the last actor
    indexer_enabled = True
    obs_min_dis = 0  # indexer_enabled must be True, 0: turn  off associative memory, 1e-3: distance
    projected_dim = 8
    projected_hidden_dim = 32
    use_hash_index = True
    gamma = 0.99
    sync_every = log_every  # in second
    sync_mode = 2  # 0: sync by pipe, 1: sync by file, 2: sync by both pipe and file
    projector_types = [None, "random", "cnn", "rnn"]
    projector = projector_types[1]  # select None to disable random projection
    e_greedy = [0.1, 1]
    optimal_graph_path = None
    statistic_crossing_obs = True
    build_dag = False
    start_over = True  # break loop and start over for adj mat multification
    max_vp_iter = 500  # num or float("inf")
    accessable_rate = 0.2  # minimum rate to treat a state is accessable
    draw_graph = False  # whether draw matplotlib figure for the graph


class Profile(Context):
    C = Context

    profiles = dict()
    for i in range(1, 27):
        profiles[i] = str(i)
    
    current_profile = sys.argv[1] if len(sys.argv) > 1 else "1"
    
    for i in range(1, 27):
        if current_profile == str(i):
            if C.env_type in C.env_types[0:4]:
                C.env_name = C.env_name_list[int(current_profile)]
            if C.env_type in C.env_types[4:6]:
                C.env_name = f"{C.env_type}_original"

    # C.clean = True

    C.optimal_graph_path = C.model_dir + f'{C.env_name}-optimal.pkl'
