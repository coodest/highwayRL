import sys
from src.util.tools import IO


class Context:
    # common
    work_dir = './'
    asset_dir = work_dir + "assets/"
    out_dir = work_dir + "output/"
    log_dir = out_dir + "log/"
    summary_dir = out_dir + "summary/"
    model_dir = out_dir + "model/"
    result_dir = out_dir + "result/"
    video_dir = out_dir + "video/"
    sync_dir = out_dir + "sync/"
    env_dir = out_dir + "env/"
    out_dirs = [out_dir, log_dir, model_dir, result_dir, video_dir, sync_dir, env_dir]
    log_every = 20
    gpus = [0]  # [0, 1]
    prio_gpu = gpus[0]  # first device in gpu list

    # env
    total_frames = [1e7, 2e5, 4e7][0]  # default 1e7
    env_types = [
        "atari",  # 0
        "maze",  # 1
        "toy_text",  # 2
        "box_2d",  # 3
        "sokoban",  # 4
        "football",  # 5
        "bullet",  # 6
    ]
    env_type = env_types[0]
    render = False  # whether test actor to render the env
    render_every = 5
    # atari
    env_name = None
    max_train_episode_steps = [108000, 1000, 27000][0]
    max_eval_episode_steps = [108000, 1000, 27000][0]
    max_random_ops = [0, 10, 20, 30][0]  # 30, to control wheter the env is random initialized
    random_init_ops = [  # control wheter the env is random initialized
        {"max_random_ops": 0},  # diasble random ops
        {"max_random_ops": 30, "ops_option": [0]},
        {"max_random_ops": 30, "ops_option": "all"},
    ][0]
    num_action_repeats = 4  # equivelent to frame skip
    stack_frames = 2
    screen_size = 84
    sticky_action = False

    # agent
    num_actor = len(gpus) * 8
    head_actor = num_actor - 1  # the last actor
    # agent:projector
    projector = [
        "raw",  # 0
        "random",  # 1
        "rnn",  # 2
        "n-rnn",  # 3
        "sha256_hash",  # 4
        "multiple_hash",  # 5
    ][2]
    projected_dim = 8
    projected_hidden_dim = 32
    # agent:graph
    alpha = 1.0
    gamma = [0.99, 1][1]  # discount factor
    sync_every = log_every  # in second
    e_greedy = [0.1, 1]
    optimal_graph_path = None
    statistic_crossing_obs = True
    build_dag = False
    start_over = True  # break loop and start over for adj mat multification
    max_vp_iter = 1e8  # num or float("inf")
    min_accessable_prob = 0  # minimum prob. to treat a state is accessable
    draw_graph = False  # whether draw matplotlib figure for the graph
    graph_sanity_check = True


class Profile(Context):
    C = Context

    profiles = dict()
    for i in range(1, 27):
        profiles[i] = str(i)
    
    current_profile = sys.argv[1] if len(sys.argv) > 1 else "1"
    
    if C.env_type in C.env_types[0]:
        C.env_name_list = IO.read_file(f"{C.asset_dir}atari.txt")
        C.env_name = C.env_name_list[int(current_profile)]
    if C.env_type in C.env_types[1]:
        C.env_name = f"{C.env_type}_original"
    if C.env_type in C.env_types[2]:
        C.env_name_list = IO.read_file(f"{C.asset_dir}toy_text.txt")
        C.env_name = C.env_name_list[int(current_profile)]
    if C.env_type in C.env_types[3]:
        C.env_name_list = IO.read_file(f"{C.asset_dir}box_2d.txt")
        C.env_name = C.env_name_list[int(current_profile)]
    if C.env_type in C.env_types[4]:
        C.env_name_list = IO.read_file(f"{C.asset_dir}sokoban.txt")
        C.env_name = C.env_name_list[int(current_profile)]

    C.optimal_graph_path = C.model_dir + f'{C.env_name}-optimal.pkl'
