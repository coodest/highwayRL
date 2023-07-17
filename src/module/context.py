import sys
from src.util.tools import IO, Logger


class Context:
    # common
    work_dir = "./"
    asset_dir = work_dir + "assets/"
    out_dir = f"{work_dir}output/{Logger.get_date()}/"
    log_dir = out_dir + "log/"
    summary_dir = out_dir + "summary/"
    model_dir = out_dir + "model/"
    result_dir = out_dir + "result/"
    video_dir = out_dir + "video/"
    env_dir = out_dir + "env/"
    out_dirs = [out_dir, log_dir, result_dir, model_dir, video_dir, env_dir]
    gpus = [0]  # [0, 1]
    prio_gpu = gpus[0]  # first device in gpu list
    start_stage = [0, 1, 2][0]

    # env
    total_frames = [1e7, 5e6, 1e6, 1e5][0]  # default 1e7
    env_type = [
        "maze",  # 0
        "toy_text",  # 1
        "football",  # 2
        "atari",  # 3
        "box_2d",  # 4
        "sokoban",  # 5
        "bullet",  # 6
        "mujoco",  # 7
    ][3]
    render = [False, True][0]  # whether test actor to render the env
    render_every = 5
    env_name = None
    deterministic = True  # env with/without randomness
    random_init_ops = [  # control wheter the env is random initialized
        {"max_random_ops": 0},  # diasble random ops
        {"max_random_ops": 30, "ops_option": [0]},
        {"max_random_ops": 30, "ops_option": "all"},
    ][0]
    num_action_repeats = 1

    # agent
    # agent:actor
    num_actor = len(gpus) * 2
    head_actor = num_actor - 1  # to report and evaluate
    stick_on_graph = 0.0  # ratio to stick on current high-score traj
    target_total_rewrad = None
    average_window = 10
    # agent:policy:projector
    projector_types = [
        "raw",  # 0(tuple of raw obs/state): for football, maze, toy_text, sokoban
        "random_rnn",  # 1： for atari,
        "historical_hash",  # 2： for atari,
        "ae",  # 3: for atari,
        "seq",   # 4: for atari,
    ]
    projector = projector_types[0]
    projected_dim = 8
    projected_hidden_dim = 32
    hashing = [False, True][0]
    # agent:policy:graph
    max_node_draw = 500
    min_traj_reward = None
    gamma = [0.99, 1][1]  # discount factor
    sync_every = 20  # in trajectories
    e_greedy = [0.1, 1]
    max_vp_iter = 1e8  # num or float("inf")


class Profile(Context):
    C = Context
    
    if len(sys.argv) < 1:
        Logger.log("profile_id required")
        exit(0)

    current_profile = sys.argv[1]

    if C.env_type == "football":
        C.num_actor = len(C.gpus) * 16
        C.head_actor = C.num_actor - 1
        C.projector = C.projector_types[0]
        C.target_total_rewrad = 2.0
        C.hashing = False
        C.min_traj_reward = 1.2
        C.gamma = 0.99
        C.e_greedy = [0.1, 1]
        C.deterministic = True
        C.num_action_repeats = 1
        reward_type = ["scoring", "scoring,checkpoints"][1]
    if C.env_type == "atari":
        C.total_frames = [1e7, 5e6, 1e6, 1e5][0]  # default 1e7
        # C.num_actor = len(C.gpus) * 4
        C.num_actor = len(C.gpus) * 8
        C.head_actor = C.num_actor - 1
        # C.projector = C.projector_types[4]
        C.projector = C.projector_types[1]
        C.target_total_rewrad = None
        C.hashing = True
        C.min_traj_reward = None
        C.gamma = [0.99, 1, 1 - 1e-8][1]
        C.num_action_repeats = 4  # equivelent to frame skip
        C.e_greedy = [0.1, 1]
        C.stick_on_graph = 0.0
        max_train_episode_steps = [108000, 1000, 27000][0]
        max_eval_episode_steps = [108000, 1000, 27000][0]
        stack_frames = 4
        screen_size = 84
        sticky_action = False
    
    C.env_name = IO.read_file(f"{C.asset_dir}{C.env_type}.txt")[int(current_profile)]

    C.optimal_graph_path = f'{C.model_dir}{C.env_name}-optimal.pkl'
