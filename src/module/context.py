import argparse


class Context:
    """
    the control panel of options with default values
    """
    # common
    work_dir = "./"
    asset_dir = work_dir + "assets/"
    out_dir = None
    log_dir = None
    summary_dir = None
    model_dir = None
    result_dir = None
    video_dir = None
    env_dir = None
    out_dirs = None
    dataset_dir = None
    gpus = [0]  # [0, 1]
    prio_gpu = gpus[0]  # first device in gpu list
    stages = [
        [False, True][1],
        [False, True][0],
    ]
    wandb_enabled = [False, True][0]  # if enabled, './wandb_key file must exist and valid
    summary_enabled = [False, True][0]

    # env
    total_frames = None  # default 1e7
    env_type = None
    render = [False, True][0]  # whether test actor to render the env, disable no-respoding-dialog: gsettings set org.gnome.mutter check-alive-timeout 60000
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
    num_actor = len(gpus) * 8
    head_actor = num_actor - 1  # to report and evaluate
    target_total_rewrad = None
    average_window = 10
    # agent:policy:projector
    projector_types = [
        "raw",  # 0(tuple of raw obs/state): for football, maze, toy_text
        "random_rnn",  # 1： for atari,
        "historical_hash",  # 2： for atari,
        "ae",  # 3: for atari,
        "seq",   # 4: for atari,
        "linear",  # 5: for football
    ]
    projector = projector_types[0]
    projected_dim = 8
    projected_hidden_dim = 32
    hashing = None
    save_transition = [False, True][0]
    log_every = 1
    # agent:policy:graph
    load_graph = False
    max_node_draw = 500
    min_traj_reward = None
    reward_filter_ratio = None
    gamma = None  # discount factor
    sync_every = 10  # in trajectories
    sync_increase = 1  # multipled to sync_every after every sync
    e_greedy = [0.1, 1]
    max_vp_iter = 1e8  # num or float("inf")
    income_types = ["total_reward", "expected_return"]
    income = income_types[0]
    stick_on_modes = ["ratio", "value"]
    stick_on_mode = stick_on_modes[0]
    stick_on_graph = [0., 0.]  # default not stick on
    stick_on_graph_inc = 0.
    negative_reward_filter = False
    # agent:policy:dnn
    dnn_types = [
        "dqn",
        "dqn-q",
    ]
    dnn = None
    batch_size = None
    lr = None
    early_stop = None


class Profile(Context):
    """
    the profile of runs with customized options
    """
    
    C = Context

    parser = argparse.ArgumentParser(description='HighwayRL')
    parser.add_argument('--run', default=0, help='game index')
    parser.add_argument('--env_name', default="", help='env name')
    parser.add_argument('--keep_dir', action='store_true', help='renew output dir or not')
    parser.add_argument('--env_type', default=C.env_type, choices=[
        "maze",  # 0
        "toy_text",  # 1
        "football",  # 2
        "atari",  # 3
    ], help='type of the env')
    args, unk_args = parser.parse_known_args()

    run = args.run
    keep_dir = [args.keep_dir, True][0]
    C.env_name = args.env_name
    C.env_type = str(args.env_type)
    C.out_dir = f"{C.work_dir}output/{C.env_type}-{C.env_name}/run-{run}/"
    C.log_dir = C.out_dir + "log/"
    C.summary_dir = C.out_dir + "summary/"
    C.model_dir = C.out_dir + "model/"
    C.result_dir = C.out_dir + "result/"
    C.video_dir = C.out_dir + "video/"
    C.env_dir = C.out_dir + "env/"
    C.dataset_dir = C.out_dir + "dataset/"
    C.out_dirs = [C.out_dir, C.log_dir, C.result_dir, C.model_dir, C.video_dir, C.env_dir, C.dataset_dir]

    if C.env_type == "maze":
        C.total_frames = [1e6, 1e8][0]
        C.num_actor = len(C.gpus) * 8
        C.head_actor = C.num_actor - 1
        C.projector = C.projector_types[0]
        C.gamma = 0.99
        C.hashing = False
        C.deterministic = True
        C.sync_every = 10
        sync_increase = 2
        C.log_every = 1
        max_train_episode_steps = [2000, 4000, 6000, 10000][2]
        max_eval_episode_steps = [2000, 4000, 6000, 10000][2]
        C.dnn = C.dnn_types[1]
        C.total_epoch = 1000
        C.batch_size = 32
        C.lr = 1e-2
        C.early_stop = 0.90
    if C.env_type == "toy_text":
        C.total_frames = [1e6, 1e8][0]
        C.num_actor = len(C.gpus) * 8
        C.head_actor = C.num_actor - 1
        C.projector = C.projector_types[0]
        C.gamma = 0.99
        C.hashing = False
        C.deterministic = True
        C.sync_every = 20
        sync_increase = 5
        C.log_every = 500
        C.render = False
        max_episode_steps = [200, 400, 600, 1000][0]
        C.dnn = C.dnn_types[1]
        C.total_epoch = 20000
        C.batch_size = 48
        C.lr = 1e-4
        C.early_stop = float("inf")
    if C.env_type == "football":
        C.total_frames = [1e6, 1e5][0]
        C.num_actor = len(C.gpus) * 8
        C.head_actor = C.num_actor - 1
        C.projector = C.projector_types[5]
        C.projected_dim = 128
        C.sync_every = 50
        C.target_total_rewrad = 2.0
        C.hashing = False
        C.min_traj_reward = 1.1
        C.gamma = 0.99
        C.e_greedy = [0.1, 0.4]
        C.income = C.income_types[1]
        C.deterministic = True
        C.num_action_repeats = 1
        C.dnn = C.dnn_types[1]
        C.total_epoch = 20000
        C.batch_size = 256
        C.lr = 1e-4
        C.early_stop = 1.95
        C.stick_on_graph = [0.3, 0.3]
        reward_type = ["scoring", "scoring,checkpoints"][1]
    if C.env_type == "atari":
        C.total_frames = [1e7, 5e6, 1e6, 1e5][2]  # default 1e7
        C.num_actor = len(C.gpus) * 8
        C.head_actor = C.num_actor - 1
        C.sync_every = 2
        C.projector = C.projector_types[1]
        C.target_total_rewrad = None
        C.hashing = False
        C.deterministic = True
        C.min_traj_reward = None
        C.gamma = [0.99, 1, 1 - 1e-8, 0.999999][3]
        C.num_action_repeats = 4  # equivelent to frame skip
        C.e_greedy = [[0.1, 1], [0.1, 0.9]][0]
        C.reward_filter_ratio = None
        C.dnn = C.dnn_types[1]
        C.total_epoch = 1000
        C.batch_size = 2560
        C.lr = 1e-4
        C.early_stop = float("inf")
        if C.env_name == "Pong":
            C.stick_on_mode = C.stick_on_modes[1]
        else:
            C.stick_on_mode = C.stick_on_modes[0]
        C.stick_on_graph = [0.75, 0.9]  # starting/end ratio to stick on current high-score traj
        C.stick_on_graph_inc = [0.0, 5e-3][1]  # increaments of 'stick_on_graph' after each epi.
        C.sticky_action = False
        C.negative_reward_filter = True
        max_train_episode_steps = [108000, 1000, 27000][0]
        max_eval_episode_steps = [108000, 1000, 27000][0]
        stack_frames = 4
        screen_size = 84
