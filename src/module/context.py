class TGNArgs:
    bs = 200  # Batch_size
    prefix = "tgn-attn"  # Prefix to name the checkpoints
    n_neighbors = 10  # Number of neighbors to sample
    n_head = 2  # Number of heads used in attention layer
    n_epoch = 50  # Number of epochs
    n_layer = 1  # Number of network layers
    lr = 0.0001  # Learning rate
    drop_out = 0.1  # Dropout probability
    gpu = 0  # Idx for the gpu to use
    backprop_every = 1  # Every how many batches to back propagate gradient
    use_memory = True  # Whether to augment the model with a node memory
    embedding_module = "graph_attention"  # Type of embedding module: ["graph_attention", "graph_sum", "identity", "time"]
    message_function = "identity"  # Type of message function: ["mlp", "identity"]
    memory_updater = "gru"  # Type of memory updater: ["gru", "rnn"]
    aggregator = "last"  # Type of message
    memory_update_at_end = False  # Whether to update memory at the end or at the start of the batch
    message_dim = 100  # Dimensions of the messages
    memory_dim = 82  # Dimensions of the memory for each user
    different_new_nodes = False  # Whether to use disjoint set of new nodes for train and val
    uniform = False  # take uniform sampling from temporal neighbors
    randomize_features = False  # Whether to randomize node features
    use_destination_embedding_in_message = False  # Whether to use the embedding of the destination node as part of the message
    use_source_embedding_in_message = False  # Whether to use the embedding of the source node as part of the message
    dyrep = False  # Whether to run the dyrep model


class Context:
    # common
    work_dir = "/mnt/shard/Builds/MemRL/"
    code_dir = work_dir + "code/"
    log_dir = work_dir + "log/"
    model_dir = work_dir + "model/"
    output_dir = work_dir + "output/"
    clean = False

    # env
    num_episodes = 10
    env_type = "atari"
    env_name = "Pong"
    max_episode_steps = 108000
    max_random_noops = 0
    num_action_repeats = 2
    render_dir = None

    # agent
    # tgn
    tgn = TGNArgs


class Profile(Context):
    base = Context
    profile = ""
