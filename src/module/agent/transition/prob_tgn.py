import math
from src.module.context import Profile as P

from src.module.agent.transition.evaluation.evaluation import eval_edge_prediction
from src.module.agent.transition.model.tgn import TGN
from src.module.agent.transition.utils.utils import RandEdgeSampler, get_neighbor_finder
from src.module.agent.transition.utils.data_processing import *
import torch

from src.util.tools import *

args = P.tgn


class ProbTGN:
    def __init__(self):
        # 1. preparation
        torch.manual_seed(0)
        np.random.seed(0)
        # paths
        IO.make_dir(f"{P.model_dir}/tgn")
        self.model_save_path = f'{P.model_dir}/tgn/{args.prefix}.pth'

        # Set device
        device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

        # Initialize Model
        tgn = TGN(neighbor_finder=None, node_features=np.zeros([1, 1]),
                  edge_features=np.zeros([1, 1]), device=device,
                  n_layers=args.n_layer,
                  n_heads=args.n_head, dropout=args.drop_out, use_memory=args.use_memory,
                  message_dimension=args.message_dim, memory_dimension=args.memory_dim,
                  memory_update_at_start=not args.memory_update_at_end,
                  embedding_module_type=args.embedding_module,
                  message_function=args.message_function,
                  aggregator_type=args.aggregator,
                  memory_updater_type=args.memory_updater,
                  n_neighbors=args.n_neighbors,
                  mean_time_shift_src=None, std_time_shift_src=None,
                  mean_time_shift_dst=None, std_time_shift_dst=None,
                  use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                  use_source_embedding_in_message=args.use_source_embedding_in_message,
                  dyrep=args.dyrep
                  )
        tgn.n_node_features = args.memory_dim
        tgn.n_nodes = args.memory_size
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(tgn.parameters(), lr=args.lr)
        self.tgn = tgn.to(device)

        if args.use_memory:
            self.tgn.memory.__init_memory__()

    def train(self, graph):
        # Extract data for training, validation and testing
        node_features, edge_features = [np.array(graph.node_feats), np.array(graph.edge_feats)]
        train_data = Data(np.array(graph.src), np.array(graph.dst), np.array(graph.ts), np.array(graph.idx), np.array(graph.label))
        full_data = Data(np.array(graph.src), np.array(graph.dst), np.array(graph.ts), np.array(graph.idx), np.array(graph.label))
        num_instance = len(train_data.sources)
        num_batch = math.ceil(num_instance / args.bs)
        Logger.log('num of training instances: {}'.format(num_instance))
        Logger.log('num of batches per epoch: {}'.format(num_batch))
        # Compute time statistics
        mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
            compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

        # Initialize validation and test neighbor finder to retrieve temporal graph
        full_ngh_finder = get_neighbor_finder(full_data, args.uniform)
        # update tgn
        self.tgn.mean_time_shift_src = mean_time_shift_src
        self.tgn.std_time_shift_src = std_time_shift_src
        self.tgn.mean_time_shift_dst = mean_time_shift_dst
        self.tgn.std_time_shift_dst = std_time_shift_dst
        self.tgn.neighbor_finder = full_ngh_finder
        self.tgn.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(self.tgn.device)
        self.tgn.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(self.tgn.device)
        self.tgn.embedding_module.node_features = self.tgn.node_raw_features
        self.tgn.embedding_module.edge_features = self.tgn.edge_raw_features
        
        # Initialize negative samplers. Set seeds for validation and testing so negatives are the same
        # across different runs
        # NB: in the inductive setting, negatives are sampled only amongst other new nodes
        train_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations)
        test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations)

        train_losses = []
        start_time = time.time()
        for epoch in range(args.n_epoch):
            m_loss = []
            for k in range(0, num_batch, args.backprop_every):
                loss = 0
                self.optimizer.zero_grad()

                # Custom loop to allow to perform backpropagation only every a certain number of batches
                for j in range(args.backprop_every):
                    batch_idx = k + j

                    if batch_idx >= num_batch:
                        continue

                    start_idx = batch_idx * args.bs
                    end_idx = min(num_instance, start_idx + args.bs)
                    sources_batch, destinations_batch = \
                        train_data.sources[start_idx:end_idx], \
                        train_data.destinations[start_idx:end_idx]
                    edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
                    timestamps_batch = train_data.timestamps[start_idx:end_idx]

                    size = len(sources_batch)
                    _, negatives_batch = train_rand_sampler.sample(size)

                    with torch.no_grad():
                        pos_label = torch.ones(size, dtype=torch.float, device=device)
                        neg_label = torch.zeros(size, dtype=torch.float, device=device)

                    tgn = self.tgn.train()

                    pos_prob, neg_prob = tgn.compute_edge_probabilities(
                        sources_batch,
                        destinations_batch,
                        negatives_batch,
                        timestamps_batch,
                        edge_idxs_batch,
                        args.n_neighbors
                    )

                    loss += self.criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

                loss /= args.backprop_every

                loss.backward()
                self.optimizer.step()
                m_loss.append(loss.item())

                # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
                # the start of time
                if args.use_memory:
                    self.tgn.memory.detach_memory()

            # # Validation/ Inference
            # tgn.set_neighbor_finder(full_ngh_finder)
            # val_ap, val_auc = eval_edge_prediction(
            #     model=tgn,
            #     negative_edge_sampler=val_rand_sampler,
            #     data=val_data,
            #     n_neighbors=args.n_neighbors
            # )

            train_losses.append(np.mean(m_loss))
        Logger.log('Training pass time: {:.2f}s'.format(time.time() - start_time))
        Logger.log('Mean loss: {}'.format(np.mean(train_losses)))

        # Save results for this run
        Logger.log('Saving TGN model')
        torch.save(self.tgn.state_dict(), self.model_save_path)
        Logger.log('TGN model saved')
