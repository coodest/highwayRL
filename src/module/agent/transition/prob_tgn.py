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
        torch.manual_seed(0)
        np.random.seed(0)
        # paths
        IO.make_dir(f"{P.model_dir}/tgn")
        self.model_save_path = f'{P.model_dir}/tgn/{args.prefix}.pth'

        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
        self.tgn = None
        self.criterion = None
        self.optimizer = None

    def init_tgn(
            self,
            neighbor_finder,
            node_features,
            edge_features,
            mean_time_shift_src,
            std_time_shift_src,
            mean_time_shift_dst,
            std_time_shift_dst
    ):
        tgn = TGN(
            neighbor_finder=neighbor_finder,
            node_features=node_features,
            edge_features=edge_features, device=self.device,
            n_layers=args.n_layer,
            n_heads=args.n_head, dropout=args.drop_out,
            use_memory=args.use_memory,
            message_dimension=args.message_dim,
            memory_dimension=args.memory_dim,
            memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module,
            message_function=args.message_function,
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=args.n_neighbors,
            mean_time_shift_src=mean_time_shift_src,
            std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst,
            std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            dyrep=args.dyrep
        )
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(tgn.parameters(), lr=args.lr)
        self.tgn = tgn.to(self.device)

        if args.use_memory:
            self.tgn.memory.__init_memory__()

    def train(self, data):
        # extract data for training, validation and testing
        node_features, edge_features, train_data = data
        num_instance = len(train_data.sources)
        num_batch = math.ceil(num_instance / args.bs)
        # Compute time statistics
        mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = compute_time_statistics(train_data.sources, train_data.destinations, train_data.timestamps)
        # initialize validation and test neighbor finder to retrieve temporal graph
        train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
        # init or update tgn
        if self.tgn is None:
            self.init_tgn(train_ngh_finder, node_features, edge_features, mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst)
        else:
            self.tgn.mean_time_shift_src = mean_time_shift_src
            self.tgn.std_time_shift_src = std_time_shift_src
            self.tgn.mean_time_shift_dst = mean_time_shift_dst
            self.tgn.std_time_shift_dst = std_time_shift_dst
            self.tgn.neighbor_finder = train_ngh_finder
            self.tgn.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(self.tgn.device)
            self.tgn.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(self.tgn.device)
            self.tgn.embedding_module.node_features = self.tgn.node_raw_features
            self.tgn.embedding_module.edge_features = self.tgn.edge_raw_features
            self.tgn.embedding_module.neighbor_finder = train_ngh_finder
            self.tgn.n_nodes = self.tgn.node_raw_features.shape[0]
            self.tgn.memory.increase_memory(self.tgn.node_raw_features.shape[0])

        # initialize negative samplers. Set seeds for validation and testing so negatives are the same
        # across different runs
        # NB: in the inductive setting, negatives are sampled only amongst other new nodes
        negative_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)

        train_losses = []
        start_time = time.time()
        for epoch in range(args.n_epoch):
            m_loss = []
            for k in range(0, num_batch, args.backprop_every):
                loss = 0
                self.optimizer.zero_grad()

                # custom loop to allow to perform backpropagation only every a certain number of batches
                for j in range(args.backprop_every):
                    batch_idx = k + j

                    if batch_idx >= num_batch:
                        continue

                    start_idx = batch_idx * args.bs
                    end_idx = min(num_instance, start_idx + args.bs)
                    sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], train_data.destinations[start_idx:end_idx]
                    edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
                    timestamps_batch = train_data.timestamps[start_idx:end_idx]

                    size = len(sources_batch)
                    _, negatives_batch = negative_rand_sampler.sample(size)
                    assert len(node_features) >= np.max(negatives_batch)

                    with torch.no_grad():
                        pos_label = torch.ones(size, dtype=torch.float, device=self.device)
                        neg_label = torch.zeros(size, dtype=torch.float, device=self.device)

                    tgn = self.tgn.train()

                    pos_prob, neg_prob = tgn.compute_edge_probabilities(
                        sources_batch,
                        destinations_batch,
                        negatives_batch,
                        timestamps_batch,
                        edge_idxs_batch,
                        args.n_neighbors
                    )

                    loss += self.criterion(pos_prob.squeeze(), pos_label) + self.criterion(neg_prob.squeeze(),
                                                                                           neg_label)

                loss /= args.backprop_every

                loss.backward()
                self.optimizer.step()
                m_loss.append(loss.item())

                # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
                # the start of time
                if args.use_memory:
                    self.tgn.memory.detach_memory()

            train_losses.append(np.mean(m_loss))
        Logger.log("S{} E{} T{:.2f}s Mean loss{:.6f}".format(
            num_instance,
            num_batch,
            time.time() - start_time,
            np.mean(train_losses)
        ))

        # # Save results for this run
        # Logger.log('Saving TGN model')
        # torch.save(self.tgn.state_dict(), self.model_save_path)
        # Logger.log('TGN model saved')

    def test(self, data):
        mem_backup = self.tgn.memory.backup_memory()
        with torch.no_grad():
            tgn = self.tgn.eval()
            # use existing neighbor finder
            pos_prob, neg_prob = tgn.compute_edge_probabilities(
                data.sources,
                data.destinations,
                data.sources,
                data.timestamps,
                data.edge_idxs,
                args.n_neighbors
            )
        self.tgn.memory.restore_memory(mem_backup)
        return pos_prob
