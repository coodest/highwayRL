import math
from src.module.context import Profile as P
from src.util.tools import *

from src.module.agent.transition.evaluation.evaluation import eval_edge_prediction
from src.module.agent.transition.model.tgn import TGN
from src.module.agent.transition.utils.utils import RandEdgeSampler, get_neighbor_finder
from src.module.agent.transition.utils.data_processing import get_data, compute_time_statistics
import torch


class ProbTGN:
    @staticmethod
    def main_func(graph):
        # 1. preparation
        torch.manual_seed(0)
        np.random.seed(0)
        # Argument and global variables
        args = P.tgn
        # paths
        IO.make_dir(f"{P.model_dir}/tgn")
        model_save_path = f'{P.model_dir}/tgn/{args.prefix}.pth'

        # Extract data for training, validation and testing
        node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = get_data(
            graph,
            different_new_nodes_between_val_and_test=args.different_new_nodes,
            randomize_features=args.randomize_features
        )
        num_instance = len(train_data.sources)
        num_batch = math.ceil(num_instance / args.bs)
        Logger.log('num of training instances: {}'.format(num_instance))
        Logger.log('num of batches per epoch: {}'.format(num_batch))

        # Initialize training neighbor finder to retrieve temporal graph
        train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

        # Initialize validation and test neighbor finder to retrieve temporal graph
        full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

        # Initialize negative samplers. Set seeds for validation and testing so negatives are the same
        # across different runs
        # NB: in the inductive setting, negatives are sampled only amongst other new nodes
        train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)

        # Set device
        device_string = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
        device = torch.device(device_string)

        # Compute time statistics
        mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
            compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

        # Initialize Model
        tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
                  edge_features=edge_features, device=device,
                  n_layers=args.n_layer,
                  n_heads=args.n_head, dropout=args.drop_out, use_memory=args.use_memory,
                  message_dimension=args.message_dim, memory_dimension=args.memory_dim,
                  memory_update_at_start=not args.memory_update_at_end,
                  embedding_module_type=args.embedding_module,
                  message_function=args.message_function,
                  aggregator_type=args.aggregator,
                  memory_updater_type=args.memory_updater,
                  n_neighbors=args.n_neighbors,
                  mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
                  mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
                  use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                  use_source_embedding_in_message=args.use_source_embedding_in_message,
                  dyrep=args.dyrep)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(tgn.parameters(), lr=args.lr)
        tgn = tgn.to(device)

        train_losses = []
        start_time = time.time()

        for epoch in range(args.n_epoch):
            # Training
            # Reinitialize memory of the model at the start of each epoch
            if args.use_memory:
                tgn.memory.__init_memory__()

            # Train using only training graph
            tgn.set_neighbor_finder(train_ngh_finder)
            m_loss = []

            Logger.log('start {} epoch'.format(epoch))
            for k in range(0, num_batch, args.backprop_every):
                loss = 0
                optimizer.zero_grad()

                # Custom loop to allow to perform backpropagation only every a certain number of batches
                for j in range(args.backprop_every):
                    batch_idx = k + j

                    if batch_idx >= num_batch:
                        continue

                    start_idx = batch_idx * args.bs
                    end_idx = min(num_instance, start_idx + args.bs)
                    sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                                        train_data.destinations[start_idx:end_idx]
                    edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
                    timestamps_batch = train_data.timestamps[start_idx:end_idx]

                    size = len(sources_batch)
                    _, negatives_batch = train_rand_sampler.sample(size)

                    with torch.no_grad():
                        pos_label = torch.ones(size, dtype=torch.float, device=device)
                        neg_label = torch.zeros(size, dtype=torch.float, device=device)

                    tgn = tgn.train()

                    pos_prob, neg_prob = tgn.compute_edge_probabilities(
                        sources_batch,
                        destinations_batch,
                        negatives_batch,
                        timestamps_batch,
                        edge_idxs_batch,
                        args.n_neighbors
                    )

                    loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

                loss /= args.backprop_every

                loss.backward()
                optimizer.step()
                m_loss.append(loss.item())

                # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
                # the start of time
                if args.use_memory:
                    tgn.memory.detach_memory()

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
        torch.save(tgn.state_dict(), model_save_path)
        Logger.log('TGN model saved')
