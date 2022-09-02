from src.util.tools import IO, Logger
from src.module.agent.memory.iterator import Iterator
from src.module.context import Profile as P
from src.util.imports.numpy import np
from collections import defaultdict


class Storage:
    # obs element index
    _obs_node_ind = 0  # corresponding node of the obs
    _obs_step = 1  # position in the node
    # node element index
    _node_obs = 0  # node_obs_list[step<int>], obs hashing, needed for updating obs's node_ind and step
    _node_action = 1  # node_action_list[action_list[action<int>]]  # first action in the action list will be the best action
    _node_reward = 2  # node_reward_list[obs_reward<float>]
    _node_next = 3  # node_next_list[node_dict{node<int>: visit<int>}], corresponding to the next node for last obs's action(s)
    _node_value = 4  # node_value<float>, initialized as the sum of node reward

    def __init__(self, id) -> None:
        super().__init__()
        self.id = id
        self._obs = dict()
        self._crossing_nodes = set()
        self._node = dict()
        self._max_total_reward = dict()
        self._trajs = list()
        self.iterator = Iterator(self.id)

    def obs_action(self, obs: str) -> int:
        node_ind, step = self._obs[obs]
        # algorithm makes the first element the best action
        return self._node[node_ind][Storage._node_action][step][0]

    def obs_exist(self, obs: str) -> bool:
        return obs in self._obs
    
    def obs_size(self) -> int:
        return len(self._obs)

    def obs_node_ind(self, obs):
        return self._obs[obs][Storage._obs_node_ind]

    def obs_is_crossing(self, obs):
        if obs not in self._obs:
            return False
        ind = self._obs[obs][Storage._obs_node_ind]
        if ind in self._crossing_nodes:
            return True
        else:
            return False

    def trajs(self) -> list:
        return self._trajs
    
    def trajs_add(self, traj):
        self._trajs.append(traj)

    def node_value(self, node):
        return self._node[node][Storage._node_value]

    def node_update(self, node_ind: int, obs: list, actions: list, reward: list, next: list):
        node_value = sum(reward)  # node is treated as a whole and discount factor is not considerred within the node
        self._node[node_ind] = [obs, actions, reward, next, node_value]
        # add or update obs
        for ind, o in enumerate(obs):
            # list object reduces a lot mem comsuption
            self._obs[o] = [node_ind, ind]

    def node_next_ind(self):
        return len(self._node)

    def node_next_accessable(self, node):
        next_nodes = []
        node_dict_list = self._node[node][Storage._node_next]
        for node_dict in node_dict_list:
            total_visit = sum(node_dict.values())
            for n in node_dict:
                if node_dict[n] / total_visit > P.min_accessable_prob:
                    if n is not None:  # remove the end of the traj.
                        next_nodes.append(n)
        return next_nodes

    def node_add(self, obs: list, action: list, reward: list, next: list):
        # exising node, return 
        if obs[0] in self._obs:
            return self._obs[obs[0]][Storage._obs_node_ind]
        # add new node
        node_ind = self.node_next_ind()
        self.node_update(node_ind, obs, action, reward, next)
        return node_ind

    def node_print(self):
        for i in self._node:
            obs, action, reward, next, value = self._node[i]
            Logger.log(f"{'Cnode' if i in self._crossing_nodes else 'Rnode'}:{i} obs:{obs} action:{action} reward:{reward} next:{next} value:{value}")
            
    def node_split(self, crossing_obs, reward=None) -> int:
        """
        split of the shrunk node
        """
        # 1. deal with non-exist crossing obs (traj. self loop), add empty crossing node
        if not self.obs_exist(crossing_obs):
            crossing_node_ind = self.node_add([crossing_obs], [[]], [reward], [])
            self._crossing_nodes.add(crossing_node_ind)
            return crossing_node_ind

        # 2. collect node info
        node_ind = self._obs[crossing_obs][Storage._obs_node_ind]
        step = self._obs[crossing_obs][Storage._obs_step]
        node_obs_list = self._node[node_ind][Storage._node_obs]
        node_action_list = self._node[node_ind][Storage._node_action]
        node_reward_list = self._node[node_ind][Storage._node_reward]
        node_next_list = self._node[node_ind][Storage._node_next]
        node_length = len(node_obs_list)

        # 3. existing node
        if node_length <= 1:
            if node_ind not in self._crossing_nodes:
                # convert non-splittable road node to crossing node
                self._crossing_nodes.add(node_ind)
            return node_ind

        # 4. node split
        if step <= 0:  # crossing node is the first obs
            new_node_ind = self.node_next_ind()
            self.node_update(
                new_node_ind, 
                node_obs_list[step + 1:], 
                node_action_list[step + 1:], 
                node_reward_list[step + 1:], 
                node_next_list
            )

            crossing_node_ind = node_ind
            self.node_update(
                crossing_node_ind,
                [node_obs_list[step]], 
                [node_action_list[step]], 
                [node_reward_list[step]], 
                [{new_node_ind: 1}]
            )
            self._crossing_nodes.add(crossing_node_ind)
        elif step >= node_length - 1:  # crossing node is the last one
            crossing_node_ind = self.node_next_ind()
            self.node_update(
                crossing_node_ind,
                [node_obs_list[step]], 
                [node_action_list[step]], 
                [node_reward_list[step]], 
                node_next_list
            )
            self._crossing_nodes.add(crossing_node_ind)

            self.node_update(
                node_ind, 
                node_obs_list[: step], 
                node_action_list[: step], 
                node_reward_list[: step], 
                [{crossing_node_ind: 1}]
            )
        else:  # otherwise, crossing node between first and last
            new_node_ind = self.node_next_ind()
            self.node_update(
                new_node_ind, 
                node_obs_list[step + 1:], 
                node_action_list[step + 1:], 
                node_reward_list[step + 1:], 
                node_next_list
            )
            
            crossing_node_ind = self.node_next_ind()
            self.node_update(
                crossing_node_ind,
                [node_obs_list[step]], 
                [node_action_list[step]], 
                [node_reward_list[step]], 
                [{new_node_ind: 1}]
            )
            self._crossing_nodes.add(crossing_node_ind)

            self.node_update(
                node_ind, 
                node_obs_list[: step], 
                node_action_list[: step], 
                node_reward_list[: step], 
                [{crossing_node_ind: 1}]
            )
        
        return crossing_node_ind

    def node_value_propagate(self):
        """
        GNN-based value propagation
        """
        total_nodes = len(self._node)
        if total_nodes == 0:
            return
        adj = np.zeros([total_nodes, total_nodes], dtype=np.int8)
        rew = np.zeros([total_nodes], dtype=np.float32)
        val_0 = np.zeros([total_nodes], dtype=np.float32)
        colume_sum = np.zeros([total_nodes], dtype=np.int8)
        for node in self._node:
            rew[node] = sum(self._node[node][Storage._node_reward])
            val_0[node] = self._node[node][Storage._node_value]
            for n in self.node_next_accessable(node):
                adj[node][n] = 1
                colume_sum[n] += 1
        m1 = (np.sum(np.where(colume_sum > 1, 1, 0)) / total_nodes) * 100
        m5 = (np.sum(np.where(colume_sum > 5, 1, 0)) / total_nodes) * 100
        m10 = (np.sum(np.where(colume_sum > 10, 1, 0)) / total_nodes) * 100
        # to dertermine the graph memory is more like a graph or more like a tree
        # all 0% means it is a tree
        Logger.log(f"{m1:>4.1f}%|{m5:>4.1f}%|{m10:>4.1f}% nodes with >1|>5|>10 merging trails", color="yellow")  
        
        # value propagation
        if P.build_dag:
            adj = adj - self.iterator.build_dag(adj)
        val_n, iters, divider = self.iterator.iterate(adj, rew, val_0)
        Logger.log(f"learner value propagation: {iters} iters * {divider} batch", color="yellow")
        for ind, val in enumerate(val_n):
            self._node[ind][Storage._node_value] = val

    def node_connection_dict(self, deterministic=False):
        d = defaultdict(list)
        for n in self._node:
            node_dict_list = self._node[n][Storage._node_next]
            for node_dict in node_dict_list:
                if deterministic:
                    d[n] += [list(node_dict.keys())[0]]
                else:
                    d[n] += list(node_dict.keys())
        return d

    def crossing_nodes(self):
        return self._crossing_nodes

    def crossing_node_add_action(self, node_ind: int, action: int, next_node_ind: int) -> None:
        # corssing node only has one obs and one action list
        crossing_node_action_list = self._node[node_ind][Storage._node_action][0]
        if action in crossing_node_action_list:
            ind = crossing_node_action_list.index(action)  
            node_dict = self._node[node_ind][Storage._node_next][ind]
            if next_node_ind in node_dict:
                # existing action and next node
                node_dict[next_node_ind] += 1
            else:
                # env may (near) stochastical OR conflict obs from different states exist
                node_dict[next_node_ind] = 1
        else:
            # add action if not exist
            self._node[node_ind][Storage._node_action][0].append(action)
            self._node[node_ind][Storage._node_next].append({next_node_ind: 1})
        
        # TEMP: sanity check: state-action integrety
        self.sanity_crossing(node_ind, action, next_node_ind)

    def sanity_crossing(self, node, action, next_node):
        ms = self

        def find_p5(obs):
            a = np.where(np.array(obs, dtype=np.int8).reshape(10, -1) == 5)
            return [a[0][0], a[1][0]]

        error = False

        obs = ms._node[node][Storage._node_obs][0]
        if next_node is not None:
            next_obs = ms._node[next_node][Storage._node_obs][0]
        else:
            # print("node is none")
            next_obs = obs

        base_p5 = find_p5(obs)
        next_p5 = find_p5(next_obs)

        if action == 0:
            if next_p5[1] == base_p5[1] and (next_p5[0] == base_p5[0] or next_p5[0] == base_p5[0] - 1):
                pass
            else:
                print(f"{node} -{action}-> {next_node}")
                error = True
        if action == 1:
            if next_p5[1] == base_p5[1] and (next_p5[0] == base_p5[0] or next_p5[0] == base_p5[0] + 1):
                pass
            else:
                print(f"{node} -{action}-> {next_node}")
                error = True
        if action == 2:
            if next_p5[0] == base_p5[0] and (next_p5[1] == base_p5[1] or next_p5[1] == base_p5[1] - 1):
                pass
            else:
                print(f"{node} -{action}-> {next_node}")
                error = True
        if action == 3:
            if next_p5[0] == base_p5[0] and (next_p5[1] == base_p5[1] or next_p5[1] == base_p5[1] + 1):
                pass
            else:
                print(f"{node} -{action}-> {next_node}")
                error = True

        if error:
            Logger.log(f"error: {error}")

    def crossing_node_action_update(self):
        for crossing_node_ind in self._crossing_nodes:
            max_action_value = - float("inf")
            target_ind = None
            for ind, node_dict in enumerate(self._node[crossing_node_ind][Storage._node_next]):
                # compute the value of the action
                action_value = None  
                total_visit = sum(node_dict.values())
                for node in node_dict:
                    if node is not None:  # filter traj. end
                        prob = node_dict[node] / total_visit
                        if prob <= P.min_accessable_prob:  # filter low prob trail and fake trail (conflict obs)
                            continue
                        node_value = self._node[node][Storage._node_value]
                        if np.isnan(node_value):
                            node_value = self._node[node][Storage._node_value] = 0
                            # assert np.isnan(node_value), "nan in node values"
                        if action_value is None:
                            action_value = node_value * prob  # wighted value to select the action
                        else:
                            action_value += node_value * prob
                if action_value is None:  # obs that cannot update
                    continue
                
                if action_value > max_action_value:
                    max_action_value = action_value
                    target_ind = ind
            # NOTE: crossing node is the last obs in the traj before done (done obs no stored)
            # or the current obs is in conflict
            if target_ind is None:  
                continue
            
            # move the best action to the head of the lists
            node_action_list = self._node[crossing_node_ind][Storage._node_action][0]  # crossing node only has one list for one obs
            node_next_list = self._node[crossing_node_ind][Storage._node_next]

            target_action = node_action_list[target_ind]
            target_next = node_next_list[target_ind]

            node_action_list.remove(target_action)
            node_next_list.remove(target_next)

            node_action_list.insert(0, target_action)
            node_next_list.insert(0, target_next)
    
    def crossing_node_size(self):
        return len(self._crossing_nodes)
    
    def max_total_reward_update(self, total_reward, init_obs):
        current = - float('inf') if len(self._max_total_reward) == 0 else self.max_total_reward()
        if total_reward > current:
            self._max_total_reward.clear()
            self._max_total_reward[total_reward] = init_obs

    def max_total_reward(self):
        if len(self._max_total_reward) == 0:
            return - float("inf")
        else:
            return list(self._max_total_reward.keys())[0]

    def max_total_reward_init_obs(self):
        if len(self._max_total_reward) == 0:
            return "none"
        else:
            return self._max_total_reward[
                self.max_total_reward()
            ]
