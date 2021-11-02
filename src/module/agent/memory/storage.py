from src.util.tools import IO, Logger
from src.module.agent.memory.iterator import Iterator
import numpy as np


class Storage:
    # obs element index
    _obs_node_ind = 0  # corresponding node of the obs
    _obs_step = 1  # position in the node
    # node element index
    _node_obs = 0  # obs hashing, needed for updating obs's node_ind and step
    _node_actions = 1  # list[list[int]]
    _node_reward = 2  # list[float]
    _node_next = 3  # list[int], corresponding to the next node for last obs's action
    _node_value = 4  # initialized as the sum of node reward

    def __init__(self) -> None:
        super().__init__()
        self._obs = dict()
        self._crossing_nodes = set()
        self._node = dict()
        self._max_total_reward = dict()
        self._trajs = list()

    def obs_action(self, obs: str) -> int:
        node_ind, step = self._obs[obs]
        # algorithm makes the first element the best action
        return self._node[node_ind][Storage._node_actions][step][0]

    def obs_exist(self, obs: str) -> bool:
        return obs in self._obs
    
    def obs_size(self) -> int:
        return len(self._obs)

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

    def node_update(self, node_ind: int, obs: list, actions: list, reward: list, next: list):
        node_value = sum(reward)
        self._node[node_ind] = [obs, actions, reward, next, node_value]
        # add or update obs
        for ind, o in enumerate(obs):
            if o in self._crossing_nodes:  # to solve obs-multi-node issue
                if self._obs[o][Storage._obs_node_ind] != node_ind:
                    # TODO: env may (near) stochastical OR obs cannot indicate corresponding state
                    pass
                continue
            # list object reduces a lot mem comsuption
            self._obs[o] = [node_ind, ind]

    def node_next_ind(self):
        return len(self._node)
    
    def node_add(self, obs: list, action: list, reward: list, next: list):
        # exising node, return 
        if obs[0] in self._obs:
            return self._obs[obs[0]][Storage._obs_node_ind]
        # add new node
        node_ind = self.node_next_ind()
        self.node_update(node_ind, obs, action, reward, next)
        return node_ind
            
    def node_split(self, crossing_obs: str) -> int:
        """
        split of the shrunk node
        """
        # 1. collect node info
        node_ind = self._obs[crossing_obs][Storage._obs_node_ind]
        step = self._obs[crossing_obs][Storage._obs_step]
        node_obs = self._node[node_ind][Storage._node_obs]
        node_actions = self._node[node_ind][Storage._node_actions]
        node_reward = self._node[node_ind][Storage._node_reward]
        node_next = self._node[node_ind][Storage._node_next]
        node_length = len(node_obs)

        # 2. existing crossing node, do nothing
        if node_length <= 1:  
            return node_ind

        # 3. node split
        if step <= 0:  # crossing node is the first obs
            new_node_ind = self.node_next_ind()
            self.node_update(
                new_node_ind, 
                node_obs[step + 1:], 
                node_actions[step + 1:], 
                node_reward[step + 1:], 
                node_next
            )

            crossing_node_ind = node_ind
            self.node_update(
                crossing_node_ind,
                [node_obs[step]], 
                [node_actions[step]], 
                [node_reward[step]], 
                [new_node_ind]
            )
            self._crossing_nodes.add(crossing_node_ind)
        elif step >= node_length - 1:  # crossing node is the last one
            crossing_node_ind = self.node_next_ind()
            self.node_update(
                crossing_node_ind,
                [node_obs[step]], 
                [node_actions[step]], 
                [node_reward[step]], 
                node_next
            )
            self._crossing_nodes.add(crossing_node_ind)

            self.node_update(
                node_ind, 
                node_obs[: step], 
                node_actions[: step], 
                node_reward[: step], 
                [crossing_node_ind]
            )
        else:  # otherwise, crossing node between first and last
            new_node_ind = self.node_next_ind()
            self.node_update(
                new_node_ind, 
                node_obs[step + 1:], 
                node_actions[step + 1:], 
                node_reward[step + 1:], 
                node_next
            )
            
            crossing_node_ind = self.node_next_ind()
            self.node_update(
                crossing_node_ind,
                [node_obs[step]], 
                [node_actions[step]], 
                [node_reward[step]], 
                [new_node_ind]
            )
            self._crossing_nodes.add(crossing_node_ind)

            self.node_update(
                node_ind, 
                node_obs[: step], 
                node_actions[: step], 
                node_reward[: step], 
                [crossing_node_ind]
            )

        # NOTE: obs of different shrunk nodes may have the same obs in any crossing node, one obs may stored in many different nodes.
        # assert str(crossing_obs).startswith(self._node[crossing_node_ind][Storage._node_obs][0]), f"obs not match, {crossing_obs} - {self._node[crossing_node_ind][Storage._node_obs][0]}"
        # assert str(crossing_obs).startswith(self._node[self._obs[crossing_obs][Storage._obs_node_ind]][Storage._node_obs][0]), f"obs not match, {crossing_obs} - {self._node[crossing_node_ind][Storage._node_obs][0]}"
        # assert crossing_node_ind == self._obs[crossing_obs][Storage._obs_node_ind], f"_obs update failed, new {crossing_node_ind} - queue {self._obs[crossing_obs][Storage._obs_node_ind]} - origin {node_ind}"
        
        return crossing_node_ind

    def node_value_propagate(self):
        """
        GNN-based value propagation
        """
        total_nodes = len(self._node)
        adj = np.zeros([total_nodes, total_nodes], dtype=np.int8)
        rew = np.zeros([total_nodes], dtype=np.float32)
        val_0 = np.zeros([total_nodes], dtype=np.float32)
        for node in self._node:
            next = self._node[node][Storage._node_next]
            rew[node] = sum(self._node[node][Storage._node_reward])
            val_0[node] = self._node[node][Storage._node_value]
            for n in next:
                if n is None:
                    continue
                adj[node][n] = 1
        
        iterator = Iterator()
        val_n = iterator.iterate(adj, rew, val_0)
        for ind, val in enumerate(val_n):
            self._node[ind][Storage._node_value] = val

    def crossing_node_add_action(self, node_ind: int, action: int, next_node_ind: int) -> None:
        try:
            ind = self._node[node_ind][Storage._node_actions][0].index(action)
            if self._node[node_ind][Storage._node_next][ind] == next_node_ind:
                # existing action and next node
                return
            else:
                # TODO: env may (near) stochastical OR obs cannot indicate corresponding state 
                pass
        except ValueError:
            self._node[node_ind][Storage._node_actions][0].append(action)
            self._node[node_ind][Storage._node_next].append(next_node_ind)

    def crossing_node_action_update(self):
        for crossing_node_ind in self._crossing_nodes:
            next_nodes = self._node[crossing_node_ind][Storage._node_next]
            if next_nodes == [None]:  # NOTE: crossing node is the last obs in the traj before done (done obs no stored)
                continue

            max_next_node_value = - float("inf")
            target_ind = None
            for next_ind in range(len(next_nodes)):
                next_node_ind = next_nodes[next_ind]
                next_node_value = self._node[next_node_ind][Storage._node_value]
                if next_node_value > max_next_node_value:
                    max_next_node_value = next_node_value
                    target_ind = next_ind
            
            # make pointer to the lists
            action_list = self._node[crossing_node_ind][Storage._node_actions][0]
            next_node_list = self._node[crossing_node_ind][Storage._node_next]

            target_action = action_list[target_ind]
            target_next = next_node_list[target_ind]

            action_list.remove(target_action)
            next_node_list.remove(target_next)

            action_list.insert(0, target_action)
            next_node_list.insert(0, target_next)
    
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
