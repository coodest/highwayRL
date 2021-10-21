from src.util.imports.numpy import np


class Storage:
    # base cell index
    _best_action = 0
    _afiliated_traj = 1
    # trajectory_infos cell index
    _total_reward = 0
    _init_obs = 1
    _action = 2
    _reward = 3

    def __init__(self) -> None:
        super().__init__()
        self._obs = dict()
        self._max_total_reward = -float("inf")
        self._max_total_reward_init_obs = None
        self._crossing_obs = set()
        self._traj = dict()

    def obs_best_action(self, obs):
        return self._obs[obs][Storage._best_action]

    def obs_exist(self, obs):
        return obs in self._obs

    def obs_add(self, obs):
        self._obs[obs] = [None, dict()]  # list obj reduces mem comsuption

    def obs_update(self, obs, action, afiliated_traj, step):
        self.obs_update_action(obs, action)
        self.obs_update_afiliated_traj(obs, afiliated_traj, step)

    def obs_update_action(self, obs, action):
        self._obs[obs][Storage._best_action] = action

    def obs_update_afiliated_traj(self, obs, afiliated_traj, step):
        self._obs[obs][Storage._afiliated_traj][afiliated_traj] = step
    
    def obs_dict(self):
        return self._obs
    
    def obs_afiliated_traj(self, obs):
        return self._obs[obs][Storage._afiliated_traj]

    def crossing_obs_add(self, obs):
        self._crossing_obs.add(obs)

    def crossing_obs_union(self, new_crossing_obs):
        self._crossing_obs = self._crossing_obs.union(new_crossing_obs)

    def crossing_obs_set(self):
        return self._crossing_obs

    def traj_exist(self, traj_ind):
        return traj_ind in self._traj
    
    def traj_add_action(self, traj_ind, action):
        self._traj[traj_ind][Storage._action].append(action)

    def traj_add_reward(self, traj_ind, step, reward):
        self._traj[traj_ind][Storage._reward][step] = reward

    def traj_add(self, traj_ind, total_reward, init_obs, traj_len):
        self._traj[traj_ind] = [
            total_reward,
            init_obs,
            list(),
            np.zeros(traj_len)
        ]

    def traj_dict(self):
        return self._traj
    
    def traj_init_obs(self, traj_ind):
        return self._traj[traj_ind][Storage._init_obs]

    def traj_total_reward(self, traj_ind):
        return self._traj[traj_ind][Storage._total_reward]

    def total_reward_update(self, total_reward, init_obs):
        if total_reward > self._max_total_reward:
            self._max_total_reward = total_reward
            self._max_total_reward_init_obs = init_obs

    def max_total_reward_value(self):
        return self._max_total_reward

    def max_total_reward_init_obs_value(self):
        return self._max_total_reward_init_obs
