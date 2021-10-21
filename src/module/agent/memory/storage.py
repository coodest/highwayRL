from src.util.imports.numpy import np


class Storage:
    # base cell index
    best_action = 0
    afiliated_traj = 1
    # trajectory_infos cell index
    total_reward = 0
    init_obs = 1
    action = 2
    reward = 3

    def __init__(self) -> None:
        super().__init__()
        self.obs = dict()
        self.max_total_reward = -float("inf")
        self.max_total_reward_init_obs = None
        self.crossing_obs = set()
        self.traj = dict()

    def obs_best_action(self, obs):
        return self.obs[obs][Storage.best_action]

    def obs_exist(self, obs):
        return obs in self.obs

    def obs_add(self, obs):
        self.obs[obs] = [None, dict()]  # list obj reduces mem comsuption

    def obs_update(self, obs, action, afiliated_traj, step):
        self.obs_update_action(obs, action)
        self.obs_update_afiliated_traj(obs, afiliated_traj, step)

    def obs_update_action(self, obs, action):
        self.obs[obs][Storage.best_action] = action

    def obs_update_afiliated_traj(self, obs, afiliated_traj, step):
        self.obs[obs][Storage.afiliated_traj][afiliated_traj] = step
    
    def obs_dict(self):
        return self.obs
    
    def obs_afiliated_traj(self, obs):
        return self.obs[obs][Storage.afiliated_traj]

    def crossing_obs_add(self, obs):
        self.crossing_obs.add(obs)

    def crossing_obs_union(self, new_crossing_obs):
        self.crossing_obs = self.crossing_obs.union(new_crossing_obs)

    def crossing_obs_size(self):
        return len(self.crossing_obs)

    def traj_exist(self, traj_ind):
        return traj_ind in self.traj
    
    def traj_add_action(self, traj_ind, action):
        self.traj[traj_ind][Storage.action].append(action)

    def traj_add_reward(self, traj_ind, step, reward):
        self.traj[traj_ind][Storage.reward][step] = reward

    def traj_add(self, traj_ind, total_reward, init_obs, traj_len):
        self.traj[traj_ind] = [
            total_reward,
            init_obs,
            list(),
            np.zeros(traj_len)
        ]
    
    def traj_init_obs(self, traj_ind):
        return self.traj[traj_ind][Storage.init_obs]

    def traj_total_reward(self, traj_ind):
        return self.traj[traj_ind][Storage.total_reward]

    def total_reward_update(self, total_reward, init_obs):
        if total_reward > self.max_total_reward:
            self.max_total_reward = total_reward
            self.max_total_reward_init_obs = init_obs

    def get_max_total_reward(self):
        return self.max_total_reward

    def get_max_total_reward_init_obs(self):
        return self.max_total_reward_init_obs
