import time
from src.util.tools import Funcs, IO
from src.module.context import Profile as P



class Memory:
    def __init__(self) -> None:
        self.max_value = None
        self.max_value_init_obs = None
        self.obs_to_action = dict()

    def keys(self):
        return self.obs_to_action.keys()

    def values(self):
        return self.obs_to_action.values()

    def __getitem__(self, key):
        return self.obs_to_action[key]

    def __setitem__(self, key, value):
        self.obs_to_action[key] = value
        if self.max_value is None:
            self.max_value = value[1]
            self.max_value_init_obs = key
        elif value[1] > self.max_value:
            self.max_value = value[1]
            self.max_value_init_obs = key

    

class OptimalGraph:
    def __init__(self, id, is_head) -> None:
        self.id = id
        self.is_head = is_head
        # dict of observation to action with value [action, reward]
        self.oa = Memory()
        self.increments = Memory()
        self.last_sync = time.time()

    def get_action(self, obs):
        if obs in self.oa:
            return self.oa[obs][0]
        else:
            return None

    def store_increments(self, trajectory, total_reward):
        for last_obs, pre_action in trajectory:
            if last_obs not in self.oa:
                self.increments[last_obs] = [pre_action, total_reward]
            elif self.oa[last_obs][1] < total_reward:
                self.increments[last_obs] = [pre_action, total_reward]

    def sync(self):
        if not self.is_head:
            # write increments (not head)
            IO.write_disk_dump(P.result_dir + f'{self.id}.pkl', self.increments)
            self.increments = Memory()
            self.oa = IO.stick_read_disk_dump(P.result_dir + 'target.pkl')
            IO.write_disk_dump(P.result_dir + f'{self.id}.finish', ["finish"])
        else:
            # read increments (head)
            for i in range(P.num_actor):
                if self.id == i:
                    continue  # head has no increments stored
                inc = IO.stick_read_disk_dump(P.result_dir + f'{i}.pkl')
                for last_obs in inc:
                    if last_obs not in self.oa:
                        self.oa[last_obs] = inc[last_obs]
                    elif self.oa[last_obs][1] < inc[last_obs][1]:
                        self.oa[last_obs] = inc[last_obs]

            # write target (head)
            IO.write_disk_dump(P.result_dir + 'target.pkl', self.oa)

            # remove increments (head)
            for i in range(P.num_actor):
                if self.id == i:
                    continue
                IO.stick_read_disk_dump(P.result_dir + f'{i}.finish')

            IO.renew_dir(P.result_dir)
            IO.write_disk_dump(P.model_dir + 'optimal.pkl', self.oa)
