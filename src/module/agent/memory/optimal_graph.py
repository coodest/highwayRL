import time

from gym.core import ObservationWrapper
from src.util.tools import IO
from src.module.context import Profile as P


class Memory(dict):
    def __init__(self) -> None:
        super().__init__()
        self.max_value = None
        self.max_value_init_obs = None
    
    def update_max(self, value, init_obs):
        if self.max_value is None:
            self.max_value = value
            self.max_value_init_obs = init_obs
        elif value > self.max_value:
            self.max_value = value
            self.max_value_init_obs = init_obs


class OptimalGraph:
    def __init__(self, id, is_head) -> None:
        self.id = id
        self.is_head = is_head
        # dict of observation to action with value [action, reward]
        self.main = Memory()
        self.increments = Memory()
        self.last_sync = time.time()

    def get_action(self, obs):
        if obs in self.main:
            return self.main[obs][0]
        else:
            return None

    def store_increments(self, trajectory, total_reward):
        for last_obs, pre_action, obs in trajectory:
            if P.add_obs:
                info = [pre_action, total_reward, obs]
            else:
                info = [pre_action, total_reward]

            if last_obs not in self.main:
                self.increments[last_obs] = info
                self.increments.update_max(total_reward, last_obs)
            elif self.main[last_obs][1] < total_reward:
                self.increments[last_obs] = info
                self.increments.update_max(total_reward, last_obs)

    def sync(self):
        if not self.is_head:
            # write increments (not head)
            IO.write_disk_dump(P.result_dir + f'{self.id}.pkl', self.increments)
            self.increments = Memory()
            self.main = IO.stick_read_disk_dump(P.result_dir + 'target.pkl')
            IO.write_disk_dump(P.result_dir + f'{self.id}.finish', ["finish"])
        else:
            # read increments (head)
            for i in range(P.num_actor):
                if self.id == i:
                    continue  # head has no increments stored
                inc = IO.stick_read_disk_dump(P.result_dir + f'{i}.pkl')
                for last_obs in inc:
                    if last_obs not in self.main:
                        self.main[last_obs] = inc[last_obs]
                        self.main.update_max(inc[last_obs][1], last_obs)
                    elif self.main[last_obs][1] < inc[last_obs][1]:
                        self.main[last_obs] = inc[last_obs]
                        self.main.update_max(inc[last_obs][1], last_obs)

            # write target (head)
            IO.write_disk_dump(P.result_dir + 'target.pkl', self.main)

            # remove increments (head)
            for i in range(P.num_actor):
                if self.id == i:
                    continue
                IO.stick_read_disk_dump(P.result_dir + f'{i}.finish')

            IO.renew_dir(P.result_dir)
            IO.write_disk_dump(P.model_dir + 'optimal.pkl', self.main)
