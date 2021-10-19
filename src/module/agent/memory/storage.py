class Storeage(dict):
    def __init__(self) -> None:
        super().__init__()
        self.max_value = -float("inf")
        self.max_value_init_obs = None
        self.crossing_obs = set()


class OptimalStorage(Storeage):
    def __init__(self) -> None:
        super().__init__()

    def update_max(self, value, obs):
        if value > self.max_value:
            self.max_value = value
            self.max_value_init_obs = obs


class TransitionStorage(Storeage):
    def __init__(self) -> None:
        super().__init__()
        self.ends = set()

    def update_max(self, value):
        if value > self.max_value:
            self.max_value = value

    def add_end(self, end):
        self.ends.add(end)


class StorageCell:
    def __init__(self, action) -> None:
        pass


class OptimalStorageCell(StorageCell):
    def __init__(self, action, total_action) -> None:
        super().__init__(action)
        self.best_action = action
        self.total_reward = total_action


class TransitionStorageCell(StorageCell):
    def __init__(self, action, parents, reward) -> None:
        super().__init__(action)
        self.action = action
        self.parents = parents
        self.reward = reward
