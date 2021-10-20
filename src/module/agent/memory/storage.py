class Storeage(dict):
    def __init__(self) -> None:
        super().__init__()
        self.max_value = -float("inf")
        self.max_value_init_obs = None
        self.crossing_obs = set()


class OptimalStorage(Storeage):
    # base cell index
    best_action = 0
    afiliated_trajectories = 1
    # trajectory_infos cell index
    value = 0
    init_obs = 1
    action = 2
    reward = 3

    def __init__(self) -> None:
        super().__init__()
        self.trajectory_infos = dict()

    def update_max(self, value, obs):
        if value > self.max_value:
            self.max_value = value
            self.max_value_init_obs = obs


class TransitionStorage(Storeage):
    # base cell index
    action = 0
    parents = 1
    reward = 2

    def __init__(self) -> None:
        super().__init__()
        self.ends = set()

    def update_max(self, value):
        if value > self.max_value:
            self.max_value = value

    def add_end(self, end):
        self.ends.add(end)

