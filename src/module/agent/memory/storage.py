class Storeage(dict):
    def __init__(self) -> None:
        super().__init__()
        self.max_value = - float('inf')
        self.max_value_init_obs = None
        

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

    def update_max(self, value):
        if value > self.max_value:
            self.max_value = value
