class StrategyManager:
    def __init__(self):
        self.strategies = {}

    def add_strategy(self, name, config):
        self.strategies[name] = config

    def get_strategies(self):
        return list(self.strategies.keys())

    def activate_strategy(self, name):
        if name in self.strategies:
            # Implement activation logic here
            return True
        return False
