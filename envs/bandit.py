import numpy as np


class Bandit:
    def __init__(self, n_bandit):
        self.n_bandit = n_bandit

    def reset(self):
        pass

    def step(self, action):
        assert action.shape[0] == self.n_bandit
        return np.sum(action)

