from modules import TwoLayerModule
from solvers.commnet import Solver

config = {
    'n_bandits': 5,
    'player_pool_size': 500,
    'd_player_embed': 128,
    'n_epoch': int(1e6),
    'n_steps': 2,
    'channels': [128, 128],
    'module': TwoLayerModule,
    'learning_rate': 1e-3
}

haha = Solver(**config)
haha.train()