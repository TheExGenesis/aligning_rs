from itertools import product, combinations
import numpy as np

# network params
_k = 30
N = _N = 500
_m = 2
_c = 0
_gamma = 1

# game params
_R = 1
_P = 0
_T = 2  # T€[0,2]
_S = -1  # S€[-1,1]


def genTSParams(M):
    t_ = np.linspace(0, 2, M)
    s_ = np.linspace(-1, 1, M)
    _ts = list(product(t_, s_))
    return _ts


# evo params
_episode_n = 100
_te = 1
_ta = 0.5
_W = _te/_ta
_W2 = 0
_beta = 0.001
