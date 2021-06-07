# %%
from defaultParams import _N, _m, _c, _gamma, _k
import numpy as np
import os
import graph_tool as gt
import graph_tool.generation as gen
from games import C, D

# %%
'''Init'''


def initStrats(N):
    return np.random.choice([C, D], N)


def initPayoffs(N):
    return np.zeros(N)


'''Graph'''
# returns the number k


def uniform(k):
    return k
# returns a number betwee 1 and k


def randSample(max):
    accept = False
    while not accept:
        k = np.random.randint(1, max+1)
        accept = np.random.random() < 1.0/k
    return k
# returns a new scale free graph


def initScaleFreeGraph(N=_N, m=_m, c=_c, gamma=_gamma):
    return gen.price_network(N=N, m=m, c=c, gamma=gamma, directed=False)
# returns a new random graph


def initRandomGraph(N=_N, k=_k):
    return gt.generation.random_graph(N, lambda x: randSample(k), directed=False)
# returns a new random graph with a uniform degree distribution


def initUniformRandomGraph(N=_N, k=30):
    return gt.generation.random_graph(N, lambda x: uniform(k), directed=False)
# sample one neighbor of node id in a graph, return none if no neighbors
