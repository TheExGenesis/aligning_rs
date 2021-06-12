# %%
from utils import flatten, pipe
from init import *
from mediators import *
from optimized.cyFns import *
import numpy as np
N = 500
graph = initUniformRandomGraph(N)
strats = initStrats(N)
x, y = crandint(0, N-1), crandint(0, N-1)
# %%

%timeit useNoMed(graph, strats, y, x)
%timeit useGoodMed(graph, strats, y, x)
%timeit useBadMed(graph, strats, y, x)
%timeit useRandomMed(graph, strats, y, x)
%timeit useFairMed(graph, strats, y, x)
# %%
%timeit useLocalGoodMed(graph, strats, y, x)
%timeit useLocalBadMed(graph, strats, y, x)
%timeit useLocalRandomMed(graph, strats, y, x)
%timeit useLocalFairMed(graph, strats, y, x)

# %%
%timeit sampleStratEligible(graph, strats, strats[y], x)

# %%


# get first degree neighbors of the nodes ids
flattenUnique = pipe(flatten, set, list)


def isFinished(graph, triedList):
    return len(triedList) == graph.num_vertices()


def getNeighbors(graph, ids):
    return flattenUnique([graph.get_all_neighbors(id) for id in ids])


def filterEgoSample(graph, filterFn, ids):
    rng = np.random.default_rng()
    rng.shuffle(ids)  # shuffle ids to check the neighbors for each randomly
    total_neighbors = []
    for id in ids:
        neighbors = graph.get_all_neighbors(id)
        eligible = list(filter(filterFn, neighbors))
        if len(eligible) > 0:  # if eligible, take a random one
            return eligible[crandint(0, len(eligible)-1)]
        else:  # if none eligible, add neighbors to the total pool
            np.concatenate((total_neighbors, neighbors), axis=None)
            continue
    if len(total_neighbors) == graph.num_vertices():
        return None
    return filterEgoSample(graph, filterFn, neighbors)


# exclude first neighbors
def filterEgoSampleUnique(graph, filterFn, id):
    first_neighbors = set([id]).union(set(graph.get_all_neighbors(id)))
    def excludeFirstNeighbors(x): return not x in first_neighbors
    return filterEgoSample(graph, lambda x: excludeFirstNeighbors(x) and filterFn(x), [id])


def sampleEgoStrat(graph, strats, strat, id):
    ineligible_ids = np.argwhere(strats != strat).flatten()
    first_neighbors = set([id]).union(set(graph.get_all_neighbors(id)))
    to_exclude = first_neighbors.union(ineligible_ids)
    def excludeFirstNeighborsAndWrongStrat(x): return not (x in to_exclude)
    return filterEgoSampleUnique(graph, excludeFirstNeighborsAndWrongStrat, id)


# %%
%timeit filterEgoSample(graph, lambda z: strats[z] == strats[y], [x])
%timeit sampleEgoStrat(graph, strats, strats[x], y)
# %%
