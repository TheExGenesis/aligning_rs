# sampling made more efficient using numba
from utils import flatten, pipe
import numpy as np
from optimized.cyFns import crandint
from numba import njit, jit


# get first degree neighbors of the nodes ids

flattenUnique = pipe(flatten, set, list)


def getNeighbors(graph, ids):
    return flattenUnique([graph.get_all_neighbors(id) for id in ids])
# get a list of for the nth degree neighbors between 0 and n


def getNthNeighbors(graph, id, n):
    nthNeighbors = [[id]] + [[] for i in range(n)]
    for i in range(n):
        nthNeighbors[i+1] = getNeighbors(graph, nthNeighbors[i])
    return nthNeighbors

# Ego Sampling

# %%


def maybeChoice(eligible): return None if not eligible else (
    np.random.choice(eligible))


def mapNth(n): return lambda xs: map(lambda x: x[n], xs)


def sampleStrat(strats, strat):
    def filterStrat(x): return x[1] == strat
    def filterStrats(xs): return filter(filterStrat, xs)
    sampleIt = pipe(enumerate, filterStrats, mapNth(0), list, maybeChoice)
    return sampleIt(strats)


# %%


def isFinished(graph, triedList):
    return len(triedList) == graph.num_vertices()
# sample neighbors of x filtered by a function prioritizing closest neighbors and moving outward in rings if none available

# old less efficient fns
# def filterEgoSample(graph, filterFn, ids):
#     neighbors = getNeighbors(graph, ids)
#     elligibleNeighbors = list(filter(filterFn, neighbors))
#     if not elligibleNeighbors:
#         if isFinished(graph, neighbors):
#             return None
#         return filterEgoSample(graph, filterFn, neighbors)
#     return np.random.choice(elligibleNeighbors)


# # exclude first neighbors
# def filterEgoSampleUnique(graph, filterFn, id):
#     def excludeFirstNeighbors(x): return x not in (
#         [id] + list(graph.get_all_neighbors(id)))
#     return filterEgoSample(graph, lambda x: excludeFirstNeighbors(x) and filterFn(x), [id])


# def sampleEgoStrat(graph, strats, strat, id):
#     return filterEgoSampleUnique(graph, lambda x: strats[x] == strat, id)

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
    ineligible_ids = np.argwhere(np.array(strats) != strat).flatten()
    first_neighbors = set([id]).union(set(graph.get_all_neighbors(id)))
    to_exclude = first_neighbors.union(ineligible_ids)
    def excludeFirstNeighborsAndWrongStrat(x): return not (x in to_exclude)
    return filterEgoSampleUnique(graph, excludeFirstNeighborsAndWrongStrat, id)


''' Neighbor Sampling'''


def sampleNeighbor(graph, id):
    neighbors = graph.get_all_neighbors(id)
    try:
        neigh = np.random.choice(neighbors)
        return neigh
    except:
        return None
# sample one neighbor of node id1 in a graph, excluding id2 and its neighbors


def sampleNeighborUnique(graph, id1, id2):
    neighbors = list(set(graph.get_all_neighbors(id1)) -
                     set(graph.get_all_neighbors(id2)) - set([id2]))
    return np.random.choice(neighbors) if neighbors else None
# sample any second degree neighbor of node id by random sampling a firt degree neighbor and then sampling its neighbors


def sampleSecondNeighbor(graph, id):
    firstNeighbor = sampleNeighbor(graph, id)
    secondNeighbor = sampleNeighbor(graph, firstNeighbor)
    return secondNeighbor
# sample any second degree neighbor of node id excluding id's first degree neighbors


def sampleSecondNeighborUnique(graph, id):
    firstNeighbors = graph.get_all_neighbors(id)
    for fN in np.random.shuffle(firstNeighbors):
        secondNeighbor = sampleNeighborUnique(graph, fN, id)
        if secondNeighbor:
            return secondNeighbor
    return None
# checks if node has only 1 edge


def isLonely(graph, x):
    return graph.vertex(x).out_degree() <= 1
# SIDE EFFECT: replaces the graph's edge (x,y) with edge (x,z)

# neighbors of b eligible for rewiring a to


def eligibleNewFriends(graph, b, a):
    return list(set(graph.get_all_neighbors(b)) - set(graph.get_all_neighbors(a)) - set([a]))
# samples a node of a given strat from the list strats, returns its id

# %%

# old , less efficient
# def sampleStratEligible(graph, strats, strat, x):
#     def filterStrat(x): return x[1] == strat
#     def filterStrats(xs): return filter(filterStrat, xs)
#     ofStrat = pipe(enumerate, filterStrats, mapNth(0), list)(strats)
#     if not ofStrat:
#         return None
#     else:
#         eligible = list(
#             set(ofStrat) - set(graph.get_all_neighbors(x)) - set([x]))
#         if not eligible:
#             return None
#         return np.random.choice(eligible)

# Samples only from a certain strategy


@njit
def eligibleStrat(neighbors, strats, strat, x):
    ofStrat = (np.array(strats) == strat).nonzero()[0]
    if ofStrat.shape[0] == 0:
        return None
    else:
        eligible = list(set(ofStrat) - neighbors - set([x]))
        # if len(eligible) <= 0:
        #     return None
        return eligible


def sampleStratEligible(graph, strats, strat, x):
    eligible = eligibleStrat(
        set(graph.get_all_neighbors(x)), strats, strat, x)
    return eligible[crandint(0, len(eligible)-1)] if eligible and len(eligible) > 0 else None


# Samples only from a certain strategy and from a certain mediator (Xclusive)
@njit
def eligibleStratX(neighbors, strats, medStrats, strat, medStrat, x):
    ofStrat = np.logical_and(
        strats == strat, medStrats == medStrat).nonzero()[0]
    if ofStrat.shape[0] == 0:
        return None
    else:
        eligible = list(set(ofStrat) - neighbors - set([x]))
        # if len(eligible) <= 0:
        #     return None
        return eligible


def sampleStratEligibleX(graph, strats, medStrats, strat, medStrat, x):
    neighbors = set(graph.get_all_neighbors(x))
    eligible = eligibleStratX(neighbors, strats, medStrats, strat, medStrat, x)
    return eligible[crandint(0, len(eligible)-1)] if eligible and len(eligible) > 0 else None
