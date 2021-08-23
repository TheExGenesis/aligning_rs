from utils import flatten, pipe
import numpy as np
from optimized.cyFns import crandint, randFloat
from numba import njit, jit

# get first degree neighbors of the nodes ids

flattenUnique = pipe(flatten, set, list)


def getNeighbors(graph, ids):
    return flattenUnique([list(graph.neighbors(id)) for id in ids])
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


def filterEgoSample(graph, filterFn, ids):
    rng = np.random.default_rng()
    rng.shuffle(ids)  # shuffle ids to check the neighbors for each randomly
    total_neighbors = []
    for id in ids:
        neighbors = list(graph.neighbors(id))
        eligible = list(filter(filterFn, neighbors))
        if len(eligible) > 0:  # if eligible, take a random one
            return eligible[crandint(0, len(eligible)-1)]
        else:  # if none eligible, add neighbors to the total pool
            np.concatenate((total_neighbors, neighbors), axis=None)
            continue
    if len(total_neighbors) == graph.number_of_nodes():
        return None
    return filterEgoSample(graph, filterFn, neighbors)


# exclude first neighbors
def filterEgoSampleUnique(graph, filterFn, id):
    first_neighbors = set([id]).union(set(graph.neighbors(id)))
    def excludeFirstNeighbors(x): return not x in first_neighbors
    return filterEgoSample(graph, lambda x: excludeFirstNeighbors(x) and filterFn(x), [id])


def sampleEgoStrat(graph, strats, strat, id):
    ineligible_ids = (np.array(strats) != strat).nonzero()[0]
    first_neighbors = set([id]).union(set(graph.neighbors(id)))
    to_exclude = first_neighbors.union(ineligible_ids)
    def excludeFirstNeighborsAndWrongStrat(x): return not (x in to_exclude)
    return filterEgoSampleUnique(graph, excludeFirstNeighborsAndWrongStrat, id)


''' Neighbor Sampling'''


def sampleNeighbor(graph, id):
    try:
        return random.sample(list(graph.neighbors(id)))
    except:
        return None
# sample one neighbor of node id1 in a graph, excluding id2 and its neighbors


def sampleNeighborUnique(graph, id1, id2):
    neighbors = list(set(graph.neighbors(id1)) -
                     set(graph.neighbors(id2)) - set([id2]))
    return np.random.choice(neighbors) if neighbors else None
# sample any second degree neighbor of node id by random sampling a firt degree neighbor and then sampling its neighbors


def sampleSecondNeighbor(graph, id):
    firstNeighbor = sampleNeighbor(graph, id)
    secondNeighbor = sampleNeighbor(graph, firstNeighbor)
    return secondNeighbor
# sample any second degree neighbor of node id excluding id's first degree neighbors


def sampleSecondNeighborUnique(graph, id):
    firstNeighbors = np.fromiter(graph.neighbors(id), dtype=np.intc)
    for fN in np.random.shuffle(firstNeighbors):
        secondNeighbor = sampleNeighborUnique(graph, fN, id)
        if secondNeighbor:
            return secondNeighbor
    return None
# checks if node has only 1 edge


def isLonely(graph, x):
    return graph.degree[x] <= 1
# SIDE EFFECT: replaces the graph's edge (x,y) with edge (x,z)

# neighbors of b eligible for rewiring a to


def eligibleNewFriends(graph, b, a):
    return list(set(graph.neighbors(b)) - set(graph.neighbors(a)) - set([a]))
# samples a node of a given strat from the list strats, returns its id


@njit
def _sampleStratEligible(strats, strat, x):
    ofStrat = (np.array(strats) == strat).nonzero()[0]  # .astype(np.int32)
    return ofStrat


def sampleStratEligible(graph, strats, strat, x):
    ofStrat = _sampleStratEligible(strats, strat, x)
    if ofStrat.shape[0] == 0:
        return None
    neighbors = set(graph.neighbors(x))
    eligible = list(set(ofStrat) - neighbors - set([x]))
    if len(eligible) <= 0:
        return None
    return eligible[crandint(0, len(eligible)-1)]


# Samples only from a certain strategy and from a certain mediator (Xclusive)
@njit
def _sampleStratEligibleX(strats, medStrats, strat, medStrat, x):
    ofStrat = np.logical_and(
        strats == strat, medStrats == medStrat).nonzero()[0].astype(np.int32)
    return ofStrat


def sampleStratEligibleX(graph, strats, medStrats, strat, medStrat, x):
    ofStrat = _sampleStratEligibleX(strats, medStrats, strat, medStrat, x)
    if ofStrat.shape[0] == 0:
        return None
    # eligible = list(set(ofStrat) - neighbors - set([x]))
    neighbors = set(graph.neighbors(x))
    pre_el = set(ofStrat) - neighbors
    eligible = list(pre_el - set([x]))
    if len(eligible) <= 0:
        return None
    return eligible[crandint(0, len(eligible)-1)]
