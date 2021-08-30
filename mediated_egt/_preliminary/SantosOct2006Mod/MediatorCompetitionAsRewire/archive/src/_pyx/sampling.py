from utils import flatten, pipe
import numpy as np
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


def filterEgoSample(graph, filterFn, ids):
    neighbors = getNeighbors(graph, ids)
    elligibleNeighbors = list(filter(filterFn, neighbors))
    if not elligibleNeighbors:
        if isFinished(graph, neighbors):
            return None
        return filterEgoSample(graph, filterFn, neighbors)
    return np.random.choice(elligibleNeighbors)


# exclude first neighbors
def filterEgoSampleUnique(graph, filterFn, id):
    def excludeFirstNeighbors(x): return x not in (
        [id] + list(graph.get_all_neighbors(id)))
    return filterEgoSample(graph, lambda x: excludeFirstNeighbors(x) and filterFn(x), [id])


def sampleEgoStrat(graph, strats, strat, id):
    return filterEgoSampleUnique(graph, lambda x: strats[x] == strat, id)


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
    return graph.vertex(x).out_degree() == 1
# SIDE EFFECT: replaces the graph's edge (x,y) with edge (x,z)

# neighbors of b eligible for rewiring a to


def eligibleNewFriends(graph, b, a):
    return list(set(graph.get_all_neighbors(b)) - set(graph.get_all_neighbors(a)) - set([a]))
# samples a node of a given strat from the list strats, returns its id

# %%


def sampleStratEligible(graph, strats, strat, x):
    def filterStrat(x): return x[1] == strat
    def filterStrats(xs): return filter(filterStrat, xs)
    ofStrat = pipe(enumerate, filterStrats, mapNth(0), list)(strats)
    if not ofStrat:
        return None
    else:
        eligible = list(
            set(ofStrat) - set(graph.get_all_neighbors(x)) - set([x]))
        if not eligible:
            return None
        return np.random.choice(eligible)
