from sampling import eligibleNewFriends, sampleStratEligible, sampleEgoStrat, filterEgoSampleUnique
from games import C, D
import numpy as np
'''Mediators'''
# No mediator, recommends a neighbor of y


def useNoMed(graph, strats, y, x):
    eligible = eligibleNewFriends(graph, y, x)
    return None if not eligible else np.random.choice(eligible)

# Recommends a random cooperator


def useGoodMed(graph, strats, y, x):
    z = sampleStratEligible(graph, strats, C, x)
    return None if not z else z

# Recommends a random defector


def useBadMed(graph, strats, y, x):
    z = sampleStratEligible(graph, strats, D, x)
    return None if not z else z

# Recommends a random node


def useRandomMed(graph, strats, y, x):
    return np.random.choice(list(set(graph.get_vertices()) - set(graph.get_all_neighbors(x)) - set([x])))

# Recs D to D and C to C


def useFairMed(graph, strats, y, x):
    z = sampleStratEligible(graph, strats, strats[int(x)], x)
    return None if not z else z


''' Ego mediators '''


def useLocalGoodMed(graph, strats, y, x):
    return sampleEgoStrat(graph, strats, C, x)


def useLocalBadMed(graph, strats, y, x):
    return sampleEgoStrat(graph, strats, D, x)


def useLocalRandomMed(graph, strats, y, x):
    return filterEgoSampleUnique(graph, lambda x: True, x)


def useLocalFairMed(graph, strats, y, x):
    return sampleEgoStrat(graph, strats, strats[x], x)


'''Mediator strats'''
GOOD_MED = 'GOOD_MED'
BAD_MED = 'BAD_MED'
NO_MED = 'NO_MED'
RANDOM_MED = 'RANDOM_MED'
FAIR_MED = 'FAIR_MED'

_medSet = [NO_MED, GOOD_MED, FAIR_MED, RANDOM_MED, BAD_MED]


def initMedStrats(N, medSet):
    return [np.random.choice(medSet) for i in range(N)]


medDict = {NO_MED: useNoMed, GOOD_MED: useGoodMed,
           BAD_MED: useBadMed, RANDOM_MED: useRandomMed, FAIR_MED: useFairMed}


def useMed(medStrat, graph, strats, medStrats, y, x):
    return medDict[medStrat](graph, strats, y, x)
