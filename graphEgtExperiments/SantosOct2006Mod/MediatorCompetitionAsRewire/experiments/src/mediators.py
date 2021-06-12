from sampling import eligibleNewFriends, sampleStratEligible, sampleEgoStrat, filterEgoSampleUnique
from games import C, D
import numpy as np
from optimized.cyFns import crandint
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
    eligible = list(set(range(len(strats))) -
                    set(graph.get_all_neighbors(x)) - set([x]))
    return None if len(eligible) <= 0 else eligible[crandint(0, len(eligible)-1)]

# Recs D to D and C to C


def useFairMed(graph, strats, y, x):
    z = sampleStratEligible(graph, strats, strats[x], x)
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

_medSet = [NO_MED, GOOD_MED, BAD_MED, RANDOM_MED, FAIR_MED]


def initMedStrats(N, medSet):
    return [np.random.choice(medSet) for i in range(N)]


str2Med = {NO_MED: useNoMed, GOOD_MED: useGoodMed,
           BAD_MED: useBadMed, RANDOM_MED: useRandomMed, FAIR_MED: useFairMed}

int2Med = {0: useNoMed, 1: useGoodMed,
           2: useBadMed, 3: useRandomMed, 4: useFairMed}
int2MedName = {0: "NO_MED", 1: "GOOD_MED",
               2: "BAD_MED", 3: "RANDOM_MED", 4: "FAIR_MED"}
medName2Int = {NO_MED: 0, GOOD_MED: 1,
               BAD_MED: 2, RANDOM_MED: 3, FAIR_MED: 4}


def useMed(medStrat, graph, strats, medStrats, y, x):
    return int2Med[medStrat](graph, strats, y, x)
