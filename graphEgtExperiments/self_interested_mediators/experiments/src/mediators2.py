from math import floor, inf
from sampling2 import eligibleNewFriends, eligibleStrat, eligibleStratX, sampleEgoStrat, filterEgoSampleUnique
from games import C, D
import numpy as np
from optimized.cyFns import crandint
from numba import njit
'''Mediators'''
# No mediator, recommends a neighbor of y


def useNoMed(graph, strats, y, x, neighs_x, neighs_y):
    eligible = eligibleNewFriends(graph, y, x)
    return None if not eligible else eligible[crandint(0, len(eligible)-1)]

# Recommends a random cooperator


def useGoodMed(graph, strats, y, x, neighs_x, neighs_y):
    eligible = eligibleStrat(set(neighs_x), strats, C, x)
    return None if not eligible else eligible[crandint(0, len(eligible)-1)]

# Recommends a random defector


def useBadMed(graph, strats, y, x, neighs_x, neighs_y):
    eligible = eligibleStrat(set(neighs_x), strats, D, x)
    return None if not eligible else eligible[crandint(0, len(eligible)-1)]

# Recommends a random node


def useRandomMed(graph, strats, y, x, neighs_x, neighs_y):
    eligible = list(set(range(len(strats))) -
                    set(neighs_x) - set([x]))
    return None if len(eligible) <= 0 else eligible[crandint(0, len(eligible)-1)]

# Recs D to D and C to C


def useFairMed(graph, strats, y, x, neighs_x, neighs_y):
    eligible = eligibleStrat(set(neighs_x), strats, strats[x], x)
    return None if not eligible else eligible[crandint(0, len(eligible)-1)]


'''Exclusive mediators, only rec their own users'''

# No mediator, recommends a neighbor of y, not exclusive, exactly the same bc no mediator

# Recommends a random cooperator


def useGoodMedX(graph, strats, medStrats, y, x, neighs_x, neighs_y):
    eligible = eligibleStratX(set(neighs_x), strats,
                              medStrats, C, medStrats[x], x)
    return None if not eligible else eligible[crandint(0, len(eligible)-1)]

# Recommends a random defector


def useBadMedX(graph, strats, medStrats, y, x, neighs_x, neighs_y):
    eligible = eligibleStratX(set(neighs_x), strats,
                              medStrats, D, medStrats[x], x)
    return None if not eligible else eligible[crandint(0, len(eligible)-1)]

# Recommends a random node


@njit
def _useRandomMedX(neighbors, strats, medStrats, y, x):
    pre_el = (medStrats == medStrats[x]).nonzero()[0]
    eligible = list(set(pre_el) - neighbors - set([x]))
    return eligible


def useRandomMedX(graph, strats, medStrats, y, x, neighs_x, neighs_y):
    eligible = _useRandomMedX(set(neighs_x), strats, medStrats, y, x)
    return None if len(eligible) <= 0 else eligible[crandint(0, len(eligible)-1)]


# Recs D to D and C to C


def useFairMedX(graph, strats, medStrats, y, x, neighs_x, neighs_y):
    eligible = eligibleStratX(set(neighs_x), strats, medStrats,
                              strats[x], medStrats[x], x)
    return None if not eligible else eligible[crandint(0, len(eligible)-1)]


''' Ego mediators '''


def useLocalGoodMed(graph, strats, y, x, neighs_x, neighs_y):
    return sampleEgoStrat(graph, strats, C, x)


def useLocalBadMed(graph, strats, y, x, neighs_x, neighs_y):
    return sampleEgoStrat(graph, strats, D, x)


def useLocalRandomMed(graph, strats, y, x, neighs_x, neighs_y):
    return filterEgoSampleUnique(graph, lambda x: True, x)


def useLocalFairMed(graph, strats, y, x, neighs_x, neighs_y):
    return sampleEgoStrat(graph, strats, strats[x], x)


'''Mediator strats'''
NO_MED = 'NO_MED'
GOOD_MED = 'GOOD_MED'
BAD_MED = 'BAD_MED'
RANDOM_MED = 'RANDOM_MED'
FAIR_MED = 'FAIR_MED'

GOOD_MED_X = 'GOOD_MED_X'
BAD_MED_X = 'BAD_MED_X'
RANDOM_MED_X = 'RANDOM_MED_X'
FAIR_MED_X = 'FAIR_MED_X'

GOOD_MED_LOCAL = 'GOOD_MED_LOCAL'
BAD_MED_LOCAL = 'BAD_MED_LOCAL'
RANDOM_MED_LOCAL = 'RANDOM_MED_LOCAL'
FAIR_MED_LOCAL = 'FAIR_MED_LOCAL'


_medSet = ["NO_MED",
           "GOOD_MED",
           "BAD_MED",
           "RANDOM_MED",
           "FAIR_MED",
           "GOOD_MED_X",
           "BAD_MED_X",
           "RANDOM_MED_X",
           "FAIR_MED_X",
           'GOOD_MED_LOCAL',
           'BAD_MED_LOCAL',
           'RANDOM_MED_LOCAL',
           'FAIR_MED_LOCAL']


def initMedStratsSmall(N, medSet, baseline_med, baseline_proportion=0.1):
    medStrats = np.zeros(N, dtype=np.intc)
    seed_size = floor(N * baseline_proportion)
    seed_population_init = np.zeros(seed_size, dtype=np.intc)
    seed_population_split = np.array_split(seed_population_init, len(medSet))
    for i in range(len(medSet)):
        seed_population_split[i].fill(medSet[i])
    np.concatenate(seed_population_split, axis=0, out=seed_population_init)
    baseline_population = np.full(N-seed_size, baseline_med, dtype=np.intc)
    medStrats = np.concatenate(
        (seed_population_init, baseline_population), axis=0)
    np.random.shuffle(medStrats)
    return medStrats


def initMedStrats(N, medSet):
    return np.random.choice(medSet, N)


# use this as source of truth
str2Med = {"NO_MED": useNoMed, "GOOD_MED": useGoodMed,
           "BAD_MED": useBadMed, "RANDOM_MED": useRandomMed,
           "FAIR_MED": useFairMed,
           "GOOD_MED_X": useGoodMedX,
           "BAD_MED_X": useBadMedX, "RANDOM_MED_X": useRandomMedX,
           "FAIR_MED_X": useFairMedX,
           'GOOD_MED_LOCAL': useLocalGoodMed,
           'BAD_MED_LOCAL': useLocalBadMed, 'RANDOM_MED_LOCAL': useLocalRandomMed,
           'FAIR_MED_LOCAL': useLocalFairMed}
int2Med = {i: medFn for i, (medName, medFn) in enumerate(str2Med.items())}
# int2Med = {0: useNoMed, 1: useGoodMed,
#            2: useBadMed, 3: useRandomMed, 4: useFairMed}
int2MedName = {i: medName for i,
               (medName, medFn) in enumerate(str2Med.items())}
# int2MedName = {0: "NO_MED", 1: "GOOD_MED",
#                2: "BAD_MED", 3: "RANDOM_MED", 4: "FAIR_MED"}
medName2Int = {medName: i for i,
               (medName, medFn) in enumerate(str2Med.items())}
# medName2Int = {NO_MED: 0, GOOD_MED: 1,
#                BAD_MED: 2, RANDOM_MED: 3, FAIR_MED: 4}


non_exclusive = [medName2Int[name]
                 for name in ["NO_MED", "GOOD_MED", "BAD_MED", "RANDOM_MED", "FAIR_MED"]]
exclusive = [medName2Int[name]
             for name in ["NO_MED", "GOOD_MED_X", "BAD_MED_X", "RANDOM_MED_X", "FAIR_MED_X"]]
local_meds = [medName2Int[name]
              for name in ["NO_MED", "GOOD_MED_LOCAL", "BAD_MED_LOCAL", "RANDOM_MED_LOCAL", "FAIR_MED_LOCAL"]]


exclusive_medset = [medName2Int[name]
                    for name in ["GOOD_MED_X", "BAD_MED_X", "RANDOM_MED_X", "FAIR_MED_X"]]


def useMed(medStrat, graph, strats, medStrats, y, x, neighs_x, neighs_y):
    if medStrat in exclusive_medset:
        return int2Med[medStrat](graph, strats, medStrats,  y, x, neighs_x, neighs_y)
    else:
        return int2Med[medStrat](graph, strats, y, x, neighs_x, neighs_y)
