from defaultParams import _T, _R, _S, _P
import numpy as np

'''Games'''
C = 1
D = 0

# Dictionary of game weights given T and S. R=1, P=0


def makeWeightDict(t, s):
    return {'R': _R, 'P': _P, 'T': t, 'S': s}


# Returns a dictionary that maps the strategies played to payoffs received for a set of 4 payoff metric
# T€[0,2] S€[-1,1]
def makeDilemma(R=_R, P=_P, T=_T, S=_S):
    return np.array([[[P, P], [T, S]], [[S, T], [R, R]]])

# Returns a dictionary that maps the strategies played to payoffs received for T and S (canonical form of a 2x2)


def makeTSDilemma(t, s):
    return makeDilemma(**makeWeightDict(t, s))


# dilemma = makeDilemma()


# Returns the payoff as a function of the dilemma, the ids and strats of the players
def playDilemma(dilemma, strats, id1, id2):
    return dilemma[strats[id1]][strats[id2]]
# cumulativePayoffs :: graph -> [strat] # Cumulative Payoff of 1 round of a node playing all its neighbors


def nodeCumPayoffs(dilemma, neighbors, strats, x):
    total = 0
    for y in neighbors:
        total += playDilemma(dilemma, strats, x, y)[0]
    return total
# Pair of cumulative payoffs


def pairCumPayoffs(dilemma, neighbors, strats, a, b):
    return [nodeCumPayoffs(dilemma, neighbors, strats, a), nodeCumPayoffs(dilemma, neighbors, strats, b)]

# return a list of payoffs for each node


def calcPayoffs(dilemma, graph, strats):
    return [nodeCumPayoffs(dilemma, graph.get_all_neighbors(x), strats, x) for x in graph.get_vertices()]
