from defaultParams import _T, _R, _S, _P


'''Games'''
C = 1
D = 0

# Dictionary of game weights given T and S. R=1, P=0


def makeWeightDict(t, s):
    return {'R': _R, 'P': _P, 'T': t, 'S': s}


# Returns a dictionary that maps the strategies played to payoffs received for a set of 4 payoff metric
# T€[0,2] S€[-1,1]
def makeDilemma(R=_R, P=_P, T=_T, S=_S):
    return {C: {C: [R, R], D: [S, T]}, D: {C: [T, S], D: [P, P]}}

# Returns a dictionary that maps the strategies played to payoffs received for T and S (canonical form of a 2x2)


def makeTSDilemma(t, s):
    return makeDilemma(**makeWeightDict(t, s))


# dilemma = makeDilemma()


# Returns the payoff as a function of the dilemma, the ids and strats of the players
def playDilemma(dilemma, strats, id1, id2):
    return dilemma[strats[id1]][strats[id2]]
# cumulativePayoffs :: graph -> [strat] # Cumulative Payoff of 1 round of a node playing all its neighbors


def nodeCumPayoffs(dilemma, graph, strats, x):
    try:
        payoffs = [playDilemma(dilemma, strats, x, y)[0]
                   for y in graph.get_all_neighbors(x)]
        return sum(payoffs)
    except:
        return 0
# Pair of cumulative payoffs


def pairCumPayoffs(dilemma, graph, strats, a, b):
    return [nodeCumPayoffs(dilemma, graph, strats, a), nodeCumPayoffs(dilemma, graph, strats, b)]
