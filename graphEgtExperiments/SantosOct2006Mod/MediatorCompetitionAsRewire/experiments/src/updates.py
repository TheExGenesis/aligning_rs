import numpy as np
from sampling import isLonely
from mediators import useMed
from random import random
from games import C, D

'''Structural Update'''
# Makes a tie update object


def rewireEdge(graph, x, y, z):
    graph.remove_edge(graph.edge(x, y))
    graph.add_edge(x, z)
    return graph


def TieUpdate(x, y, z):
    return {"updateType": "rewire", "x": x, "old": y, "new": z}

# Asks for a recommendation and keeps the same tie if there are no other valid nodes


def decideRewire(graph, strats, x, y, medStrats):
    if isLonely(graph, y):
        return TieUpdate(x, y, y)  # enforcing graph connectedness
    z = useMed(medStrats[x], graph, strats, medStrats, y, x)
    if not z:
        return TieUpdate(x, y, y)  # enforcing graph connectedness
    return TieUpdate(x, y, z)
# Decides whether a rewire should happen based on x and y's strats. If x is satisfied, nothing happens


def calcStructuralUpdate(graph, strats, _x, _y, p, medStrats):
    x, y = int(_x), int(_y)
    if (strats[x] == C and strats[y] == D):
        doRewire = random() < p
        if doRewire:
            return decideRewire(graph, strats, x, y, medStrats)
        else:
            return TieUpdate(x, y, y)
    elif (strats[x] == D and strats[y] == D):
        keepX = random() < p
        args = [x, y] if keepX else [y, x]
        return decideRewire(graph, strats, args[0], args[1], medStrats)
    return TieUpdate(x, y, y)
# Applies a tie update to the graph


def updateTies(graph, tieUpdate):
    return rewireEdge(graph, tieUpdate["x"], tieUpdate["old"], tieUpdate["new"])


'''Strategy Evolution'''
# fermi formula, pairwise comparison


def fermi(beta, fitness_diff):
    return 1. / (1. + np.exp(-1 * beta * np.clip(fitness_diff, 0, None), dtype=np.float64))
#
# def calcK(graph, x, y) = max(graph.vertex(x).out_degree(), graph.vertex(y).out_degree())
# def calcD(T=_T, S=_S) = max(T, 1) - min(S, 0)
# def transProb(calcK, P, x, y) = (P[y] - P[x]) / (calcK(x, y) * calcD())

# Returns a strategy update object
# updateStrat :: graph -> [strat] -> [float] -> id -> strat


def calcStrategyUpdate(strats, x, y, p):
    doChangeStrat = random() < p
    new = strats[y] if doChangeStrat else strats[x]
    return {"updateType": "strat", "x": x, "old": strats[x], "new": new}
# Applies a strategy update to the graph


def updateStrat(strats, update):
    strats[update["x"]] = update["new"]
    return strats


'''Mediator Evolution/Competition'''


def calcMedUpdate(medStrats, x, y, p):
    doChangeStrat = random() < p
    new = medStrats[y] if doChangeStrat else medStrats[x]
    return {"updateType": "mediator", "x": x, "old": medStrats[x], "new": new}


def updateMed(medStrats, update):
    medStrats[update["x"]] = update["new"]
    return medStrats
