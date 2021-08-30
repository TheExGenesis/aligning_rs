
cimport numpy as cnp
import sys
from pathlib import Path
from updates import calcStrategyUpdate, updateStrat, calcStructuralUpdate, updateTies, calcMedUpdate, updateMed
from sampling import sampleNeighbor
from games import pairCumPayoffs, makeTSDilemma, nodeCumPayoffs
from mediators import initMedStrats, _medSet, useNoMed, useGoodMed, useBadMed, useRandomMed, useFairMed, useMed, NO_MED, GOOD_MED, BAD_MED, RANDOM_MED, FAIR_MED
from init import initUniformRandomGraph, initPayoffs, initStrats
from defaultParams import _episode_n, _N, _W, _W2, _k, _T, _S
from defaultParams import *
from functools import partial
from math import floor, inf
from collections import Counter
from copy import deepcopy
import pandas as pd
from updates import updateTies, calcStructuralUpdate, TieUpdate
from sampling import sampleNeighbor, isLonely
from egt_io import saveRes, makeCompetitionName
from evolution import runCompetitionExperiment, runEvolutionCompetitionEp
import numpy as np
from math import inf
from time import time
import array
from optimized.cyFns import *
from cpython cimport array
from libc.math cimport exp
from libc.math cimport log, exp
from libc.stdlib cimport rand, RAND_MAX


# ctypedef enum UpdateType: M EDIATOR, STRAT, REWIRE


ctypedef struct Update:
    int updateType
    int x
    int old
    int new

def isLonely(graph, x):
    return graph.vertex(x).out_degree() <= 1


medDict = {0: useNoMed, 1: useGoodMed,
           2: useBadMed, 3: useRandomMed, 4: useFairMed}


def cy_useMed(int medStrat, graph, int[:] strats, int y, int x):
    return medDict[medStrat](graph, strats, y, x)


def cy_rewireEdge(graph, int x, int y, int z):
    graph.remove_edge(graph.edge(x, y))
    graph.add_edge(x, z)
    return graph

def cy_updateTies(graph, Update tieUpdate):
    return cy_rewireEdge(graph, tieUpdate.x, tieUpdate.old, tieUpdate.new)

def cy_decideRewire(graph, int [:] strats, int x, int y, int [:] medStrats):
    cdef Update tieUpdate
    if isLonely(graph, y):
        tieUpdate = [MEDIATOR, x, y, y]
        return tieUpdate  # enforcing graph connectedness
    z = cy_useMed(medStrats[x], graph, strats, y, x)
    if not z:
        tieUpdate = [MEDIATOR, x, y, y]
        return tieUpdate  # enforcing graph connectedness
    tieUpdate = [MEDIATOR, x, y, z]
    return tieUpdate
# # Decides whether a rewire should happen based on x and y's strats. If x is satisfied, nothing happens


def cy_calcStructuralUpdate(graph, int[:] strats, int x, int y, float p, int[:] medStrats):
    cdef Update tieUpdate
    cdef bint doRewire
    if (strats[x] == C and strats[y] == D):
        doRewire = rand()/RAND_MAX < p
        if doRewire:
            return cy_decideRewire(graph, strats, x, y, medStrats)
        else:
            tieUpdate = [MEDIATOR, x, y, y]
            return tieUpdate
    elif (strats[x] == D and strats[y] == D):
        keepX = rand()/RAND_MAX < p
        if keepX:
            return cy_decideRewire(graph, strats, x, y, medStrats)
        else:
            return cy_decideRewire(graph, strats, y, x, medStrats)
    tieUpdate = [MEDIATOR, x, y, y]
    return tieUpdate
# # Applies a tie update to the graph

def full_cy_runEvolutionCompetitionEp(int N, float beta, float W, float W2, float[:, :, :] dilemma, graph, int[:] medStrats, int[:] strats, history, int _x, bint saveHistory=False):
    cdef int x = _x
    cdef long[:] neighs_x = graph.get_all_neighbors(x)
    cdef int _y = neighs_x[crandint(0, len(neighs_x)-1)]  # sample neighbor
    # cdef int x = int(_x)
    # cdef int y = int(_y)
    cdef int y = _y
    cdef long[:] neighs_y = graph.get_all_neighbors(y)
    cdef float px = cy_nodeCumPayoffs(dilemma, neighs_x, strats, _x)
    cdef float py = cy_nodeCumPayoffs(dilemma, neighs_y, strats, _y)
    cdef float p = cy_fermi(beta, py - px)
    cdef float r = rand() / RAND_MAX
    cdef bint doMedUpdate = r * (1+W2) > 1
    if doMedUpdate:
        medUpdate = cy_calcMedUpdate(medStrats, x, y, p)
        if saveHistory:
            history.append(medUpdate)
        medStrats = cy_updateMed(medStrats, medUpdate)
    else:
        doStratUpdate = (rand() / RAND_MAX) * (1+W) <= 1
        if doStratUpdate:
            stratUpdate = cy_calcStrategyUpdate(strats, x, y, p)
            if saveHistory:
                history.append(stratUpdate)
            strats = cy_updateStrat(strats, stratUpdate)
        else:
            graphUpdate = cy_calcStructuralUpdate(graph, strats, _x, _y, p, medStrats)
            if saveHistory:
                history.append(graphUpdate)
            graph = cy_updateTies(graph, graphUpdate)
    return strats, graph, history


# def cy_runEvolutionCompetitionEp( N,  beta,  W,  W2, dilemma, graph, medStrats, strats, history, _x, saveHistory=False):
def cy_runEvolutionCompetitionEp(int N, float beta, float W, float W2, float[:, :, :] dilemma, graph, int[:] medStrats, int[:] strats, history, int _x, bint saveHistory=False):
    cdef int x = _x
    cdef long[:] neighs_x = graph.get_all_neighbors(x)
    cdef int y = neighs_x[crandint(0, len(neighs_x)-1)]  # sample neighbor
    # cdef int x = int(_x)
    # cdef int y = int(_y)
    cdef long[:] neighs_y = graph.get_all_neighbors(y)
    cdef float px = cy_nodeCumPayoffs(dilemma, neighs_x, strats, _x)
    cdef float py = cy_nodeCumPayoffs(dilemma, neighs_y, strats, y)
    cdef float p = cy_fermi(beta, py - px)
    cdef float r = rand() / RAND_MAX
    cdef bint doMedUpdate = r * (1+W2) > 1
    if doMedUpdate:
        medUpdate = calcMedUpdate(medStrats, x, y, p)
        if saveHistory:
            history.append(medUpdate)
        medStrats = updateMed(medStrats, medUpdate)
    else:
        doStratUpdate = (rand() / RAND_MAX) * (1+W) <= 1
        if doStratUpdate:
            stratUpdate = calcStrategyUpdate(strats, x, y, p)
            if saveHistory:
                history.append(stratUpdate)
            strats = updateStrat(strats, stratUpdate)
        else:
            graphUpdate = calcStructuralUpdate(graph, strats, _x, y, p, medStrats)
            if saveHistory:
                history.append(graphUpdate)
            graph = updateTies(graph, graphUpdate)
    return strats, graph, history

def cy_genericRunEvolution(runEvolutionEp, int N, int episode_n, float W, float W2, float[:, :, :] dilemma, int[:] medStrats, int[:] strats, float beta=0.001, _graph=None, int k=30, history=None, saveHistory=False):
    history = [] if history == None else history
    cdef int[:] initialStrats = strats[:]
    cdef int[:] initialMedStrats = medStrats[:]
    totalPayoffs = initPayoffs(N)
#     history = [] #[{updateType: "strat", x: N, old: {C,D}, new: {C,D}}, {updateType: "rewire", xy: (x,y), old: (x,y), new: (x,z)}]
    graph = deepcopy(_graph) if _graph else initUniformRandomGraph(
        N=N, k=(k if k else _k))
    cdef int i = 0
    cdef int x = 0
    for i in range(episode_n):
        x = crandint(0, N-1)
        strats, graph, history = cy_runEvolutionCompetitionEp(N, beta, W, W2, dilemma, graph, medStrats, strats, history, x, False)
        # strats, graph, history = runEvolutionEp(
        #     N, beta, W, W2, dilemma, graph, medStrats, strats, history, x, False)
        # N, beta, W, W2, dilemma, graph, medStrats, strats, history, x, saveHistory)
        if i % 10000 == 0:
            print(f"i: {i}")
        #     medEvoDone = any([x == N for x in Counter(medStrats).values()])
        #     stratEvoDone = any([x == N for x in Counter(strats).values()])
        #     if medEvoDone and stratEvoDone:
        #         break
        #     if W2 == inf and medEvoDone:
        #         break
        #     if W == 0 and stratEvoDone:
        #         break
    return {"graph": graph, "history": pd.DataFrame(history, dtype="category"), "initStrats": initialStrats, "finalStrats": strats[:], "initMedStrats": initialMedStrats,  "medStrats": medStrats[:]}


# Initializes strats, payoffs, history, graph and runs many evolution episodes, each for a random node x
# runEvolution :: int -> graph -> [strat] -> [float] -> [[strat]]
cy_runEvolutionCompetition = partial(
    cy_genericRunEvolution, cy_runEvolutionCompetitionEp)


def cy_runCompetitionExperiment(int N=_N, int episode_n=_episode_n, float W=_W, float W2=_W2, graph=None, ts=(_T, _S), medStrats=None, strats=None, float beta=0.005, int k=30, medSet=_medSet, history=None, bint saveHistory=False):
    cdef int[:] _strats = strats[:] if strats else cy_initStrats(N)[:]
    cdef int[:] _medStrats = medStrats[:] if medStrats else cy_initMedStrats(N, medSet)[:]
    experimentResults = cy_runEvolutionCompetition(N, episode_n, W, W2, cy_makeTSDilemma(
        *ts), _medStrats, _strats, beta, deepcopy(graph), k, history, saveHistory=saveHistory)
    return experimentResults



# %%

def cy_competitionFullEvo(episode_n = 1000000, medSet=[NO_MED, GOOD_MED], save=False):
    n_trials = 1
    N = 500
    W1 = 1
    W2 = 1
    ts = (2, -1)
    beta = 0.005
    k = 30 
    runs = [cy_runCompetitionExperiment(N=N, episode_n=episode_n, W=W1, W2=W2, ts=ts, beta=beta,
                                        k=k, saveHistory=True, history=[], medSet=medSet) for i in range(n_trials)]
    if save:
        saveRes(runs,   makeCompetitionName(
            medSet, N, episode_n, ts, beta, W1, W2, k)) 
    return runs

# start = time()
# res = cy_competitionFullEvo([NO_MED, GOOD_MED], save=False)
# end = time()
# print(f"cy_competitionFullEvo took {end-start}ms")
# # %%
# medSet = [medStrat2Int["NO_MED"], medStrat2Int["GOOD_MED"], medStrat2Int["BAD_MED"]]
# k = 30
# N = 500
# beta = 0.005
# x = np.random.randint(N-1)
# W = 1
# W2 = 1
# strats = cy_initStrats(N)
# medStrats = cy_initMedStrats(N, medSet)
# dilemma = cy_makeTSDilemma(2, -1)
# graph = initUniformRandomGraph(N=N, k=k)
# epArgs = {"N": N, "beta": beta, "W": W, "W2": W2, "dilemma": dilemma,
#           "graph": graph, "medStrats": medStrats, "strats": strats, "history": [], "_x": x, }

# # cy_runEvolutionCompetitionEp(**epArgs)
# #%%
