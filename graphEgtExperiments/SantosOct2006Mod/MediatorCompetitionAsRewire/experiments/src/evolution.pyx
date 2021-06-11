import cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, exp
from libc.math cimport exp
from cpython cimport array
from optimized.cyFns import *
import array
from time import time
from math import inf
import numpy as np
from egt_io import saveRes, makeCompetitionName
from sampling import sampleNeighbor, isLonely
from updates import updateTies, calcStructuralUpdate, TieUpdate
import pandas as pd
from copy import deepcopy
from collections import Counter
from math import floor, inf
from functools import partial
from defaultParams import *
from defaultParams import _episode_n, _N, _W, _W2, _k, _T, _S
from init import initUniformRandomGraph, initPayoffs, initStrats
from mediators import _medSet
from mediators import *
from games import pairCumPayoffs, makeTSDilemma, nodeCumPayoffs
from sampling import sampleNeighbor
from updates import calcStrategyUpdate, updateStrat, calcStructuralUpdate, updateTies, calcMedUpdate, updateMed
from pathlib import Path
import sys
cimport numpy as cnp


# ctypedef enum UpdateType: M EDIATOR, STRAT, REWIRE


ctypedef struct Update:
    int updateType
    int x
    int old
    int new

# def cy_runEvolutionCompetitionEp( N,  beta,  W,  W2, dilemma, graph, medStrats, strats, history, _x, saveHistory=False):


def cy_runEvolutionCompetitionEp(int N, float beta, float W1, float W2, float[:, :, :] dilemma, graph, int[:] medStrats, int[:] strats, history, int _x, bint saveHistory=False):
    cdef int x = _x
    cdef long[:] neighs_x = graph.get_all_neighbors(x)
    cdef int idx = crandint(0, len(neighs_x)-1)
    cdef int y = neighs_x[idx]  # sample neighbor
    cdef long[:] neighs_y = graph.get_all_neighbors(y)
    cdef float px = cy_nodeCumPayoffs(dilemma, neighs_x, strats, _x)
    cdef float py = cy_nodeCumPayoffs(dilemma, neighs_y, strats, y)
    cdef float p = cy_fermi(beta, py - px)
    cdef int rint1, rint2
    cdef float r
    rint1 = rand()
    r = rint1 / float(RAND_MAX)
    cdef bint doMedUpdate = r * (1+W2) > 1
    # print(
    #     f"updating. doMedUpdate {str(doMedUpdate)} W1={W2} updateProb={rint1 * (1+W2)} r={r} rint={rint1} RAND_MAX={RAND_MAX}")
    if doMedUpdate:
        medUpdate = calcMedUpdate(medStrats, x, y, p)
        if saveHistory:
            history.append(medUpdate)
        medStrats = updateMed(medStrats, medUpdate)
    else:
        rint2 = rand()
        r = rint2 / float(RAND_MAX)
        doStratUpdate = r * (1+W1) <= 1
        # print(f"updating. doStratUpdate {str(doStratUpdate)} W1={W1} r={r} rint={rint2} RAND_MAX={RAND_MAX}")
        if doStratUpdate:
            stratUpdate = calcStrategyUpdate(strats, x, y, p)
            if saveHistory:
                history.append(stratUpdate)
                #print(f"saveHistory {history}")
            strats = updateStrat(strats, stratUpdate)
        else:
            graphUpdate = calcStructuralUpdate(
                graph, strats, _x, y, p, medStrats)
            if saveHistory:
                history.append(graphUpdate)
                #print(f"saveHistory {history}")
            graph = updateTies(graph, graphUpdate)
    return graph, history


def cy_genericRunEvolution(int N, int episode_n, float W1, float W2, float[:, :, :] dilemma, int[:] medStrats, int[:] strats, float beta=0.001, _graph=None, int k=30, history=None, saveHistory=False):
    history = [] if history == None else history
    cdef int[:] initialStrats, initialMedStrats
    initialStrats = np.empty(N, dtype=np.intc)
    initialMedStrats = np.empty(N, dtype=np.intc)
    initialStrats[...] = strats
    initialMedStrats[...] = medStrats
    totalPayoffs = initPayoffs(N)
    graph = _graph
    graph.set_fast_edge_removal(True)
    cdef int i = 0
    cdef int x = 0
    for i in range(episode_n):
        x = crandint(0, N-1)
        # strats, graph, history = cy_runEvolutionCompetitionEp(N, beta, W1, W2, dilemma, graph, medStrats, strats, history, x, False)
        graph, history = cy_runEvolutionCompetitionEp(
            N, beta, W1, W2, dilemma, graph, medStrats, strats, history, x, saveHistory)
        # cy_runEvolutionCompetitionEp(
        #     N, beta, W1, W2, dilemma, graph, medStrats, strats, history, x, False)
        if i % 5000 == 0:
            # print(f"i {i}")
            medEvoDone = any([x == N for x in Counter(medStrats).values()])
            stratEvoDone = any([x == N for x in Counter(strats).values()])
            if medEvoDone and stratEvoDone:
                break
            if W2 == inf and medEvoDone:
                break
            if W1 == 0 and stratEvoDone:
                break
    return {"graph": graph,
            "history": pd.DataFrame(history, dtype="category"),
            "initStrats": np.asarray(initialStrats, dtype=np.intc),
            "finalStrats": np.asarray(strats, dtype=np.intc),
            "initMedStrats": np.asarray(initialMedStrats, dtype=np.intc),
            "medStrats": np.asarray(medStrats, dtype=np.intc)}
# I should be able to take strats and medStrats but for debugging purposes, I'm making it make them from scratch every
def cy_runCompetitionExperiment(int N=_N, int episode_n=_episode_n, float W1=_W, float W2=_W2, graph=None, ts=(_T, _S), int[:] medStrats=None, int[:] strats=None, float beta=0.005, int k=30, medSet=_medSet, history=None, bint saveHistory=False):
    print(f"cy_runCompetitionExperiment {ts}")
    # cdef int[:] _strats
    # _strats[...] = cy_initStrats(N) if strats is None else strats
    # cdef int[:] _medStrats
    # _medStrats[...] = cy_initMedStrats(N, medSet) if medStrats is None else medStrats
    cdef float[:, :, :] dilemma
    dilemma = cy_makeTSDilemma(ts[0], ts[1])
    _graph = deepcopy(graph) if graph else initUniformRandomGraph(
        N=N, k=(k if k else _k))
    experimentResults = cy_genericRunEvolution(
        N, episode_n, W1, W2, dilemma, cy_initMedStrats(N, medSet), cy_initStrats(N), beta, deepcopy(_graph), k, history, saveHistory=saveHistory)
    return experimentResults

def cy_continueCompetitionExperiment(graph, int[:] medStrats, int[:] strats, int N=_N, int episode_n=_episode_n, float W1=_W, float W2=_W2, ts=(_T,_S), float beta=0.005, int k=30, medSet=_medSet, history=None, bint saveHistory=False):
    print(f"cy_continueCompetitionExperiment {ts}")
    # cdef int[:] _strats
    # _strats[...] = cy_initStrats(N) if strats is None else strats
    # cdef int[:] _medStrats
    # _medStrats[...] = cy_initMedStrats(N, medSet) if medStrats is None else medStrats
    cdef float[:, :, :] dilemma
    dilemma = cy_makeTSDilemma(ts[0], ts[1])
    _graph = deepcopy(graph) if graph else initUniformRandomGraph(
        N=N, k=(k if k else _k))
    experimentResults = cy_genericRunEvolution(
        N, episode_n, W1, W2, dilemma, medStrats, strats, beta, deepcopy(_graph), k, history, saveHistory=saveHistory)
    return experimentResults