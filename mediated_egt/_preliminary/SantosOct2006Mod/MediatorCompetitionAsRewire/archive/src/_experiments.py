# %%
import line_profiler
from Cython.Compiler.Options import get_directive_defaults
directive_defaults = get_directive_defaults()

directive_defaults['linetrace'] = True
directive_defaults['binding'] = True
%load_ext line_profiler
%load_ext Cython
# %%
%%cython -a
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
cimport numpy as cnp
import sys
from pathlib import Path
from updates import calcStrategyUpdate, updateStrat, calcStructuralUpdate, updateTies, calcMedUpdate, updateMed
from sampling import sampleNeighbor
from games import pairCumPayoffs, makeTSDilemma, nodeCumPayoffs
from mediators import *
from mediators import _medSet
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
from evolution import *
import numpy as np
from math import inf
from time import time
import array
from optimized.cyFns import *
from cpython cimport array
from libc.math cimport exp
from libc.math cimport log, exp
from libc.stdlib cimport rand, RAND_MAX
import cython



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def cy_runEvolutionCompetitionEp(int N, float beta, float W1, float W2, float[:, :, :] dilemma, graph, int[:] medStrats, int[:] strats, history, int _x, bint saveHistory=False):
    cdef int x = _x
    cdef long[:] neighs_x = graph.get_all_neighbors(x)
    cdef int y = neighs_x[crandint(0, len(neighs_x)-1)]  # sample neighbor
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
        doStratUpdate = (rand() / RAND_MAX) * (1+W1) <= 1
        if doStratUpdate:
            stratUpdate = calcStrategyUpdate(strats, x, y, p)
            if saveHistory:
                history.append(stratUpdate)
            strats = updateStrat(strats, stratUpdate)
        else:
            graphUpdate = calcStructuralUpdate(
                graph, strats, _x, y, p, medStrats)
            if saveHistory:
                history.append(graphUpdate)
            graph = updateTies(graph, graphUpdate)
    return strats, graph, history


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def cy_genericRunEvolution(int N, int episode_n, float W1, float W2, float[:, :, :] dilemma, int[:] medStrats, int[:] strats, float beta=0.001, _graph=None, int k=30, history=None, saveHistory=False):
    history = [] if history == None else history
    cdef int[:] initialStrats = strats[:]
    cdef int[:] initialMedStrats = medStrats[:]
    totalPayoffs = initPayoffs(N)
    graph = _graph
    cdef int i = 0
    cdef int x = 0
    for i in range(episode_n):
        x = crandint(0, N-1)
        strats, graph, history = cy_runEvolutionCompetitionEp(
            N, beta, W1, W2, dilemma, graph, medStrats, strats, history, x, False)
        if i % 1000 == 0:
            print(f"i {i}")
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
            "initStrats": np.asarray(initialStrats, dtype=np.int32),
            "finalStrats": np.asarray(strats[:], dtype=np.int32),
            "initMedStrats": np.asarray(initialMedStrats, dtype=np.int32),
            "medStrats": np.asarray(medStrats[:], dtype=np.int32)}


# # Initializes strats, payoffs, history, graph and runs many evolution episodes, each for a random node x
# # runEvolution :: int -> graph -> [strat] -> [float] -> [[strat]]


# def cy_runCompetitionExperiment(int N=_N, int episode_n=_episode_n, float W1=_W, float W2=_W2, graph=None, ts=(_T, _S), medStrats=None, strats=None, float beta=0.005, int k=30, medSet=_medSet, history=None, bint saveHistory=False):
#     cdef int[:] _strats = strats[:] if strats else cy_initStrats(N)[:]
#     cdef int[:] _medStrats = medStrats[:] if medStrats else cy_initMedStrats(N, medSet)[:]
#     cdef float[:, :, :] dilemma = cy_makeTSDilemma(ts[0], ts[1])
#     _graph = deepcopy(graph) if graph else initUniformRandomGraph(
#         N=N, k=(k if k else _k))
#     experimentResults = cy_genericRunEvolution(
#         N, episode_n, W1, W2, dilemma, _medStrats, _strats, beta, deepcopy(_graph), k, history, saveHistory=saveHistory)
#     return experimentResults

# # run cy_runCompetitionExperiment and save it. Takes a dir_path as an additional argument
# def saveCompetitionExperiment(N=_N,episode_n=_episode_n, W1=_W, W2=_W2, graph=None, ts=(_T, _S), medStrats=None, strats=None, beta=0.005, k=30, medSet=_medSet, history=None, saveHistory=False, dir_path="./data"):
#     # print(f"running experiment. medSet {medSet} ts {ts}")
#     run = cy_runCompetitionExperiment(N=N, episode_n=episode_n, W1=W1, W2=W2, ts=ts, beta=beta, k=k, saveHistory=True, history=[], medSet=medSet)
#     print(f"saving {medSet, N, episode_n, ts, beta, W1, W2, k}")
#     saveRes(run, makeCompetitionName(medSet, N, episode_n, ts, beta, W1, W2, k))
# # %%
# def cy_competitionFullEvo(episode_n=100000, medSet=[0, 1], save=False):
#     n_trials = 1
#     N = 500
#     W1 = 1
#     W2 = 1
#     ts = (2, -1)
#     beta = 0.005
#     k = 30
#     runs = [cy_runCompetitionExperiment(N=N, episode_n=episode_n, W1=W1, W2=W2, ts=ts, beta=beta,
#                                         k=k, saveHistory=True, history=[], medSet=medSet) for i in range(n_trials)]
#     if save:
#         print(f"saving {medSet, N, episode_n, ts, beta, W1, W2, k}")
#         saveRes(runs,   makeCompetitionName(
#             medSet, N, episode_n, ts, beta, W1, W2, k))
#     return runs
# %%
# Sinlge mediator, 1 trial, simulations over many games
def tsMatrixSim(med=0, M=5):
    med=0
    M=5
    ts = (2, -1)
    gameParams = genTSParams(M)
    medSet = [med]
    n_trials = 1
    episode_n = 100000
    N = 500
    W1 = 1
    W2 = 0
    beta = 0.005
    k = 30

    runs = {ts: saveCompetitionExperiment(N=N, episode_n=episode_n, W1=W1, W2=W2, ts=ts, beta=beta, k=k, saveHistory=False, history=[], medSet=medSet) for ts in gameParams}
    return runs


# %%
cy_competitionFullEvo(save=True)
