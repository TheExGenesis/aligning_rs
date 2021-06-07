from libc.math cimport exp
from games import pairCumPayoffs, makeTSDilemma, nodeCumPayoffs
from math import inf
import numpy as np
from evolution import runCompetitionExperiment, runEvolutionCompetitionEp
from egt_io import saveRes, makeCompetitionName
from mediators import NO_MED, GOOD_MED, BAD_MED
from init import initStrats, initUniformRandomGraph
from games import makeTSDilemma, pairCumPayoffs
from mediators import initMedStrats, useMed
from sampling import sampleNeighbor, isLonely
from updates import updateTies, calcStructuralUpdate, TieUpdate
import pandas as pd
from copy import deepcopy
from collections import Counter
from math import floor, inf
from functools import partial

from defaultParams import _episode_n, _N, _W, _W2, _k, genTSParams, _T, _S
from init import initUniformRandomGraph, initPayoffs, initStrats
from mediators import initMedStrats, _medSet, useNoMed
from games import pairCumPayoffs, makeTSDilemma
from sampling import sampleNeighbor
from updates import calcStrategyUpdate, updateStrat, calcStructuralUpdate, updateTies, calcMedUpdate, updateMed
from pathlib import Path
import sys

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, exp

cdef int crandint(int lower, int upper) except -1:
    return (rand() % (upper - lower + 1)) + lower

cimport numpy as np

# from cyFns import fermi


cdef inline float float_max(float a, float b): return a if a >= b else b
cdef inline float float_min(float a, float b): return a if a <= b else b


cdef inline float clamp(float val, float minval, float maxval): return float_max(minval, float_min(val, maxval))
cdef inline float clip(float val, float minval): return float_max(minval, val)


cdef float fermi(float beta, float fitness_diff):
    cdef float clipDiff = clip(fitness_diff, 0)
    cdef float exponent = -1 * beta * clipDiff
    cdef float exponential = exp(exponent)
    # cdef float exponential = np.exp(exponent, dtype=np.float64)
    cdef float p = 1. / (1. + exponential)
    return p


def runEvolutionCompetitionEp(int N, float beta, float W, float W2, dilemma, graph, medStrats, strats, history, int _x, bint saveHistory=False):
    cdef int _y = sampleNeighbor(graph, _x)
    # cdef int x = int(_x)
    # cdef int y = int(_y)
    cdef int x = _x
    cdef int y = _y
    cdef float px = nodeCumPayoffs(dilemma, graph, strats, _x)
    cdef float py = nodeCumPayoffs(dilemma, graph, strats, _y)
    cdef float p = fermi(beta, py - px)
    cdef bint doMedUpdate = rand() * (1+W2) > 1
    if doMedUpdate:
        medUpdate = calcMedUpdate(beta, graph, dilemma, medStrats, x, y, p)
        if saveHistory:
            history.append(medUpdate)
        medStrats = updateMed(medStrats, medUpdate)
    else:
        doStratUpdate = np.random.rand() * (1+W) <= 1
        if doStratUpdate:
            stratUpdate = calcStrategyUpdate(
                beta, graph, dilemma, strats, x, y, p)
            if saveHistory:
                history.append(stratUpdate)
            strats = updateStrat(strats, stratUpdate)
        else:
            graphUpdate = calcStructuralUpdate(
                beta, graph, dilemma, strats, _x, _y, p, medStrats)
            if saveHistory:
                history.append(graphUpdate)
            graph = updateTies(graph, graphUpdate)
    return strats, graph, history


def genericRunEvolution(runEvolutionEp, int N, int episode_n, float W, float W2, dilemma, medStrats, strats, beta=0.001, _graph=None, k=None, history=None, saveHistory=False):
    history = [] if history == None else history
    print(
        f"starting runEvolution, history len= {len(history)}, {'saving history, ' if saveHistory else ''} N={N}, episode_n={episode_n}")
    initialStrats = deepcopy(strats)
    initialMedStrats = deepcopy(medStrats)
    totalPayoffs = initPayoffs(N)
#     history = [] #[{updateType: "strat", x: N, old: {C,D}, new: {C,D}}, {updateType: "rewire", xy: (x,y), old: (x,y), new: (x,z)}]
    graph = deepcopy(_graph) if _graph else initUniformRandomGraph(
        N=N, k=(k if k else _k))
    cdef int start = 0
    cdef int step = 1
    cdef int stop = episode_n
    cdef int i = start - step
    cdef int length = len(range(start, stop, step))
    cdef int x = 0
    print("lol wooooooop")
    for _ in range(length):
        i += step
        x = crandint(0, N-1)
        strats, graph, history = runEvolutionEp(
            N, beta, W, W2, dilemma, graph, medStrats, strats, history, x, False)
            # N, beta, W, W2, dilemma, graph, medStrats, strats, history, x, saveHistory)
        if i % 5000 == 0:
            medEvoDone = any([x == N for x in Counter(medStrats).values()])
            stratEvoDone = any([x == N for x in Counter(strats).values()])
            if medEvoDone and stratEvoDone:
                break
            if W2 == inf and medEvoDone:
                break
            if W == 0 and stratEvoDone:
                break
    return {"graph": graph, "history": pd.DataFrame(history, dtype="category"), "initStrats": initialStrats, "finalStrats": deepcopy(strats), "initMedStrats": initialMedStrats,  "medStrats": deepcopy(medStrats)}


# Initializes strats, payoffs, history, graph and runs many evolution episodes, each for a random node x
# runEvolution :: int -> graph -> [strat] -> [float] -> [[strat]]
runEvolutionCompetition = partial(
    genericRunEvolution, runEvolutionCompetitionEp)


def cy_runCompetitionExperiment(int N=_N, int episode_n=_episode_n, float W=_W, float W2=_W2, graph=None, ts=(_T, _S), medStrats=None, strats=None, float beta=0.005, int k=30, medSet=_medSet, history=None, bint saveHistory=False):
    strats = deepcopy(strats) if strats else initStrats(N)
    medStrats = deepcopy(medStrats) if medStrats else initMedStrats(N, medSet)
    experimentResults = runEvolutionCompetition(N, episode_n, W, W2, makeTSDilemma(
        *ts), deepcopy(medStrats), deepcopy(strats), beta, deepcopy(graph), k, history, saveHistory=saveHistory)
    return experimentResults

# %%


def competitionFullEvo(medSet, save=False):
    n_trials = 1
    N = 500
    episode_n = 100
    W1 = 1
    W2 = 1
    ts = (2, -1)
    beta = 0.005
    k = 30
    print("lol going")
    runs = [cy_runCompetitionExperiment(N=N, episode_n=episode_n, W=W1, W2=W2, ts=ts, beta=beta,
                                        k=k, saveHistory=True, history=[], medSet=medSet) for i in range(n_trials)]
    if save:
        saveRes(runs, makeCompetitionName(
            medSet, N, episode_n, ts, beta, W1, W2, k))
    return runs

