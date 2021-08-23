# evolution using networkx
# %%
from graph_tool import spectral
import networkx as nx
from typing import Dict
from optimized.cyFns import *
import array
from time import time
from math import inf
import numpy as np
from egt_io import saveRes, makeCompetitionName, timestamp
from sampling_nx import sampleNeighbor, isLonely
from updates import updateTies, calcStructuralUpdate, TieUpdate, fermi
import pandas as pd
from copy import deepcopy
from collections import Counter
from math import floor, inf
from functools import partial
from defaultParams import *
from defaultParams import _episode_n, _N, _W, _W2, _k, _T, _S
from init import initUniformRandomGraph, initPayoffs, initStrats
from mediators_nx import _medSet
from mediators_nx import *
from games import pairCumPayoffs, makeTSDilemma, nodeCumPayoffs
from sampling_nx import sampleNeighbor
from updates import calcStrategyUpdate, updateStrat, calcStructuralUpdate, updateTies, calcMedUpdate, updateMed
from pathlib import Path
import sys
from numba import njit, jit
from numba.typed import List
from defaultParams import _T, _R, _S, _P
import random
import graph_tool as gt
# UPDATES


@njit
def sampleOne(a):
    return np.random.default_rng().choice(a, 1, replace=False)[0]


@njit
def fermi(beta, fitness_diff):
    clipDiff = max(0, fitness_diff)
    exponent = -1 * clipDiff * beta
    exponential = np.exp(exponent)
    return 1 / (1 + exponential)


# GAMES


def makeWeightDict(t, s):
    return {'R': _R, 'P': _P, 'T': t, 'S': s}


def makeDilemma(R=_R, P=_P, T=_T, S=_S):
    return np.array([[[P, P], [T, S]], [[S, T], [R, R]]])


def makeTSDilemma(t, s):
    return makeDilemma(**makeWeightDict(t, s))

# Returns the payoff as a function of the dilemma, the ids and strats of the players


@njit
def playDilemma(dilemma, strats, id1, id2):
    return dilemma[strats[id1]][strats[id2]]
# cumulativePayoffs :: graph -> [strat] # Cumulative Payoff of 1 round of a node playing all its neighbors


@njit
def nodeCumPayoffs(dilemma, neighbors, strats, x):
    total = 0
    for y in neighbors:
        total += playDilemma(dilemma, strats, x, y)[0]
    return total
# EVOLUTION


# def runEvolutionCompetitionEp( N,  beta,  W,  W2, dilemma, graph, medStrats, strats, history, _x, saveHistory=False):
update2Int = {"strat": 0, "rewire": 1, "mediator": 2}
int2Update = {0: "strat", 1: "rewire", 2: "mediator"}


def update2Array(update): return [
    update["updateType"], update["x"], update["old"], update["new"]]


def array2Update(array): return {
    "updateType": int2Update[array[0]], "x": array[1], "old": array[2], "new": array[3]}


def decideRewire(graph, strats, x, y, medStrats):
    if isLonely(graph, y):
        return [1, x, y, y]   # enforcing graph connectedness
    z = useMed(medStrats[x], graph, strats, medStrats, y, x)
    if not z:
        return [1, x, y, y]  # enforcing graph connectedness
    return [1, x, y, z]


def calcStructuralUpdate(graph, strats, x, y, p, medStrats):
    sx, sy = strats[x], strats[y]
    if (sx == C and sy == D):
        doRewire = randFloat() < p
        if doRewire:
            return decideRewire(graph, strats, x, y, medStrats)
        else:
            return [1, x, y, y]
    elif (sx == D and sy == D):
        keepX = randFloat() < p
        args = [x, y] if keepX else [y, x]
        return decideRewire(graph, strats, args[0], args[1], medStrats)
    return [1, x, y, y]


def rewireEdge(graph, x, y, z):
    graph.remove_edge(x, y)
    graph.add_edge(x, z)
    return graph


@njit
def initEp():
    return
# using netowrkx graphs


def runEvolutionCompetitionEp(N, beta, W1, W2, dilemma, graph, medStrats, strats, history, x, saveHistory=False):
    neighs_x = np.fromiter(graph.neighbors(x), dtype=np.intc)
    # y = neighs_x[random.randint(0, len(neighs_x)-1)]
    y = neighs_x[crandint(0, len(neighs_x)-1)]
    neighs_y = np.fromiter(graph.neighbors(y), dtype=np.intc)
    # neighs_x = list(graph.neighbors(x))
    # y = random.sample(neighs_x, 1)[0]
    # neighs_y = list(graph.neighbors(y))
    px = nodeCumPayoffs(dilemma, neighs_x, strats, x)
    py = nodeCumPayoffs(dilemma, neighs_y, strats, y)
    doMedUpdate = randFloat() * (1+W2) > 1
    did_rewire = 0
    if doMedUpdate:
        p_med = fermi(beta*10, py - px)
        doChangeStrat = randFloat() < p_med
        sx = medStrats[x]
        new = medStrats[y] if doChangeStrat else sx
        medUpdate = [0, x, sx, new]
        # medUpdate = calcMedUpdate(medStrats, x, y, p_med)
        if saveHistory:
            history.append(medUpdate)
        medStrats[medUpdate[1]] = medUpdate[3]
    else:
        r = randFloat()
        doStratUpdate = randFloat() * (1+W1) <= 1
        if doStratUpdate:
            p_strat = cy_fermi(beta, py - px)
            doChangeStrat = randFloat() < p_strat
            sx = strats[x]
            new = strats[y] if doChangeStrat else sx
            stratUpdate = [2, x, sx, new]
            # stratUpdate = {"updateType": "strat","x": x, "old": sx, "new": new}
            # stratUpdate = calcStrategyUpdate(strats, x, y, p_strat)
            if saveHistory:
                history.append(stratUpdate)
            strats[stratUpdate[1]] = stratUpdate[3]
            # strats = updateStrat(strats, stratUpdate)
        else:
            p_rewire = cy_fermi(beta, py - px)
            graphUpdate = calcStructuralUpdate(
                graph, strats, x, y, p_rewire, medStrats)
            if saveHistory:
                history.append(graphUpdate)
            graph = rewireEdge(
                graph, graphUpdate[1], graphUpdate[2], graphUpdate[3]) if graphUpdate[2] != graphUpdate[3] else graph  # only update if old and new are different
            # did_rewire = 1
            # track rewire attempts, which happen if x is unsatisfied
            did_rewire = 1 if strats[y] == D else 0
    return graph, history, did_rewire


def genericRunEvolution(N, episode_n, W1, W2, dilemma,  medStrats, strats, beta=0.005, graph=None, k=30, history=None, saveHistory=False, endOnStratConverge=False, log=True):
    # history = [] if history == None else history
    history = [Dict for x in range(0)] if history == None else history
    initialStrats = np.empty(N, dtype=np.intc)
    initialMedStrats = np.empty(N, dtype=np.intc)
    initialStrats = strats[:]
    initialMedStrats = medStrats[:]
    totalPayoffs = initPayoffs(N)
    graph = graph
    graph.set_fast_edge_removal(True)
    timestep = 0
    x = 0
    t, s = dilemma[0][1]
    rewire_n = 0
    xs = np.random.randint(N-1, size=episode_n)
    for x, i in enumerate(xs):
        timestep = i
        graph, history, did_rewire = runEvolutionCompetitionEp(
            N, beta, W1, W2, dilemma, graph, medStrats, strats, history, x, saveHistory)
        rewire_n += did_rewire
        if log and i % 5000 == 0:
            medEvoDone = any([x == N for x in Counter(medStrats).values()])
            stratEvoDone = any([x == N for x in Counter(strats).values()])
            if (medEvoDone and stratEvoDone) or (endOnStratConverge and stratEvoDone):
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
            "medStrats": np.asarray(medStrats, dtype=np.intc),
            "rewire_n": rewire_n,
            "stop_n": timestep,
            "timestamp": timestamp(),
            "params": {"N": N, "episode_n": episode_n, "W1": W1, "W2": W2, "t": t, "s": s, "beta": beta, "k": k, "medSet": np.unique(initialMedStrats), "endOnStratConverge": endOnStratConverge}}


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

# I should be able to take strats and medStrats but for debugging purposes, I'm making it make them from scratch every


def runCompetitionExperiment(N=_N, episode_n=_episode_n, W1=_W, W2=_W2, graph=None, ts=(_T, _S), medStrats=None, strats=None, beta=0.005, k=30, medSet=_medSet, history=None, saveHistory=False, endOnStratConverge=True, smallMedInit=False):
    dilemma = makeTSDilemma(ts[0], ts[1])
    _graph = deepcopy(graph) if graph else initUniformRandomGraph(
        N=N, k=(k if k else _k))
    medStrats = initMedStratsSmall(
        N, medSet, 0) if smallMedInit else initMedStrats(N, medSet)
    experimentResults = genericRunEvolution(
        N, episode_n, W1, W2, dilemma, medStrats, initStrats(N), beta, deepcopy(_graph), k, history, saveHistory=saveHistory, endOnStratConverge=endOnStratConverge)
    return experimentResults

# def continueCompetitionExperiment(graph, int[:] medStrats, int[:] strats, int N=_N, int episode_n=_episode_n, float W1=_W, float W2=_W2, ts=(_T,_S), float beta=0.005, int k=30, medSet=_medSet, history=None, bint saveHistory=False):
#     cdef float[:, :, :] dilemma
#     dilemma = makeTSDilemma(ts[0], ts[1])
#     _graph = deepcopy(graph) if graph else initUniformRandomGraph(
#         N=N, k=(k if k else _k))
#     experimentResults = genericRunEvolution(
#         N, episode_n, W1, W2, dilemma, medStrats, strats, beta, deepcopy(_graph), k, history, saveHistory=saveHistory)
#     return experimentResults


# # %%
# N = 500
# beta = 0.005
# k = 30
# medSet = [5, 6, 7, 8]
# strats = initStrats(N)
# medStrats = initMedStratsSmall(N, medSet, 0)
# medStrats = initMedStrats(N, medSet)
# dilemma = makeTSDilemma(2, -1)
# x = np.random.randint(0, N-1)
# W1 = inf
# W2 = 0
# graph = initUniformRandomGraph(N, k=k)
# graph_nx = nx.Graph(spectral.adjacency(graph))
# graph = nx.relabel.convert_node_labels_to_integers(graph_nx)
# history = []
# runEvolutionCompetitionEp(N, beta, W1, W2, dilemma, graph,
#                           medStrats, strats, history, x, saveHistory=False)
# # %%
# % % timeit
# x = crandint(0, N-1)
# runEvolutionCompetitionEp(N, beta, 1, 0.1, dilemma, graph,
#                           medStrats, strats, history, x, saveHistory=False)


# # %%
# %load_ext line_profiler
# # profiling
# # %%
# %lprun - f  runEvolutionCompetitionEp runEvolutionCompetitionEp(N, beta, 0, 0, dilemma, graph_nx, medStrats, strats, history, x, saveHistory=False)

# #

# # %%
# # compatible with numba


# @njit
# def _sampleStratEligibleX(neighbors, strats, medStrats, strat, medStrat, x):
#     ofStrat = np.logical_and(
#         strats == strat, medStrats == medStrat).nonzero()[0]
#     if ofStrat.shape[0] == 0:
#         return None
#     else:
#         pre_el = set(ofStrat) - set(neighbors)
#         eligible = list(pre_el - set([x]))
#         if len(eligible) <= 0:
#             return None
#         return eligible


# def sampleStratEligibleX(graph, strats, medStrats, strat, medStrat, x):
#     eligible = _sampleStratEligibleX(
#         set(graph.neighbors(x)), strats, medStrats, strat, medStrat, x)
#     return eligible[crandint(0, len(eligible)-1)]
# # %%


# @njit
# def testtype(a, b):
#     return set(a)-set(b)


# # %%
# strat, medStrat = 0, 0
