from cyFns import fermi
import numpy as np
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
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path + "/pyx")
# W2 is the time ratio between med update and (rewire/strat update). W2=0 means there are no med updates


def runEvolutionCompetitionEp(N, beta, W, W2, dilemma, graph, medStrats, strats, history, _x, saveHistory=False):
    _y = sampleNeighbor(graph, _x)
    x, y = int(_x), int(_y)
    px, py = pairCumPayoffs(dilemma, graph, strats, x, y)
    p = fermi(beta, py - px)
    doMedUpdate = np.random.rand() * (1+W2) > 1
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


def genericRunEvolution(runEvolutionEp, N, episode_n, W, W2, dilemma, medStrats, strats, beta=0.001, _graph=None, k=None, history=None, saveHistory=False):
    history = [] if history == None else history
    print(
        f"starting runEvolution, history len= {len(history)}, {'saving history, ' if saveHistory else ''} N={N}, episode_n={episode_n}")
    initialStrats = deepcopy(strats)
    initialMedStrats = deepcopy(medStrats)
    totalPayoffs = initPayoffs(N)
#     history = [] #[{updateType: "strat", x: N, old: {C,D}, new: {C,D}}, {updateType: "rewire", xy: (x,y), old: (x,y), new: (x,z)}]
    graph = deepcopy(_graph) if _graph else initUniformRandomGraph(
        N=N, k=(k if k else _k))
    for i in range(episode_n):
        x = np.random.randint(N)
        strats, graph, history = runEvolutionEp(
            N, beta, W, W2, dilemma, graph, medStrats, strats, history, x, saveHistory)
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

# Runs evolution, times it


def runCompetitionExperiment(N=_N, episode_n=_episode_n, W=_W, W2=_W2, graph=None, ts=(_T, _S), medStrats=None, strats=None, beta=0.005, k=None, medSet=_medSet, history=None, saveHistory=False):
    strats = deepcopy(strats) if strats else initStrats(N)
    medStrats = deepcopy(medStrats) if medStrats else initMedStrats(N, medSet)
    experimentResults = runEvolutionCompetition(N, episode_n, W, W2, makeTSDilemma(
        *ts), deepcopy(medStrats), deepcopy(strats), beta, deepcopy(graph), k, history, saveHistory=saveHistory)
    return experimentResults

# Runs one experiment for each game configuration in a MxM matrix os T,S values


def runTSExperiment(M, N=_N, episode_n=_episode_n, W=_W, graph=None, strats=None, beta=0.005, k=None, useMediator=useNoMed, saveHistory=False):
    strats = deepcopy(strats) if strats else initStrats(N)
    ts = genTSParams(M)
    results = {(t, s): runEvolutionCompetition(N=N, episode_n=episode_n, W=W, dilemma=makeTSDilemma(t, s), strats=deepcopy(strats), beta=beta,
                                               _graph=deepcopy(graph), k=k, useMediator=useMediator, saveHistory=(True if (t, s) == (2, -1) else False)) for t, s in ts}
    return results

# Runs many experiments, one for each value in argDict


def runManyExperiments(argDict):
    return {key: runCompetitionExperiment(**args) for key, args in argDict.items()}
# Runs many TS experiments, one for each value in argDict


def runManyTSExperiments(argDict):
    return {key: runTSExperiment(**args) for key, args in argDict.items()}


def continueTSExperiment(args, res):
    prevGraphs = {k: v['graph'] for k, v in res.items()}
    prevStrats = {k: v['finalStrats'] for k, v in res.items()}
    prevHistories = {k: v['history'] for k, v in res.items()}
    ts = genTSParams(args["M"])
    results = {(t, s): runEvolutionCompetition(N=args['N'], episode_n=args['episode_n'], W=args['W'], dilemma=makeTSDilemma(t, s), strats=deepcopy(prevStrats[(
        t, s)]), beta=args['beta'], graph=deepcopy(prevGraphs[(t, s)]), k=args['k'], useMediator=args['useMediator'], history=deepcopy(prevHistories[(t, s)])) for t, s in ts}
    return results
