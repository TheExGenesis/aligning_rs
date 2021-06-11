from collections import Counter
import numpy as np
import seaborn as sns
import pandas as pd
from itertools import chain
from functools import reduce


def countUpdateTypes(history):
    updateTypes = map(lambda x: x["updateType"], history)
    return dict(Counter(updateTypes))


def nEmptyRewires(history):
    return  len(list(filter(lambda x: x["updateType"] == "rewire" and x['old'] == x['new'], history)))


def nEmptyStratChanges(history):
    return  len(list(filter(lambda x: x["updateType"] == "strat" and x['old'] == x['new'], history)))


def historyStats(history):
    print(f'countUpdateTypes: {countUpdateTypes(history)}')
    print(f'nEmptyRewires: {nEmptyRewires(history)}')
    print(f'nEmptyStratChanges: {nEmptyStratChanges(history)}')


def coopCountsRes(res):
    gameParams = list(res[0].keys())
    coopCounts = {game: list(map(lambda x: coopCount(x[game]['finalStrats']), res)) for game in gameParams}
    return coopCounts


def fractionsAbsorbCoop(N, res, absorbFraction=0.90):

    nSimulations = len(res)
    coopCounts = coopCountsRes(res)
    gameParams = list(res[0].keys())
    fractions = {game: np.count_nonzero(np.array(
        coopCounts[game]) >= N*absorbFraction)/nSimulations for game in gameParams}
    return fractions


def avgFractionCoop(N, res):
    #     d = max(1, N/1e4)
    d = 1
    coopCounts = coopCountsRes(res)
    gameParams = list(res[0].keys())
    avgs = {game: np.mean(coopCounts[game])/N for game in gameParams}
    return avgs


def pickKeyMany(key, obj):
    return lambda x: map(obj[key], x)

#just copied in Jun 11

# list of dicts to dict of lists
def list2Dict(runs):
    all_keys = set(chain(*[x.keys() for x in runs]))
    return {k: [run[k] for run in runs] for k in all_keys}

# dict of lists


def coop(runs):
    runsByGame = list2Dict(runs).items()
    N = list(runs[0].values())[0]["initStrats"].size
    return {k: np.array([run['finalStrats'] for run in gameRuns]).sum(axis=1).mean()/N for k, gameRuns in runsByGame}

# takes list of dicts of results


def medCountsDf(results):
    N = list(results[0].values())[0]["initStrats"].size
    n_trials = len(results)
    def makeMedDf(res): return pd.DataFrame(
        {k: Counter(r["medStrats"]) for k, r in res.items()}).fillna(0)
    counts = reduce(lambda x, y: x+y, [makeMedDf(res)
                    for res in results])/n_trials
    return counts/N


def plotCoop(results):
    c = coop(results)
    df = pd.DataFrame(zip(list(c.values()), *transposeList(list(c.keys()))),
                      columns=["coops", 't', 's']).pivot('s', 't', "coops").iloc[::-1]
    fig = sns.heatmap(df, annot=True, cbar=True, xticklabels=2,
                      yticklabels=2, vmin=0, vmax=1).set(title="Avg. final cooperators")

