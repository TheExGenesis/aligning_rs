from collections import Counter


def countUpdateTypes(history):
    updateTypes = history | > map$(.["updateType"])
    return Counter(updateTypes) | > dict


def nEmptyRewires(history):
    return history | > filter$(x -> x["updateType"] == "rewire" and x['old'] == x['new']) | > list | > len


def nEmptyStratChanges(history):
    return history | > filter$(x -> x["updateType"] == "strat" and x['old'] == x['new']) | > list | >  len


def historyStats(history):
    print(f'countUpdateTypes: {countUpdateTypes(history)}')
    print(f'nEmptyRewires: {nEmptyRewires(history)}')
    print(f'nEmptyStratChanges: {nEmptyStratChanges(history)}')


def coopCountsRes(res):
    gameParams = list(res[0].keys())
    coopCounts = {game: list(map(x -> coopCount(x[game]['finalStrats']), res)) for game in gameParams}
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
    return map$(obj[key])
