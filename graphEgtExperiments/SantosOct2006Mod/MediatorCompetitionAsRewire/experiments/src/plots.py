# Plots
import ternary
from graph_tool.draw import *
import pandas as pd
from collections import Counter
import numpy as np
import graph_tool as gt
import seaborn as sns
from defaultParams import _T, _R, _S, _P, C, D
from utils import transposeList


def drawGraph(graph, strats=None, mplfig=None):
    strats = strats if strats else [None for i in range(graph.num_vertices())]
    strat2Color = {C: 'green', D: 'red', None: 'gray'}
    color = graph.new_vertex_property("string")
    for i, strat in enumerate(strats):
        color[graph.vertex(i)] = strat2Color[strat]
    if mplfig:
        print('drawing graph as subplot')
        graph_draw(graph, bg_color="white",
                   vertex_fill_color=color, mplfig=mplfig)
    else:
        graph_draw(graph, bg_color="white", vertex_fill_color=color)


# plot graph with colored edges


def plotGraph2(n, graph, strats):
    def edgeStratColor(e):
        c = (strats[int(e.target())] == C) + (strats[int(e.source())] == C)
        return {0: 'red', 1: 'orange', 2: 'green'}[c]

    vIds = graph.new_vertex_property("string")
    stratColors = graph.new_vertex_property("string")
    edgeColors = graph.new_edge_property("string")
    for i in range(n):
        stratColors[i] = {C: 'green', D: 'red'}[strats[i]]
        vIds[i] = f'{i}'
    for e in graph.edges():
        edgeColors[e] = edgeStratColor(e)
    deg = graph.degree_property_map("out")
    deg.a = 25 * (np.sqrt(deg.a/n))+3
    pos = fruchterman_reingold_layout(graph)
    graph_draw(graph, pos=pos, vertex_text=vIds, vertex_font_size=16,
               vertex_fill_color=stratColors, vertex_size=deg, edge_color=edgeColors)


def plotHist(_list, bins):
    return pd.Series(_list).plot.hist(bins=bins)


# plot for different initial K, for different beta
def finalCoopsByW(res, game=(2.0, -1.0)):
    return pd.DataFrame({"W": manyTsRes.keys(), "cooperators": [Counter(manyTsRes[w][game]['episodes'][-1])[C] for w in manyTsRes.keys()]}).plot.line(x="W", y="cooperators")


def plotDegreeLog(yFn, graph, title='', xlabel="$k$", ylabel="$NP(k)$"):
    hist = gt.stats.vertex_hist(graph, 'out')
    y = yFn(hist[0])
    err = np.sqrt(y)
    err[err >= y] = y[err >= y] - 1e-2
    plt.plot(hist[1][:-1], y, "o", label="degree")
#     plt.errorbar(hist[1][:-1], y, fmt="o", yerr=err,label="in")
    plt.xlabel(xlabel)
    plt.ylabel("$NP(k)$")
    plt.tight_layout()
    ax = plt.gca()
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim(1e-3, 1.5)
    ax.set_xlim(0.8, 1e3)
    return ax


def plotDD(graph):
    return plotDegreeLog(lambda y: y/graph.num_vertices(), graph)


def plotCDD(graph):
    return plotDegreeLog(lambda y: np.flip(np.cumsum(np.flip(y)))/graph.num_vertices(), graph, ylabel="$D(k)$")


def avgSquares(graph):
    counts, degrees = gt.stats.vertex_hist(graph, 'out')
    return np.sum([(degrees[i]**2)*counts[i]
                   for i in range(len(counts))])/graph.num_vertices()


def squaredAvg(graph):
    gt.stats.vertex_average(graph, 'out')[0]**2


def heterogeneity(graph):
    return avgSquares(graph) - squaredAvg(graph)


def plotLandscape(ts, vals, axis=None, valName=''):
    size = 4
    M = len(vals)
    df = pd.DataFrame(zip(vals, *transposeList(ts)),
                      columns=[valName, 't', 's'])
    df = df.pivot('s', 't', valName).iloc[::-1]
    if not axis:
        # Sample figsize in inches
        fig, ax = plt.subplots(figsize=(size+1, size))
        ax = sns.heatmap(df, annot=True, cbar=True, xticklabels=2,
                         yticklabels=2, ax=ax)  # .set_title(title)
        #  ax = sns.heatmap(df, annot=False, cbar=True, xticklabels=2, yticklabels=2)#.set_title(title)
        plt.show()
    else:
        ax = sns.heatmap(df, annot=False, cbar=True, xticklabels=2,
                         yticklabels=2, ax=axis)  # .set_title(title)
    return ax


def coopCount(finalStrats):
    return Counter(finalStrats)['C']


def coopLandscape(ts, res, title='', axis=None):
    cCounts = [coopCount(r['finalStrats']) for r in res]
    return plotLandscape(ts, cCounts, axis=axis, valName='coop counts')


def heterogeneityLandscape(ts, res, title='', axis=None):
    hVals = [heterogeneity(r['graph']) for r in res]
    return plotLandscape(ts, hVals, axis=axis, valName='heterogeneity')


def maxDegreeLandscape(ts, res, title='', axis=None):
    hVals = [maxDegree(r['graph']) for r in res]
    return plotLandscape(ts, hVals, axis=axis, valName='max degree (k)')


#  cCounts = [Counter(r['episodes'][-1])['C'] for r in res]
#     M = len(cCounts)
#     df = pd.DataFrame(zip(cCounts, *transposeList(ts)), columns=['count', 't', 's'])
#     df = df.pivot('s', 't', 'count').iloc[::-1]
#     if not axis:
#         ax = sns.heatmap(df, annot=False, cbar=True, xticklabels=2, yticklabels=2)#.set_title(title)
#         plt.show()
#     else:
#         ax = sns.heatmap(df, annot=False, cbar=True, xticklabels=2, yticklabels=2, ax=axis)#.set_title(title)
#     return ax

def hist2StratCount(old, new):
    d = {D: {C: 1, D: 0},  C: {C: 0, D: -1}}
    return d[old][new]


def historyToStratCounts(initStrat, history):
    N = len(initStrat)
    initC = Counter(initStrat)[C]
    # steps = history | > filter$(lambda x: x["updateType"] == "strat") | > map$(lambda x: hist2StratCount(x['old'], x['new'])) | >list
    steps = list(map(lambda x: hist2StratCount(x['old'], x['new']), filter(
        lambda x: x["updateType"] == "strat", history)))
    Cs = np.cumsum(steps)+initC
    Ds = map(lambda c: N-c, Cs)
    return [{C: c, D: d} for c, d in zip(Cs, Ds)]


def plotStratEvo(initStrat, history, nSteps=500):
    stratCounts = historyToStratCounts(initStrat, history)
    stepSize = int(max(np.floor(len(stratCounts)/nSteps), 1))
    pd.DataFrame(stratCounts[::stepSize]).plot.line(
        color={C: 'orange', D: 'blue'})


def dfStratCounts(run):
    stratSet = [C, D]
    initCounts = Counter(run["initStrats"])
    N = len(run["initStrats"])
    dfHist = pd.DataFrame(run['history'])
    dfStratHist = dfHist[dfHist["updateType"]
                         == "strat"].drop(columns=["updateType"])
    for strat in stratSet:
        colName = strat+"_delta"
        dfStratHist[colName] = 0
        dfStratHist.loc[dfStratHist['new'].values == strat, colName] += 1
        dfStratHist.loc[dfStratHist['old'].values == strat, colName] -= 1
    dfStratHist = dfStratHist[dfStratHist.columns.difference(['old', 'new'])]
    dfStratCounts = dfStratHist[[
        strat+"_delta" for strat in stratSet]].cumsum()
    for strat in stratSet:
        dfStratCounts[strat+"_delta"] += initCounts[strat]
    dfStratCounts = dfStratCounts.rename(
        columns={strat+"_delta": strat+"_count" for strat in stratSet})
    return dfStratCounts/N


def plotStratCounts(run):
    stratCounts = dfStratCounts(run)
    stratCounts.plot.line(color={"C_count": 'orange', "D_count": 'blue'})

    '''Simplex plots'''


def simplexPath(episodes, title=""):
    # Sample trajectory plot
    figure, tax = ternary.figure(scale=1.0)
    figure.set_size_inches(15, 15, forward=True)
    tax.boundary()
    tax.gridlines(multiple=0.2, color="black")
    tax.set_title(title, fontsize=20)
    points = []
    # Plot the data
    tax.plot(episodes, linewidth=2.0, label="Curve")
    tax.ticks(axis='lbr', multiple=0.2, linewidth=1, tick_formats="%.1f")
    tax.legend()
    tax.show()


def simplexPaths(runs, title=''):
    fontsize = 10
    figure, tax = ternary.figure(scale=1.0)
    figure.set_size_inches(15, 15, forward=True)
    tax.boundary()
    tax.gridlines(multiple=0.2, color="black")
    tax.set_title(title, fontsize=20, y=1.08)
    points = []
    tax.right_corner_label(NO_MED, fontsize=fontsize)
    tax.top_corner_label(GOOD_MED, fontsize=fontsize)
    tax.left_corner_label(FAIR_MED, fontsize=fontsize)

    # Plot the data
    for run in runs:
        tax.plot(run, linewidth=2.0, label="Curve")
    tax.ticks(axis='lbr', multiple=0.2, linewidth=1, tick_formats="%.1f")
    tax.legend()
    tax.show()


def medPropSimplex(runs, title=''):
    vecRuns = map(lambda run: map(makePropVector3, run), runs)
    simplexPaths(vecRuns, title)
    return


def dfMedDeltas(run, medSet=None):
    if medSet == None:
        initCounts = Counter(run["initMedStrats"])
        medSet = sorted(initCounts.keys())
    dfHist = pd.DataFrame(run['history'])
    dfMedHist = dfHist[dfHist["updateType"] ==
                       "mediator"].drop(columns=["updateType"])
    for med in medSet:
        colName = med+"_delta"
        dfMedHist[colName] = 0
        dfMedHist.loc[dfMedHist['new'].values == med, colName] += 1
        dfMedHist.loc[dfMedHist['old'].values == med, colName] -= 1
    return dfMedHist[dfMedHist.columns.difference(['old', 'new'])]


def dfMedCounts(run):
    initCounts = Counter(run["initMedStrats"])
    medSet = sorted(initCounts.keys())
    N = len(run["initMedStrats"])
    dfMedHist = dfMedDeltas(run, medSet)
    dfMedCounts = dfMedHist[[medName+"_delta" for medName in medSet]].cumsum()
    for med in medSet:
        dfMedCounts[med+"_delta"] += initCounts[med]
    dfMedCounts = dfMedCounts.rename(
        columns={med+"_delta": med+"_count" for med in medSet})
    return dfMedCounts/N


def plotMedEvolution(run, nSteps=500):
    stepSize = int(max(np.floor(len(run['initMedStrats'])/nSteps), 1))
    return simplexPath(dfMedCounts(run)[::stepSize].to_numpy())


def plotMedEvolutions(runs, nSteps=500):
    stepSize = int(max(np.floor(len(runs[0]['initMedStrats'])/nSteps), 1))
    return simplexPaths([dfMedCounts(run)[::stepSize].to_numpy() for run in runs], "")


def plotMedEvolutions(runs, nSteps=500):
    stepSize = int(max(np.floor(len(runs[0]['initMedStrats'])/nSteps), 1))
    return simplexPaths([dfMedCounts(run)[::stepSize].to_numpy() for run in runs], "")


def plotKeysCoopByW(unif_res):
    return pd.DataFrame({key: [Counter(x['episodes'][-1])[C] for w, x in res.items()] for key, res in unif_res.items()}, index=list(unif_res.values())[0].keys()).plot.line()


def plotKeysMaxKByW(unif_res):
    return pd.DataFrame({key: [maxDegree(x['graph']) for w, x in res.items()] for key, res in unif_res.items()}, index=list(unif_res.values())[0].keys()).plot.line()


def plotBetasCoopByW(unif_res):
    return pd.DataFrame({beta: [Counter(x['episodes'][-1])[C] for w, x in res.items()] for beta, res in unif_res.items()}, index=list(unif_res.values())[0].keys()).plot.line()


def plotBetasMaxKByW(unif_res):
    return pd.DataFrame({beta: [maxDegree(x['graph']) for w, x in res.items()] for beta, res in unif_res.items()}, index=list(unif_res.values())[0].keys()).plot.line()


def maxDegree(graph):
    return np.max(graph.get_out_degrees(graph.get_vertices()))

# Multi Plots


# graphs don't show up if we pass them the axis
def graphBeforeAfter(graph0, strats0, graph1, strats1):
    size = 4
    xn = 2
    yn = 1
    fig, ax = plt.subplots(yn, xn, figsize=(yn*size, xn*size))
    drawGraph(graph0, strats=strats0, mplfig=ax[0])
    drawGraph(graph1, strats=strats1, mplfig=ax[1])
    plt.show()


def plot1D(res, plotFn, title='', keyName=''):
    size = 4
    items = res.items()
    n = len(items)
    fig, ax = plt.subplots(1, n, figsize=((n+1)*size, size))
    fig.suptitle(title, fontsize=16, y=1.08)
    for i, (k, v) in enumerate(items):
        plt.subplot(1, n, i+1)
        ax = plotFn(k, v, keyName)


def plot2D(M, res, plotFn, title='', keyName=''):
    size = 4
    items = res.items()
    n = len(items)
    fig, ax = plt.subplots(M, M, figsize=(M*size, M*size))
    fig.suptitle(title, fontsize=16, y=1.08)
    for i, (k, v) in enumerate(items):
        plt.subplot(M, M, M*M-i)
        ax = plotFn(k, v, keyName)


def plotDDFn(k, v, keyName=''):
    #     graph, totalPayoffs, episodes = v
    ax = plotDD(v['graph'])
    ax.set_title(f'DD {keyName}={k}')
    return ax


def plotCDDFn(k, v, keyName=''):
    ax = plotCDD(v['graph'])
    ax.set_title(f'CDD {keyName}={k}')
    return ax


def plotStratEvoFn(k, v, keyName=''):
    #     graph, totalPayoffs, episodes = v
    stratCounts = historyToStratCounts(v['initStrats'], v['history'])
    ax = pd.DataFrame(stratCounts).plot(
        color={C: 'orange', D: 'blue'}, ax=plt.gca())
    ax.set_title(f'{keyName}={k}')
    return ax


def plotHistFn(k, v, keyName=''):
    #     graph, totalPayoffs, episodes = v
    ax = plotHist(*gt.stats.vertex_hist(v['graph'], 'out'))
    return ax


def plotCoopLandscapeFn(k, v, keyName=''):
    #     ts,res = v
    #     ax = coopLandscape(v['ts'], v['results'], title='', axis=plt.gca())
    ax = coopLandscape(v.keys(), v.values(), title='', axis=plt.gca())
    ax.set_title(f'{keyName}={k}')
    return ax


def plotHetLandscapeFn(k, v, keyName=''):
    #     ts,res = v
    #     ax = heterogeneityLandscape(v['ts'], v['results'], title='', axis=plt.gca())
    ax = heterogeneityLandscape(v.keys(), v.values(), title='', axis=plt.gca())
    ax.set_title(f'{keyName}={k}')
    return ax


def plotMaxDegreeLandscapeFn(k, v, keyName=''):
    ax = maxDegreeLandscape(v.keys(), v.values(), title='', axis=plt.gca())
    ax.set_title(f'{keyName}={k}')
    return ax

# for i in range(M):
#     plot2D(M{ts[M*i+j]:res for j,res in enumerate(results[M*i:M*(i+1)])}, plotStratEvoFn, keyName='t,s')


def plotStratMatrix(tsExpRes):
    _results = orderTsMatrixPlot(list(tsExpRes.values()), M)
#     _results = tsExpRes.values()
#     _ts = tsExpRes.keys()
    _ts = orderTsMatrixPlot(list(tsExpRes.keys()), M)
    tsRes = {ts: res for ts, res in reversed(zip(_ts, _results))}
#     tsRes = {ts:res for ts,res in zip(_ts, _results)}
    return plot2D(M, tsRes, plotStratEvoFn, keyName='t,s')


def plotCDDMatrix(tsExpRes):
    _results = orderTsMatrixPlot(list(tsExpRes.values()), M)
#     _results = tsExpRes.values()
#     _ts = tsExpRes.keys()
    _ts = orderTsMatrixPlot(list(tsExpRes.keys()), M)
    tsRes = {ts: res for ts, res in reversed(zip(_ts, _results))}
#     tsRes = {ts:res for ts,res in zip(_ts, _results)}
    return plot2D(M, tsRes, plotCDDFn, keyName='t,s')
