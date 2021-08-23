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
from analysis import *
from mediators import *


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
    stratName = ["C", "D"]
    initCounts = Counter(run["initStrats"])
    N = len(run["initStrats"])
    dfHist = pd.DataFrame(run['history'])
    dfStratHist = dfHist[dfHist["updateType"]
                         == "strat"].drop(columns=["updateType"])
    for strat in stratSet:
        colName = stratName[strat]+"_delta"
        dfStratHist[colName] = 0
        dfStratHist.loc[dfStratHist['new'].values == strat, colName] += 1
        dfStratHist.loc[dfStratHist['old'].values == strat, colName] -= 1
    dfStratHist = dfStratHist[dfStratHist.columns.difference(['old', 'new'])]
    dfStratCounts = dfStratHist[[
        stratName[strat]+"_delta" for strat in stratSet]].cumsum()
    for strat in stratSet:
        dfStratCounts[stratName[strat]+"_delta"] += initCounts[strat]
    dfStratCounts = dfStratCounts.rename(
        columns={stratName[strat]+"_delta": stratName[strat]+"_count" for strat in stratSet})
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
        colName = int2MedName[med]+"_delta"
        dfMedHist[colName] = 0
        dfMedHist.loc[dfMedHist['new'].values == med, colName] += 1
        dfMedHist.loc[dfMedHist['old'].values == med, colName] -= 1
    return dfMedHist[dfMedHist.columns.difference(['old', 'new'])]


def dfMedCounts(run):
    initCounts = Counter(run["initMedStrats"])
    medSet = sorted(initCounts.keys())
    N = len(run["initMedStrats"])
    dfMedHist = dfMedDeltas(run, medSet)
    dfMedCounts = dfMedHist[[int2MedName[med] +
                             "_delta" for med in medSet]].cumsum()
    for med in medSet:
        dfMedCounts[int2MedName[med]+"_delta"] += initCounts[med]
    dfMedCounts = dfMedCounts.rename(
        columns={int2MedName[med]+"_delta": int2MedName[med]+"_count" for med in medSet})
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


# from exp_all_analysis


def plot_heatmap(df, vmin, vmax, yticklabels, xticklabels, title='', ax=None, interpolate=True):
    if not interpolate:
        return sns.heatmap(df, vmin=vmin, vmax=vmax, yticklabels=yticklabels, xticklabels=xticklabels, annot=True, square=True, ax=ax).set(title=title)
    else:
        if not ax:
            print("not ax")
            ax = plt.gca()
        im = ax.imshow(df, interpolation='spline36', vmin=vmin, vmax=vmax)
        # im = ax.imshow(df, interpolation='lanczos', vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_xticklabels(xticklabels)
        ax.set_yticklabels(yticklabels)
        ax.set_title(title)
        cbar = ax.figure.colorbar(im, ax=ax)
        # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        return im


def plot_competition(medSet, df):
    interpolate = False
    df_mean = df.groupby(["meds", "ws"]).mean()
    n = 5
    size = 5
    fig, ax = plt.subplots(1, n, figsize=((n+1)*size, size))
    df_pivoted = df_mean.loc[medSet].pivot(
        index=[("params", "W1")], columns=[("params", "W2")])
    w1s, w2s = [round(x, 3) for x in df_pivoted[("agents", "coop_freq")].index], [
        round(x, 3) for x in df_pivoted[("agents", "coop_freq")].columns]
    plot_heatmap(df_pivoted[("agents", "coop_freq")], vmin=0, vmax=1, yticklabels=w1s,
                 xticklabels=w2s, title=f"Avg. final #cooperators", ax=ax[0], interpolate=interpolate)
    plot_heatmap(df_pivoted[("net", "heterogeneity")], vmin=df_mean[('net', 'heterogeneity')].min(), vmax=df_mean[('net', 'heterogeneity')].max(
    ), yticklabels=w1s, xticklabels=w2s, title=f"Avg. final degree heterogeneity", ax=ax[1], interpolate=interpolate)
    plot_heatmap(df_pivoted[("net", "k_max")], vmin=df_mean[('net', 'k_max')].min(), vmax=df_mean[('net', 'k_max')].max(
    ), yticklabels=w1s, xticklabels=w2s, title=f"Avg. final max degree", ax=ax[2], interpolate=interpolate)
    # rewire per opportunity: rewire numbers adjusted to stop time and W parameters.
    df_piv_rewire_n = (df_pivoted[("net", "rewire_n")]).divide(1-1/(df_pivoted[("net", "rewire_n")].index+1),
                                                               axis=0) / 1/(df_pivoted[("net", "rewire_n")].columns+1) / df_pivoted[("net", "stop_n")]
    plot_heatmap(df_piv_rewire_n, vmin=0, vmax=1, yticklabels=w1s, xticklabels=w2s,
                 title=f"Avg. final #rewires / opportunity", ax=ax[3], interpolate=interpolate)
    plot_heatmap(df_pivoted[("net", "stop_n")], vmin=df_mean[('net', 'stop_n')].min(), vmax=df_mean[('net', 'stop_n')].max(
    ), yticklabels=w1s, xticklabels=w2s, title=f"Avg. final stop time", ax=ax[4], interpolate=interpolate)
    plt.suptitle(f"Mediators: {medSet}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# plot a row of heatmaps for a given mediator set and w1
def plot_ts_row(w1, medSet, df_mean_ts, axs=None):
    interpolate = True

    def plot_ts_row_w_axis(axs):
        ax = axs
        df_pivoted = df_mean_ts.loc[medSet].loc[w1].pivot(
            index=[("game", "s")], columns=[("game", "t")]).sort_index(ascending=False)
        ss, ts = ['%.2f' % elem for elem in df_pivoted[("agents", "coop_freq")].index], [
            '%.2f' % elem for elem in df_pivoted[("agents", "coop_freq")].columns]
        plot_heatmap(df_pivoted[("agents", "coop_freq")], vmin=0, vmax=1, yticklabels=ss,
                     xticklabels=ts, title=f"Avg. final #cooperators", ax=ax[0], interpolate=interpolate)
        plot_heatmap(df_pivoted[("net", "heterogeneity")], vmin=df_mean_ts[('net', 'heterogeneity')].min(), vmax=df_mean_ts[('net', 'heterogeneity')].max(
        ), yticklabels=ss, xticklabels=ts, title=f"Avg. final degree heterogeneity", ax=ax[1], interpolate=interpolate)
        plot_heatmap(df_pivoted[("net", "k_max")], vmin=df_mean_ts[('net', 'k_max')].min(), vmax=df_mean_ts[('net', 'k_max')].max(
        ), yticklabels=ss, xticklabels=ts, title=f"Avg. final max degree", ax=ax[2], interpolate=interpolate)
        # rewire per opportunity: rewire numbers adjusted to stop time and W parameters.
        df_piv_rewire_n = (df_pivoted[("net", "rewire_n")]) / \
            w1 / df_pivoted[("net", "stop_n")]
        plot_heatmap(df_piv_rewire_n, vmin=0, vmax=1, yticklabels=ss, xticklabels=ts,
                     title=f"Avg. final #rewires / opportunity", ax=ax[3], interpolate=interpolate)
        plot_heatmap(df_pivoted[("net", "stop_n")], vmin=df_mean_ts[('net', 'stop_n')].min(), vmax=df_mean_ts[('net', 'stop_n')].max(
        ), yticklabels=ss, xticklabels=ts, title=f"Avg. final stop time", ax=ax[4], interpolate=interpolate)
    if axs != None and axs.all() != None:
        # if axs.all() != None:
        plot_ts_row_w_axis(axs)
    else:
        size = 5
        n = 5
        fig, new_axs = plt.subplots(1, n, figsize=(
            (n+1)*size, size), constrained_layout=False)
        plt.suptitle(f"Mediators: {medSet}, w1={w1}")
        df_mean_ts = df.groupby(["meds", ("params", "W1"), "ts"]).mean()
        plot_ts_row_w_axis(new_axs)
        fig.tight_layout(rect=[0, 0, 1, 0.95])


# plot ts for a given set of mediators and each w1
def plot_ts(medSet, df):
    filter_inf = True
    df = df if not filter_inf else df[df[("params", "W1")] != inf]
    n = 5
    size = 5
    df_mean_ts = df.groupby(["meds", ("params", "W1"), "ts"]).mean()
    w1s = df_mean_ts.index.levels[1]
    fig, axs = plt.subplots(len(w1s), n, figsize=(
        (n+1)*size, len(w1s) * size), constrained_layout=False)
    for i, w1 in enumerate(w1s):
        plot_ts_row(w1, medSet, df_mean_ts, axs[i])
    plt.suptitle(f"Mediators: {medSet}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
# Makes a pandas bar plot of mediator frequencies for each w parameter combination


def plot_bar_grid(medSet, df):
    df_mean = df.groupby(["meds", "ws"]).mean()
    size = 5
    w1s = [round(x, 3)
           for x in pd.unique(df_mean.loc[medSet][("params", "W1")])]
    w2s = [round(x, 3)
           for x in pd.unique(df_mean.loc[medSet][("params", "W2")])]
    fig, axs = plt.subplots(len(w1s), len(
        w2s), figsize=((len(w1s)+1)*size, len(w2s)*size))
    for i, w1 in enumerate(w1s):
        for j, w2 in enumerate(w2s):
            med_freq_ax = df_mean.loc[medSet].loc[[(w1, w2)]]['med_freqs'].plot.bar(
                ax=axs[i, j], xlabel="mediators", ylim=(0, 1), xticks=[], label=str())
            med_freq_ax.set_title(
                f"Mediator freqs w1={w1} w2={w2}", color='black')
            med_freq_ax.legend(bbox_to_anchor=(1.0, 1.0))
            med_freq_ax.plot()
    plt.suptitle(f"Mediator frquency: {medSet}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# Makes a pandas violin plot of mediator frequencies for each w parameter combination


def plot_violin_grid(medSet, df):
    df_mean = df.groupby(["meds", "ws"]).mean()
    size = 5
    w1s = [round(x, 3)
           for x in pd.unique(df_mean.loc[medSet][("params", "W1")])]
    w2s = [round(x, 3)
           for x in pd.unique(df_mean.loc[medSet][("params", "W2")])]
    fig, axs = plt.subplots(len(w1s), len(
        w2s), figsize=((len(w1s)+1)*size, len(w2s)*size))
    plt.suptitle(f"Mediator frquency: {medSet}")
    for i, w1 in enumerate(w1s):
        for j, w2 in enumerate(w2s):
            df_plot = df[(df['meds'] == medSet) & (df['ws'] == (w1, w2))][[
                ("med_freqs", x) for x in medSet]].stack().reset_index()
            plt.figure(figsize=(10, 5))
            ax = sns.violinplot(x="level_1", y="med_freqs",
                                data=df_plot, inner=None, ax=axs[i, j]).set_title(f"Mediator freqs w1={w1} w2={w2}", color='black')
            ax = sns.swarmplot(x="level_1", y="med_freqs", data=df_plot,
                               color="white", edgecolor="gray", size=2,  ax=axs[i, j])
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
