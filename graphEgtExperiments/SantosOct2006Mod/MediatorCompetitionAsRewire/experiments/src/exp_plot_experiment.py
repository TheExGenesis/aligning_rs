# %%
from utils import transposeList
from mediators import int2MedName
from egt_io import *
from analysis import *
import pickle
import matplotlib.pyplot as plt


# takes a dict with medSets as keys contianing lists of dicts with w-pairs as keys, containing experiment results
# plots cooperation and mediator freqs
def plotExperiment(results, episode_n=1000000, n_trials=11):
    for i, (medSet, res) in enumerate(results.items()):
        experiment_name = makeCompetitionName(
            {"medSet": [int2MedName[med] for med in medSet], "n_eps": episode_n, "n_trials": n_trials})+"_"+timestamp()
        n = 2
        size = 4
        fig, ax = plt.subplots(1, n, figsize=((n+1)*size, size))
        plt.suptitle(experiment_name)
        c = coop(res)
        coop_df = pd.DataFrame(zip(list(c.values()), *transposeList(list(c.keys()))),
                               columns=["coops", 'w1s', 'w2s']).pivot('w2s', 'w1s', "coops").iloc[::-1]
        coop_df = coop(res).transpose().unstack()
        med_count_df = medCountsDf(res).transpose().unstack()
        plt.subplot(1, n, 1)
        sns.heatmap(med_count_df[medSet[1]], annot=True, cbar=True, xticklabels=2,
                    yticklabels=2, vmin=0, vmax=1).set(title=f"Avg. final counts of {int2MedName[medSet[1]]}")
        plt.subplot(1, n, 2)
        sns.heatmap(coop_df, annot=True, cbar=True, xticklabels=2,
                    yticklabels=2, vmin=0, vmax=1).set(title=f"Avg. final coop")
        # plt.savefig(f'../plots/med_competition/{experiment_name}.png')


filename = "/mnt/c/Users/frsc/Documents/Projects/aligning_rs/graphEgtExperiments/SantosOct2006Mod/MediatorCompetitionAsRewire/experiments/data/med_competition/overnight_medSets-[most_pairs]_n_eps-1000000_n_trials-10_w1s-w2s-0.5-3_Jun-11-2021_1054.pkl"
results = pickle.load(open(filename, "rb"))
plotExperiment(results)
