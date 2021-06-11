import pickle
import matplotlib.pyplot as plt
filename="/home/fcarvalho/Documents/Projects/aligning_rs/graphEgtExperiments/SantosOct2006Mod/MediatorCompetitionAsRewire/experiments/data/overnight_medSets-[most_pairs]_n_eps-1000000_n_trials-10_w1s-w2s-0.5-3_Jun-11-2021_1054/Jun-11-2021_1054/overnight_medSets-[most_pairs]_n_eps-1000000_n_trials-10_w1s-w2s-0.5-3_Jun-11-2021_1054.pkl"
results=pickle.load(open(filename, "rb"))

from analysis import *
from egt_io import *
from mediators import int2MedName
from utils import transposeList

episode_n=1000000
n_trials=11
for i,(medSet, res) in enumerate(results.items()):
        experiment_name = makeCompetitionName(
        {"medSet": [int2MedName[med] for med in medSet], "n_eps": episode_n, "n_trials": n_trials})+"_"+timestamp()
        c=coop(res)
        coop_df = pd.DataFrame(zip(list(c.values()), *transposeList(list(c.keys()))),
                           columns=["coops", 'w1s', 'w2s']).pivot('w2s', 'w1s', "coops").iloc[::-1]
        print(coop_df)
        med_count_df = medCountsDf(res)
        print(med_count_df)
        plt.figure(i)  # creates a new figure
        sns.heatmap(coop_df, annot=True, cbar=True, xticklabels=2,
                        yticklabels=2, vmin=0, vmax=1).set(title=f"Avg. final coop, {[int2MedName[med] for med in medSet]}, ep_n={episode_n}, n_trials={n_trials}")
        plt.savefig(f'../plots/med_competition/{experiment_name}.png')