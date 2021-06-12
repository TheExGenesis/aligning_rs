# %%
from exp_dataframe import makeDataframeFromSingleMedRuns
from egt_io import *


# gather individual runs and assemble them into a dict

# load overnight (meds->trials->ws->res) experiment


def loadOvernight():
    results = loadExperiment("../data/overnight", filename2MedPair)
    n_trials = 10
    episode_n = 1000000
    w1s = [0.5, 1, 2, 3]
    w2s = [0.5, 1, 2, 3]
    experiment_name = "overnight_"+makeCompetitionName(
        {"medSets": "[most_pairs]", "n_eps": episode_n, "n_trials": n_trials, "w1s-w2s": w1s})+"_"+timestamp()
    saveRes(results, experiment_name,
            dir_path=f"../data/{experiment_name}")


# load single med experiments

# returns med as a string
def filename2Med(fn):
    import re
    try:
        matches = re.search(r".*medSet-([0-9A-Z_]+)_n_eps.*", fn)
        med1 = matches.group(1)
        return med1
    except AttributeError as error:
        print(f"filename: {fn}")
        print(f"error: {error.args}")
        raise error

# %%
