from defaultParams import int2MedStrat
from copy import deepcopy
import os
import pickle
import time
import os
import fnmatch
import shutil
import pandas as pd


def timestamp():
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    return timestamp

# Save to files


def pToStr(param):
    return f"{str(param[0])}-{str(param[-1])}" if isinstance(param, (list, tuple)) else str(param)


def makePklPath(name, dir_path='./data'):
    return f"{dir_path}/{name}.pkl"


def renameDuplicate(makePklPath, name, i=0, dir_path='./data'):
    print(f"renameDuplicate {i}")
    _name = deepcopy(name)
    if i != 0:
        _name = f'{_name} ({i})'
    if os.path.exists(makePklPath(_name, dir_path)):
        return renameDuplicate(makePklPath, name, i+1, dir_path)
    return _name

# "only works if all values and keys are printable"


def dict2Filename(d):
    return "_".join([f"{pToStr(k)}-{pToStr(v)}" for k, v in d.items()])


def makeExperimentName(useMediator, N, M, episode_n, beta, W, k):
    baseName = f"{useMediator.__name__}_N-{pToStr(N)}_M-{pToStr(M)}_episoden-{pToStr(episode_n)}_beta-{pToStr(beta)}_W-{pToStr(W)}_k-{pToStr(k)}"
    return baseName


# def makeCompetitionName(medSet, N, episode_n, ts, beta, W1, W2, k):
#     baseName = f"{'-'.join(map(lambda x: int2MedStrat[x], medSet))}_N-{pToStr(N)}_episoden-{pToStr(episode_n)}_beta-{pToStr(beta)}_W1-{pToStr(W1)}_W2-{pToStr(W2)}_k-{pToStr(k)}_ts-{pToStr(ts)}"
#     return baseName

def makeCompetitionName(args):
    return dict2Filename(args)


def saveRes(res, _name, dir_path='./data'):
    dir_path = f"{dir_path}/{timestamp()}"
    name = renameDuplicate(makePklPath, _name, dir_path=dir_path)
    path = makePklPath(name, dir_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(path, "wb+") as file:
        pickle.dump(res, file)
        print(f"saved {name}.pkl")


def saveDf(df, _name, dir_path='./data'):
    dir_path = f"{dir_path}/{timestamp()}"
    name = renameDuplicate(makePklPath, _name, dir_path=dir_path)
    path = f"{dir_path}/{name}.csv"
    os.makedirs(dir_path, exist_ok=True)
    df.to_csv(path)
    print(f"saved {name}.csv")


def loadRes(name, dir_path='./data'):
    with open(f"{dir_path}/{name}.pkl", "rb") as file:
        res = pickle.load(file)
        print(f"loaded {name}.pkl")
        return res


def loadResFn(filename, dir_path='./data'):
    with open(f"{dir_path}/{filename}", "rb") as file:
        res = pickle.load(file)
        print(f"loaded {filename}")
        return res

# newly copied in Jun 11:


def filename2W(fn):
    import re
    return int(re.search(r".*W1-([0-9]+).*", fn).group(1))


def filename2MedPair(fn):
    import re
    from mediators import medName2Int
    try:
        matches = re.search(r".*medSet-([0-9A-Z_]+)-([0-9A-Z_]+)\.pkl", fn)
        med1 = matches.group(1)
        med2 = matches.group(2)
        return (medName2Int[med1], medName2Int[med2])
    except AttributeError as error:
        print(f"filename: {fn}")
        print(f"error: {error.args}")
        raise error


def filename2Med(fn):
    import re
    try:
        matches = re.search(r".*medSet-([0-9A-Z_]+)\.pkl", fn)
        med1 = matches.group(1)
        return med1
    except AttributeError as error:
        print(f"filename: {fn}")
        print(f"error: {error.args}")
        raise error


# loads a pickle
def loadPickle(path):
    import pickle
    with open(path, "rb") as file:
        res = pickle.load(file)
        print(f"loaded {path}")
        return res

#  dict comprehension where values are lists keys are given by applying a function fn to items
# Make a list into a dict, get keys by applying fn to list values


def dictOfLists(my_list, fn=lambda x: x):
    new_dict = {}
    for value in my_list:
        key = fn(value)
        if key in new_dict:
            new_dict[key].append(value)
        else:
            new_dict[key] = [value]
    return new_dict

# recursively find all pickles in a dir


def getAllPickles(dir_name):
    return [os.path.join(root, name)
            for root, dirs, files in os.walk(dir_name)
            for name in files
            if name.endswith(".pkl")]
# load all filenames in dir and its subdirs into 1 experiment dict where keys are W values

# key_fn is a function to extract a key from the filename


def loadExperiment(dir_name, key_fn):
    filenames = getAllPickles(dir_name)
    res = {k: [loadPickle(fn) for fn in fns]
           for k, fns in dictOfLists(filenames, key_fn).items()}
    return res


def loadExperimentDf(dir_name):
    all_csvs = [os.path.join(root, name)
                for root, dirs, files in os.walk(dir_name)
                for name in files
                if name.endswith(".csv")]
    return [pd.read_csv(fn, header=[0, 1], index_col=0) for fn in all_csvs]


def loadExperimentPkl(dir_name):
    return [loadPickle(fn) for fn in getAllPickles(dir_name)]
