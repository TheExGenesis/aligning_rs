from defaultParams import int2MedStrat
from copy import deepcopy
import os
import pickle
import time
import os
import fnmatch
import shutil


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
