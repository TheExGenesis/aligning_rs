# Save to files
def pToStr(param):
    return f"{param[0]}-{param[-1]}" if isinstance(param, list) else str(param)


def makePath(name):
    return f"./data/{name}.pkl"


def renameDuplicate(makePath, name, i=0):
    print(f"renameDuplicate {i}")
    _name = deepcopy(name)
    if i != 0:
        _name = f'{_name} ({i})'
    if os.path.exists(makePath(_name)):
        return renameDuplicate(makePath, name, i+1)
    return _name


def makeExperimentName(useMediator, N, M, episode_n, beta, W, k):
    baseName = f"{useMediator.__name__}_N-{pToStr(N)}_M-{pToStr(M)}_episoden-{pToStr(episode_n)}_beta-{pToStr(beta)}_W-{pToStr(W)}_k-{pToStr(k)}"
    return baseName


def makeCompetitionName(medSet, N, episode_n, ts, beta, W1, W2, k):
    baseName = f"{'-'.join(medSet)}_N-{pToStr(N)}_episoden-{pToStr(episode_n)}_beta-{pToStr(beta)}_W1-{pToStr(W1)}_W2-{pToStr(W2)}_k-{pToStr(k)}"
    return baseName


def saveRes(res, _name):
    name = renameDuplicate(makePath, _name)
    os.makedirs('./data', exist_ok=True)
    path = makePath(name)
    with open(path, "wb+") as file:
        pickle.dump(res, file)
        print(f"saved {name}.pkl")


def loadRes(name):
    with open(f"./data/{name}.pkl", "rb") as file:
        res = pickle.load(file)
        print(f"loaded {name}.pkl")
        return res


def loadResFn(filename):
    with open(f"./data/{filename}", "rb") as file:
        res = pickle.load(file)
        print(f"loaded {filename}")
        return res
