import numpy as np
import helper
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

np.set_printoptions(precision=3, suppress=True)

def t_to_p(tt, dd=4):
    pval = stats.t.sf(np.abs(tt), dd) * 2
    return pval

# 24, 4 [alpha=.05]
critics = [2.0639, 2.7764, 2.57]
corrs = np.linspace(0.1, 0.6, 6)
clfs = ["gnb", "knn", "cart"]
pairs = [(0, 1), (0, 2), (1, 2)]

tests = [
    (0, "non-parametric correction"),
    (1, "parametric-correction (corr=.1)"),
    (1, "parametric-correction (corr=.2)"),
    (1, "parametric-correction (corr=.3)"),
    (1, "parametric-correction (corr=.4)"),
    (1, "parametric-correction (corr=.5)"),
    (1, "parametric-correction (corr=.6)"),
    (1, "regular"),
    (2, "cv52cft")
]
colors = {1: "black", 2: "blue", 3: "red"}

datasets = helper.datasets()

# Dataset, pair, test, 3-cases
all_cases = np.zeros((len(datasets), len(pairs), len(tests), 3))

for d_id, dataset in enumerate(datasets):
    scores = np.load("scores/%s.npy" % dataset[1])

    for pid, pair in enumerate(pairs):
        for t_id, (critic_id, test_name) in enumerate(tests):
            if t_id not in [0,5,7,8]:
                continue
            fname = "\\resizebox{.25\\textwidth}{!}{\input{tables/%s_%i_%i.tex}}" % (dataset[1], pid, t_id)
            print(fname)
