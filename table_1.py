"""
Table presenting number of cases.
"""

import numpy as np
import helper

tests = [
    (1, "without correction"),
    (1, "parametric (corr=.1)"),
    (1, "parametric (corr=.2)"),
    (1, "parametric (corr=.3)"),
    (1, "parametric (corr=.4)"),
    (1, "parametric (corr=.5)"),
    (1, "parametric (corr=.6)"),
    (0, "non-parametric"),
]
clfs = ["GNB", "kNN", "CART"]
pairs = [(0, 1), (0, 2), (1, 2)]

# Dataset, pair, test, 3-cases
cases = np.load("cases.npy")
cases[:, :, [0, -1], :] = cases[:, :, [-1, 0], :]

# Mask of non-zero cases
mask = cases > 0

# Number of supported research hypotheses
n_hyp = np.sum(mask.astype(int), axis=3)

datasets = helper.datasets()

res_n_hyp = n_hyp.reshape((18 * 3, 8)).T

print(res_n_hyp)

for i, (_, test_name) in enumerate(tests):
    vector = res_n_hyp[i]

    c = [
        np.sum(vector == 1),
        np.sum(vector == 2),
        np.sum(vector == 3),
    ]
    print("%30s & %2i & %2i & %2i \\\\" % ("\\emph{%s}" % test_name, c[0], c[1], c[2]))
exit()

for db_idx, dataset in enumerate(datasets):
    """ Pair x test """
    db_hyp = n_hyp[db_idx]

    print(db_hyp)

    continue

    sub_entries = []
    for p_idx, pair in enumerate(pairs):
        sub_entry = " & ".join(["%i" % _ for _ in db_hyp[p_idx]])
        sub_entries.append(sub_entry)

    sub_entries = " & ".join(sub_entries)
    entry = "%20s & %s \\\\" % ("\\emph{%s}" % dataset[1], sub_entries)

    print(entry)
    # exit()
