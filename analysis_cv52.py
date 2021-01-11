import numpy as np
import helper
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(precision=3, suppress=True)

# 24, 4 [alpha=.05]
critics = [2.0639, 2.7764]
corrs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

clfs = ["GNB", "kNN", "CART"]
pairs = [(0, 1), (0, 2), (1, 2)]

limit = 10000000000

datasets = helper.datasets()

for d_id, dataset in enumerate(datasets):
    scores = np.load("scores/%s_cv52.npy" % dataset[1])
    print("# Dataset %s" % dataset[0])

    for pid, pair in enumerate(pairs):
        print("\t- pair %i vs %i" % pair)
        n_replications, n_repetitions, n_splits, n_clfs = scores.shape

        print("%i x %i x %i" % (n_replications, n_repetitions, n_splits))

        # Firstly Corrected resampled t-test
        print("\t\tCalculate tests 0")
        t_stats_1 = np.zeros(n_replications)[:limit]
        for i, replication in enumerate(scores):
            a = replication[:, :, pair[0]]
            b = replication[:, :, pair[1]]
            t = helper.cv52cft(a, b)
            #print(t,np.mean(a),np.mean(b),np.mean(a)>np.mean(b))

            t_stats_1[i] = t[0] if np.mean(a) > np.mean(b) else -t[0]
            if i == limit - 1:
                break

        print(t_stats_1.shape, t_stats_1)

        np.save("tests/%s_p%i_f" % (dataset[1], pid), t_stats_1)
