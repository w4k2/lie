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


datasets = helper.datasets()

for d_id, dataset in enumerate(datasets):
    scores = np.load("scores/%s.npy" % dataset[1])
    print("# Dataset %s" % dataset[0])

    for pid, pair in enumerate(pairs):
        print("\t- pair %i vs %i" % pair)
        n_replications, n_repetitions, n_splits, n_clfs = scores.shape

        # Firstly Corrected resampled t-test
        print("\t\tCalculate tests 1")
        t_stats_1 = np.zeros(n_replications)
        for i, replication in enumerate(scores):
            a = replication[:, :, pair[0]]
            b = replication[:, :, pair[1]]
            t = helper.t_test_14(a, b)
            t_stats_1[i] = t[0]

        np.save("tests/%s_p%i_t0" % (dataset[1], pid), t_stats_1)

        # Secondly resampled t-test
        print("\t\tCalculate tests 2")
        for corr_idx, corr in enumerate(corrs):
            print("\t\t- corr %.1f" % corr)
            t_stats_2 = np.zeros(n_replications * n_repetitions)
            second_scores = scores.reshape(
                (n_replications * n_repetitions, n_splits, n_clfs)
            )
            for i, replication in enumerate(second_scores):
                a = replication[:, pair[0]]
                b = replication[:, pair[1]]
                t = helper.t_test_13(a, b, corr=0.6)
                t_stats_2[i] = t[0]

            np.save("tests/%s_p%i_t%i" % (dataset[1], pid, corr_idx + 1), t_stats_2)

        # Secondly resampled t-test
        print("\t\tCalculate tests 3")
        t_stats_3 = np.zeros(n_replications * n_repetitions)
        second_scores = scores.reshape(
            (n_replications * n_repetitions, n_splits, n_clfs)
        )
        for i, replication in enumerate(second_scores):
            a = replication[:, pair[0]]
            b = replication[:, pair[1]]
            t = helper.t_test_rel(a, b)
            t_stats_3[i] = t[0]

        np.save("tests/%s_p%i_t7" % (dataset[1], pid), t_stats_3)
