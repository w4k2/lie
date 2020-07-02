import numpy as np
import helper
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(precision=3, suppress=True)

# 24, 4 [alpha=.05]
critics = [2.0639, 2.7764]
corrs = np.linspace(0.1, 0.6, 6)
clfs = ["GNB", "kNN", "CART"]
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
]
colors = {1: "black", 2: "blue", 3: "red"}

datasets = helper.datasets()

for d_id, dataset in enumerate(datasets):
    print("# Dataset %i" % d_id)
    scores = np.load("scores/%s.npy" % dataset[1])

    for pid, pair in enumerate(pairs):
        print("\t- pair %i vs %i" % pair)

        for t_id, (critic_id, test_name) in enumerate(tests):
            print(t_id, critic_id, test_name)
            t_stats = np.load("tests/%s_p%i_t%i.npy" % (dataset[1], pid, t_id))
            cases = (
                np.array(
                    [
                        np.sum(t_stats < -critics[critic_id]),
                        np.sum(np.abs(t_stats) <= critics[critic_id]),
                        np.sum(t_stats > critics[critic_id]),
                    ]
                )
                / t_stats.shape
                * 100
            )

            print(cases)

            # exit()

        continue
        n_replications, n_repetitions, n_splits, n_clfs = scores.shape

        means = np.mean(scores, axis=(0, 1, 2))
        stds = np.std(scores, axis=(0, 1, 2))

        print("M", means[pair[0]], means[pair[1]], "s", stds[pair[0]], stds[pair[1]])

        fig, ax = plt.subplots(1, 1, figsize=(4.83, 3))

        j = 3
        corr = corrs[j]

        # Gather and count
        t_stats = np.load("thirteentest/%s_p%i_t%i.npy" % (dataset[1], pid, j))
        cases = (
            np.array(
                [
                    np.sum(t_stats < -critic),
                    np.sum(np.abs(t_stats) <= critic),
                    np.sum(t_stats > critic),
                ]
            )
            / t_stats.shape
            * 100
        )

        c = "black"

        # Presentation
        ma = np.nanmean(np.abs(t_stats))
        me = np.nanstd(np.abs(t_stats))

        z = 4 * me + ma
        if z < 8:
            z = 8
        elif z > 20:
            z = 30
        if np.isnan(z):
            z = 15
        # z = 15
        r = (-z, z)
        bins = 64

        print("CASES", j, "%.1f" % corr, cases, "MA", ma, me, z, len(t_stats))

        # Plot
        h = ax.hist(t_stats, bins=bins, color=c, range=r)
        hmax = np.max(h[0])
        # ax.set_title("corr = %.1f" % corr)
        ax.vlines(
            [-critic, critic],
            ymin=0,
            ymax=hmax,
            ls=":",
            label="critical values",
            color="red",
            lw=1,
        )

        # Plot symbols
        """
        if cases[0] > 0:
            ax.text(
                -0.97 * z,
                1.015 * hmax,
                "%s\n%.3f%% " % (clfs[pair[1]], cases[0]),
                ha="center",
                c=c,
            )
        if cases[1] > 0:
            ax.text(0, 1.015 * hmax, "==\n%.3f%%" % cases[1], ha="center", c=c)
        if cases[2] > 0:
            ax.text(
                0.97 * z,
                1.015 * hmax,
                "%s\n%.3f%%" % (clfs[pair[0]], cases[2]),
                ha="center",
                c=c,
            )
        """
        ax.set_yticks([])
        ax.set_ylim(0, 1 * hmax)
        ax.spines["right"].set_color("none")
        ax.spines["left"].set_color("none")
        ax.spines["top"].set_color("none")

        """
        fig.suptitle(
            "%s dataset\n%s = %.3f (+-%.2f), %s = %.3f (+-%.2f)"
            % (
                dataset[1],
                clfs[pair[0]],
                means[pair[0]],
                stds[pair[0]],
                clfs[pair[1]],
                means[pair[1]],
                stds[pair[1]],
            ),
            fontsize=12,
        )
        """

        fig.subplots_adjust(top=0.99, bottom=0.08, left=0, right=1)

        # ax[0].adjust(left=0.2, bottom=0.7)
        # plt.tight_layout()
        plt.savefig("foo.png")
        # exit()
        # plt.savefig("figures/%s_%i.png" % (dataset[1], pid))
        plt.savefig("figures/%s_%i.eps" % (dataset[1], pid))
