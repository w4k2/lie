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

# Dataset, pair, test, 3-cases
all_cases = np.zeros((len(datasets), len(pairs), len(tests), 3))


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
            all_cases[d_id, pid, t_id] = cases

            fig, ax = plt.subplots(1, 1, figsize=(4.83, 2.5))
            h = ax.hist(t_stats, bins=64, color="black", range=(-20, 20))
            hmax = np.max(h[0])
            ax.vlines(
                [-critics[critic_id], critics[critic_id]],
                ymin=0,
                ymax=hmax,
                ls=":",
                label="critical values",
                color="red",
                lw=1,
            )
            ax.set_yticks([])
            ax.set_ylim(0, 1 * hmax)
            ax.spines["right"].set_color("none")
            ax.spines["left"].set_color("none")
            ax.spines["top"].set_color("none")
            fig.subplots_adjust(top=0.99, bottom=0.1, left=0, right=1)

            plt.savefig("foo.png")
            plt.savefig("figures/%s_%i_%i.eps" % (dataset[1], pid, t_id))
            plt.savefig("figures/%s_%i_%i.png" % (dataset[1], pid, t_id))

np.save("cases", all_cases)
