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
    print("# Dataset %s" % dataset[1])
    scores = np.load("scores/%s.npy" % dataset[1])

    for pid, pair in enumerate(pairs):
        print("\t- pair %i vs %i" % pair)

        for t_id, (critic_id, test_name) in enumerate(tests):
            if critic_id != 2:
                t_stats = np.load("tests/%s_p%i_t%i.npy" % (dataset[1], pid, t_id))
            else:
                t_stats = np.load("tests/%s_p%i_f.npy" % (dataset[1], pid))
            if t_id != 0:
                t_scores = scores.reshape(-1,5,3)[:,:,pair]
            else:
                t_scores = scores.reshape(-1,25,3)


            plot_fname = "figures/%s_%i_%i.eps" % (dataset[1], pid, t_id)

            print(t_id, critic_id, test_name, plot_fname)

            """
            Table header
            """
            header = "\\scriptsize\\begin{tabularx}{.31\\textwidth}{@{\\hspace{.5em}}c@{\\hspace{.5em}}c@{\\hspace{.5em}}c|CCC}\n\\toprule\\multicolumn{6}{c}{\\textsc{%s}}\\\\\\bottomrule\n\\multicolumn{6}{@{}c@{}}{\\includegraphics[width=.31\\textwidth]{figures/%s_%i_%i.eps}}\\\\\n\\midrule & \\textsc{%s} & \\textsc{%s} & \\textsc{t} & $p$ & \\textsc{d}\\\\" % ("%s %s" % (dataset[1], test_name), dataset[1],pid, t_id, clfs[pair[0]], clfs[pair[1]])

            """
            APPROX PART
            """
            means = np.mean(t_scores,axis=(0,1))
            stds = np.std(t_scores,axis=(0,1))
            mean_t = np.nanmean(t_stats)

            approx = "$\\approx$ & %s %.3f & %s %.3f & %.2f & %.2f & ---\\\\\n& {\\tiny(%.2f)} & {\\tiny(%.2f)} & & &\\\\\\midrule" % ("\\bfseries" if mean_t > critics[critic_id] else "", means[0], "\\bfseries" if mean_t < -critics[critic_id] else "", means[1], mean_t, t_to_p(mean_t, t_scores.shape[1]-1), stds[0], stds[1])

            #print(approx)

            """
            = part
            """
            eq_idx = np.nanargmin(np.abs(t_stats))
            means = np.mean(t_scores[eq_idx], axis=0)
            stds = np.std(t_scores,axis=(0,1))
            t = t_stats[eq_idx]
            d = np.sum((t_stats > -critics[critic_id]) * (t_stats < critics[critic_id]))/len(t_stats)

            if (t > -critics[critic_id]) * (t < critics[critic_id]):
                eq = "=         & %s %.3f & %s %.3f & %.2f & %.2f & %.4f\\\\\n  & {\\tiny(%.2f)} & {\\tiny(%.2f)} & &\\\\"  % ("\\bfseries" if t > critics[critic_id] else "", means[0], "\\bfseries" if t < -critics[critic_id] else "", means[1], t, t_to_p(t, t_scores.shape[1]-1), d, stds[0], stds[1])
                #print(eq)
            else:
                eq = "-         & --- & --- & --- & --- & 0.0000\\\n\\\\&  & & & &\\\\"

            """
            - part
            """
            minus_idx = np.nanargmin(t_stats)
            means = np.mean(t_scores[minus_idx], axis=0)
            stds = np.std(t_scores,axis=(0,1))
            t = t_stats[minus_idx]
            d = np.sum((t_stats < -critics[critic_id]))/len(t_stats)

            if t < -critics[critic_id]:
                minus = "-         & %s %.3f & %s %.3f & %.2f & %.2f & %.4f\\\\\n  & {\\tiny(%.2f)} & {\\tiny(%.2f)} & &\\\\"  % ("\\bfseries" if t > critics[critic_id] else "", means[0], "\\bfseries" if t < -critics[critic_id] else "", means[1], t, t_to_p(t, t_scores.shape[1]-1), d, stds[0], stds[1])
            else:
                minus = "-         & --- & --- & --- & --- & 0.0000\\\n\\\\&  & & & &\\\\"


            """
            + part
            """
            plus_idx = np.nanargmax(t_stats)
            means = np.mean(t_scores[plus_idx], axis=0)
            stds = np.std(t_scores,axis=(0,1))
            t = t_stats[plus_idx]
            d = np.sum((t_stats > critics[critic_id]))/len(t_stats)

            if t > critics[critic_id]:
                plus = "+         & %s %.3f & %s %.3f & %.2f & %.2f & %.4f\\\\\n  & {\\tiny(%.2f)} & {\\tiny(%.2f)} & &\\\\"  % ("\\bfseries" if t > critics[critic_id] else "", means[0], "\\bfseries" if t < -critics[critic_id] else "", means[1], t, t_to_p(t, t_scores.shape[1]-1), d, stds[0], stds[1])
            else:
                plus = "+         & --- & --- & --- & --- & 0.0000\\\n\\\\&  & & & &\\\\"


            """
            ending
            """
            ending = "\\bottomrule\n\\end{tabularx}"
            table = header + "\n" + approx + "\n" + eq + "\n" + minus + "\n" + plus + ending

            # print(table)

            text_file = open("tables/%s_%i_%i.tex" % (dataset[1], pid, t_id), "wt")
            n = text_file.write(table)
            text_file.close()
