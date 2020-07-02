import numpy as np
from scipy import stats
import os, re


def datasets():
    directory = "datasets/"
    files = np.array(
        [
            (directory + x, x[:-4])
            for x in os.listdir(directory)
            if re.match("^([a-zA-Z0-9])+\.csv$", x)
        ]
    )
    return np.sort(files, axis=0)


def t_test_14(a, b):
    """
    Corrected t-test for repeated cross-validation.
    input, two 2d arrays. Repetitions x folds
    """
    J, k = a.shape  # J - repetitions, k - folds
    d = a - b
    bar_d = np.mean(d)
    bar_sigma_2 = np.var(d.reshape(-1), ddof=1)
    bar_sigma_2_mod = (1 / (J * k) + 1 / (k - 1)) * bar_sigma_2
    t_stat = bar_d / np.sqrt(bar_sigma_2_mod)
    pval = stats.t.sf(np.abs(t_stat), (k * J) - 1) * 2
    return t_stat, pval


def t_test_13(a, b, corr=0.6):
    """
    Corrected t-test for repeated cross-validation.
    input, two 2d arrays. Repetitions x folds
    """
    k = len(a)  # J - repetitions, k - folds
    d = a - b
    bar_d = np.mean(d)
    bar_sigma_2 = np.var(d.reshape(-1), ddof=1)
    bar_sigma_2_mod = (1 / (k * (1 - corr))) * bar_sigma_2
    t_stat = bar_d / np.sqrt(bar_sigma_2_mod)
    pval = stats.t.sf(np.abs(t_stat), k - 1) * 2
    return t_stat, pval


def t_test_rel(a, b):
    """
    Paired, relative t-test.
    """
    J = len(a)
    d = a - b
    bar_d = np.mean(d)
    bar_sigma_2 = np.var(d.reshape(-1), ddof=1)
    bar_sigma_2_mod = (1 / J) * bar_sigma_2
    t_stat = bar_d / np.sqrt(bar_sigma_2_mod)
    pval = stats.t.sf(np.abs(t_stat), J - 1) * 2
    return t_stat, pval
