import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import helper
import pandas as pd

clfs = {
    "GNB": GaussianNB(),
    "kNN": KNeighborsClassifier(),
    "CART": DecisionTreeClassifier(),
}
seed = 1410
np.random.seed(seed)
n_replications = 20000  # 2000 Number of full analysis replications
n_splits = 5
n_repetitions = 5

datasets = helper.datasets()

for d_id, dataset in tqdm(enumerate(datasets), ascii=True, total=len(datasets)):
    # Read dataset
    ds = pd.read_csv(dataset[0], header=None).values
    X, y = ds[:, :-1], ds[:, -1].astype("int")

    # exit()

    # Prepare scoring table
    scores = np.zeros((n_replications, n_repetitions, n_splits, len(clfs)))

    # Iterate replications
    for replication in tqdm(range(n_replications)):
        # Get random states for repetitions
        random_states = np.random.randint(100, size=n_repetitions)

        # Iterate repetitions
        for repetition, random_state in enumerate(random_states):
            skf = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )

            # Iterate folds
            for fold, (train, test) in enumerate(skf.split(X, y)):
                # Iterate classifiers
                for clf_idx, clfn in enumerate(clfs):
                    # Train and predict
                    clf = clone(clfs[clfn]).fit(X[train], y[train])
                    y_pred = clf.predict(X[test])

                    # Assess score
                    score = accuracy_score(y[test], y_pred)
                    scores[replication, repetition, fold, clf_idx] = score

    np.save("scores/%s" % dataset[1], scores)
