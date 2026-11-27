"""Loading and splitting the banknote authentication data.

Everything here is plain NumPy. The file is the one distributed by the UCI
Machine Learning Repository (dataset 267): 1372 rows, four real-valued
features taken from wavelet-transformed images of banknotes (variance,
skewness, curtosis, entropy) and a class label (0 or 1) in the last column.
"""

import os
from typing import Iterator, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.normpath(
    os.path.join(_HERE, "..", "..", "data", "data_banknote_authentication.txt")
)
FEATURE_NAMES = ["variance", "skewness", "curtosis", "entropy"]


def load_banknote(path: str = DEFAULT_PATH) -> Tuple[np.ndarray, np.ndarray]:
    """Read the CSV file. Returns (X, y): X is (1372, 4) floats, y is (1372,)
    ints in {0, 1}."""
    data = np.loadtxt(path, delimiter=",")
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y


def stratified_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_fraction: float = 0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle and split so that both parts keep the class proportions.

    Each class is shuffled and cut separately, so a 70/30 split has 70% of the
    class-0 rows and 70% of the class-1 rows in the training set.

    Returns:
        X_train, X_test, y_train, y_test
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for label in np.unique(y):
        members = np.flatnonzero(y == label)
        rng.shuffle(members)
        n_test = int(round(test_fraction * len(members)))
        test_idx.append(members[:n_test])
        train_idx.append(members[n_test:])
    train_idx = rng.permutation(np.concatenate(train_idx))
    test_idx = rng.permutation(np.concatenate(test_idx))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def stratified_k_fold(
    y: np.ndarray, k: int = 5, seed: int = 0
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_indices, validation_indices) for k stratified folds.

    The rows of each class are shuffled and dealt out to the folds in turn, so
    every fold has close to the overall class balance.
    """
    if k < 2:
        raise ValueError("k must be at least 2.")
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=int)
    for label in np.unique(y):
        members = np.flatnonzero(y == label)
        rng.shuffle(members)
        fold_of[members] = np.arange(len(members)) % k
    for fold in range(k):
        yield np.flatnonzero(fold_of != fold), np.flatnonzero(fold_of == fold)


class Standardiser:
    """Rescale each feature to zero mean and unit variance.

    The mean and standard deviation are learned from the training data only
    and then applied unchanged to the test data, so nothing about the test set
    leaks into training.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray) -> "Standardiser":
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.std_ = np.where(std == 0.0, 1.0, std)  # constant feature: leave as is
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise ValueError("Call fit() before transform().")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
