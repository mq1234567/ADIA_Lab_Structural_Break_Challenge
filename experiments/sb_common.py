"""Shared harness for the ADIA structural-break experiments.

ts_auc = the competition metric (Time-Stratified AUC): per online step t
(cross-sectional across series alive at t), pair-count weighted average.
"""
import numpy as np
import pandas as pd

DATA = "/Users/minqi/Documents/ADIA_Lab_Structural_Break_Challenge"


def _auc_pair_stat(scores, y):
    """(concordant pairs incl. 0.5*ties, n_pos*n_neg) within one stratum."""
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0, 0.0
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(scores))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float(ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0), float(n_pos * n_neg)


def ts_auc(pred, y_true):
    """Time-Stratified AUC. pred, y_true: Series indexed by (id, time)."""
    pred = pred.reindex(y_true.index)
    t = y_true.groupby(level="id").cumcount().to_numpy()
    y = y_true.to_numpy().astype(int)
    p = pred.to_numpy()
    order = np.argsort(t, kind="mergesort")
    t, y, p = t[order], y[order], p[order]
    num = den = 0.0
    i = 0
    while i < len(t):
        j = i
        while j + 1 < len(t) and t[j + 1] == t[i]:
            j += 1
        c, w = _auc_pair_stat(p[i:j + 1], y[i:j + 1])
        num += c
        den += w
        i = j + 1
    return num / den


def ts_auc_arrays(p, y, t):
    """Same metric from raw arrays (p: scores, y: 0/1, t: online step index)."""
    order = np.argsort(t, kind="mergesort")
    t, y, p = t[order], y[order], p[order]
    num = den = 0.0
    i = 0
    while i < len(t):
        j = i
        while j + 1 < len(t) and t[j + 1] == t[i]:
            j += 1
        c, w = _auc_pair_stat(p[i:j + 1], y[i:j + 1])
        num += c
        den += w
        i = j + 1
    return num / den


def load_train():
    X = pd.read_parquet(f"{DATA}/X_train.parquet")
    y = pd.read_parquet(f"{DATA}/y_train.parquet")["target"]
    return X, y


def load_reduced():
    X = pd.read_parquet(f"{DATA}/X_test.reduced.parquet")
    y = pd.read_parquet(f"{DATA}/y_test.reduced.parquet")["target"]
    return X, y


def holdout_ids(all_ids, n_val=2000, seed=7):
    """Fixed series-level split for fast experiment iteration."""
    ids = np.unique(np.asarray(all_ids))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ids)
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])
