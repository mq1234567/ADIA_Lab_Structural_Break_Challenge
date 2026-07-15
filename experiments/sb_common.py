"""Shared harness: TS-AUC metric + data loaders.

TS-AUC (Time-Stratified AUC) = the competition metric. At each online step,
take the cross-sectional AUC across the series alive at that step; then average
across steps weighted by the pair count n_pos * n_neg.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DATA = "/Users/minqi/Documents/ADIA_Lab_Structural_Break_Challenge"


def ts_auc_arrays(p, y, t):
    """TS-AUC from raw arrays: scores p, 0/1 labels y, per-row step index t."""
    df = pd.DataFrame({"p": np.asarray(p), "y": np.asarray(y, np.int8), "t": np.asarray(t)})
    num = den = 0.0
    for _, g in df.groupby("t", sort=False):
        n_pos = int(g["y"].sum()); n_neg = len(g) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        w = n_pos * n_neg
        num += roc_auc_score(g["y"], g["p"]) * w
        den += w
    return num / den


def ts_auc(pred, y_true):
    """TS-AUC on Series indexed by (id, time). Step = per-id cumcount."""
    p = pred.reindex(y_true.index).to_numpy()
    y = y_true.to_numpy()
    t = y_true.groupby(level="id").cumcount().to_numpy()
    return ts_auc_arrays(p, y, t)


def load_train():
    X = pd.read_parquet(f"{DATA}/X_train.parquet")
    y = pd.read_parquet(f"{DATA}/y_train.parquet")["target"]
    return X, y


def load_reduced():
    X = pd.read_parquet(f"{DATA}/X_test.reduced.parquet")
    y = pd.read_parquet(f"{DATA}/y_test.reduced.parquet")["target"]
    return X, y


def load_bank(path=f"{DATA}/experiments/feat_final_train.parquet", cols=None):
    """Read the cached feature bank -> (X float32 ndarray, MultiIndex, column list)."""
    F = pd.read_parquet(path, columns=cols)
    return F.to_numpy(np.float32), F.index, list(F.columns)


def load_y():
    """Full (id, time) online-step targets, for labels + TS-AUC step strata."""
    return pd.read_parquet(f"{DATA}/y_train.parquet")["target"]


# ---- modeling utilities -----------------------------------------------------

def rank01(a):
    """Map scores to [0, 1] ranks — used to average bag members before scoring."""
    return np.argsort(np.argsort(a)) / (len(a) - 1.0)


def eq_series_weight(index):
    """1 / (#rows for the series), normalized to mean 1 — each SERIES counts equally."""
    gs = pd.Series(1, index=index).groupby(level="id").transform("size").to_numpy()
    w = 1.0 / gs
    return w / w.mean()


def true_step(index, y_full):
    """True 0-based online step for each row (TS-AUC strata), from the full targets."""
    return y_full.groupby(level="id").cumcount().reindex(index).to_numpy()
