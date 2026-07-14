"""Modeling helpers shared by the pipeline scripts: the LightGBM recipe and the
sample-weighting / step-index utilities. Import-safe (no side effects)."""
import numpy as np
import pandas as pd

# the tuned LightGBM recipe (equal-series weighting is the big win, see REPORT.md)
PARAMS = dict(objective="binary", n_estimators=500, learning_rate=0.05,
              num_leaves=63, min_child_samples=300, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbosity=-1)


def rank01(a):
    """Map scores to [0,1] ranks — used to average bag members before scoring."""
    return np.argsort(np.argsort(a)) / (len(a) - 1.0)


def eq_series_weight(index):
    """1 / (#rows for the series), normalized to mean 1 -> each SERIES counts equally."""
    gs = pd.Series(1, index=index).groupby(level="id").transform("size").to_numpy()
    w = 1.0 / gs
    return w / w.mean()


def true_step(index, y_full):
    """True 0-based online step for each row (for TS-AUC strata), from the full targets."""
    step = pd.Series(y_full.groupby(level="id").cumcount().to_numpy(), index=y_full.index)
    return step.reindex(index).to_numpy()
