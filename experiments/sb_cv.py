"""Series-level K-fold cross-validation in true TS-AUC, run on the cached feature
bank (no re-featurize). The single fixed split is too noisy to trust ~0.005 gains;
cv_ts_auc(mean) is the decision metric for the model-tuning stage. Import-safe."""
import numpy as np
import pandas as pd
import lightgbm as lgb
import sb_common as C
from sb_model import PARAMS, rank01, eq_series_weight, true_step

TRAIN_BANK = f"{C.DATA}/experiments/feat_final_train.parquet"


def load_bank(path=TRAIN_BANK, cols=None):
    """Read the cached bank -> (X float32 ndarray, MultiIndex, column list)."""
    F = pd.read_parquet(path, columns=cols)
    return F.to_numpy(np.float32), F.index, list(F.columns)


def load_y():
    """Full (id, time) targets, used for labels + TS-AUC step strata."""
    return pd.read_parquet(f"{C.DATA}/y_train.parquet")["target"]


def series_folds(index, n_splits=5, seed=0):
    """Yield (train_mask, val_mask) boolean arrays, split by series id (no leakage)."""
    ids = index.get_level_values("id").to_numpy()
    uniq = np.random.default_rng(seed).permutation(np.unique(ids))
    for fold in np.array_split(uniq, n_splits):
        vm = np.isin(ids, fold)
        yield ~vm, vm


def cv_ts_auc(X, index, y_full, params=PARAMS, seeds=(0, 1, 2), n_splits=5,
              seed=0, feat_cols=None, verbose=False):
    """Series-fold CV. Per fold: seed-bag LightGBM (eq-series weights) -> rank01 avg
    -> TS-AUC on the held fold. feat_cols = optional integer column subset.
    Returns (mean, std, per_fold_list)."""
    if feat_cols is not None:
        X = X[:, feat_cols]
    y = y_full.reindex(index).to_numpy().astype(np.int8)
    step = true_step(index, y_full)
    scores = []
    for k, (tr, va) in enumerate(series_folds(index, n_splits, seed)):
        wtr = eq_series_weight(index[tr])
        preds = []
        for s in seeds:
            m = lgb.LGBMClassifier(random_state=s, **params).fit(
                X[tr], y[tr], sample_weight=wtr)
            preds.append(m.predict_proba(X[va])[:, 1])
        p = np.mean([rank01(q) for q in preds], axis=0)
        sc = C.ts_auc_arrays(p, y[va], step[va])
        scores.append(sc)
        if verbose:
            print("fold %d  ts_auc=%.4f  (ntr=%d nva=%d)" % (
                k, sc, int(tr.sum()), int(va.sum())), flush=True)
    scores = np.array(scores)
    return float(scores.mean()), float(scores.std()), scores.tolist()
