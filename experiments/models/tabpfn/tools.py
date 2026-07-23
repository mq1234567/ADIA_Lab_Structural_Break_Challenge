from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

PATH = "/Users/minqi/Documents/ADIA_Lab_Structural_Break_Challenge/"

def emit_train_datasets(X_train: pd.DataFrame, y_train_index: pd.DataFrame):
    """Generator matching lgb.ipynb `train()`: yields
    ``(dataset_id, x_hist, x_online, tau)`` for every series in `X`.

    - `X`: MultiIndex(id, time) frame with cols `value`, `period` (1=hist, 2=online).
    - `y_idx`: id-indexed frame with col `tau_index` (-1 = no break).
    - `x_hist`, `x_online`: float numpy arrays.
    - `tau`: int (offset into online segment, 0 indexed) if there was a break, else None.
    """
    output = []
    tau_map = y_train_index["tau_index"].to_dict()
    for dataset_id, sub in X_train.groupby(level="id", sort=False):
        vals = sub["value"].to_numpy()
        per = sub["period"].to_numpy()
        x_hist = vals[per == 1]
        x_online = vals[per == 2]
        t = int(tau_map[dataset_id])
        tau = None if t == -1 else t
        output.append((dataset_id, x_hist, x_online, tau))
    
    return output

def emit_infer_datasets(X: pd.DataFrame):
    """Generator matching `infer()`: yields ``(dataset_id, x_historical, x_online)``
    for every series in `X`. No label — inference-time shape only.
    - `X`: MultiIndex(id, time) frame with cols `value`, `period` (1=hist, 2=online).
    """
    output = []
    for dataset_id, sub in X.groupby(level="id", sort=False):
        vals = sub["value"].to_numpy()
        per = sub["period"].to_numpy()

        x_hist = vals[per == 1]
        x_online = vals[per == 2]
        output.append((dataset_id, x_hist, x_online))

    return output


def local_ts_auc(y_test: pd.DataFrame, prediction: pd.DataFrame) -> float:
    from sklearn.metrics import roc_auc_score
    merged = prediction.merge(
    y_test,
    how="left",
    left_index=True,
    right_index=True,
    )

    # Add the online step index (0, 1, 2, ...).
    merged["time_online"] = merged.groupby("id").cumcount()

    # Weighted per-step AUC.
    weighted_auc_sum = 0.0
    total_weight     = 0.0

    for t, group in merged.groupby("time_online"):
        labels = group["target"].values
        scores = group["prediction"].values

        n_pos = int(labels.sum())
        n_neg = int((1 - labels).sum())
        if n_pos == 0 or n_neg == 0:
            continue

        auc_t  = float(roc_auc_score(labels, scores))
        weight = float(n_pos * n_neg)

        weighted_auc_sum += weight * auc_t
        total_weight     += weight

    ts_auc = weighted_auc_sum / total_weight if total_weight > 0 else 0.5
    return ts_auc


if __name__ == "__main__":
    pass