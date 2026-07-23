"""Build the full-resolution CV cache: every series, every online step.

Why full resolution
-------------------
The older caches baked in one split — 8,000 "train" series stored at only ~40
log-spaced steps, 2,000 "val" series stored at every step. K-fold needs each series
to play BOTH roles (validate once, train in the other folds), and a series stored at
40 steps cannot be validated the way TS-AUC actually scores: at every online step.

So this stores features for ALL series at EVERY online step, plus an
`is_sampled_step` mask marking the ~40 log-spaced steps the deployed trainer uses.
`cv_tools.run_cv` then does:

    train rows = train_row_mask=is_sampled_step   (coarse, matches deployment)
    val   rows = all rows of held-out series      (full, matches the metric)

One cache, any fold assignment, no rebuild per experiment.

Leakage properties (verified, see tools/stream_state.py):
  * features are causal — truncating a stream leaves earlier steps bit-identical,
  * `reset()` fully clears state, so rows do not depend on series ordering,
  * `time` is a per-series counter (every series starts at 0), so there is no shared
    calendar and no cross-series time leakage to guard against,
  * rows are written sorted by (id, time), which `cv_tools.steps_from_index` assumes.

Output (~1.0 GB, npz, uncompressed for fast reload):
    X                float32 [n_rows, 50]   feature matrix
    y                int8    [n_rows]       1 if a break has occurred by this step
    sid              int64   [n_rows]       series id
    time             int64   [n_rows]       within-series time index
    is_sampled_step  bool    [n_rows]       the ~40 log-spaced training steps
    cols             str     [50]           feature names

Usage:  python build_cache.py [--out PATH] [--samples-per 40] [--limit N]
"""
from __future__ import annotations

import argparse
import os
import sys
import time as _time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.abspath(os.path.join(HERE, ".."))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
from stream_state import StreamState, log_steps  # noqa: E402

DATA = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
DEFAULT_OUT = os.path.join(HERE, "cv_cache_full.npz")


def build(out_path: str = DEFAULT_OUT, samples_per: int = 40, limit: int | None = None) -> None:
    t0 = _time.time()
    # Fail now, not after 5 minutes of feature building: the save is the last step,
    # so an unwritable destination otherwise costs the whole run.
    outdir = os.path.dirname(os.path.abspath(out_path))
    if not os.path.isdir(outdir):
        raise RuntimeError(f"output dir does not exist: {outdir}")
    print("loading raw data ...", flush=True)
    X = pd.read_parquet(os.path.join(DATA, "X_train.parquet"))
    y_full = pd.read_parquet(os.path.join(DATA, "y_train.parquet"))["target"]

    ids_all = X.index.get_level_values("id").to_numpy()
    times_all = X.index.get_level_values("time").to_numpy()
    vals = X["value"].to_numpy(np.float64)
    per = X["period"].to_numpy()
    uids, starts = np.unique(ids_all, return_index=True)   # np.unique sorts -> (id, time) order
    bounds = np.append(starts, len(ids_all))
    if limit:
        uids = uids[:limit]

    # ---- pass 1: which series are usable, and how many rows in total ------
    keep, n_online = [], []
    for k in range(len(uids)):
        s, e = bounds[k], bounds[k + 1]
        m_on = per[s:e] == 2
        n_on = int(m_on.sum())
        n_hist = (e - s) - n_on
        ok = n_on > 0 and n_hist >= 8          # same filter the experiments used
        keep.append(ok)
        n_online.append(n_on if ok else 0)
    keep = np.array(keep); n_online = np.array(n_online)
    total = int(n_online.sum())
    proto = StreamState()
    ncol = proto.ncol
    print(f"  {keep.sum():,} usable series of {len(uids):,} | {total:,} online rows "
          f"| {ncol} features -> {total * ncol * 4 / 1e9:.2f} GB", flush=True)

    # ---- pass 2: fill preallocated arrays ---------------------------------
    # Preallocated (not a list of rows): 5M x 50 as a python list would cost several GB
    # of object overhead before it ever reached numpy.
    Xf = np.empty((total, ncol), np.float32)
    sid = np.empty(total, np.int64)
    tim = np.empty(total, np.int64)
    sampled = np.zeros(total, bool)

    st = StreamState()
    w = 0
    for k in range(len(uids)):
        if not keep[k]:
            continue
        s, e = bounds[k], bounds[k + 1]
        m_on = per[s:e] == 2
        x_hist = vals[s:e][~m_on]
        x_on = vals[s:e][m_on]
        t_on = times_all[s:e][m_on]
        L = len(x_on)

        st.reset(x_hist)                        # full state clear -> no cross-series bleed
        for i in range(L):
            Xf[w + i] = st.update(x_on[i])      # copied by the float32 assignment
        sid[w:w + L] = uids[k]
        tim[w:w + L] = t_on
        sampled[w + log_steps(L, samples_per)] = True
        w += L

        if (k + 1) % 2000 == 0:
            print(f"  {k + 1:,}/{len(uids):,} series | {w:,} rows | "
                  f"{_time.time() - t0:.0f}s", flush=True)
    assert w == total, f"row count mismatch: wrote {w}, expected {total}"

    # ---- labels: align by (id, time) rather than trusting row order -------
    index = pd.MultiIndex.from_arrays([sid, tim], names=["id", "time"])
    y = y_full.reindex(index)
    if y.isna().any():
        raise RuntimeError(f"{int(y.isna().sum())} rows have no label — (id,time) misalignment")
    y = y.to_numpy().astype(np.int8)

    print(f"saving -> {out_path}", flush=True)
    np.savez(out_path, X=Xf, y=y, sid=sid, time=tim,
             is_sampled_step=sampled, cols=np.array(proto.feature_names))
    print(f"done: {total:,} rows, {keep.sum():,} series, "
          f"{sampled.sum():,} sampled train rows, pos-rate {y.mean():.3f} | "
          f"{_time.time() - t0:.0f}s | {os.path.getsize(out_path) / 1e9:.2f} GB", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--samples-per", type=int, default=40,
                    help="log-spaced steps flagged as training rows (deployed trainer uses 40)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N series (smoke test)")
    a = ap.parse_args()
    build(a.out, a.samples_per, a.limit)


if __name__ == "__main__":
    main()
