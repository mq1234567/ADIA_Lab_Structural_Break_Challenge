"""Dev harness for the incremental streaming submission: build training rows by
streaming StreamState, fit the GBM bag, then full-stream the reduced test set to
get local TS-AUC + per-step timing. Not the submission file (that's a notebook)."""
import time, numpy as np, pandas as pd, lightgbm as lgb
import sb_common as C
from sb_stream import StreamState
from sb_model import PARAMS, rank01

N_TRAIN = 6000        # subsample series for the dev loop
SAMPLES_PER = 30      # log-spaced online steps sampled per series for training


def iter_series(X):
    ids = X.index.get_level_values("id").to_numpy()
    times = X.index.get_level_values("time").to_numpy()
    vals = X["value"].to_numpy(np.float64); per = X["period"].to_numpy()
    uids, starts = np.unique(ids, return_index=True); b = np.append(starts, len(ids))
    for k, sid in enumerate(uids):
        s, e = b[k], b[k + 1]; onl = per[s:e] == 2
        yield sid, vals[s:e][~onl], vals[s:e][onl]


def log_steps(L, m):
    if L <= m:
        return np.arange(L)
    return np.unique(np.round(np.expm1(np.linspace(0, np.log1p(L - 1), m))).astype(int))


def stream_rows(X, ids, tau, sample=None):
    """Stream the given series -> (rows, labels, sids, times, feat_cols).
    sample=m -> keep only log-spaced m steps (training); None -> every step (eval)."""
    st = StreamState(); idset = set(ids.tolist())
    rows, labels, sids, times, cols = [], [], [], [], None
    for sid, xh, xo in iter_series(X):
        if sid not in idset:
            continue
        st.reset(xh); L = len(xo); ti = tau.get(sid, -1)
        want = set(log_steps(L, sample).tolist()) if sample else None
        for k in range(L):
            f = st.update(xo[k])
            if want is not None and k not in want:
                continue
            if cols is None:
                cols = list(f.keys())
            rows.append([f[c] for c in cols])
            labels.append(1 if (ti != -1 and k >= ti) else 0)
            sids.append(sid); times.append(k)
    return (np.asarray(rows, np.float32), np.asarray(labels, np.int8),
            np.asarray(sids), np.asarray(times), cols)


print("loading train...", flush=True)
X, y = C.load_train()
tau = pd.read_parquet(f"{C.DATA}/y_train_index.parquet")["tau_index"].to_dict()
tr_ids, va_ids = np.load("tr_ids.npy"), np.load("va_ids.npy")

t0 = time.time()
Xtr, ytr, sids, _, cols = stream_rows(X, tr_ids[:N_TRAIN], tau, sample=SAMPLES_PER)
cmap = dict(zip(*np.unique(sids, return_counts=True)))
w = np.array([1.0 / cmap[s] for s in sids]); w /= w.mean()
print("train rows %d from %d series  %.0fs  (%d feats)" % (
    len(Xtr), len(np.unique(sids)), time.time() - t0, len(cols)), flush=True)

models = [lgb.LGBMClassifier(random_state=s, **PARAMS).fit(Xtr, ytr, sample_weight=w) for s in (0, 1, 2)]

# eval: stream the 2000-series holdout, score EVERY step (stable, ~ compares to 0.5793)
tt = time.time()
Xva, yva, sva, tva, _ = stream_rows(X, va_ids, tau)
p = np.mean([rank01(m.predict_proba(Xva)[:, 1]) for m in models], axis=0)
score = C.ts_auc_arrays(p, yva, tva)
print("HOLDOUT(2000) stream TS-AUC = %.4f   (batch-bank subsample 0.5793 / full-res ~0.606)" % score, flush=True)
print("infer timing: %.1f us/step  -> 10M steps = %.2f h (x1 core)" % (
    1e6 * (time.time() - tt) / len(Xva), 10e6 * ((time.time() - tt) / len(Xva)) / 3600), flush=True)
