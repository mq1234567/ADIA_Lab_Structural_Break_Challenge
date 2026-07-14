"""Verify the submission notebook's train()/infer() end-to-end without crunch:
exec the notebook's code cells, simulate the crunch datasets format, run
train -> infer, score TS-AUC + measure real per-step infer timing."""
import json, time, tempfile, os, numpy as np, pandas as pd
import sb_common as C

# ---- exec the notebook's code cells (skip crunch/pip/test cells) ----
nb = json.load(open(f"{C.DATA}/submission.ipynb"))
import math, joblib, lightgbm as lgb
from typing import Iterable, List, Optional, Tuple
g = {"math": math, "os": os, "joblib": joblib, "np": np, "pd": pd, "lgb": lgb,
     "Iterable": Iterable, "List": List, "Optional": Optional, "Tuple": Tuple}
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if any(k in src for k in ("pip install", "crunch", "prediction/prediction")):
        continue
    exec(compile(src, "<cell>", "exec"), g)
train, infer = g["train"], g["infer"]
print("loaded train/infer from notebook", flush=True)

# ---- simulate crunch datasets from local parquet ----
X, y = C.load_train()
tau = pd.read_parquet(f"{C.DATA}/y_train_index.parquet")["tau_index"].to_dict()
tr_ids, va_ids = np.load("tr_ids.npy"), np.load("va_ids.npy")

def series(X, sid):
    s = X.loc[sid]; xh = s[s["period"] == 1]["value"].to_numpy(np.float64)
    xo = s[s["period"] == 2]["value"].to_numpy(np.float64); return xh, xo

def train_ds(ids):
    for sid in ids:
        xh, xo = series(X, sid)
        ti = tau.get(sid, -1)
        yield int(sid), xh, xo, (None if ti == -1 else int(ti))

tmp = tempfile.mkdtemp()
t0 = time.time()
train(list(train_ds(tr_ids[:6000])), tmp)
print("train %.0fs -> %s" % (time.time() - t0, os.listdir(tmp)), flush=True)

# ---- infer on a subset of holdout, score + time ----
EVAL = va_ids[:400]
def infer_ds(ids):
    for sid in ids:
        xh, xo = series(X, sid); yield xh, xo

t0 = time.time(); scores, steps, ys = [], [], []
gen = infer(infer_ds(EVAL), tmp)
next(gen)  # consume readiness yield
for sid in EVAL:
    xh, xo = series(X, sid); ti = tau.get(sid, -1)
    for k in range(len(xo)):
        scores.append(next(gen)); steps.append(k)
        ys.append(1 if (ti != -1 and k >= ti) else 0)
dt = time.time() - t0
auc = C.ts_auc_arrays(np.array(scores), np.array(ys, np.int8), np.array(steps))
print("INFER holdout[400] TS-AUC = %.4f  (full-res, ~0.60 target)" % auc, flush=True)
print("infer timing %.1f us/step -> 10M = %.2f h (x1 core)" % (
    1e6 * dt / len(scores), 10e6 * (dt / len(scores)) / 3600), flush=True)
