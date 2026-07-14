"""Validate slow-lr TUNED params + a decorrelated-GBM blend on the real eval sets.
Reduced test uses its OWN labels (load_reduced), NOT train y (ids 10000-10099)."""
import numpy as np, pandas as pd, lightgbm as lgb, time
import sb_common as C, sb_cv as cv
from sb_model import PARAMS, rank01, eq_series_weight, true_step

X, idx, cols = cv.load_bank(); y = cv.load_y()
TUNED = {**PARAMS, 'learning_rate': 0.01, 'n_estimators': 2500, 'reg_alpha': 1.0, 'reg_lambda': 3.0}
tr_ids = np.load('tr_ids.npy'); ids = idx.get_level_values('id').to_numpy()
mtr = np.isin(ids, tr_ids); yv = y.reindex(idx).to_numpy().astype(np.int8)
wtr = eq_series_weight(idx[mtr])

# holdout: ids are a subset of train -> use train y
Fh = pd.read_parquet('feat_final_holdout.parquet')[cols]
Xh = Fh.to_numpy(np.float32); yh = y.reindex(Fh.index).to_numpy().astype(np.int8)
th = true_step(Fh.index, y)
# reduced: separate label file
Xr_df, yr_s = C.load_reduced()
Fr = pd.read_parquet('feat_final_reduced.parquet')[cols]
Xr = Fr.to_numpy(np.float32); yr = yr_s.reindex(Fr.index).to_numpy().astype(np.int8)
trd = true_step(Fr.index, yr_s)

def bag(params, nseed=3):
    ph, pr = [], []
    for s in range(nseed):
        m = lgb.LGBMClassifier(random_state=s, **params).fit(X[mtr], yv[mtr], sample_weight=wtr)
        ph.append(m.predict_proba(Xh)[:, 1]); pr.append(m.predict_proba(Xr)[:, 1])
    return np.mean([rank01(q) for q in ph], 0), np.mean([rank01(q) for q in pr], 0)

t0 = time.time()
bh, br = bag(PARAMS); print('BASELINE holdout %.4f (0.5793) | reduced %.4f (0.5764)  %.0fs' % (
    C.ts_auc_arrays(bh, yh, th), C.ts_auc_arrays(br, yr, trd), time.time()-t0), flush=True)
gh, gr = bag(TUNED); print('TUNED    holdout %.4f (0.5793) | reduced %.4f (0.5764)  %.0fs' % (
    C.ts_auc_arrays(gh, yh, th), C.ts_auc_arrays(gr, yr, trd), time.time()-t0), flush=True)

# decorrelated 2nd GBM diversity member (shallow, low colsample)
DIV = dict(objective='binary', n_estimators=400, learning_rate=0.05, num_leaves=15,
           min_child_samples=200, colsample_bytree=0.4, subsample=0.7, subsample_freq=1,
           reg_lambda=1.0, n_jobs=-1, verbosity=-1)
d = lgb.LGBMClassifier(random_state=99, **DIV).fit(X[mtr], yv[mtr], sample_weight=wtr)
dh = rank01(d.predict_proba(Xh)[:, 1]); dr = rank01(d.predict_proba(Xr)[:, 1])

# blend weight tuned on holdout tune-half, reported on report-half + reduced
hid = Fh.index.get_level_values('id').to_numpy(); hu = np.unique(hid)
tune = set(np.random.default_rng(1).permutation(hu)[:len(hu)//2].tolist())
mt = np.isin(hid, list(tune)); mrep = ~mt
bl = lambda a, b, w: rank01((1-w)*a + w*b)
bw, bb = 0, -1
for w in np.linspace(0, 0.6, 13):
    sc = C.ts_auc_arrays(bl(gh, dh, w)[mt], yh[mt], th[mt])
    if sc > bb: bb, bw = sc, w
g_rep = C.ts_auc_arrays(gh[mrep], yh[mrep], th[mrep])
b_rep = C.ts_auc_arrays(bl(gh, dh, bw)[mrep], yh[mrep], th[mrep])
print('BLEND w=%.2f  holdout-report %.4f -> %.4f (%+.4f) | reduced %.4f -> %.4f (%+.4f)' % (
    bw, g_rep, b_rep, b_rep-g_rep,
    C.ts_auc_arrays(gr, yr, trd), C.ts_auc_arrays(bl(gr, dr, bw), yr, trd),
    C.ts_auc_arrays(bl(gr, dr, bw), yr, trd) - C.ts_auc_arrays(gr, yr, trd)), flush=True)
