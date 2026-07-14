"""Confirm the consolidated single-file bank (1072 features = calibrated +
descriptive + hypothesis + spectral + context) reproduces the union. Memory-safe
(8 GB): stream to parquet, evaluate on the log-subsample holdout.
Targets: union (two-parquet) 0.5821 ; 330-calibrated-alone 0.5773 (subsample bag3)."""
import sys, time, gc
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "/Users/minqi/Documents/ADIA_Lab_Structural_Break_Challenge/experiments")
import sb_common as C, sb_bank as B
from sb_model import PARAMS, rank01, eq_series_weight, true_step

tr_ids, va_ids = np.load("tr_ids.npy"), np.load("va_ids.npy")
Xr, yr = C.load_reduced()
B.build_bank(Xr, out_path="feat_final_reduced.parquet", full_resolution=True, block_size=100)
del Xr; gc.collect()

X, _ = C.load_train()
t0 = time.time()
B.build_bank(X, out_path="feat_final_train.parquet", block_size=300, verbose=True)
print("final train written %.0fs" % (time.time() - t0), flush=True)
Xva = X[np.isin(X.index.get_level_values("id").to_numpy(), va_ids)]
del X; gc.collect()
B.build_bank(Xva, out_path="feat_final_holdout.parquet", block_size=200)
del Xva; gc.collect()

y_full = pd.read_parquet(f"{C.DATA}/y_train.parquet")["target"]
F = pd.read_parquet("feat_final_train.parquet"); cols = list(F.columns)
ids = F.index.get_level_values("id").to_numpy()
mtr, mva = np.isin(ids, tr_ids), np.isin(ids, va_ids)
yv = y_full.reindex(F.index).to_numpy().astype(np.int8)
th = true_step(F.index, y_full)
wtr, w_all = eq_series_weight(F.index[mtr]), eq_series_weight(F.index)
Xall = F.to_numpy(np.float32); del F; gc.collect()

Fh = pd.read_parquet("feat_final_holdout.parquet")[cols]
yh = y_full.reindex(Fh.index).to_numpy().astype(np.int8); thh = true_step(Fh.index, y_full)
Xh = Fh.to_numpy(np.float32); del Fh; gc.collect()

preds, imp = [], np.zeros(len(cols))
for s in (0, 1, 2):
    m = lgb.LGBMClassifier(random_state=s, **PARAMS).fit(Xall[mtr], yv[mtr], sample_weight=wtr)
    preds.append(m.predict_proba(Xh)[:, 1]); imp += m.booster_.feature_importance("gain")
ir = pd.Series(imp, index=cols); ir.sort_values(ascending=False).to_csv("final_gain_ranking.csv")
print("FINAL subsample-holdout: seed0 %.4f | bag3 %.4f   (union 0.5821 / cal-alone 0.5773)" % (
    C.ts_auc_arrays(preds[0], yh, thh),
    C.ts_auc_arrays(np.mean([rank01(p) for p in preds], axis=0), yh, thh)), flush=True)
print("gain share cal_ vs rest: %.2f / %.2f" % (
    ir[[c for c in cols if c.startswith("cal_")]].sum() / ir.sum(),
    ir[[c for c in cols if not c.startswith("cal_")]].sum() / ir.sum()), flush=True)
print("top 12:", ir.sort_values(ascending=False).head(12).round(0).to_dict(), flush=True)
del Xh; gc.collect()

Fr = pd.read_parquet("feat_final_reduced.parquet")[cols]
Xrr = Fr.to_numpy(np.float32)
rp = [rank01(lgb.LGBMClassifier(random_state=s, **PARAMS)
             .fit(Xall, yv, sample_weight=w_all).predict_proba(Xrr)[:, 1]) for s in (0, 1, 2)]
print("FINAL reduced bag3 = %.4f   (union 0.5676 / baseline 0.5719)" %
      C.ts_auc(pd.Series(np.mean(rp, axis=0), index=Fr.index), yr.reindex(Fr.index)), flush=True)
