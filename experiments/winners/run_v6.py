"""v6 pipeline: causality test, build features, holdout model, reduced test."""
import numpy as np, pandas as pd, time
import lightgbm as lgb
import sb_common as C
from sb_features_v6 import build_features_v6, V6_NEW

# causality
Xr, yr = C.load_reduced()
for sid in Xr.index.get_level_values('id').unique()[:3]:
    Xs = Xr.loc[[sid]]
    full = build_features_v6(Xs)
    onl_times = Xs.loc[sid].query('period == 2').index
    cut = onl_times[len(onl_times) // 2]
    prefix = build_features_v6(Xs[Xs.index.get_level_values('time') <= cut])
    pd.testing.assert_frame_equal(full.loc[prefix.index], prefix)
print('causality OK', flush=True)

t0 = time.time()
Fr = build_features_v6(Xr)
Fr.to_parquet('feat_reduced_v6.parquet')
print('reduced v6:', Fr.shape, 'in %.0fs' % (time.time() - t0), flush=True)

t0 = time.time()
X, y = C.load_train()
F = build_features_v6(X)
print('train v6:', F.shape, 'in %.0fs | NaN: %s' % (time.time() - t0, F.isna().any().any()), flush=True)
F.to_parquet('feat_train_v6.parquet')
del X

tr_ids = np.load('tr_ids.npy'); va_ids = np.load('va_ids.npy')
ids = F.index.get_level_values('id').to_numpy()
mtr, mva = np.isin(ids, tr_ids), np.isin(ids, va_ids)
yv = y.reindex(F.index).to_numpy()
Ln = pd.Series(1, index=F.index).groupby(level='id').transform('size').to_numpy()
va_idx = F.index[mva]
yva_s = pd.Series(yv[mva], index=va_idx)

print('\nper-feature TS-AUC (holdout) of new v6 features:', flush=True)
Fva = F[mva]
for c in V6_NEW:
    print('  %-16s %.4f' % (c, C.ts_auc(Fva[c], yva_s)), flush=True)

params = dict(objective='binary', n_estimators=500, learning_rate=0.05,
              num_leaves=63, min_child_samples=300, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbosity=-1)
Xtr, ytr = F[mtr].to_numpy(np.float32), yv[mtr]
Xva = F[mva].to_numpy(np.float32)
w0 = 1.0 / Ln[mtr]; w0 = w0 / w0.mean()

def rank01(a):
    return np.argsort(np.argsort(a)) / (len(a) - 1.0)

t0 = time.time()
preds = []
for s in (0, 1, 2):
    m = lgb.LGBMClassifier(random_state=s, **params).fit(Xtr, ytr, sample_weight=w0)
    preds.append(pd.Series(m.predict_proba(Xva)[:, 1], index=va_idx))
    if s == 0:
        print('v6_eq seed0: holdout TS-AUC = %.4f (%ds)'
              % (C.ts_auc(preds[0], yva_s), time.time() - t0), flush=True)
        imp = pd.Series(m.booster_.feature_importance('gain'), index=F.columns)
        print('top gain:', (imp / imp.sum()).sort_values(ascending=False).head(12).round(3).to_dict(), flush=True)
bag = pd.Series(np.mean([rank01(p.to_numpy()) for p in preds], axis=0), index=va_idx)
print('v6_eq bag3 : holdout TS-AUC = %.4f' % C.ts_auc(bag, yva_s), flush=True)

# train on all, reduced test
w_all = 1.0 / Ln; w_all = w_all / w_all.mean()
Xall = F.to_numpy(np.float32)
rp = []
for s in (0, 1, 2):
    m = lgb.LGBMClassifier(random_state=s, **params).fit(Xall, yv, sample_weight=w_all)
    rp.append(rank01(m.predict_proba(Fr.to_numpy(np.float32))[:, 1]))
pr = pd.Series(np.mean(rp, axis=0), index=Fr.index)
print('REDUCED TEST TS-AUC (v6 bag3): %.4f' % C.ts_auc(pr, yr), flush=True)
