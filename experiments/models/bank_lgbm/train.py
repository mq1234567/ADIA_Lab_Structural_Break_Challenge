"""Reproducer for the bank_lgbm model.

(Re)builds the three feature parquets in experiments/ if missing, then trains
a 3-seed LightGBM bag and saves model_seed{0,1,2}.txt into this directory.
Also writes final_gain_ranking.csv back into experiments/.
"""
import gc, json, os, sys, time
import numpy as np, pandas as pd, lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, EXPERIMENTS)
import sb_common as C, sb_bank as B                                    # noqa: E402

PARAMS = json.load(open(os.path.join(HERE, "params.json")))
FEAT_TRAIN = os.path.join(EXPERIMENTS, "feat_final_train.parquet")
FEAT_HOLD = os.path.join(EXPERIMENTS, "feat_final_holdout.parquet")
FEAT_RED = os.path.join(EXPERIMENTS, "feat_final_reduced.parquet")


def build_features_if_missing():
    tr_ids = np.load(os.path.join(EXPERIMENTS, "tr_ids.npy"))
    va_ids = np.load(os.path.join(EXPERIMENTS, "va_ids.npy"))
    if not os.path.exists(FEAT_RED):
        Xr, _ = C.load_reduced()
        B.build_bank(Xr, out_path=FEAT_RED, full_resolution=True, block_size=100)
        del Xr; gc.collect()
    if not (os.path.exists(FEAT_TRAIN) and os.path.exists(FEAT_HOLD)):
        X, _ = C.load_train()
        t0 = time.time()
        B.build_bank(X, out_path=FEAT_TRAIN, block_size=300, verbose=True)
        print("train bank written %.0fs" % (time.time() - t0), flush=True)
        Xva = X[np.isin(X.index.get_level_values("id").to_numpy(), va_ids)]
        del X; gc.collect()
        B.build_bank(Xva, out_path=FEAT_HOLD, block_size=200)
        del Xva; gc.collect()
    return tr_ids, va_ids


def main():
    tr_ids, va_ids = build_features_if_missing()

    y_full = C.load_y()
    F = pd.read_parquet(FEAT_TRAIN); cols = list(F.columns)
    ids = F.index.get_level_values("id").to_numpy()
    mtr = np.isin(ids, tr_ids)
    yv = y_full.reindex(F.index).to_numpy().astype(np.int8)
    wtr = C.eq_series_weight(F.index[mtr])
    w_all = C.eq_series_weight(F.index)
    Xall = F.to_numpy(np.float32); del F; gc.collect()

    Fh = pd.read_parquet(FEAT_HOLD)[cols]
    yh = pd.Series(y_full.reindex(Fh.index).to_numpy().astype(np.int8), index=Fh.index)
    Xh = Fh.to_numpy(np.float32); h_idx = Fh.index; del Fh; gc.collect()

    preds, imp = [], np.zeros(len(cols))
    for s in (0, 1, 2):
        m = lgb.LGBMClassifier(random_state=s, **PARAMS).fit(
            Xall[mtr], yv[mtr], sample_weight=wtr)
        m.booster_.save_model(os.path.join(HERE, f"model_seed{s}.txt"))
        preds.append(pd.Series(m.predict_proba(Xh)[:, 1], index=h_idx))
        imp += m.booster_.feature_importance("gain")

    pd.Series(imp, index=cols).sort_values(ascending=False).to_csv(
        os.path.join(EXPERIMENTS, "final_gain_ranking.csv"))
    bag_h = pd.Series(np.mean([C.rank01(p.to_numpy()) for p in preds], axis=0), index=h_idx)
    print("HOLDOUT (2000 series):  seed0 %.4f | bag3 %.4f" % (
        C.ts_auc(preds[0], yh), C.ts_auc(bag_h, yh)), flush=True)
    del Xh; gc.collect()

    # reduced test uses ALL train series for fitting (paper-style final model)
    _, yr = C.load_reduced()
    Fr = pd.read_parquet(FEAT_RED)[cols]
    Xrr = Fr.to_numpy(np.float32)
    rp = [C.rank01(lgb.LGBMClassifier(random_state=s, **PARAMS)
                    .fit(Xall, yv, sample_weight=w_all)
                    .predict_proba(Xrr)[:, 1]) for s in (0, 1, 2)]
    pred_r = pd.Series(np.mean(rp, axis=0), index=Fr.index)
    print("REDUCED (100 series, noisy):  bag3 = %.4f" %
          C.ts_auc(pred_r, yr.reindex(Fr.index)), flush=True)


if __name__ == "__main__":
    main()
