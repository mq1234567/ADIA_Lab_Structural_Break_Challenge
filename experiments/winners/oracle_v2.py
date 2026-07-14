"""Extended oracle experiment: does the winner's feature battery raise the
detection ceiling on the REAL-TIME dataset?

Old battery (previous run): 0.7219 series-level AUC (5-fold CV).
Adds hand-implemented versions of the winning solution's key families:
entropy/complexity (sample/permutation/spectral entropy, Higuchi FD, Hjorth),
tsfresh-style change_quantiles / ratio_beyond_r_sigma / reoccurring values,
RBF-kernel MMD (ruptures CostRbf analog), AR(5) cross-prediction residuals,
CUMSUM-view trend features.
"""
import math
import numpy as np, pandas as pd, time
import lightgbm as lgb
from sklearn.model_selection import cross_val_predict, KFold
import sb_common as C
from sb_common import _auc_pair_stat


def mid_u(sorted_arr, x):
    n = len(sorted_arr)
    return 0.5 * (np.searchsorted(sorted_arr, x, 'left')
                  + np.searchsorted(sorted_arr, x, 'right')) / n


def samp_en(x, m=2, cap=400):
    x = np.asarray(x[:cap], dtype=np.float64)
    n = len(x)
    if n < m + 3:
        return np.nan
    r = 0.2 * x.std()
    if r <= 0:
        return np.nan
    def count(mm):
        emb = np.lib.stride_tricks.sliding_window_view(x, mm)
        d = np.abs(emb[:, None, :] - emb[None, :, :]).max(-1)
        iu = np.triu_indices(len(emb), 1)
        return float((d[iu] <= r).sum())
    B = count(m)
    A = count(m + 1)
    if A == 0 or B == 0:
        return np.nan
    return -np.log(A / B)


def perm_en(x, order=3):
    x = np.asarray(x)
    if len(x) < order + 1:
        return np.nan
    emb = np.lib.stride_tricks.sliding_window_view(x, order)
    pat = np.argsort(emb, axis=1)
    codes = pat @ (order ** np.arange(order))
    _, cnt = np.unique(codes, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log(p)).sum() / np.log(math.factorial(order)))


def spec_en(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 8:
        return np.nan
    p = np.abs(np.fft.rfft(x - x.mean())) ** 2
    p = p[1:]
    if p.sum() <= 0:
        return np.nan
    p = p / p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def higuchi_fd(x, kmax=10):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < kmax * 3:
        return np.nan
    L, ks = [], []
    for k in range(1, kmax + 1):
        Lk = []
        for m0 in range(k):
            idx = np.arange(m0, n, k)
            if len(idx) < 2:
                continue
            Lm = np.abs(np.diff(x[idx])).sum() * (n - 1) / (len(idx) - 1) / k
            Lk.append(Lm)
        if Lk:
            L.append(np.log(np.mean(Lk) + 1e-12))
            ks.append(np.log(1.0 / k))
    if len(L) < 3:
        return np.nan
    return float(np.polyfit(ks, L, 1)[0])


def hjorth(x):
    x = np.asarray(x, dtype=np.float64)
    dx = np.diff(x)
    ddx = np.diff(dx)
    s0, s1 = x.std(), dx.std()
    if s0 <= 0 or s1 <= 0:
        return np.nan, np.nan
    mob = s1 / s0
    comp = (ddx.std() / s1) / mob
    return float(mob), float(comp)


def change_q_var(x, ql, qh, isabs=True):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 10:
        return np.nan
    lo, hi = np.quantile(x, ql), np.quantile(x, qh)
    inside = (x >= lo) & (x <= hi)
    ok = inside[:-1] & inside[1:]
    d = np.diff(x)[ok]
    if len(d) < 5:
        return np.nan
    return float(np.abs(d).var() if isabs else d.var())


def mmd_rbf(a, b, cap=200, rng=None):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if rng is not None:
        if len(a) > cap:
            a = a[rng.choice(len(a), cap, replace=False)]
        if len(b) > cap:
            b = b[rng.choice(len(b), cap, replace=False)]
    pool = np.concatenate([a, b])
    med = np.median(np.abs(pool[:, None] - pool[None, :])) + 1e-9
    g = 1.0 / (2 * med * med)
    def K(u, v):
        return np.exp(-g * (u[:, None] - v[None, :]) ** 2)
    kaa = K(a, a); kbb = K(b, b); kab = K(a, b)
    na, nb = len(a), len(b)
    return float((kaa.sum() - na) / (na * (na - 1)) + (kbb.sum() - nb) / (nb * (nb - 1))
                 - 2 * kab.mean())


def ar_fit(x, lags=5):
    n = len(x)
    if n < lags + 10:
        return None, np.nan
    X = np.column_stack([x[lags - 1 - i:n - 1 - i] for i in range(lags)] + [np.ones(n - lags)])
    ytg = x[lags:]
    beta, *_ = np.linalg.lstsq(X, ytg, rcond=None)
    resid = ytg - X @ beta
    return beta, float(resid.std())


def ar_pred_resid(x_from, x_to, lags=5):
    beta, rs = ar_fit(x_from, lags)
    if beta is None or len(x_to) < lags + 10:
        return np.nan, np.nan
    n = len(x_to)
    X = np.column_stack([x_to[lags - 1 - i:n - 1 - i] for i in range(lags)] + [np.ones(n - lags)])
    e = x_to[lags:] - X @ beta
    return float(e.std()), rs


def battery(ref, seg, rng):
    """Old 15-stat battery + winner-family additions. All on (ref, post-seg)."""
    d_r = ref - ref.mean(); sd = d_r.std() + 1e-9
    zr, zs = d_r / sd, (seg - ref.mean()) / sd
    u = mid_u(np.sort(ref), seg)
    u2 = mid_u(np.sort(np.abs(d_r)), np.abs(seg - ref.mean()))
    dzr, dzs = np.diff(zr), np.diff(zs)
    o = {}
    # --- old battery (same as previous oracle run) ---
    o['u_mean'] = u.mean() - 0.5
    o['u_disp'] = np.abs(u - 0.5).mean() - 0.25
    o['u2_mean'] = u2.mean() - 0.5
    o['logvar'] = np.log(zs.var() + 1e-9)
    o['dvar'] = np.log((dzs.var() + 1e-9) / (dzr.var() + 1e-9))
    o['absdz'] = np.log((np.abs(dzs).mean() + 1e-9) / (np.abs(dzr).mean() + 1e-9))
    o['ac1'] = np.corrcoef(zs[1:], zs[:-1])[0, 1] - np.corrcoef(zr[1:], zr[:-1])[0, 1]
    o['ac2'] = np.corrcoef(zs[2:], zs[:-2])[0, 1] - np.corrcoef(zr[2:], zr[:-2])[0, 1]
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        o['cdf%g' % q] = (u <= q).mean() - q
    o['tail'] = ((u < 0.05) | (u > 0.95)).mean() - 0.10
    o['runs'] = (np.sign(zs[1:]) == np.sign(zs[:-1])).mean() - (np.sign(zr[1:]) == np.sign(zr[:-1])).mean()
    o['jump'] = (np.abs(dzs) > np.quantile(np.abs(dzr), 0.95)).mean() - 0.05
    o['seg_len'] = np.log(len(seg)); o['ref_len'] = np.log(len(ref))
    o['skew'] = ((zs - zs.mean()) ** 3).mean() / (zs.std() ** 3 + 1e-9) - (zr ** 3).mean()
    o['kurt'] = ((zs - zs.mean()) ** 4).mean() / (zs.std() ** 4 + 1e-9) - (zr ** 4).mean()
    # --- NEW: entropy / complexity (winner family 7) ---
    o['sampen_diff'] = samp_en(zs) - samp_en(zr[-400:])
    o['permen_diff'] = perm_en(zs) - perm_en(zr)
    o['specen_diff'] = spec_en(zs) - spec_en(zr)
    o['higuchi_diff'] = higuchi_fd(zs) - higuchi_fd(zr)
    mr, cr = hjorth(zr); ms, cs = hjorth(zs)
    o['hjorth_mob_diff'] = ms - mr
    o['hjorth_comp_diff'] = cs - cr
    # --- NEW: tsfresh-style (winner family 8) ---
    o['chq_var_28'] = np.log((change_q_var(zs, .2, .8) + 1e-9) / (change_q_var(zr, .2, .8) + 1e-9))
    o['chq_var_46'] = np.log((change_q_var(zs, .4, .6) + 1e-9) / (change_q_var(zr, .4, .6) + 1e-9))
    o['beyond_1s'] = (np.abs(zs) > 1).mean() - (np.abs(zr) > 1).mean()
    o['beyond_3s'] = (np.abs(zs) > 3).mean() - (np.abs(zr) > 3).mean()
    ur_, uc_ = np.unique(ref, return_counts=True)
    us_, usc_ = np.unique(seg, return_counts=True)
    o['reocc_diff'] = (usc_ > 1).mean() - (uc_ > 1).mean()
    # --- NEW: kernel MMD (ruptures CostRbf analog, winner family 10) ---
    o['mmd_rbf'] = mmd_rbf(zr, zs, rng=rng)
    # --- NEW: AR(5) cross-prediction (winner family 9) ---
    e_fwd, rs_in = ar_pred_resid(zr, zs)
    o['ar_xpred_ratio'] = np.log((e_fwd + 1e-9) / (rs_in + 1e-9)) if np.isfinite(e_fwd) else np.nan
    br, _ = ar_fit(zr); bs, _ = ar_fit(zs)
    o['ar_coef_dist'] = float(np.linalg.norm(br - bs)) if br is not None and bs is not None else np.nan
    # --- NEW: CUMSUM-view trend (winner transform) ---
    cs_r, cs_s = np.cumsum(zr), np.cumsum(zs)
    for nm, v in (('ref', cs_r), ('seg', cs_s)):
        x_ = np.arange(len(v))
        b1, b0 = np.polyfit(x_, v, 1)
        det = v - (b1 * x_ + b0)
        o[f'cum_r2_{nm}'] = 1 - det.var() / (v.var() + 1e-9)
        o[f'cum_detvol_{nm}'] = det.std() / (len(v) + 1e-9) ** 0.5
    o['cum_r2_diff'] = o['cum_r2_seg'] - o['cum_r2_ref']
    o['cum_detvol_ratio'] = np.log((o['cum_detvol_seg'] + 1e-9) / (o['cum_detvol_ref'] + 1e-9))
    return o


def main():
    X, _ = C.load_train()
    yidx = pd.read_parquet('/Users/minqi/Documents/ADIA_Lab_Structural_Break_Challenge/y_train_index.parquet')
    ids_all = X.index.get_level_values('id').to_numpy()
    vals = X['value'].to_numpy(np.float64)
    per = X['period'].to_numpy()
    uids, starts = np.unique(ids_all, return_index=True)
    bounds = np.append(starts, len(ids_all))
    rng0 = np.random.default_rng(0)
    sub = set(rng0.choice(uids, 6000, replace=False).tolist())

    rows, labels = [], []
    t0 = time.time()
    for k, sid in enumerate(uids):
        if sid not in sub:
            continue
        s, e = bounds[k], bounds[k + 1]
        v, p = vals[s:e], per[s:e]
        ref, onl = v[p == 1], v[p == 2]
        ti = int(yidx.loc[sid, 'tau_index'])
        seg, lab = (onl, 0) if ti < 0 else (onl[ti:], 1)
        if len(seg) < 20:
            continue
        rng = np.random.default_rng(int(sid))
        rows.append(battery(ref, seg, rng))
        labels.append(lab)
    df = pd.DataFrame(rows)
    lab = np.array(labels)
    print('n = %d | built in %.0fs | %d features' % (len(df), time.time() - t0, df.shape[1]))

    med = df.median()
    dff = df.fillna(med)

    configs = {
        'mine (lr.05 n400 31lv)': dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                       min_child_samples=50, reg_lambda=1.0),
        'winner (lr.005 n3600 29lv)': dict(n_estimators=3600, learning_rate=0.005, num_leaves=29,
                                           reg_alpha=3.0, reg_lambda=3.0),
    }
    base = dict(objective='binary', colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                verbosity=-1, n_jobs=-1, random_state=0)
    old_cols = [c for c in df.columns if c in (
        'u_mean','u_disp','u2_mean','logvar','dvar','absdz','ac1','ac2','tail','runs','jump',
        'seg_len','ref_len','skew','kurt','cdf0.1','cdf0.25','cdf0.5','cdf0.75','cdf0.9')]
    for feats_name, cols in [('old battery', old_cols), ('extended battery', list(df.columns))]:
        for cfg_name, ov in configs.items():
            m = lgb.LGBMClassifier(**{**base, **ov})
            pred = cross_val_predict(m, dff[cols].to_numpy(np.float32), lab,
                                     cv=KFold(5, shuffle=True, random_state=0),
                                     method='predict_proba')[:, 1]
            c, w = _auc_pair_stat(pred, lab)
            print('%-18s | %-26s AUC = %.4f' % (feats_name, cfg_name, c / w))

    # per-feature oracle AUC of new features
    print()
    new_cols = [c for c in df.columns if c not in old_cols]
    for c_ in new_cols:
        v = dff[c_].to_numpy()
        cc, ww = _auc_pair_stat(np.abs(v - np.median(v)), lab)
        print('  |%-18s| %.4f' % (c_, cc / ww))


if __name__ == '__main__':
    main()
