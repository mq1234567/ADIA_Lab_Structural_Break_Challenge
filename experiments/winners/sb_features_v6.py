"""v6 = v5 + causal ports of the strongest UNTRIED ideas from the classic
challenge's winning solutions (validated on this dataset by the extended
oracle test, which lifted the known-tau ceiling 0.722 -> 0.754):

- AR(5) cross-prediction residuals (BoNing-Gu family 9; oracle 0.639 alone):
  AR(5) fit on reference, online one-step residual e_t; family = e^2 change.
- Second-difference variance (Hjorth complexity direction; oracle ~0.61).
- Ring / quiet-state occupancy (Alphabot f15/f16/f26/f33): 1{|z|<=0.1} and
  1{1<=|z|<=2} occupancy families.
- Corridor change-variance (tsfresh change_quantiles; oracle 0.609): dz^2
  restricted to the reference's central [q20, q80] corridor.
- Permutation entropy (order 3) expanding diff vs reference (oracle 0.598).
- Multi-scale max aggregation (Alphabot Fisher/min-p idea): max |stat| over
  rolling windows {20, 50, 100, 200}.
- Boundary-local Welch t on |z| (Alphabot f01): first 50 online vs last 50
  reference observations.

All causal; scans/expanding stats reuse the v3 machinery with per-series
long-run-variance null calibration.
"""
import numpy as np
import pandas as pd

from sb_features_v3 import _bartlett_lrv, _scan, MIN_AGE
from sb_features_v4 import ref_stats_v4
from sb_features_v5 import series_features_v5, FEATURE_NAMES_V5
from sb_features import EPS

AR_LAGS = 5
BND_W = 50

V6_NEW = [
    "exp_ar5", "scan_ar5",
    "exp_d2var", "scan_d2var",
    "exp_ring0", "scan_ring0",
    "exp_ring12", "scan_ring12",
    "exp_chq", "scan_chq",
    "permen_diff_z",
    "ms_max_mean_z", "ms_max_u_z", "ms_max_logvar",
    "bnd_abs_t", "bnd_t",
]
FEATURE_NAMES_V6 = FEATURE_NAMES_V5 + V6_NEW


def _perm_codes(x):
    """Order-3 permutation pattern code (0..7, 6 valid) per sliding triple."""
    a, b, c = x[:-2], x[1:-1], x[2:]
    return (4 * (a < b) + 2 * (b < c) + (a < c)).astype(np.int64)


def _perm_entropy_hist(codes):
    cnt = np.bincount(codes, minlength=8).astype(np.float64)
    p = cnt / max(cnt.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def ref_stats_v6(ref):
    rs = ref_stats_v4(ref)
    ref = np.asarray(ref, dtype=np.float64)
    z = (ref - rs["mean"]) / rs["std"]

    # AR(5) on reference
    n = len(z)
    if n > AR_LAGS + 20:
        X = np.column_stack([z[AR_LAGS - 1 - i:n - 1 - i] for i in range(AR_LAGS)]
                            + [np.ones(n - AR_LAGS)])
        beta, *_ = np.linalg.lstsq(X, z[AR_LAGS:], rcond=None)
        e = z[AR_LAGS:] - X @ beta
    else:
        beta = np.zeros(AR_LAGS + 1)
        e = z.copy()
    rs["ar5_beta"] = beta
    rs["ar5_msq"] = float((e * e).mean()) + EPS
    rs["lrv_ar5"] = _bartlett_lrv(e * e / rs["ar5_msq"] - 1.0)

    d2 = np.diff(z, 2)
    rs["d2_msq"] = float((d2 * d2).mean()) + EPS
    rs["lrv_d2var"] = _bartlett_lrv(d2 * d2 / rs["d2_msq"] - 1.0)

    ring0 = (np.abs(z) <= 0.1).astype(np.float64)
    ring12 = ((np.abs(z) >= 1.0) & (np.abs(z) <= 2.0)).astype(np.float64)
    rs["ring0_p"] = float(ring0.mean())
    rs["ring12_p"] = float(ring12.mean())
    rs["lrv_ring0"] = _bartlett_lrv(ring0 - rs["ring0_p"])
    rs["lrv_ring12"] = _bartlett_lrv(ring12 - rs["ring12_p"])

    q20, q80 = np.quantile(z, 0.2), np.quantile(z, 0.8)
    rs["chq_lo"], rs["chq_hi"] = float(q20), float(q80)
    inside = (z >= q20) & (z <= q80)
    ok = inside[:-1] & inside[1:]
    dz = np.diff(z)
    dsel = dz[ok] if ok.sum() >= 10 else dz
    rs["chq_msq"] = float((dsel * dsel).mean()) + EPS
    chq_inc = np.where(ok, dz * dz / rs["chq_msq"] - 1.0, 0.0)
    rs["lrv_chq"] = _bartlett_lrv(chq_inc)

    rs["permen_ref"] = _perm_entropy_hist(_perm_codes(z))

    tail = np.abs(z[-BND_W:])
    rs["bnd_abs_mean"] = float(tail.mean())
    rs["bnd_abs_var"] = float(tail.var()) + EPS
    tail_z = z[-BND_W:]
    rs["bnd_mean"] = float(tail_z.mean())
    rs["bnd_var"] = float(tail_z.var()) + EPS
    return rs


def series_features_v6(rs, x):
    base = series_features_v5(rs, x)   # includes v2..v5 columns

    x = np.asarray(x, dtype=np.float64)
    L = len(x)
    n = np.arange(1, L + 1, dtype=np.float64)
    sq_n = np.sqrt(n)
    z = (x - rs["mean"]) / rs["std"]
    inv_sqrt_age = np.concatenate(([0.0], 1.0 / np.sqrt(np.arange(1, L + 1))))

    # seed with last values of the reference (known constants -> causal)
    zm2, zm1 = rs["z_tail"]
    # for AR(5) / d2 / corridor we need the last 5 reference z values; rebuild
    # them from the stored sorted array is impossible -> keep simple: seed with
    # zm2, zm1 only and zero-pad older lags (constant across prefixes -> still
    # prefix-invariant / causal).
    ze = np.concatenate(([0.0, 0.0, 0.0, zm2, zm1], z))

    def fam(inc, lrv):
        sd = np.sqrt(max(lrv, EPS))
        S = np.concatenate(([0.0], np.cumsum(inc / sd)))
        exp = (S[1:] / n) * sq_n
        scan, _ = _scan(S, inv_sqrt_age)
        return exp, scan

    # AR(5) one-step residuals
    beta = rs["ar5_beta"]
    lag_mat = np.column_stack([ze[5 - 1 - i:5 - 1 - i + L] for i in range(AR_LAGS)]
                              + [np.ones(L)])
    e = z - lag_mat @ beta
    exp_ar5, scan_ar5 = fam(e * e / rs["ar5_msq"] - 1.0, rs["lrv_ar5"])

    d2 = ze[5:] - 2.0 * ze[4:-1] + ze[3:-2]          # second difference at t
    exp_d2, scan_d2 = fam(d2 * d2 / rs["d2_msq"] - 1.0, rs["lrv_d2var"])

    ring0 = (np.abs(z) <= 0.1).astype(np.float64)
    ring12 = ((np.abs(z) >= 1.0) & (np.abs(z) <= 2.0)).astype(np.float64)
    exp_r0, scan_r0 = fam(ring0 - rs["ring0_p"], rs["lrv_ring0"])
    exp_r12, scan_r12 = fam(ring12 - rs["ring12_p"], rs["lrv_ring12"])

    dz = ze[5:] - ze[4:-1]
    inside = (z >= rs["chq_lo"]) & (z <= rs["chq_hi"])
    inside_prev = np.concatenate(([True], inside[:-1]))
    ok = inside & inside_prev
    chq_inc = np.where(ok, dz * dz / rs["chq_msq"] - 1.0, 0.0)
    exp_chq, scan_chq = fam(chq_inc, rs["lrv_chq"])

    # expanding permutation entropy (order 3) vs reference
    codes = _perm_codes(np.concatenate(([zm2, zm1], z)))   # one code per online t
    onehot = np.zeros((L, 8))
    onehot[np.arange(L), codes] = 1.0
    cum = onehot.cumsum(axis=0)
    tot = cum.sum(axis=1, keepdims=True)
    p = cum / np.maximum(tot, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    permen_diff_z = np.where(n >= 20, (H - rs["permen_ref"]) * sq_n, 0.0)

    # multi-scale max over the rolling-window stats already in `base`
    names = FEATURE_NAMES_V5
    def col(c):
        return base[:, names.index(c)].astype(np.float64)
    ms_mean = np.max(np.abs(np.stack([col(f"roll{w}_mean_z") for w in (20, 50, 100, 200)])), axis=0)
    ms_u = np.max(np.abs(np.stack([col(f"roll{w}_u_z") for w in (20, 50, 100, 200)])), axis=0)
    ms_var = np.max(np.abs(np.stack([col(f"roll{w}_logvar") for w in (20, 50, 100, 200)])), axis=0)

    # boundary-local Welch t (first <=50 online vs last 50 reference)
    m = np.minimum(n, BND_W)
    az = np.abs(z)
    ca, ca2 = np.cumsum(az), np.cumsum(az * az)
    cz, cz2 = np.cumsum(z), np.cumsum(z * z)
    iW = int(min(BND_W, L)) - 1
    def frozen(c, c2):
        s = np.where(n <= BND_W, c, c[iW])
        s2 = np.where(n <= BND_W, c2, c2[iW])
        mu = s / m
        va = np.clip(s2 / m - mu ** 2, EPS, None)
        return mu, va
    mu_a, va_a = frozen(ca, ca2)
    mu_z, va_z = frozen(cz, cz2)
    bnd_abs_t = (mu_a - rs["bnd_abs_mean"]) / np.sqrt(va_a / m + rs["bnd_abs_var"] / BND_W)
    bnd_t = (mu_z - rs["bnd_mean"]) / np.sqrt(va_z / m + rs["bnd_var"] / BND_W)

    new = np.column_stack([
        exp_ar5, scan_ar5, exp_d2, scan_d2, exp_r0, scan_r0,
        exp_r12, scan_r12, exp_chq, scan_chq, permen_diff_z,
        ms_mean, ms_u, ms_var, bnd_abs_t, bnd_t,
    ]).astype(np.float32)
    return np.hstack([base, new])


def build_features_v6(X, ids_subset=None):
    ids_all = X.index.get_level_values("id").to_numpy()
    times_all = X.index.get_level_values("time").to_numpy()
    vals = X["value"].to_numpy(dtype=np.float64)
    per = X["period"].to_numpy()

    uids, starts = np.unique(ids_all, return_index=True)
    bounds = np.append(starts, len(ids_all))
    keep = set(ids_subset) if ids_subset is not None else None

    mats, out_ids, out_times = [], [], []
    for k, sid in enumerate(uids):
        if keep is not None and sid not in keep:
            continue
        s, e = bounds[k], bounds[k + 1]
        p = per[s:e]
        v = vals[s:e]
        t = times_all[s:e]
        onl = p == 2
        x = v[onl]
        if len(x) == 0:
            continue
        mats.append(series_features_v6(ref_stats_v6(v[~onl]), x))
        out_ids.append(np.full(len(x), sid))
        out_times.append(t[onl])

    F = np.vstack(mats)
    idx = pd.MultiIndex.from_arrays(
        [np.concatenate(out_ids), np.concatenate(out_times)], names=["id", "time"])
    return pd.DataFrame(F, index=idx, columns=FEATURE_NAMES_V6)
