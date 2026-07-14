"""Structural-break feature bank — large, winner-style, flat.

Idea (from the 2nd-place classic solution): make several TRANSFORMED VIEWS of the
series, compute a battery of DESCRIPTIVE STATISTICS over the whole thing and over
trailing windows, and compare the live stream to its own history. Add a few classic
two-sample HYPOTHESIS TESTS, a FREQUENCY-DOMAIN view, and change-point statistics.
Gradient-boosted trees shrug off redundant/weak columns, so we generate ~900+
features and let the model sort them out.

The only wrinkle vs. the winner: they compute ONE number per series (they look at
the end); we must emit a number at EVERY online step, causally. That is exactly
what pandas `.expanding()` / `.rolling()` give us — running stats that only ever
look backward — so causality is automatic and the code stays plain. `test_causality`
double-checks that a feature at step t never depends on anything after t.

Read order: helpers -> build_views() -> prep_reference() -> the feature blocks
(descriptive / hypothesis / spectral / changepoint / context) -> series_features()
glues them -> build_bank() is the driver.
"""
import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-9
CLIP = 1e4
WINDOWS = (20, 50, 100, 200, 500)          # trailing "periods"
SPEC_W = 128                               # frequency-analysis window
LRV_LAGS = 30
EXPAND_STATS = ["mean", "std", "skew", "kurt", "min", "max", "median"]
ROLL_STATS = ["mean", "std", "skew", "kurt"]


def mid_rank(sorted_ref, x):
    """Percentile of each x within the sorted reference (0..1), ties averaged."""
    n = len(sorted_ref)
    return 0.5 * (np.searchsorted(sorted_ref, x, "left")
                  + np.searchsorted(sorted_ref, x, "right")) / n


def bartlett_lrv(v, lags=LRV_LAGS):
    """Per-series 'natural wobble' scale, accounting for autocorrelation."""
    v = np.asarray(v, np.float64); v = v - v.mean(); n = len(v)
    if n < 10:
        return max(v.var(), EPS)
    g0 = float((v * v).mean()); K = min(lags, n // 5); lrv = g0
    for l in range(1, K + 1):
        lrv += 2.0 * (1.0 - l / (K + 1.0)) * float((v[l:] * v[:-l]).mean())
    return float(max(lrv, 0.05 * g0, EPS))


# =========================================================================
# transformed VIEWS of the series (a DataFrame, one column per view)
# =========================================================================
def build_views(x, ref):
    """Causal transformed views of the standardized series, as a DataFrame.
    diff / rolling views use pandas ops (backward-looking) so they stay causal."""
    z = pd.Series((np.asarray(x, np.float64) - ref["mean"]) / ref["std"])
    u = pd.Series(mid_rank(ref["sorted"], x))
    return pd.DataFrame({
        "z": z,
        "abs": z.abs(),
        "sign": np.sign(z),
        "square": z ** 2,
        "logabs": np.log(z.abs() + EPS),
        "cumsum": z.cumsum(),
        "diff": z.diff().fillna(0.0),
        "diff2": z.diff().diff().fillna(0.0),
        "rank": u,
        "rank_centered": u - 0.5,
        "rollmean16": z.rolling(16, min_periods=1).mean(),
        "rollstd16": z.rolling(16, min_periods=1).std().fillna(0.0),
    })


def prep_reference(reference):
    """Precompute once per series: standardization, sorted values, and the
    descriptive stats of every view over the whole reference (for comparisons)."""
    reference = np.asarray(reference, np.float64)
    mean, std = reference.mean(), reference.std() + EPS
    ref = {"vals": reference, "n": len(reference), "mean": mean, "std": std,
           "sorted": np.sort(reference), "z": (reference - mean) / std}
    ref["seed"] = ref["z"][-1] if len(reference) else 0.0
    RV = build_views(reference, ref)
    R = RV.agg(EXPAND_STATS)                                    # rows=stat, cols=view
    R.loc["q25"], R.loc["q75"] = RV.quantile(0.25), RV.quantile(0.75)
    R.loc["range"] = R.loc["max"] - R.loc["min"]
    R.loc["cov"] = R.loc["std"] / (R.loc["mean"].abs() + EPS)
    R.loc["autocorr"] = RV.apply(lambda s: s.autocorr(1)).fillna(0.0)
    ref["ref_stats"] = R

    # constants + per-family null long-run variances for the calibrated battery
    z = ref["z"]; nz = len(z)
    ref["z_tail"] = z[-5:] if nz >= 5 else np.concatenate([np.zeros(5 - nz), z])
    ref["ar5"] = fit_ar5(z)
    rc = online_channels(reference[5:], ref, z[:5])          # reference null channels
    az, dz, v = rc["az"], rc["dz"], float((rc["z"] ** 2).mean()) + EPS
    ref["ac1"] = float((rc["z"] * rc["lag1"]).mean() / v)
    ref["ac2"] = float((rc["z"] * rc["lag2"]).mean() / v)
    ref["e_az"] = float(az.mean())
    ref["ring0_p"] = float((az <= 0.1).mean())
    ref["ring12_p"] = float(((az >= 1) & (az <= 2)).mean())
    ref["dz_msq"] = float((dz * dz).mean()) + EPS
    ref["dz_mabs"] = float(np.abs(dz).mean()) + EPS
    ref["dz_q95"] = float(np.quantile(np.abs(dz), 0.95))
    ref["jump_p"] = float((np.abs(dz) > ref["dz_q95"]).mean())
    ref["runs_p"] = float((np.sign(rc["z"]) == np.sign(rc["lag1"])).mean())
    ref["ar5_msq"] = float((rc["e5"] ** 2).mean()) + EPS
    ref["lrv"] = {name: bartlett_lrv(a) for name, a in increments(rc, ref).items()}
    return ref


# =========================================================================
# FEATURE BLOCK 1 — descriptive stats over views x windows, vs the reference
# =========================================================================
def descriptive_features(online, ref):
    V = build_views(online, ref)
    R = ref["ref_stats"]
    feats = {}

    exp = V.expanding()
    est = {s: getattr(exp, s)() for s in EXPAND_STATS}
    est["q25"], est["q75"] = exp.quantile(0.25), exp.quantile(0.75)
    est["range"] = est["max"] - est["min"]
    est["cov"] = est["std"] / (est["mean"].abs() + EPS)
    lag = V.shift(1)
    est["autocorr"] = ((V * lag).expanding().mean() - est["mean"] ** 2) / (est["std"] ** 2 + EPS)
    for sname, df in est.items():
        r = R.loc[sname]
        for v in V.columns:
            col = df[v].to_numpy()
            feats[f"{v}_{sname}"] = col
            feats[f"{v}_{sname}_vs_ref"] = col - r[v]
            feats[f"{v}_{sname}_ratio"] = col / (abs(r[v]) + EPS)

    for w in WINDOWS:
        roll = V.rolling(w, min_periods=1)
        for sname in ROLL_STATS:
            df = getattr(roll, sname)()
            r = R.loc[sname]
            for v in V.columns:
                col = df[v].to_numpy()
                feats[f"{v}_{sname}_w{w}"] = col
                feats[f"{v}_{sname}_w{w}_vs_ref"] = col - r[v]
    return feats


# =========================================================================
# FEATURE BLOCK 2 — classic two-sample tests: reference vs online-so-far
# =========================================================================
def hypothesis_test_features(online, ref):
    z = (np.asarray(online, np.float64) - ref["mean"]) / ref["std"]
    n = np.arange(1, len(z) + 1, dtype=np.float64)
    u = mid_rank(ref["sorted"], online)
    feats = {}
    # F-test: online variance vs reference variance (=1 in standardized units)
    var_on = np.clip(np.cumsum(z * z) / n - (np.cumsum(z) / n) ** 2, EPS, None)
    feats["ftest_stat"] = np.log(var_on)
    with np.errstate(all="ignore"):
        feats["ftest_pval"] = 2.0 * stats.f.sf(np.maximum(var_on, 1 / var_on),
                                                np.maximum(n - 1, 1), ref["n"] - 1)
    # Levene-style: robust dispersion (spread of |z - ref median|)
    amed = abs(np.median(ref["z"]))
    a = np.abs(z - amed)
    mu_a = np.cumsum(a) / n
    va_a = np.clip(np.cumsum(a * a) / n - mu_a ** 2, EPS, None)
    ar = np.abs(ref["z"] - amed)
    feats["levene_t"] = (mu_a - ar.mean()) / np.sqrt(va_a / n + ar.var() / ref["n"] + EPS)
    # KS two-sample: biggest gap between the online and reference CDFs
    qgrid = np.linspace(0.02, 0.98, 49)
    ghat = np.cumsum((u[:, None] <= qgrid[None, :]).astype(float), axis=0) / n[:, None]
    ks = np.max(np.abs(ghat - qgrid[None, :]), axis=1)
    n_eff = n * ref["n"] / (n + ref["n"])
    feats["ks_stat"] = ks
    feats["ks_pval"] = np.exp(-2.0 * (np.sqrt(n_eff) * ks) ** 2)
    # Mann-Whitney: location, via mean rank of online within reference
    mw_z = (np.cumsum(u - 0.5) / n) * np.sqrt(12.0 * n_eff)
    feats["mannwhitney_z"] = mw_z
    feats["mannwhitney_pval"] = 2.0 * stats.norm.sf(np.abs(mw_z))
    return feats


# =========================================================================
# FEATURE BLOCK 3 — frequency domain (trailing-window FFT vs the reference)
# =========================================================================
def spectral_features(online, ref):
    z = (np.asarray(online, np.float64) - ref["mean"]) / ref["std"]
    L = len(z)

    def describe(win):
        P = np.abs(np.fft.rfft(win, axis=1)) ** 2
        P[:, 0] = 0.0
        p = P / (P.sum(axis=1, keepdims=True) + EPS)
        freqs = np.fft.rfftfreq(win.shape[1])
        plogp = np.zeros_like(p); m = p > 0; plogp[m] = p[m] * np.log(p[m])
        ent = -plogp.sum(axis=1) / np.log(max(P.shape[1], 2))
        centroid = (p * freqs[None, :]).sum(axis=1)
        dom = freqs[np.argmax(P, axis=1)]
        low = p[:, freqs <= 0.25].sum(axis=1)
        return np.column_stack([ent, centroid, dom, low])

    Wr = min(SPEC_W, len(ref["z"]))
    if Wr >= 8:
        rd = describe(np.lib.stride_tricks.sliding_window_view(ref["z"], Wr))
        rmu, rsd = rd.mean(axis=0), rd.std(axis=0) + EPS
    else:
        rmu, rsd = np.zeros(4), np.ones(4)
    D = np.zeros((L, 4))
    if L >= SPEC_W:
        D[SPEC_W - 1:] = describe(np.lib.stride_tricks.sliding_window_view(z, SPEC_W))
    for t in range(15, min(SPEC_W - 1, L)):
        D[t] = describe(z[: t + 1][None, :])[0]
    Dz = (D - rmu[None, :]) / rsd[None, :]
    return {"spectral_entropy": Dz[:, 0], "spectral_centroid": Dz[:, 1],
            "spectral_domfreq": Dz[:, 2], "spectral_lowband": Dz[:, 3]}


# =========================================================================
# FEATURE BLOCK 4 — calibrated change-point battery (the strong block)
# For ~20 key increments, divide by the series' own null wobble (bartlett_lrv on
# the reference) and track calibrated running scores: expanding mean-z, GLR scan
# (biggest shift at any unknown point) + its running peak, two trailing-window
# z-scores, and a threshold-crossing fraction. A pooled GLR shares one change-
# point across all increments and reads off each one's post-change segment.
# =========================================================================
def fit_ar5(z):
    """AR(5) coefficients [b1..b5, intercept] fit on the reference (least squares)."""
    n = len(z)
    if n <= 25:
        return np.zeros(6)
    X = np.column_stack([z[4 - i:n - 1 - i] for i in range(5)] + [np.ones(n - 5)])
    beta, *_ = np.linalg.lstsq(X, z[5:], rcond=None)
    return beta


def online_channels(x, ref, seed5):
    """Causal channels for the calibrated battery (diff/lag/AR seeded from the
    reference tail so the first online points are causal)."""
    x = np.asarray(x, np.float64); L = len(x)
    z = (x - ref["mean"]) / ref["std"]
    s = np.asarray(seed5, np.float64)
    if len(s) < 5:
        s = np.concatenate([np.zeros(5 - len(s)), s])
    ze = np.concatenate([s[-5:], z])
    lag1, lag2 = ze[4:4 + L], ze[3:3 + L]
    lagmat = np.column_stack([ze[4 - i:4 - i + L] for i in range(5)] + [np.ones(L)])
    return {"z": z, "az": np.abs(z), "u": mid_rank(ref["sorted"], x),
            "dz": z - lag1, "lag1": lag1, "lag2": lag2,
            "e5": z - lagmat @ ref["ar5"], "L": L}


def increments(ch, ref):
    """Centered increment sequences (E_ref[.] ~= 0) for the calibrated families."""
    z, az, u, dz, lag1, lag2, e5 = (ch["z"], ch["az"], ch["u"], ch["dz"],
                                    ch["lag1"], ch["lag2"], ch["e5"])
    inc = {
        "mean": z, "var": z * z - 1.0, "absdev": az - ref["e_az"],
        "rank": u - 0.5, "rank_disp": np.abs(u - 0.5) - 0.25,
        "tail": ((u < 0.05) | (u > 0.95)).astype(float) - 0.10,
        "autocorr1": z * lag1 - ref["ac1"], "autocorr2": z * lag2 - ref["ac2"],
        "sign_runs": (np.sign(z) == np.sign(lag1)).astype(float) - ref["runs_p"],
        "roughness": dz * dz / ref["dz_msq"] - 1.0,
        "absdz": np.abs(dz) / ref["dz_mabs"] - 1.0,
        "jump": (np.abs(dz) > ref["dz_q95"]).astype(float) - ref["jump_p"],
        "ring0": (az <= 0.1).astype(float) - ref["ring0_p"],
        "ring12": ((az >= 1) & (az <= 2)).astype(float) - ref["ring12_p"],
        "ar5_var": e5 * e5 / ref["ar5_msq"] - 1.0,
    }
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        inc[f"cdf{q:g}"] = (u <= q).astype(float) - q
    return inc


def calibrated_features(online, ref):
    ch = online_channels(online, ref, ref["z_tail"])
    inc = increments(ch, ref)
    L = ch["L"]
    n = np.arange(1, L + 1, dtype=np.float64); sqn = np.sqrt(n); idx = np.arange(L)
    isa = np.concatenate(([0.0], 1.0 / np.sqrt(np.arange(1, L + 1))))
    age = np.arange(1, L + 1)[:, None] - np.arange(0, L + 1)[None, :]
    Wt = np.where(age >= 3, isa[np.clip(age, 1, None)], 0.0)
    l50, l200 = np.maximum(idx + 1 - 50, 0), np.maximum(idx + 1 - 200, 0)

    feats = {}; S_all = {}
    for name, a in inc.items():
        S = np.concatenate(([0.0], np.cumsum(a / np.sqrt(ref["lrv"][name])))); S1 = S[1:]
        S_all[name] = S
        scan = (np.abs(S1[:, None] - S[None, :]) * Wt).max(axis=1)
        feats[f"cal_{name}_exp"] = S1 / sqn
        feats[f"cal_{name}_scan"] = scan
        feats[f"cal_{name}_scanpeak"] = np.maximum.accumulate(scan)
        feats[f"cal_{name}_roll50"] = (S1 - S[l50]) / np.sqrt(np.minimum(n, 50))
        feats[f"cal_{name}_roll200"] = (S1 - S[l200]) / np.sqrt(np.minimum(n, 200))
        feats[f"cal_{name}_frac"] = np.cumsum((scan > 2.5).astype(float)) / n
    # pooled shared change-point across all increments
    G = np.zeros((L, L + 1))
    for S in S_all.values():
        D = (S[1:, None] - S[None, :]) * Wt; G += D * D
    khat = G.argmax(axis=1); rows = np.arange(L); ia = isa[np.clip(rows + 1 - khat, 1, None)]
    for name, S in S_all.items():
        feats[f"cal_seg_{name}"] = (S[rows + 1] - S[khat]) * ia
    nf = float(len(S_all)); seglen = (rows + 1 - khat).astype(float)
    feats["cal_glr_strength"] = (G.max(axis=1) - nf) / np.sqrt(2 * nf)
    feats["cal_glr_peak"] = np.maximum.accumulate(feats["cal_glr_strength"])
    feats["cal_glr_seg_len"] = np.log1p(seglen)
    feats["cal_glr_seg_frac"] = seglen / (rows + 1)
    return feats


# =========================================================================
# FEATURE BLOCK 5 — context (fixed facts about the series' history)
# =========================================================================
def context_features(online, ref):
    R = ref["ref_stats"]
    vals = {"ref_log_n": np.log(ref["n"]), "ref_kurt": R.loc["kurt", "z"],
            "ref_skew": R.loc["skew", "z"], "ref_autocorr": R.loc["autocorr", "z"],
            "ref_std_raw": ref["std"]}
    return {k: np.full(len(online), v) for k, v in vals.items()}


# =========================================================================
# glue: all features for one series
# =========================================================================
def series_features(reference, online):
    ref = prep_reference(reference)
    f = descriptive_features(online, ref)
    f.update(hypothesis_test_features(online, ref))
    f.update(spectral_features(online, ref))
    f.update(calibrated_features(online, ref))
    f.update(context_features(online, ref))
    return f


# =========================================================================
# driver — assemble the bank over all series
# =========================================================================
def default_step_grid():
    fib = [0, 1, 2, 4, 7, 12, 20, 33, 54, 88, 143, 232, 376, 609, 986]
    lin = [9, 29, 49, 74, 99, 149, 199, 274, 349, 449, 549, 699, 849, 998]
    return np.array(sorted(set(fib + lin)), dtype=np.int64)


def _iter_series(X, ids_subset):
    ids_all = X.index.get_level_values("id").to_numpy()
    times_all = X.index.get_level_values("time").to_numpy()
    vals = X["value"].to_numpy(np.float64)
    per = X["period"].to_numpy()
    uids, starts = np.unique(ids_all, return_index=True)
    bounds = np.append(starts, len(ids_all))
    keep = set(ids_subset) if ids_subset is not None else None
    for k, sid in enumerate(uids):
        if keep is not None and sid not in keep:
            continue
        s, e = bounds[k], bounds[k + 1]
        onl = per[s:e] == 2
        x = vals[s:e][onl]
        if len(x) == 0:
            continue
        yield sid, vals[s:e][~onl], x, times_all[s:e][onl]


def build_bank(X, ids_subset=None, out_path=None, full_resolution=False,
               step_grid=None, block_size=500, include=None, verbose=False):
    """Wide causal feature bank. Returns a DataFrame unless out_path is given."""
    import time
    if step_grid is None:
        step_grid = default_step_grid()
    schema, writer, frames = None, None, []
    bmat, bids, btimes = [], [], []
    t0 = time.time(); count = 0

    def keep_mask(L):
        if full_resolution:
            return np.ones(L, bool)
        m = np.zeros(L, bool); m[step_grid[step_grid < L]] = True; return m

    def flush():
        nonlocal writer, schema
        if not bmat:
            return
        idx = pd.MultiIndex.from_arrays([np.concatenate(bids), np.concatenate(btimes)],
                                        names=["id", "time"])
        df = pd.DataFrame(np.vstack(bmat), index=idx, columns=schema)
        if out_path:
            import pyarrow as pa, pyarrow.parquet as pq
            tbl = pa.Table.from_pandas(df)
            if writer is None:
                writer = pq.ParquetWriter(out_path, tbl.schema)
            writer.write_table(tbl)
        else:
            frames.append(df)
        bmat.clear(); bids.clear(); btimes.clear()

    for sid, ref, x, times in _iter_series(X, ids_subset):
        cols = series_features(ref, x)
        if include is not None:
            cols = {k: v for k, v in cols.items() if k in include}
        names = list(cols.keys())
        if schema is None:
            schema = names
        elif names != schema:
            raise RuntimeError(f"column mismatch on id {sid}")
        M = np.column_stack([cols[k] for k in names])
        M = np.nan_to_num(M, nan=0.0, posinf=CLIP, neginf=-CLIP)
        np.clip(M, -CLIP, CLIP, out=M)
        mask = keep_mask(len(x))
        bmat.append(M[mask].astype(np.float32))
        bids.append(np.full(int(mask.sum()), sid)); btimes.append(times[mask])
        count += 1
        if count % block_size == 0:
            flush()
            if verbose:
                print("  %d series | %.0fs" % (count, time.time() - t0), flush=True)
    flush()
    if out_path:
        if writer is not None:
            writer.close()
        return None
    return pd.concat(frames) if frames else pd.DataFrame()


def test_causality(X, n_series=3, verbose=True):
    """A feature at step t must be identical whether or not the series is
    truncated after t (no peeking at the future)."""
    ids = X.index.get_level_values("id").unique()[:n_series]
    for sid in ids:
        Xs = X.loc[[sid]]
        full = build_bank(Xs, full_resolution=True)
        onl = Xs.loc[sid].query("period == 2").index
        cut = onl[len(onl) // 2]
        pref = build_bank(Xs[Xs.index.get_level_values("time") <= cut], full_resolution=True)
        pd.testing.assert_frame_equal(full.loc[pref.index], pref)
    if verbose:
        n_feat = build_bank(X.loc[[ids[0]]], full_resolution=True).shape[1]
        print("causality OK (%d series, %d features)" % (len(ids), n_feat), flush=True)
    return True
