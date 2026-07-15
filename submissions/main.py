#%pip install crunch-cli --upgrade --quiet --progress-bar off
#!crunch setup-notebook structural-break-real-time bopyO0zQvKLwr0i2HmfjqV06


import math, os
from typing import Iterable, List, Optional, Tuple
import joblib, numpy as np, pandas as pd
import lightgbm as lgb
import crunch
#crunch_tools = crunch.load_notebook()


#"""Incremental O(1)/O(window) streaming features for the Real-Time submission.
#
#Here every feature updates in constant time as each online point arrives. The SAME StreamState
#is used in train() (to generate training rows) and infer() (to stream), so the
#feature vectors match exactly. Pure numpy/python, no scipy.
#"""
import numpy as np
from bisect import bisect_left, bisect_right
from collections import deque


# ---- reference (static, computed once per series) ------------------------
def _autocorr(x, lag):
    if len(x) <= lag:
        return 0.0
    a, b = x[:-lag], x[lag:]
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def _fit_ar(x, p=5):
    """Least-squares AR(p) with intercept on the reference. Returns (coefs, resid_var)."""
    if len(x) <= p + 5:
        return None, 1.0
    rows = np.lib.stride_tricks.sliding_window_view(x, p + 1)
    Y = rows[:, -1]
    Xd = np.column_stack([np.ones(len(rows)), rows[:, :-1][:, ::-1]])  # [1, x_{t-1..t-p}]
    beta, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
    resid = Y - Xd @ beta
    return beta, float(resid.var()) if len(resid) else 1.0


class StreamState:
    """Per-series streaming feature extractor. reset() once, update(x) per point."""

    VAR_MIN = 20  # guard: log-var features are unstable for tiny n (see memory)

    def __init__(self, bins=16, roll_windows=(50, 200), ewma_alphas=(0.05, 0.2), ar_p=5):
        self.bins = bins; self.rw = roll_windows; self.alphas = ewma_alphas; self.p = ar_p

    def reset(self, reference):
        r = np.asarray(reference, dtype=np.float64)
        self.mu_h = float(r.mean()) if len(r) else 0.0
        self.sd_h = max(float(r.std(ddof=1)) if len(r) > 1 else 1.0, 1e-8)
        self.var_h = self.sd_h ** 2
        self.ref_n = len(r)
        z = (r - self.mu_h) / self.sd_h
        self.ref_skew = float((z ** 3).mean())
        self.ref_kurt = float((z ** 4).mean() - 3.0)
        self.ref_ac1 = _autocorr(r, 1); self.ref_ac2 = _autocorr(r, 2)
        self.sorted_ref = np.sort(r)
        self.edges = np.quantile(r, np.linspace(0, 1, self.bins + 1)[1:-1]) if len(r) else np.zeros(self.bins - 1)
        self.ar, self.ar_var_h = _fit_ar(r, self.p)
        # streaming accumulators
        self.n = 0
        self.mean = self.M2 = self.M3 = self.M4 = 0.0
        self.sx = self.sxx = self.sxy = 0.0          # for online autocorr(1)
        self.sxy2 = 0.0; self.hist = deque(maxlen=3)  # last points for ac2 / AR
        self.cumsum = self.cmax = self.cmin = 0.0
        self.cusum_p = self.cusum_n = self.cusum_max = 0.0
        self.tail2 = self.tail3 = 0
        self.binc = np.zeros(self.bins)
        self.rank_sum = 0.0
        self.jumps = 0; self.signruns = 0; self.prev = None
        self.ar_resid_ss = 0.0; self.ar_n = 0
        self.arbuf = deque(maxlen=self.p)
        self.roll = {w: (deque(maxlen=w), [0.0, 0.0]) for w in self.rw}  # buf, [sum, sumsq]
        self.ew = {a: self.mu_h for a in self.alphas}
        self.ewv = {a: 0.0 for a in self.alphas}
        return self

    def update(self, x):
        x = float(x); self.n += 1; n = self.n
        sd_h, mu_h = self.sd_h, self.mu_h
        # --- running moments (Welford up to 4th) ---
        d = x - self.mean; dn = d / n; dn2 = dn * dn; term = d * dn * (n - 1)
        self.mean += dn
        self.M4 += term * dn2 * (n * n - 3 * n + 3) + 6 * dn2 * self.M2 - 4 * dn * self.M3
        self.M3 += term * dn * (n - 2) - 3 * dn * self.M2
        self.M2 += term
        var = self.M2 / (n - 1) if n > 1 else 0.0
        std = np.sqrt(var)
        # --- standardized-by-reference point ---
        zx = (x - mu_h) / sd_h
        # --- cumsum of centered residuals + range ---
        self.cumsum += zx
        self.cmax = max(self.cmax, self.cumsum); self.cmin = min(self.cmin, self.cumsum)
        # --- CUSUM (two-sided, slack 0.5) ---
        self.cusum_p = max(0.0, self.cusum_p + zx - 0.5)
        self.cusum_n = max(0.0, self.cusum_n - zx - 0.5)
        self.cusum_max = max(self.cusum_max, self.cusum_p, self.cusum_n)
        # --- tail mass ---
        if abs(zx) > 2.0: self.tail2 += 1
        if abs(zx) > 3.0: self.tail3 += 1
        # --- jumps / sign runs ---
        if self.prev is not None:
            if abs(x - self.prev) > 2.0 * sd_h: self.jumps += 1
            if (x - mu_h) * (self.prev - mu_h) < 0: self.signruns += 1
        # --- online autocorr(1) running ---
        if self.prev is not None:
            self.sxy += x * self.prev
        self.sx += x; self.sxx += x * x
        # --- rank vs reference (Mann-Whitney) ---
        lo = bisect_left(self.sorted_ref, x); hi = bisect_right(self.sorted_ref, x)
        self.rank_sum += 0.5 * (lo + hi) / max(self.ref_n, 1)
        # --- KS histogram (equal-freq ref bins) ---
        self.binc[np.searchsorted(self.edges, x)] += 1
        # --- AR residual vs reference-fit AR ---
        if self.ar is not None and len(self.arbuf) == self.p:
            lag = np.array(self.arbuf, dtype=np.float64)[::-1]  # x_{t-1..t-p}
            pred = self.ar[0] + self.ar[1:] @ lag
            r = x - pred; self.ar_resid_ss += r * r; self.ar_n += 1
        self.arbuf.append(x)
        # --- EWMA mean/var ---
        for a in self.alphas:
            m0 = self.ew[a]; self.ew[a] = (1 - a) * m0 + a * x
            self.ewv[a] = (1 - a) * (self.ewv[a] + a * (x - m0) ** 2)
        # --- rolling windows via ring buffer ---
        rollf = {}
        for w, (buf, acc) in self.roll.items():
            if len(buf) == buf.maxlen:
                old = buf[0]; acc[0] -= old; acc[1] -= old * old
            buf.append(x); acc[0] += x; acc[1] += x * x
            m = acc[0] / len(buf); v = max(acc[1] / len(buf) - m * m, 0.0)
            rollf["roll%d_meanshift" % w] = (m - mu_h) / sd_h
            rollf["roll%d_logvar" % w] = np.log((v + 1e-8) / self.var_h)
        self.prev = x
        return self._emit(var, std, zx, rollf)

    def _emit(self, var, std, zx, rollf):
        n = self.n; sd_h = self.sd_h
        se = sd_h / np.sqrt(max(n, 1))
        emp = np.cumsum(self.binc) / n
        ks = float(np.max(np.abs(emp - np.linspace(1.0 / self.bins, 1.0, self.bins)))) if n else 0.0
        # online autocorr(1)
        if n > 2:
            mx = self.sx / n
            cov = self.sxy / (n - 1) - mx * mx * n / (n - 1)
            ac1 = cov / var if var > 0 else 0.0
        else:
            ac1 = 0.0
        vg = n >= self.VAR_MIN
        f = {
            "n_online": np.log1p(n),
            "mean_shift_z": (self.mean - self.mu_h) / max(se, 1e-8),
            "last_z": zx,
            "online_logvar": np.log((var + 1e-8) / self.var_h) if vg else 0.0,
            "online_std": std / sd_h,
            "online_skew": self.M3 / n / (var ** 1.5 + 1e-8) if vg else 0.0,
            "online_kurt": (self.M4 / n / (var ** 2 + 1e-8) - 3.0) if vg else 0.0,
            "cumsum_range": (self.cmax - self.cmin) / np.sqrt(max(n, 1)),
            "cusum_max": self.cusum_max / np.sqrt(max(n, 1)),
            "tail2_frac": self.tail2 / n,
            "tail3_frac": self.tail3 / n,
            "jump_freq": self.jumps / n,
            "signrun_freq": self.signruns / n,
            "ks_stat": ks,
            "rank_dev": self.rank_sum / n - 0.5,
            "ac1_online": ac1,
            "ac1_shift": ac1 - self.ref_ac1,
            "ar_resid_logratio": np.log((self.ar_resid_ss / self.ar_n + 1e-8) / (self.ar_var_h + 1e-8))
                                 if self.ar_n >= self.VAR_MIN else 0.0,
            # reference context (static, but useful splits)
            "ref_log_n": np.log(self.ref_n + 1.0),
            "ref_skew": self.ref_skew, "ref_kurt": self.ref_kurt, "ref_ac1": self.ref_ac1,
        }
        for a in self.alphas:
            ew_se = sd_h * np.sqrt(a / (2 - a))
            f["ewma%.2f_z" % a] = (self.ew[a] - self.mu_h) / max(ew_se, 1e-8)
            f["ewma%.2f_logvar" % a] = np.log((self.ewv[a] + 1e-8) / self.var_h) if vg else 0.0
        f.update(rollf)
        return f


#FEATURE_NAMES = None  # filled on first update by the caller if needed


def _log_steps(L, m):
    if L <= m: return np.arange(L)
    return np.unique(np.round(np.expm1(np.linspace(0, np.log1p(L-1), m))).astype(int))

def train(datasets: List[Tuple[int, List[float], List[float], Optional[int]]],
          model_directory_path: str):
    params = dict(objective='binary', n_estimators=500, learning_rate=0.05, num_leaves=63,
                  min_child_samples=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_lambda=1.0, n_jobs=-1, verbosity=-1)
    samples_per = 40   # log-spaced online steps sampled per series for training
    n_seeds = 3
    st = StreamState()
    rows, labels, sids, cols = [], [], [], None
    for dataset_id, x_hist, x_online, tau in datasets:
        xo = np.asarray(x_online, dtype=np.float64)
        st.reset(x_hist); L = len(xo)
        want = set(_log_steps(L, samples_per).tolist())
        for k in range(L):
            f = st.update(xo[k])
            if k not in want: continue
            if cols is None: cols = list(f.keys())
            rows.append([f[c] for c in cols])
            labels.append(1 if (tau is not None and k >= tau) else 0)
            sids.append(dataset_id)
    Xtr = np.asarray(rows, np.float32); ytr = np.asarray(labels, np.int8)
    sids = np.asarray(sids)
    cmap = dict(zip(*np.unique(sids, return_counts=True)))       # equal-series weights
    w = np.array([1.0 / cmap[s] for s in sids]); w /= w.mean()
    models = [lgb.LGBMClassifier(random_state=s, **params).fit(Xtr, ytr, sample_weight=w)
              for s in range(n_seeds)]
    boosters = [m.booster_ for m in models]                       # fast per-row predict
    joblib.dump({'boosters': boosters, 'cols': cols}, os.path.join(model_directory_path, 'model.joblib'))
    print('trained: %d rows, %d feats, %d series' % (len(Xtr), len(cols), len(cmap)))


def infer(datasets: Iterable[Tuple[List[float], Iterable[float]]],
          model_directory_path: str):
    m = joblib.load(os.path.join(model_directory_path, 'model.joblib'))
    boosters, cols = m['boosters'], m['cols']
    ncol = len(cols)
    st = StreamState()
    yield  # signal readiness
    for x_historical, x_online in datasets:
        st.reset(x_historical)
        vec = np.empty((1, ncol), np.float64)
        for point in x_online:
            f = st.update(point)
            for j, c in enumerate(cols):
                vec[0, j] = f[c]
            s = 0.0
            for b in boosters:
                s += b.predict(vec)[0]
            yield float(s / len(boosters))


# @crunch/keep:on
INFER_PARALLELISM = 4
# @crunch/keep:off


#crunch_tools.test()


from sklearn.metrics import roc_auc_score
#prediction = pd.read_parquet('prediction/prediction.parquet')
#y_test = pd.read_parquet('data/y_test.reduced.parquet')
#m = prediction.merge(y_test, how='left', left_index=True, right_index=True)
#m['t'] = m.groupby('id').cumcount()
#num = den = 0.0
#for t, g in m.groupby('t'):
#    yv = g['target'].values; sv = g['prediction'].values
#    npos = int(yv.sum()); nneg = len(yv) - npos
#    if npos == 0 or nneg == 0: continue
#    num += npos*nneg*roc_auc_score(yv, sv); den += npos*nneg
#print('Local TS-AUC: %.4f' % (num/den if den else 0.5))


#crunch_tools.submit(message='incremental streaming GBM')
