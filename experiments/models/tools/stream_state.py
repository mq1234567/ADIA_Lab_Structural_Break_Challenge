"""Causal streaming feature extractor — single source of truth.

This is the same StreamState the crunch submission runs
(`submissions/submit#3-catboost_varcusum`), lifted out of the notebooks so the
cache builder, the experiments and the submission cannot drift apart.

Every feature is CAUSAL: state is updated one point at a time and never looks
ahead, so the vector emitted at online step t depends only on the reference
period plus online[:t]. That property is what makes k-fold CV honest, and it is
verified two ways: truncating the stream leaves earlier steps bit-identical, and
`reset()` fully clears state so features are independent of the order series are
processed in.

Feature groups (50 total):
  base (22)   level/scale/shape vs the reference, CUSUM, KS, rank, AR residual
  ewma (4)    EWMA level + log-variance at two decay rates
  roll (4)    trailing-window mean shift + log-variance at two windows
  spec (3)    spectral entropy / centroid vs the reference spectrum
  dep  (10)   higher-lag autocorrelation + extra AR-order residual ratios
  extra (5)   var_cusum_max, kuiper, cvm, skew_shift, kurt_shift
  glr  (2)    var_glr, var_glr_peak  (variance changepoint, best split)

NOTE: `update()` returns an internal buffer that is overwritten on the next call.
Callers that keep the row must copy it (`out.copy()`).
"""
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import deque

import numpy as np


# ---------------------------------------------------------------------------
# reference-period helpers (computed once per series in reset())
# ---------------------------------------------------------------------------
def _autocorr(x, lag):
    if len(x) <= lag:
        return 0.0
    a, b = x[:-lag], x[lag:]
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def _fit_ar(x, p=5):
    """Least-squares AR(p) with intercept. Returns (coefs, residual variance)."""
    if len(x) <= p + 5:
        return None, 1.0
    rows = np.lib.stride_tricks.sliding_window_view(x, p + 1)
    Y = rows[:, -1]
    Xd = np.column_stack([np.ones(len(rows)), rows[:, :-1][:, ::-1]])
    beta, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
    resid = Y - Xd @ beta
    return beta, float(resid.var()) if len(resid) else 1.0


def _norm_psd(seg, win):
    """Hann-windowed one-sided PSD, normalised to sum 1. None if degenerate."""
    x = (seg - seg.mean()) * win
    sp = np.fft.rfft(x)
    p = (sp.real ** 2 + sp.imag ** 2)[1:]        # drop DC
    s = p.sum()
    if s <= 0 or not np.isfinite(s):
        return None
    return p / s


def _spec_ec(p, freqs, logk):
    return -float((p * np.log(p + 1e-12)).sum()) * logk, float((freqs * p).sum())


def log_steps(L, m=40):
    """The m log-spaced online steps the deployed trainer samples from a length-L stream.

    Log spacing because early steps carry most of the TS-AUC weight (every series is
    still alive there) while late steps are few and highly correlated.
    """
    if L <= m:
        return np.arange(L)
    return np.unique(np.round(np.expm1(np.linspace(0, np.log1p(L - 1), m))).astype(int))


# ---------------------------------------------------------------------------
class StreamState:
    """Incremental O(1)/O(window) feature extractor. `reset(reference)` then `update(x)`."""

    VAR_MIN = 20          # below this many online points, variance-based features stay 0
    SPEC_W = 128          # spectral window
    AC_LAGS_NEW = (2, 3, 5, 10)
    AR_ORDERS_NEW = (2, 10)
    MAXLAG = 10
    GLR_MIN = 30          # variance-GLR needs enough points for a meaningful split

    _BASE = (
        "n_online", "mean_shift_z", "last_z", "online_logvar", "online_std",
        "online_skew", "online_kurt", "cumsum_range", "cusum_max",
        "tail2_frac", "tail3_frac", "jump_freq", "signrun_freq",
        "ks_stat", "rank_dev", "ac1_online", "ac1_shift", "ar_resid_logratio",
        "ref_log_n", "ref_skew", "ref_kurt", "ref_ac1",
    )
    _SPEC = ("spec_entropy", "spec_entropy_shift", "spec_centroid_shift")
    _EXTRA = ("var_cusum_max", "kuiper", "cvm", "skew_shift", "kurt_shift")
    _GLR = ("var_glr", "var_glr_peak")

    def __init__(self, bins=16, roll_windows=(50, 200), ewma_alphas=(0.05, 0.2), ar_p=5):
        self.bins = bins; self.rw = tuple(roll_windows)
        self.alphas = tuple(ewma_alphas); self.p = ar_p
        fn = list(self._BASE)
        for a in self.alphas:
            fn += ["ewma%.2f_z" % a, "ewma%.2f_logvar" % a]
        for w in self.rw:
            fn += ["roll%d_meanshift" % w, "roll%d_logvar" % w]
        self._ew_off = len(self._BASE)
        self._roll_off = self._ew_off + 2 * len(self.alphas)
        self._spec_off = len(fn); fn += list(self._SPEC)
        self._dep_off = len(fn)
        self._dep_names = []
        for L in self.AC_LAGS_NEW:
            self._dep_names += ["ac%d_online" % L, "ac%d_shift" % L]
        for pp in self.AR_ORDERS_NEW:
            self._dep_names += ["ar%d_resid_logratio" % pp]
        fn += self._dep_names
        self._extra_off = len(fn); fn += list(self._EXTRA)
        self._glr_off = len(fn); fn += list(self._GLR)
        self.feature_names = fn; self.ncol = len(fn)
        self._out = np.zeros(self.ncol, dtype=np.float64)
        self._bin_targets = np.linspace(1.0 / self.bins, 1.0, self.bins)
        W = self.SPEC_W
        self._spec_win = np.hanning(W)
        self._spec_freqs = np.arange(1, W // 2 + 1, dtype=np.float64) / (W // 2)
        self._spec_logk = 1.0 / math.log(len(self._spec_freqs))
        self._glr_fracs = np.linspace(0.1, 0.9, 12)

    # -- per-series setup ---------------------------------------------------
    def reset(self, reference):
        """Clear ALL state and precompute reference statistics.

        Must clear everything `update()` touches: leftover state would make features
        depend on which series was processed before, which under a k-fold split is
        leakage. (Verified: features are identical run-alone vs run-after-another.)
        """
        r = np.asarray(reference, dtype=np.float64)
        self.mu_h = float(r.mean()) if len(r) else 0.0
        self.sd_h = max(float(r.std(ddof=1)) if len(r) > 1 else 1.0, 1e-8)
        self.var_h = self.sd_h ** 2; self.ref_n = len(r)
        self._inv_refn = 1.0 / max(self.ref_n, 1)
        z = (r - self.mu_h) / self.sd_h if len(r) else r
        self.ref_skew = float((z ** 3).mean()) if len(r) else 0.0
        self.ref_kurt = float((z ** 4).mean() - 3.0) if len(r) else 0.0
        self.ref_ac1 = _autocorr(r, 1)
        self.ref_log_n = math.log(self.ref_n + 1.0)
        self.ref_ac = {L: _autocorr(r, L) for L in self.AC_LAGS_NEW}
        self.sorted_ref = np.sort(r).tolist() if len(r) else []
        edges = (np.quantile(r, np.linspace(0, 1, self.bins + 1)[1:-1])
                 if len(r) else np.zeros(self.bins - 1))
        self.edges = edges.tolist()
        self.ar, self.ar_var_h = _fit_ar(r, self.p)
        self.arN = {pp: _fit_ar(r, pp) for pp in self.AR_ORDERS_NEW}
        self.ref_spec_ent = 0.0; self.ref_spec_cen = 0.0
        if len(r) >= self.SPEC_W:
            W = self.SPEC_W; acc = np.zeros(W // 2); cnt = 0
            for s in range(0, len(r) - W + 1, W // 2):
                p = _norm_psd(r[s:s + W], self._spec_win)
                if p is not None:
                    acc += p; cnt += 1
            if cnt:
                pr = acc / acc.sum()
                self.ref_spec_ent, self.ref_spec_cen = _spec_ec(
                    pr, self._spec_freqs, self._spec_logk)
        self.n = 0
        self.mean = self.M2 = self.M3 = self.M4 = 0.0
        self.sx = self.sxx = self.sxy = 0.0
        self.cmax = self.cmin = self.cumsum = 0.0
        self.cusum_p = self.cusum_n = self.cusum_max = 0.0
        self.cusum_var_p = self.cusum_var_n = self.cusum_var_max = 0.0
        self.pref_s1 = [0.0]; self.pref_s2 = [0.0]; self.var_glr_peak_v = 0.0
        self.tail2 = self.tail3 = 0
        self.binc = np.zeros(self.bins); self.rank_sum = 0.0
        self.jumps = 0; self.signruns = 0; self.prev = None
        self.ar_resid_ss = 0.0; self.ar_n = 0; self.arbuf = deque(maxlen=self.p)
        self.roll = {w: (deque(maxlen=w), [0.0, 0.0]) for w in self.rw}
        self.ew = {a: self.mu_h for a in self.alphas}
        self.ewv = {a: 0.0 for a in self.alphas}
        self.specbuf = deque(maxlen=self.SPEC_W)
        self._last_spec_ent = 0.0; self._last_spec_cen = 0.0
        self.lagbuf = deque(maxlen=self.MAXLAG)
        self.acP = {L: 0.0 for L in self.AC_LAGS_NEW}
        self.acC = {L: 0 for L in self.AC_LAGS_NEW}
        self.arNbuf = {pp: deque(maxlen=pp) for pp in self.AR_ORDERS_NEW}
        self.arNss = {pp: 0.0 for pp in self.AR_ORDERS_NEW}
        self.arNn = {pp: 0 for pp in self.AR_ORDERS_NEW}
        return self

    # -- variance changepoint ----------------------------------------------
    def _var_glr(self, L):
        """Max Gaussian variance-change likelihood ratio over candidate split points.

        For a split k: compare 'two segments with different variance' against 'one
        variance throughout'. Prefix sums make each candidate O(1), so scanning a
        12-point grid stays cheap enough to stream.
        """
        s1, s2 = self.pref_s1, self.pref_s2
        tot1 = s1[L]; tot2 = s2[L]
        v0 = tot2 / L - (tot1 / L) ** 2
        if v0 <= 1e-9:
            return 0.0
        best = 0.0; lv0 = math.log(v0 + 1e-8)
        for frac in self._glr_fracs:
            k = int(frac * L)
            if k < 8 or k > L - 8:
                continue
            m1 = s1[k] / k; v1 = s2[k] / k - m1 * m1
            nk = L - k; m2 = (tot1 - s1[k]) / nk; v2 = (tot2 - s2[k]) / nk - m2 * m2
            if v1 <= 1e-9 or v2 <= 1e-9:
                continue
            g = L * lv0 - k * math.log(v1 + 1e-8) - nk * math.log(v2 + 1e-8)
            if g > best:
                best = g
        return best

    # -- one online point ---------------------------------------------------
    def update(self, x):
        """Absorb one online point and return the feature row.

        WARNING: returns an internal buffer reused across calls — copy it to keep it.
        """
        x = float(x); self.n += 1; n = self.n
        sd_h = self.sd_h; mu_h = self.mu_h; var_h = self.var_h
        d = x - self.mean; dn = d / n; dn2 = dn * dn; term = d * dn * (n - 1)
        self.mean += dn
        self.M4 += term * dn2 * (n * n - 3 * n + 3) + 6 * dn2 * self.M2 - 4 * dn * self.M3
        self.M3 += term * dn * (n - 2) - 3 * dn * self.M2; self.M2 += term
        var = self.M2 / (n - 1) if n > 1 else 0.0
        std = math.sqrt(var) if var > 0 else 0.0
        zx = (x - mu_h) / sd_h
        self.pref_s1.append(self.pref_s1[-1] + zx)
        self.pref_s2.append(self.pref_s2[-1] + zx * zx)
        self.cumsum += zx
        if self.cumsum > self.cmax: self.cmax = self.cumsum
        if self.cumsum < self.cmin: self.cmin = self.cumsum
        cp = self.cusum_p + zx - 0.5; cn = self.cusum_n - zx - 0.5
        self.cusum_p = cp if cp > 0.0 else 0.0
        self.cusum_n = cn if cn > 0.0 else 0.0
        if self.cusum_p > self.cusum_max: self.cusum_max = self.cusum_p
        if self.cusum_n > self.cusum_max: self.cusum_max = self.cusum_n
        vinc = zx * zx - 1.0
        vp = self.cusum_var_p + vinc - 0.5; vn = self.cusum_var_n - vinc - 0.5
        self.cusum_var_p = vp if vp > 0.0 else 0.0
        self.cusum_var_n = vn if vn > 0.0 else 0.0
        if self.cusum_var_p > self.cusum_var_max: self.cusum_var_max = self.cusum_var_p
        if self.cusum_var_n > self.cusum_var_max: self.cusum_var_max = self.cusum_var_n
        azx = zx if zx >= 0 else -zx
        if azx > 2.0: self.tail2 += 1
        if azx > 3.0: self.tail3 += 1
        if self.prev is not None:
            if abs(x - self.prev) > 2.0 * sd_h: self.jumps += 1
            if (x - mu_h) * (self.prev - mu_h) < 0: self.signruns += 1
            self.sxy += x * self.prev
        self.sx += x; self.sxx += x * x
        lo = bisect_left(self.sorted_ref, x); hi = bisect_right(self.sorted_ref, x)
        self.rank_sum += 0.5 * (lo + hi) * self._inv_refn
        self.binc[bisect_right(self.edges, x)] += 1
        if self.ar is not None and len(self.arbuf) == self.p:
            lag = np.array(self.arbuf, dtype=np.float64)[::-1]
            pred = self.ar[0] + self.ar[1:] @ lag
            r_ = x - pred; self.ar_resid_ss += r_ * r_; self.ar_n += 1
        self.arbuf.append(x)
        lb = self.lagbuf
        for L in self.AC_LAGS_NEW:
            if len(lb) >= L:
                self.acP[L] += x * lb[-L]; self.acC[L] += 1
        lb.append(x)
        for pp in self.AR_ORDERS_NEW:
            beta, _vh = self.arN[pp]; buf = self.arNbuf[pp]
            if beta is not None and len(buf) == pp:
                lag = np.array(buf, dtype=np.float64)[::-1]
                pred = beta[0] + beta[1:] @ lag
                r_ = x - pred; self.arNss[pp] += r_ * r_; self.arNn[pp] += 1
            buf.append(x)
        for a in self.alphas:
            m0 = self.ew[a]
            self.ew[a] = (1 - a) * m0 + a * x
            self.ewv[a] = (1 - a) * (self.ewv[a] + a * (x - m0) ** 2)
        for w, (buf, acc) in self.roll.items():
            if len(buf) == buf.maxlen:
                old = buf[0]; acc[0] -= old; acc[1] -= old * old
            buf.append(x); acc[0] += x; acc[1] += x * x
        self.specbuf.append(x)
        if len(self.specbuf) == self.SPEC_W:
            p = _norm_psd(np.array(self.specbuf, dtype=np.float64), self._spec_win)
            if p is not None:
                self._last_spec_ent, self._last_spec_cen = _spec_ec(
                    p, self._spec_freqs, self._spec_logk)
        self.prev = x

        out = self._out; se = sd_h / math.sqrt(n)
        emp = np.cumsum(self.binc) / n
        ks = float(np.max(np.abs(emp - self._bin_targets)))
        if n > 2 and var > 0:
            mx = self.sx / n
            cov = self.sxy / (n - 1) - mx * mx * n / (n - 1); ac1 = cov / var
        else:
            ac1 = 0.0
        vg = n >= self.VAR_MIN; sqrt_n = math.sqrt(n)
        out[0] = math.log1p(n); out[1] = (self.mean - mu_h) / max(se, 1e-8); out[2] = zx
        out[3] = math.log((var + 1e-8) / var_h) if vg else 0.0; out[4] = std / sd_h
        out[5] = (self.M3 / n) / (var ** 1.5 + 1e-8) if vg else 0.0
        out[6] = ((self.M4 / n) / (var ** 2 + 1e-8) - 3.0) if vg else 0.0
        out[7] = (self.cmax - self.cmin) / sqrt_n; out[8] = self.cusum_max / sqrt_n
        out[9] = self.tail2 / n; out[10] = self.tail3 / n
        out[11] = self.jumps / n; out[12] = self.signruns / n
        out[13] = ks; out[14] = self.rank_sum / n - 0.5; out[15] = ac1
        out[16] = ac1 - self.ref_ac1
        out[17] = (math.log((self.ar_resid_ss / self.ar_n + 1e-8)
                            / (self.ar_var_h + 1e-8)) if self.ar_n >= self.VAR_MIN else 0.0)
        out[18] = self.ref_log_n; out[19] = self.ref_skew
        out[20] = self.ref_kurt; out[21] = self.ref_ac1
        off = self._ew_off
        for ai, a in enumerate(self.alphas):
            ew_se = sd_h * math.sqrt(a / (2 - a))
            out[off + 2 * ai] = (self.ew[a] - mu_h) / max(ew_se, 1e-8)
            out[off + 2 * ai + 1] = math.log((self.ewv[a] + 1e-8) / var_h) if vg else 0.0
        off = self._roll_off
        for wi, w in enumerate(self.rw):
            buf, acc = self.roll[w]; L = len(buf)
            m = acc[0] / L; v = acc[1] / L - m * m
            if v < 0.0: v = 0.0
            out[off + 2 * wi] = (m - mu_h) / sd_h
            out[off + 2 * wi + 1] = math.log((v + 1e-8) / var_h)
        off = self._spec_off
        out[off] = self._last_spec_ent
        out[off + 1] = self._last_spec_ent - self.ref_spec_ent
        out[off + 2] = self._last_spec_cen - self.ref_spec_cen
        off = self._dep_off; j = 0; mean_all = self.mean
        for L in self.AC_LAGS_NEW:
            if self.acC[L] > 0 and var > 0 and vg:
                acL = (self.acP[L] / self.acC[L] - mean_all * mean_all) / var
            else:
                acL = 0.0
            out[off + j] = acL; out[off + j + 1] = acL - self.ref_ac[L]; j += 2
        for pp in self.AR_ORDERS_NEW:
            out[off + j] = (math.log((self.arNss[pp] / self.arNn[pp] + 1e-8)
                                     / (self.arN[pp][1] + 1e-8))
                            if self.arNn[pp] >= self.VAR_MIN else 0.0); j += 1
        eo = self._extra_off
        out[eo] = self.cusum_var_max / sqrt_n if vg else 0.0
        out[eo + 1] = float(np.max(emp - self._bin_targets) + np.max(self._bin_targets - emp))
        out[eo + 2] = float(np.mean((emp - self._bin_targets) ** 2))
        out[eo + 3] = out[5] - self.ref_skew if vg else 0.0
        out[eo + 4] = out[6] - self.ref_kurt if vg else 0.0
        go = self._glr_off
        glr = self._var_glr(n) if (vg and n >= self.GLR_MIN) else 0.0
        if glr > self.var_glr_peak_v:
            self.var_glr_peak_v = glr
        out[go] = glr / n; out[go + 1] = self.var_glr_peak_v / n
        return out


# back-compat alias: the notebooks/submission call it StreamStateV3
StreamStateV3 = StreamState
