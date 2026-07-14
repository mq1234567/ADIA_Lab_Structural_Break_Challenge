"""Plain-English catalog of every feature sb_bank produces. Generated from the
code (run series_features on a synthetic series) so it never drifts. Run:
    python sb_catalog.py
writes feature_catalog.csv + feature_catalog.md next to the code."""
import os
import numpy as np, pandas as pd
import sb_bank as B

# --- descriptive block: names are {view}_{stat}[_w{W}][_vs_ref|_ratio] --------
VIEW = {
    "z": "standardized value", "abs": "magnitude |z|", "sign": "sign of z",
    "square": "squared value z^2", "logabs": "log-magnitude", "cumsum": "running sum",
    "diff": "step-to-step change", "diff2": "curvature (2nd difference)",
    "rank": "percentile rank vs history", "rank_centered": "centered percentile rank",
    "rollmean16": "local 16-point mean", "rollstd16": "local 16-point volatility",
}
STAT = {  # stat token -> (category, meaning)
    "mean": ("statistical", "average level"),
    "median": ("statistical", "robust average level"),
    "min": ("statistical", "running minimum"),
    "max": ("statistical", "running maximum"),
    "skew": ("statistical", "lopsidedness (skew)"),
    "kurt": ("statistical", "heavy-tailedness (kurtosis)"),
    "std": ("volatility", "spread / std"),
    "range": ("volatility", "max-min range"),
    "cov": ("volatility", "coefficient of variation"),
    "q25": ("distributional", "25th percentile"),
    "q75": ("distributional", "75th percentile"),
    "autocorr": ("dependence", "lag-1 autocorrelation"),
}
VARIANT = {
    "": "raw running value",
    "_vs_ref": "difference vs the reference-segment value",
    "_ratio": "ratio to the reference-segment value",
}
# --- calibrated change-point block: cal_{increment}_{reducer} -----------------
CAL_INC = {
    "mean": "mean shift", "var": "variance shift", "absdev": "robust spread shift",
    "rank": "rank/location shift", "rank_disp": "rank dispersion", "tail": "tail occupancy",
    "autocorr1": "lag-1 dependence", "autocorr2": "lag-2 dependence",
    "sign_runs": "sign persistence", "roughness": "step-change bumpiness",
    "absdz": "size of step changes", "jump": "big-jump frequency",
    "ring0": "quiet-band occupancy", "ring12": "active-band occupancy",
    "ar5_var": "AR(5) residual variance", "cdf0.1": "CDF at 10th pctile",
    "cdf0.25": "CDF at 25th pctile", "cdf0.5": "CDF at median",
    "cdf0.75": "CDF at 75th pctile", "cdf0.9": "CDF at 90th pctile",
}
CAL_RED = {
    "exp": "calibrated running mean-z", "scan": "GLR change-point scan",
    "scanpeak": "strongest scan evidence so far", "roll50": "recent-50 calibrated mean",
    "roll200": "recent-200 calibrated mean", "frac": "fraction of steps over threshold",
}
SPECIAL = {
    "spectral_entropy": ("spectral", "spread of the frequency content (spectral entropy)"),
    "spectral_centroid": ("spectral", "center-of-mass frequency"),
    "spectral_domfreq": ("spectral", "dominant frequency"),
    "spectral_lowband": ("spectral", "low-frequency power fraction"),
    "ftest_stat": ("hypothesis", "F-test statistic (log variance ratio)"),
    "ftest_pval": ("hypothesis", "F-test p-value for equal variance"),
    "levene_t": ("hypothesis", "Levene robust equal-spread test"),
    "ks_stat": ("hypothesis", "KS distance between live and history distributions"),
    "ks_pval": ("hypothesis", "KS test p-value"),
    "mannwhitney_z": ("hypothesis", "Mann-Whitney location test (z-score)"),
    "mannwhitney_pval": ("hypothesis", "Mann-Whitney p-value"),
    "cal_glr_strength": ("changepoint", "pooled 'did it break' detector strength"),
    "cal_glr_peak": ("changepoint", "strongest pooled break evidence so far"),
    "cal_glr_seg_len": ("changepoint", "log length of estimated post-break segment"),
    "cal_glr_seg_frac": ("changepoint", "fraction of stream estimated post-break"),
}


def describe(name):
    if name in SPECIAL:
        return SPECIAL[name]
    if name.startswith("ref_"):
        return "context", f"fixed history fact: {name[4:]} (context, not break evidence)"
    if name.startswith("cal_seg_"):
        return "changepoint", f"{name[8:]} measured on the estimated post-break segment"
    if name.startswith("cal_"):
        for red in sorted(CAL_RED, key=len, reverse=True):
            if name.endswith("_" + red):
                inc = name[4: -len(red) - 1]
                d = CAL_INC.get(inc, inc)
                return "changepoint", f"[{CAL_RED[red]}] of [{d}]"
        return "changepoint", name
    # descriptive: strip variant, optional window, then {view}_{stat}
    variant = ""
    for suf in ("_vs_ref", "_ratio"):
        if name.endswith(suf):
            variant, name = suf, name[: -len(suf)]
            break
    win = ""
    if "_w" in name and name.rsplit("_w", 1)[1].isdigit():
        base, win = name.rsplit("_w", 1)
        name = base
    for stat in sorted(STAT, key=len, reverse=True):
        if name.endswith("_" + stat):
            view = name[: -len(stat) - 1]
            if view in VIEW:
                cat, meaning = STAT[stat]
                where = f" over the last {win} points" if win else " (expanding)"
                return cat, f"{meaning} of [{VIEW[view]}]{where}, {VARIANT[variant]}"
    return "other", name


def main():
    rng = np.random.default_rng(0)
    cols = list(B.series_features(rng.standard_normal(400), rng.standard_normal(300)).keys())
    rows = [dict(feature=n, category=(c := describe(n))[0], plain_english=c[1]) for n in cols]
    cat = pd.DataFrame(rows)
    cat.to_csv("feature_catalog.csv", index=False)
    order = ["statistical", "distributional", "dependence", "volatility",
             "changepoint", "hypothesis", "spectral", "context", "other"]
    with open("feature_catalog.md", "w") as f:
        f.write(f"# Feature bank catalog — {len(cat)} features\n\n")
        f.write("Descriptive names are `{view}_{stat}[_w{window}][_vs_ref|_ratio]` "
                "(e.g. `abs_std_w100_vs_ref`). Calibrated change-point names are "
                "`cal_{increment}_{reducer}` (e.g. `cal_var_scan`). Plus standalone "
                "hypothesis-test / spectral / context features.\n\n")
        for c in order:
            sub = cat[cat.category == c]
            if len(sub) == 0:
                continue
            f.write(f"\n## {c} ({len(sub)} features)\n\n| feature | meaning |\n|---|---|\n")
            for _, r in sub.iterrows():
                f.write(f"| `{r.feature}` | {r.plain_english} |\n")
    print(f"wrote feature_catalog.csv + .md — {len(cat)} features")
    print(cat.category.value_counts().to_string())


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
