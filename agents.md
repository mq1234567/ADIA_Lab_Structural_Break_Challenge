# AGENTS.md — ADIA Lab Structural Break Challenge (Real-Time Edition)

Context notes for agents working in this repo. Last updated 2026-07-14.

## 1. The competition

- **Host:** CrunchDAO × ADIA Lab, 2026. Page: https://hub.crunchdao.com/competitions/structural-break-real-time
  (that page is a JS SPA — WebFetch returns only nav chrome. Use the docs:
  https://docs.crunchdao.com/competitions/competitions/adia-lab-structural-break-challenge)
- **Scientific question:** did the data-generating process of a time series change (a *structural
  break*), or is the variation just noise? A break = a fundamental change in the DGP (mean, variance,
  distribution, autocorrelation, functional form). It can be abrupt or gradual and **may take some
  observations after the true break point to become detectable.**
- **What "Real-Time Edition" changes vs. the classic 2025 challenge:** the classic version asked for
  *one* confidence score per series (did a break happen anywhere after the known boundary). This
  edition is **online / sequential**: you get a historical reference segment up front, then the
  online segment is revealed **one observation at a time**, and after *each* new observation you emit
  a cumulative confidence in `[0,1]` that a break **has already occurred by now**.
- **Metric: Time-Stratified AUC (TS-AUC)** — per online step `t`, a cross-sectional AUC across the
  series alive at `t`, averaged weighted by pair count `w(t)=n_pos·n_neg`. Threshold-free; ~0.5 =
  random, 1.0 = perfect. **A score that is constant across series at a fixed step (e.g. elapsed time)
  scores 0.5** — you only win by discriminating series *against each other at the same step*. See §4b.
- **Submission model:** submit Python code with `train()` and `infer()`. Inference is **sequential —
  you cannot read all of X_test and predict at once** (no look-ahead across the online stream).
  Compute budget ~15 hours/week. Outputs must be deterministic.

## 2. Data files in this directory

| File | Shape | What it is |
|------|-------|-----------|
| `X_train.parquet` | 35,036,464 × 2 | All series values. MultiIndex `(id, time)`, cols `value` (float32), `period` (int: 1=reference/pre, 2=online/post-boundary) |
| `y_train.parquet` | 5,036,517 × 1 | Per-online-timestep target. MultiIndex `(id, time)`, col `target` ∈ {0,1}. Only covers period-2 timesteps |
| `y_train_index.parquet` | 10,000 × 2 | Per-series break location. Index `id`, cols `tau_index`, `tau` |
| `X_test.reduced.parquet` | 346,019 × 2 | Local eval set, 100 series (ids 10000–10099), same schema as X_train |
| `y_test.reduced.parquet` | 50,983 × 1 | Targets for the 100 local eval series |
| `y_test_index.reduced.parquet` | 100 × 2 | Break locations for the 100 local eval series |

Full test set is 10,000 series server-side; only the 100-series "reduced" set is local.

## 3. Data structure — verified facts

- **10,000 training series.** Total length 1,055–5,962 obs (mean/median ≈ 3,504).
- **`period`:** `1` = historical reference segment (given up front, ~mean 3,000 obs/series);
  `2` = online segment revealed one-at-a-time (length 10–999, mean ≈ 504). The **boundary** is the
  period 1→2 switch and is known to you.
- **Break labels (`y_train_index`):**
  - `tau = -1` and `tau_index = -1` → **no break** in this series.
  - otherwise `tau` = absolute `time` of the true break; `tau_index` = its offset **into the online
    segment** (i.e. `tau = boundary_time + tau_index`; verified on id 0: boundary 1192, tau_index 72,
    tau 1264).
  - **Break rate ≈ 50.3%** no-break vs 49.7% break — balanced.
  - break offset `tau_index` into online segment: min 0, median ~184, max 984. **29 series break at
    the very first online obs (`tau_index=0`).**
- **Per-timestep target (`y_train`):** for online time `t`, `target = 1` iff `t >= tau` (break already
  happened), else `0`. No-break series are all 0. Overall `target` mean ≈ 0.256. Every series appears
  in y_train. This is exactly the online "has a break occurred yet" label you must rank with.
- **Values:** roughly standardized *per series* (per-series ~zero mean, unit-ish scale; 1/50/99
  percentiles ≈ −2.36 / 0 / 2.39). Global std (7.15) is inflated by **across-series scale
  differences** — normalize per series (and per period) before comparing.

## 4. Starting insights / baseline findings

Quick experiments (see section 6 to reproduce):

- **Naive series-level detectors are weak.** Comparing the *whole* online segment to the reference:
  - mean-shift z-score → AUC ≈ **0.543**
  - variance-ratio (|log σ₂/σ₁|) → AUC ≈ **0.537**
  - combined → AUC ≈ **0.545**
- **Implication:** breaks are mostly NOT simple mean/variance jumps. On break series only ~31% show
  |mean shift| > 0.1 and ~28% show a std ratio outside [0.83, 1.2]. The signal likely lives in
  **distributional shape, autocorrelation/spectral structure, or functional form** — go beyond first
  two moments (KS/Cramér–von Mises two-sample tests, AR/spectral features, higher moments, CUSUM).
- **The real task is easier than that series-level number suggests**, because scoring is per-timestep
  and threshold-free: a good online statistic that grows once `t` passes `tau` will rank post-break
  timesteps above pre-break ones even if the per-series effect is small. Frame it as **online
  change-point detection**: at each `t`, test "online-so-far vs reference" and output a monotone-ish
  cumulative score (e.g. running max of a CUSUM / test statistic so confidence never drops after
  evidence appears).

## 4b. Modeling pipeline — `pipeline.ipynb` (steps 1–4 built)

Reusable pipeline, separate from `investigate.ipynb`. Two shared pieces: **`build_features(X)`**
(causal per-timestep feature matrix) and **`score_predictions(pred, y)`** (TS-AUC harness,
numpy rank-sum, no scipy/sklearn; `pooled_auc` kept for diagnostics).

- **Step 1 — eval harness:** `ts_auc` (Time-Stratified AUC = competition metric; random→0.50,
  perfect→1.00, series-constant→0.50) + `score_predictions` (aligns `(id,time)` preds to online rows).
- **Step 2 — baselines:** `causal_baseline(X)` = expanding mean-shift z + |log var-ratio|, variance
  guarded until `VAR_MIN=20` points.
- **Step 3 — `build_features`:** 7 causal features (`n_online`, `last_z`, `exp_mean_shift_z`,
  `exp_logvar`, `roll_mean_shift`, `roll_logstd`, `evidence_cummax`). Includes a **leakage test**
  (prefix-truncation invariance) — keep it; it's the guardrail against look-ahead features.
- **Step 4 — model:** LightGBM, **GroupKFold(5) by `id`**, OOF scored vs baselines. (Needs the
  `structural-break` conda env for `lightgbm`/`scikit-learn` — can't run under system python.)

### ⚠️ THE METRIC IS TIME-STRATIFIED AUC (TS-AUC) — not pooled AUC

At each online step `t` (the `t`-th online observation, aligned across series) TS-AUC computes a
**cross-sectional AUC across the series alive at step `t`** (positive = break already occurred),
then averages `AUC(t)` weighted by pair count `w(t)=n_pos(t)·n_neg(t)`. Equivalent to pooling only
*within-step* pos/neg pairs. `score_predictions`/`ts_auc` in `pipeline.ipynb` implement this;
`pooled_auc` is kept for diagnostics only.

Consequences (these correct earlier notes):

- **`n_online` scores exactly 0.5 under TS-AUC** — it's identical across all series at a fixed step,
  so it can't discriminate. The metric is *designed* to neutralize any series-independent score.
  (The earlier "~0.68 baseline" was a pooled-AUC artifact from cross-step comparisons — **ignore it**.)
- **The real baseline is the value detector ≈ 0.55**, and features must be **comparable across series
  at the same step** → normalize by each series' own reference stats (as the features do).
- Per-feature TS-AUC on reduced set: `exp_mean_shift_z` ≈ 0.56, `|exp_logvar|` ≈ 0.54,
  `roll_mean_shift` ≈ 0.52; `n_online` = 0.50 (was 0.69 pooled). Genuine value-detection is the game.
- **Bug caught earlier:** a CUSUM baseline scored 0.57 (pooled) only because `log(variance)` blows up
  at `n=1` (single-point variance=0) + `cummax` lock. Guard small-`n` variance (`VAR_MIN=20`).

## 4c. Optimization campaign (see experiments/ + REPORT.md). Current best 0.606 holdout.

Full feature/model optimization lives in `experiments/`. The old `sb_features*.py` chain was
replaced by a single flat feature bank `sb_bank.py`. All TS-AUC on a fixed 2000-series holdout.

- **Equal-series sample weighting (w = 1/L_i per row) is the single biggest modeling win**
  (+0.02): long series dominate unweighted log-loss.
- **Best batch model: 1072-feature bank (`sb_bank.py`), LightGBM 3-seed bag, eq-series weights,
  trained on a log-spaced online-step subsample → 0.606 holdout.** Progression: 7 feats 0.533 →
  30 feats 0.561 → +eq-weights 0.589 → flat bank 0.606. New families that paid off: trailing-window
  spectral entropy/centroid, causal KS / F-test / Mann-Whitney, AR(5)-residual.
- **Oracle ceiling**: with the TRUE break point known, the battery reaches 0.722 series-level AUC.
  Causal last-step equivalent ≈ 0.607, so we are near the causal ceiling — the ~0.11 gap is
  break-point uncertainty, not missing statistics. Breaks are mostly variance/tail/shape changes;
  mean-shift, skew, kurtosis are weak.
- **Dead ends measured (don't retry)**: feature selection (any subset ≤ all-features), denser/full-res
  training (log-subsample is better), deeper/more trees, lambdarank by step, TS-pair weighting,
  per-step-band specialist models, grace-band label cleaning, cummax post-processing, MMD-RBF,
  CUSUM-view features, deep learning, synthetic augmentation.
- Hyperparameters near-optimal at: lr 0.05, 500 trees, 63 leaves, min_child 300, subsample 0.8,
  colsample 0.8, reg_lambda 1.

## 4d. Submission — streaming model (2026-07-14)

The batch bank recomputes over the whole online buffer each step (~120 ms/step → ~330 h) and **cannot
be submitted**. The submission is a streaming model, `experiments/sb_stream.py`: a `StreamState` class
with `reset(reference)` + `update(x)->features`, 30 O(1)/ring-buffer features, used identically to build
training rows and to stream inference so vectors match. **Scores 0.603 holdout (≈ batch bank), runs
~3 h per 10M points** (budget 15 h/week). Deliverables: `submission.ipynb` (root), `submissions/submission1.ipynb`.

- **Reduced-100 eval is unreliable** (stream scores 0.52 there, 95% range ~[0.42, 0.61]); trust the
  2000-series holdout.
- **Cloud "Run failed" gotcha:** `# @crunch/keep:on` is a toggle — it exports every following line to
  the cloud script until `# @crunch/keep:off`. Keep `crunch_tools.test()`/`submit()` in their own
  cells, outside any keep block, or the cloud run crashes with `NameError: crunch_tools is not defined`.

## 5. Suggested next steps

Feature engineering is exhausted at ~0.61 (causal ceiling). Headroom is on the modeling side:

1. **Learned sequence model** (small TCN/Transformer over the online stream, trained for the ranking
   metric). Highest upside — learns break signatures instead of fixed statistics.
2. **More streaming features** in `sb_stream.py` (more AR lags, more EWMA speeds, trailing-window
   spectral) — cheap incremental gains; keep every feature O(1)/ring-buffer and re-run the leakage test.
3. **Slower/tuned trees** (low lr, many trees) — ~+0.01 in offline CV.
4. **Stacking / model diversity** (2025 winners used TabPFN as a feature; blocked — needs an online token).

## 6. Environment / gotchas

- `python3` = 3.10.4 at `/usr/local/bin/python3`. `pandas`, `numpy`, `pyarrow` (24.0.0) available.
  **`scipy` is NOT installed** and there's **no GPU** — implement stats (AUC, ranks, KS) in numpy or
  `pip install scipy`.
- **Not a git repo.** No CLAUDE.md. Parquet files are large (`X_train` ≈ 218 MB, 35 M rows) — load
  with column/row filtering or per-id groupby; don't materialize everything naively.
- Reading a single series: `X.loc[id]` on the MultiIndex; split by `period` column.
- Scratchpad for temp files:
  `/private/tmp/claude-501/-Users-minqi-Documents-ADIA-Lab-Structural-Break-Challenge/37e86991-c218-4c3f-ba40-e7c7ff82e596/scratchpad`

## 7. Reproduce the baseline numbers

Load `X_train`/`y_train_index`, sample series, and for each compare period-2 vs period-1 with a
mean-shift z, |log variance-ratio|, characterize post-vs-pre moments on break series, and score with a
numpy Mann–Whitney AUC (`rank(score)` → `(R1 − n1(n1+1)/2)/(n1·n0)`). Exact snippets are in the chat
transcript that created this file; re-run against the reduced test set for a realistic estimate.
