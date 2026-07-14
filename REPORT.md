# Report — ADIA Structural Break Challenge (Real-Time)

**Last updated:** 2026-07-14

## 1. The task in one paragraph

Each time series has a reference segment, then an online segment revealed one point
at a time. After every online point we output a score in `[0,1]`: "has the process
already broken by now?" Scoring is **Time-Stratified AUC (TS-AUC)** — at each online
step, rank the series against each other (broken vs not), then average across steps.
Random guessing = 0.50. A score that is the same for every series at a step (like
elapsed time) also = 0.50, so we only win by telling series apart *at the same step*.

## 2. Where we landed

| Model | TS-AUC (2000-series holdout) |
|---|---|
| Value detector (mean-shift + variance ratio) | 0.53 |
| Old pipeline (7 features + LightGBM) | 0.53 |
| Batch feature bank (1072 features, 3-seed bag) | **0.606** |
| **Streaming model — what we actually submit** (30 features) | **0.603** |
| Oracle ceiling (true break point known) | 0.72 |

The streaming model matches the big batch model, and it runs online within budget.
We are essentially at the causal ceiling (~0.607): with the break point *unknown*,
0.61 is about the best any statistic can do on this data. The 0.72 oracle number is
only reachable if you already know when the break happened, which we don't.

## 3. What actually moved the score

- **Weight each series equally (1/length per row): +0.02.** Biggest single win. Long
  series otherwise dominate training; the metric treats every series equally.
- **Right feature families.** These breaks are mostly **variance / tail / shape**
  changes, not simple jumps in the average. Top features: tail mass, log-variance,
  spectral entropy, cumulative-sum range, AR-residual, distribution (KS) shift.
  Mean-shift and skew/kurtosis are weak on their own.
- **Seed bagging (3 seeds): +0.002.** Small but real.
- **Train on a log-spaced sample of online steps**, not every step. The even spread
  matches how the metric weights steps, and it beats dense training.

## 4. What we submit

The big batch model recomputes over the whole online buffer every step — about
120 ms/step, ~330 hours for the full test set. Not usable.

So the submission is a **streaming model** (`experiments/sb_stream.py`): one
`StreamState` object updates 30 features in constant time per point, and the *same*
object builds the training rows and runs inference, so the two always match. It scores
**0.603 holdout** and runs in **~3 hours per 10M points** (budget is 15 h/week).

Deliverable notebooks: `submission.ipynb` (repo root) and `submissions/submission1.ipynb`.

**Cloud "Run failed" fix (2026-07-14):** `# @crunch/keep:on` is a toggle — everything
after it exports to the cloud script until `# @crunch/keep:off`. `crunch_tools.test()`
was in the same kept cell as `INFER_PARALLELISM`, got exported, and crashed with
`NameError: crunch_tools is not defined`. Fixed: the keep block now wraps only
`INFER_PARALLELISM`; `test()` and `submit()` sit in their own separate cells.

## 5. Don't retry these (measured, all neutral or worse)

- Feature selection (any subset scored ≤ using all features — the model handles
  redundancy fine).
- Denser / full-resolution training (worse than the log-spaced sample).
- Deeper or more trees, lambdarank grouped by step, per-step specialist models,
  grouped-pair weighting, label cleaning near the break.
- Forcing the score to only go up (cummax) — the metric compares within a step, so
  this just throws away the ability to cancel a false alarm.
- MMD/RBF and cumulative-sum-view features (weak on this data).
- Deep learning and synthetic-data augmentation (also failed for the 2025 winners).

## 6. Notes on evaluation

- **Trust the 2000-series holdout, not the 100-series reduced set.** The reduced set
  is tiny and noisy — the streaming model scores 0.52 there but its 95% range is
  roughly [0.42, 0.61], fully consistent with the 0.60 holdout. Don't act on the
  reduced number.
- Every feature is causal (reference + online up to now only) and passes a
  leakage test. Re-run that test after any feature change.

## 7. If we push further (headroom is on the modeling side, not features)

Hand-built statistics are exhausted at ~0.61. Ranked ideas:

1. **Learned sequence model** (small TCN/Transformer over the online stream), trained
   directly for the ranking metric. Highest upside — learns break signatures instead
   of fixed test statistics.
2. **More streaming features** in the submitted model (more AR lags, more EWMA
   speeds, trailing-window spectral) — cheap, incremental gains.
3. **Slower/tuned trees** (low learning rate, many trees) — ~+0.01 in offline CV.
4. **Model diversity / stacking** (the 2025 winners used TabPFN as a feature; blocked
   for us — it needs an online auth token).
