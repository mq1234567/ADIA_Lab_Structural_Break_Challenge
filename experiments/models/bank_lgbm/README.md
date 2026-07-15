# bank_lgbm — batch feature bank + 3-seed LightGBM

Current best offline model. 1072 features from `sb_bank.py` fed into LightGBM,
3-seed bag, equal-series row weighting.

## Score (TS-AUC)

| eval set | score | source |
|---|---|---|
| 2000-series holdout | **0.606** | `train.py` (this dir) |
| 100-series reduced test | ~0.57 | idem; noisy at n=100 (see REPORT §6) |

The streaming submission (`sb_stream.py`, not persisted here) matches within
noise at 0.603 holdout and is what actually gets submitted.

## Contents

- `params.json` — LightGBM hyperparameters (the tuned config)
- `train.py` — reproducer: (re)builds feature parquets, trains the 3-seed bag,
  saves `model_seed{0,1,2}.txt`, writes `final_gain_ranking.csv` back into
  `experiments/`
- `model_seed{0,1,2}.txt` — trained LightGBM boosters (~3.4 MB each)

## Reproducing

```bash
cd experiments/models/bank_lgbm
python train.py
```

Requires the cached feature parquets in `experiments/`
(`feat_final_train.parquet`, `feat_final_holdout.parquet`,
`feat_final_reduced.parquet`) — `train.py` will regenerate them from the raw
data if missing. Also expects `experiments/tr_ids.npy` / `va_ids.npy`.

## Key modeling choices (see REPORT §3)

- **Equal-series weighting** (`1 / L_i` per row) — the biggest single win, +0.02.
- **Log-spaced online steps** during training (from `sb_bank.default_step_grid`).
- **3-seed bag** with rank-averaged predictions — small but consistent +0.002.
- **Same PARAMS** used by the streaming submission for consistency.
