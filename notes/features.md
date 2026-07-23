
## Selected streaming features (~50) family


# Feature bank catalog — 1072 features

Descriptive names are `{view}_{stat}[_w{window}][_vs_ref|_ratio]` (e.g. `abs_std_w100_vs_ref`). Calibrated change-point names are `cal_{increment}_{reducer}` (e.g. `cal_var_scan`). Plus standalone hypothesis-test / spectral / context features.

## Features importance ranking (LightGBM gain importance)
- experiments/final_gain_ranking.csv
- Top features
	1. cumsum_range
	2. spectral_entropy
	3. cal_ar5_var_scan
	4. ref_skew
	5. ref_log_n