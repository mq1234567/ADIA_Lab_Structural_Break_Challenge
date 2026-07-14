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

## statistical (576 features)

| feature | meaning |
|---|---|
| `z_mean` | average level of [standardized value] (expanding), raw running value |
| `z_mean_vs_ref` | average level of [standardized value] (expanding), difference vs the reference-segment value |
| `z_mean_ratio` | average level of [standardized value] (expanding), ratio to the reference-segment value |
| `abs_mean` | average level of [magnitude |z|] (expanding), raw running value |
| `abs_mean_vs_ref` | average level of [magnitude |z|] (expanding), difference vs the reference-segment value |
| `abs_mean_ratio` | average level of [magnitude |z|] (expanding), ratio to the reference-segment value |
| `sign_mean` | average level of [sign of z] (expanding), raw running value |
| `sign_mean_vs_ref` | average level of [sign of z] (expanding), difference vs the reference-segment value |
| `sign_mean_ratio` | average level of [sign of z] (expanding), ratio to the reference-segment value |
| `square_mean` | average level of [squared value z^2] (expanding), raw running value |
| `square_mean_vs_ref` | average level of [squared value z^2] (expanding), difference vs the reference-segment value |
| `square_mean_ratio` | average level of [squared value z^2] (expanding), ratio to the reference-segment value |
| `logabs_mean` | average level of [log-magnitude] (expanding), raw running value |
| `logabs_mean_vs_ref` | average level of [log-magnitude] (expanding), difference vs the reference-segment value |
| `logabs_mean_ratio` | average level of [log-magnitude] (expanding), ratio to the reference-segment value |
| `cumsum_mean` | average level of [running sum] (expanding), raw running value |
| `cumsum_mean_vs_ref` | average level of [running sum] (expanding), difference vs the reference-segment value |
| `cumsum_mean_ratio` | average level of [running sum] (expanding), ratio to the reference-segment value |
| `diff_mean` | average level of [step-to-step change] (expanding), raw running value |
| `diff_mean_vs_ref` | average level of [step-to-step change] (expanding), difference vs the reference-segment value |
| `diff_mean_ratio` | average level of [step-to-step change] (expanding), ratio to the reference-segment value |
| `diff2_mean` | average level of [curvature (2nd difference)] (expanding), raw running value |
| `diff2_mean_vs_ref` | average level of [curvature (2nd difference)] (expanding), difference vs the reference-segment value |
| `diff2_mean_ratio` | average level of [curvature (2nd difference)] (expanding), ratio to the reference-segment value |
| `rank_mean` | average level of [percentile rank vs history] (expanding), raw running value |
| `rank_mean_vs_ref` | average level of [percentile rank vs history] (expanding), difference vs the reference-segment value |
| `rank_mean_ratio` | average level of [percentile rank vs history] (expanding), ratio to the reference-segment value |
| `rank_centered_mean` | average level of [centered percentile rank] (expanding), raw running value |
| `rank_centered_mean_vs_ref` | average level of [centered percentile rank] (expanding), difference vs the reference-segment value |
| `rank_centered_mean_ratio` | average level of [centered percentile rank] (expanding), ratio to the reference-segment value |
| `rollmean16_mean` | average level of [local 16-point mean] (expanding), raw running value |
| `rollmean16_mean_vs_ref` | average level of [local 16-point mean] (expanding), difference vs the reference-segment value |
| `rollmean16_mean_ratio` | average level of [local 16-point mean] (expanding), ratio to the reference-segment value |
| `rollstd16_mean` | average level of [local 16-point volatility] (expanding), raw running value |
| `rollstd16_mean_vs_ref` | average level of [local 16-point volatility] (expanding), difference vs the reference-segment value |
| `rollstd16_mean_ratio` | average level of [local 16-point volatility] (expanding), ratio to the reference-segment value |
| `z_skew` | lopsidedness (skew) of [standardized value] (expanding), raw running value |
| `z_skew_vs_ref` | lopsidedness (skew) of [standardized value] (expanding), difference vs the reference-segment value |
| `z_skew_ratio` | lopsidedness (skew) of [standardized value] (expanding), ratio to the reference-segment value |
| `abs_skew` | lopsidedness (skew) of [magnitude |z|] (expanding), raw running value |
| `abs_skew_vs_ref` | lopsidedness (skew) of [magnitude |z|] (expanding), difference vs the reference-segment value |
| `abs_skew_ratio` | lopsidedness (skew) of [magnitude |z|] (expanding), ratio to the reference-segment value |
| `sign_skew` | lopsidedness (skew) of [sign of z] (expanding), raw running value |
| `sign_skew_vs_ref` | lopsidedness (skew) of [sign of z] (expanding), difference vs the reference-segment value |
| `sign_skew_ratio` | lopsidedness (skew) of [sign of z] (expanding), ratio to the reference-segment value |
| `square_skew` | lopsidedness (skew) of [squared value z^2] (expanding), raw running value |
| `square_skew_vs_ref` | lopsidedness (skew) of [squared value z^2] (expanding), difference vs the reference-segment value |
| `square_skew_ratio` | lopsidedness (skew) of [squared value z^2] (expanding), ratio to the reference-segment value |
| `logabs_skew` | lopsidedness (skew) of [log-magnitude] (expanding), raw running value |
| `logabs_skew_vs_ref` | lopsidedness (skew) of [log-magnitude] (expanding), difference vs the reference-segment value |
| `logabs_skew_ratio` | lopsidedness (skew) of [log-magnitude] (expanding), ratio to the reference-segment value |
| `cumsum_skew` | lopsidedness (skew) of [running sum] (expanding), raw running value |
| `cumsum_skew_vs_ref` | lopsidedness (skew) of [running sum] (expanding), difference vs the reference-segment value |
| `cumsum_skew_ratio` | lopsidedness (skew) of [running sum] (expanding), ratio to the reference-segment value |
| `diff_skew` | lopsidedness (skew) of [step-to-step change] (expanding), raw running value |
| `diff_skew_vs_ref` | lopsidedness (skew) of [step-to-step change] (expanding), difference vs the reference-segment value |
| `diff_skew_ratio` | lopsidedness (skew) of [step-to-step change] (expanding), ratio to the reference-segment value |
| `diff2_skew` | lopsidedness (skew) of [curvature (2nd difference)] (expanding), raw running value |
| `diff2_skew_vs_ref` | lopsidedness (skew) of [curvature (2nd difference)] (expanding), difference vs the reference-segment value |
| `diff2_skew_ratio` | lopsidedness (skew) of [curvature (2nd difference)] (expanding), ratio to the reference-segment value |
| `rank_skew` | lopsidedness (skew) of [percentile rank vs history] (expanding), raw running value |
| `rank_skew_vs_ref` | lopsidedness (skew) of [percentile rank vs history] (expanding), difference vs the reference-segment value |
| `rank_skew_ratio` | lopsidedness (skew) of [percentile rank vs history] (expanding), ratio to the reference-segment value |
| `rank_centered_skew` | lopsidedness (skew) of [centered percentile rank] (expanding), raw running value |
| `rank_centered_skew_vs_ref` | lopsidedness (skew) of [centered percentile rank] (expanding), difference vs the reference-segment value |
| `rank_centered_skew_ratio` | lopsidedness (skew) of [centered percentile rank] (expanding), ratio to the reference-segment value |
| `rollmean16_skew` | lopsidedness (skew) of [local 16-point mean] (expanding), raw running value |
| `rollmean16_skew_vs_ref` | lopsidedness (skew) of [local 16-point mean] (expanding), difference vs the reference-segment value |
| `rollmean16_skew_ratio` | lopsidedness (skew) of [local 16-point mean] (expanding), ratio to the reference-segment value |
| `rollstd16_skew` | lopsidedness (skew) of [local 16-point volatility] (expanding), raw running value |
| `rollstd16_skew_vs_ref` | lopsidedness (skew) of [local 16-point volatility] (expanding), difference vs the reference-segment value |
| `rollstd16_skew_ratio` | lopsidedness (skew) of [local 16-point volatility] (expanding), ratio to the reference-segment value |
| `z_kurt` | heavy-tailedness (kurtosis) of [standardized value] (expanding), raw running value |
| `z_kurt_vs_ref` | heavy-tailedness (kurtosis) of [standardized value] (expanding), difference vs the reference-segment value |
| `z_kurt_ratio` | heavy-tailedness (kurtosis) of [standardized value] (expanding), ratio to the reference-segment value |
| `abs_kurt` | heavy-tailedness (kurtosis) of [magnitude |z|] (expanding), raw running value |
| `abs_kurt_vs_ref` | heavy-tailedness (kurtosis) of [magnitude |z|] (expanding), difference vs the reference-segment value |
| `abs_kurt_ratio` | heavy-tailedness (kurtosis) of [magnitude |z|] (expanding), ratio to the reference-segment value |
| `sign_kurt` | heavy-tailedness (kurtosis) of [sign of z] (expanding), raw running value |
| `sign_kurt_vs_ref` | heavy-tailedness (kurtosis) of [sign of z] (expanding), difference vs the reference-segment value |
| `sign_kurt_ratio` | heavy-tailedness (kurtosis) of [sign of z] (expanding), ratio to the reference-segment value |
| `square_kurt` | heavy-tailedness (kurtosis) of [squared value z^2] (expanding), raw running value |
| `square_kurt_vs_ref` | heavy-tailedness (kurtosis) of [squared value z^2] (expanding), difference vs the reference-segment value |
| `square_kurt_ratio` | heavy-tailedness (kurtosis) of [squared value z^2] (expanding), ratio to the reference-segment value |
| `logabs_kurt` | heavy-tailedness (kurtosis) of [log-magnitude] (expanding), raw running value |
| `logabs_kurt_vs_ref` | heavy-tailedness (kurtosis) of [log-magnitude] (expanding), difference vs the reference-segment value |
| `logabs_kurt_ratio` | heavy-tailedness (kurtosis) of [log-magnitude] (expanding), ratio to the reference-segment value |
| `cumsum_kurt` | heavy-tailedness (kurtosis) of [running sum] (expanding), raw running value |
| `cumsum_kurt_vs_ref` | heavy-tailedness (kurtosis) of [running sum] (expanding), difference vs the reference-segment value |
| `cumsum_kurt_ratio` | heavy-tailedness (kurtosis) of [running sum] (expanding), ratio to the reference-segment value |
| `diff_kurt` | heavy-tailedness (kurtosis) of [step-to-step change] (expanding), raw running value |
| `diff_kurt_vs_ref` | heavy-tailedness (kurtosis) of [step-to-step change] (expanding), difference vs the reference-segment value |
| `diff_kurt_ratio` | heavy-tailedness (kurtosis) of [step-to-step change] (expanding), ratio to the reference-segment value |
| `diff2_kurt` | heavy-tailedness (kurtosis) of [curvature (2nd difference)] (expanding), raw running value |
| `diff2_kurt_vs_ref` | heavy-tailedness (kurtosis) of [curvature (2nd difference)] (expanding), difference vs the reference-segment value |
| `diff2_kurt_ratio` | heavy-tailedness (kurtosis) of [curvature (2nd difference)] (expanding), ratio to the reference-segment value |
| `rank_kurt` | heavy-tailedness (kurtosis) of [percentile rank vs history] (expanding), raw running value |
| `rank_kurt_vs_ref` | heavy-tailedness (kurtosis) of [percentile rank vs history] (expanding), difference vs the reference-segment value |
| `rank_kurt_ratio` | heavy-tailedness (kurtosis) of [percentile rank vs history] (expanding), ratio to the reference-segment value |
| `rank_centered_kurt` | heavy-tailedness (kurtosis) of [centered percentile rank] (expanding), raw running value |
| `rank_centered_kurt_vs_ref` | heavy-tailedness (kurtosis) of [centered percentile rank] (expanding), difference vs the reference-segment value |
| `rank_centered_kurt_ratio` | heavy-tailedness (kurtosis) of [centered percentile rank] (expanding), ratio to the reference-segment value |
| `rollmean16_kurt` | heavy-tailedness (kurtosis) of [local 16-point mean] (expanding), raw running value |
| `rollmean16_kurt_vs_ref` | heavy-tailedness (kurtosis) of [local 16-point mean] (expanding), difference vs the reference-segment value |
| `rollmean16_kurt_ratio` | heavy-tailedness (kurtosis) of [local 16-point mean] (expanding), ratio to the reference-segment value |
| `rollstd16_kurt` | heavy-tailedness (kurtosis) of [local 16-point volatility] (expanding), raw running value |
| `rollstd16_kurt_vs_ref` | heavy-tailedness (kurtosis) of [local 16-point volatility] (expanding), difference vs the reference-segment value |
| `rollstd16_kurt_ratio` | heavy-tailedness (kurtosis) of [local 16-point volatility] (expanding), ratio to the reference-segment value |
| `z_min` | running minimum of [standardized value] (expanding), raw running value |
| `z_min_vs_ref` | running minimum of [standardized value] (expanding), difference vs the reference-segment value |
| `z_min_ratio` | running minimum of [standardized value] (expanding), ratio to the reference-segment value |
| `abs_min` | running minimum of [magnitude |z|] (expanding), raw running value |
| `abs_min_vs_ref` | running minimum of [magnitude |z|] (expanding), difference vs the reference-segment value |
| `abs_min_ratio` | running minimum of [magnitude |z|] (expanding), ratio to the reference-segment value |
| `sign_min` | running minimum of [sign of z] (expanding), raw running value |
| `sign_min_vs_ref` | running minimum of [sign of z] (expanding), difference vs the reference-segment value |
| `sign_min_ratio` | running minimum of [sign of z] (expanding), ratio to the reference-segment value |
| `square_min` | running minimum of [squared value z^2] (expanding), raw running value |
| `square_min_vs_ref` | running minimum of [squared value z^2] (expanding), difference vs the reference-segment value |
| `square_min_ratio` | running minimum of [squared value z^2] (expanding), ratio to the reference-segment value |
| `logabs_min` | running minimum of [log-magnitude] (expanding), raw running value |
| `logabs_min_vs_ref` | running minimum of [log-magnitude] (expanding), difference vs the reference-segment value |
| `logabs_min_ratio` | running minimum of [log-magnitude] (expanding), ratio to the reference-segment value |
| `cumsum_min` | running minimum of [running sum] (expanding), raw running value |
| `cumsum_min_vs_ref` | running minimum of [running sum] (expanding), difference vs the reference-segment value |
| `cumsum_min_ratio` | running minimum of [running sum] (expanding), ratio to the reference-segment value |
| `diff_min` | running minimum of [step-to-step change] (expanding), raw running value |
| `diff_min_vs_ref` | running minimum of [step-to-step change] (expanding), difference vs the reference-segment value |
| `diff_min_ratio` | running minimum of [step-to-step change] (expanding), ratio to the reference-segment value |
| `diff2_min` | running minimum of [curvature (2nd difference)] (expanding), raw running value |
| `diff2_min_vs_ref` | running minimum of [curvature (2nd difference)] (expanding), difference vs the reference-segment value |
| `diff2_min_ratio` | running minimum of [curvature (2nd difference)] (expanding), ratio to the reference-segment value |
| `rank_min` | running minimum of [percentile rank vs history] (expanding), raw running value |
| `rank_min_vs_ref` | running minimum of [percentile rank vs history] (expanding), difference vs the reference-segment value |
| `rank_min_ratio` | running minimum of [percentile rank vs history] (expanding), ratio to the reference-segment value |
| `rank_centered_min` | running minimum of [centered percentile rank] (expanding), raw running value |
| `rank_centered_min_vs_ref` | running minimum of [centered percentile rank] (expanding), difference vs the reference-segment value |
| `rank_centered_min_ratio` | running minimum of [centered percentile rank] (expanding), ratio to the reference-segment value |
| `rollmean16_min` | running minimum of [local 16-point mean] (expanding), raw running value |
| `rollmean16_min_vs_ref` | running minimum of [local 16-point mean] (expanding), difference vs the reference-segment value |
| `rollmean16_min_ratio` | running minimum of [local 16-point mean] (expanding), ratio to the reference-segment value |
| `rollstd16_min` | running minimum of [local 16-point volatility] (expanding), raw running value |
| `rollstd16_min_vs_ref` | running minimum of [local 16-point volatility] (expanding), difference vs the reference-segment value |
| `rollstd16_min_ratio` | running minimum of [local 16-point volatility] (expanding), ratio to the reference-segment value |
| `z_max` | running maximum of [standardized value] (expanding), raw running value |
| `z_max_vs_ref` | running maximum of [standardized value] (expanding), difference vs the reference-segment value |
| `z_max_ratio` | running maximum of [standardized value] (expanding), ratio to the reference-segment value |
| `abs_max` | running maximum of [magnitude |z|] (expanding), raw running value |
| `abs_max_vs_ref` | running maximum of [magnitude |z|] (expanding), difference vs the reference-segment value |
| `abs_max_ratio` | running maximum of [magnitude |z|] (expanding), ratio to the reference-segment value |
| `sign_max` | running maximum of [sign of z] (expanding), raw running value |
| `sign_max_vs_ref` | running maximum of [sign of z] (expanding), difference vs the reference-segment value |
| `sign_max_ratio` | running maximum of [sign of z] (expanding), ratio to the reference-segment value |
| `square_max` | running maximum of [squared value z^2] (expanding), raw running value |
| `square_max_vs_ref` | running maximum of [squared value z^2] (expanding), difference vs the reference-segment value |
| `square_max_ratio` | running maximum of [squared value z^2] (expanding), ratio to the reference-segment value |
| `logabs_max` | running maximum of [log-magnitude] (expanding), raw running value |
| `logabs_max_vs_ref` | running maximum of [log-magnitude] (expanding), difference vs the reference-segment value |
| `logabs_max_ratio` | running maximum of [log-magnitude] (expanding), ratio to the reference-segment value |
| `cumsum_max` | running maximum of [running sum] (expanding), raw running value |
| `cumsum_max_vs_ref` | running maximum of [running sum] (expanding), difference vs the reference-segment value |
| `cumsum_max_ratio` | running maximum of [running sum] (expanding), ratio to the reference-segment value |
| `diff_max` | running maximum of [step-to-step change] (expanding), raw running value |
| `diff_max_vs_ref` | running maximum of [step-to-step change] (expanding), difference vs the reference-segment value |
| `diff_max_ratio` | running maximum of [step-to-step change] (expanding), ratio to the reference-segment value |
| `diff2_max` | running maximum of [curvature (2nd difference)] (expanding), raw running value |
| `diff2_max_vs_ref` | running maximum of [curvature (2nd difference)] (expanding), difference vs the reference-segment value |
| `diff2_max_ratio` | running maximum of [curvature (2nd difference)] (expanding), ratio to the reference-segment value |
| `rank_max` | running maximum of [percentile rank vs history] (expanding), raw running value |
| `rank_max_vs_ref` | running maximum of [percentile rank vs history] (expanding), difference vs the reference-segment value |
| `rank_max_ratio` | running maximum of [percentile rank vs history] (expanding), ratio to the reference-segment value |
| `rank_centered_max` | running maximum of [centered percentile rank] (expanding), raw running value |
| `rank_centered_max_vs_ref` | running maximum of [centered percentile rank] (expanding), difference vs the reference-segment value |
| `rank_centered_max_ratio` | running maximum of [centered percentile rank] (expanding), ratio to the reference-segment value |
| `rollmean16_max` | running maximum of [local 16-point mean] (expanding), raw running value |
| `rollmean16_max_vs_ref` | running maximum of [local 16-point mean] (expanding), difference vs the reference-segment value |
| `rollmean16_max_ratio` | running maximum of [local 16-point mean] (expanding), ratio to the reference-segment value |
| `rollstd16_max` | running maximum of [local 16-point volatility] (expanding), raw running value |
| `rollstd16_max_vs_ref` | running maximum of [local 16-point volatility] (expanding), difference vs the reference-segment value |
| `rollstd16_max_ratio` | running maximum of [local 16-point volatility] (expanding), ratio to the reference-segment value |
| `z_median` | robust average level of [standardized value] (expanding), raw running value |
| `z_median_vs_ref` | robust average level of [standardized value] (expanding), difference vs the reference-segment value |
| `z_median_ratio` | robust average level of [standardized value] (expanding), ratio to the reference-segment value |
| `abs_median` | robust average level of [magnitude |z|] (expanding), raw running value |
| `abs_median_vs_ref` | robust average level of [magnitude |z|] (expanding), difference vs the reference-segment value |
| `abs_median_ratio` | robust average level of [magnitude |z|] (expanding), ratio to the reference-segment value |
| `sign_median` | robust average level of [sign of z] (expanding), raw running value |
| `sign_median_vs_ref` | robust average level of [sign of z] (expanding), difference vs the reference-segment value |
| `sign_median_ratio` | robust average level of [sign of z] (expanding), ratio to the reference-segment value |
| `square_median` | robust average level of [squared value z^2] (expanding), raw running value |
| `square_median_vs_ref` | robust average level of [squared value z^2] (expanding), difference vs the reference-segment value |
| `square_median_ratio` | robust average level of [squared value z^2] (expanding), ratio to the reference-segment value |
| `logabs_median` | robust average level of [log-magnitude] (expanding), raw running value |
| `logabs_median_vs_ref` | robust average level of [log-magnitude] (expanding), difference vs the reference-segment value |
| `logabs_median_ratio` | robust average level of [log-magnitude] (expanding), ratio to the reference-segment value |
| `cumsum_median` | robust average level of [running sum] (expanding), raw running value |
| `cumsum_median_vs_ref` | robust average level of [running sum] (expanding), difference vs the reference-segment value |
| `cumsum_median_ratio` | robust average level of [running sum] (expanding), ratio to the reference-segment value |
| `diff_median` | robust average level of [step-to-step change] (expanding), raw running value |
| `diff_median_vs_ref` | robust average level of [step-to-step change] (expanding), difference vs the reference-segment value |
| `diff_median_ratio` | robust average level of [step-to-step change] (expanding), ratio to the reference-segment value |
| `diff2_median` | robust average level of [curvature (2nd difference)] (expanding), raw running value |
| `diff2_median_vs_ref` | robust average level of [curvature (2nd difference)] (expanding), difference vs the reference-segment value |
| `diff2_median_ratio` | robust average level of [curvature (2nd difference)] (expanding), ratio to the reference-segment value |
| `rank_median` | robust average level of [percentile rank vs history] (expanding), raw running value |
| `rank_median_vs_ref` | robust average level of [percentile rank vs history] (expanding), difference vs the reference-segment value |
| `rank_median_ratio` | robust average level of [percentile rank vs history] (expanding), ratio to the reference-segment value |
| `rank_centered_median` | robust average level of [centered percentile rank] (expanding), raw running value |
| `rank_centered_median_vs_ref` | robust average level of [centered percentile rank] (expanding), difference vs the reference-segment value |
| `rank_centered_median_ratio` | robust average level of [centered percentile rank] (expanding), ratio to the reference-segment value |
| `rollmean16_median` | robust average level of [local 16-point mean] (expanding), raw running value |
| `rollmean16_median_vs_ref` | robust average level of [local 16-point mean] (expanding), difference vs the reference-segment value |
| `rollmean16_median_ratio` | robust average level of [local 16-point mean] (expanding), ratio to the reference-segment value |
| `rollstd16_median` | robust average level of [local 16-point volatility] (expanding), raw running value |
| `rollstd16_median_vs_ref` | robust average level of [local 16-point volatility] (expanding), difference vs the reference-segment value |
| `rollstd16_median_ratio` | robust average level of [local 16-point volatility] (expanding), ratio to the reference-segment value |
| `z_mean_w20` | average level of [standardized value] over the last 20 points, raw running value |
| `z_mean_w20_vs_ref` | average level of [standardized value] over the last 20 points, difference vs the reference-segment value |
| `abs_mean_w20` | average level of [magnitude |z|] over the last 20 points, raw running value |
| `abs_mean_w20_vs_ref` | average level of [magnitude |z|] over the last 20 points, difference vs the reference-segment value |
| `sign_mean_w20` | average level of [sign of z] over the last 20 points, raw running value |
| `sign_mean_w20_vs_ref` | average level of [sign of z] over the last 20 points, difference vs the reference-segment value |
| `square_mean_w20` | average level of [squared value z^2] over the last 20 points, raw running value |
| `square_mean_w20_vs_ref` | average level of [squared value z^2] over the last 20 points, difference vs the reference-segment value |
| `logabs_mean_w20` | average level of [log-magnitude] over the last 20 points, raw running value |
| `logabs_mean_w20_vs_ref` | average level of [log-magnitude] over the last 20 points, difference vs the reference-segment value |
| `cumsum_mean_w20` | average level of [running sum] over the last 20 points, raw running value |
| `cumsum_mean_w20_vs_ref` | average level of [running sum] over the last 20 points, difference vs the reference-segment value |
| `diff_mean_w20` | average level of [step-to-step change] over the last 20 points, raw running value |
| `diff_mean_w20_vs_ref` | average level of [step-to-step change] over the last 20 points, difference vs the reference-segment value |
| `diff2_mean_w20` | average level of [curvature (2nd difference)] over the last 20 points, raw running value |
| `diff2_mean_w20_vs_ref` | average level of [curvature (2nd difference)] over the last 20 points, difference vs the reference-segment value |
| `rank_mean_w20` | average level of [percentile rank vs history] over the last 20 points, raw running value |
| `rank_mean_w20_vs_ref` | average level of [percentile rank vs history] over the last 20 points, difference vs the reference-segment value |
| `rank_centered_mean_w20` | average level of [centered percentile rank] over the last 20 points, raw running value |
| `rank_centered_mean_w20_vs_ref` | average level of [centered percentile rank] over the last 20 points, difference vs the reference-segment value |
| `rollmean16_mean_w20` | average level of [local 16-point mean] over the last 20 points, raw running value |
| `rollmean16_mean_w20_vs_ref` | average level of [local 16-point mean] over the last 20 points, difference vs the reference-segment value |
| `rollstd16_mean_w20` | average level of [local 16-point volatility] over the last 20 points, raw running value |
| `rollstd16_mean_w20_vs_ref` | average level of [local 16-point volatility] over the last 20 points, difference vs the reference-segment value |
| `z_skew_w20` | lopsidedness (skew) of [standardized value] over the last 20 points, raw running value |
| `z_skew_w20_vs_ref` | lopsidedness (skew) of [standardized value] over the last 20 points, difference vs the reference-segment value |
| `abs_skew_w20` | lopsidedness (skew) of [magnitude |z|] over the last 20 points, raw running value |
| `abs_skew_w20_vs_ref` | lopsidedness (skew) of [magnitude |z|] over the last 20 points, difference vs the reference-segment value |
| `sign_skew_w20` | lopsidedness (skew) of [sign of z] over the last 20 points, raw running value |
| `sign_skew_w20_vs_ref` | lopsidedness (skew) of [sign of z] over the last 20 points, difference vs the reference-segment value |
| `square_skew_w20` | lopsidedness (skew) of [squared value z^2] over the last 20 points, raw running value |
| `square_skew_w20_vs_ref` | lopsidedness (skew) of [squared value z^2] over the last 20 points, difference vs the reference-segment value |
| `logabs_skew_w20` | lopsidedness (skew) of [log-magnitude] over the last 20 points, raw running value |
| `logabs_skew_w20_vs_ref` | lopsidedness (skew) of [log-magnitude] over the last 20 points, difference vs the reference-segment value |
| `cumsum_skew_w20` | lopsidedness (skew) of [running sum] over the last 20 points, raw running value |
| `cumsum_skew_w20_vs_ref` | lopsidedness (skew) of [running sum] over the last 20 points, difference vs the reference-segment value |
| `diff_skew_w20` | lopsidedness (skew) of [step-to-step change] over the last 20 points, raw running value |
| `diff_skew_w20_vs_ref` | lopsidedness (skew) of [step-to-step change] over the last 20 points, difference vs the reference-segment value |
| `diff2_skew_w20` | lopsidedness (skew) of [curvature (2nd difference)] over the last 20 points, raw running value |
| `diff2_skew_w20_vs_ref` | lopsidedness (skew) of [curvature (2nd difference)] over the last 20 points, difference vs the reference-segment value |
| `rank_skew_w20` | lopsidedness (skew) of [percentile rank vs history] over the last 20 points, raw running value |
| `rank_skew_w20_vs_ref` | lopsidedness (skew) of [percentile rank vs history] over the last 20 points, difference vs the reference-segment value |
| `rank_centered_skew_w20` | lopsidedness (skew) of [centered percentile rank] over the last 20 points, raw running value |
| `rank_centered_skew_w20_vs_ref` | lopsidedness (skew) of [centered percentile rank] over the last 20 points, difference vs the reference-segment value |
| `rollmean16_skew_w20` | lopsidedness (skew) of [local 16-point mean] over the last 20 points, raw running value |
| `rollmean16_skew_w20_vs_ref` | lopsidedness (skew) of [local 16-point mean] over the last 20 points, difference vs the reference-segment value |
| `rollstd16_skew_w20` | lopsidedness (skew) of [local 16-point volatility] over the last 20 points, raw running value |
| `rollstd16_skew_w20_vs_ref` | lopsidedness (skew) of [local 16-point volatility] over the last 20 points, difference vs the reference-segment value |
| `z_kurt_w20` | heavy-tailedness (kurtosis) of [standardized value] over the last 20 points, raw running value |
| `z_kurt_w20_vs_ref` | heavy-tailedness (kurtosis) of [standardized value] over the last 20 points, difference vs the reference-segment value |
| `abs_kurt_w20` | heavy-tailedness (kurtosis) of [magnitude |z|] over the last 20 points, raw running value |
| `abs_kurt_w20_vs_ref` | heavy-tailedness (kurtosis) of [magnitude |z|] over the last 20 points, difference vs the reference-segment value |
| `sign_kurt_w20` | heavy-tailedness (kurtosis) of [sign of z] over the last 20 points, raw running value |
| `sign_kurt_w20_vs_ref` | heavy-tailedness (kurtosis) of [sign of z] over the last 20 points, difference vs the reference-segment value |
| `square_kurt_w20` | heavy-tailedness (kurtosis) of [squared value z^2] over the last 20 points, raw running value |
| `square_kurt_w20_vs_ref` | heavy-tailedness (kurtosis) of [squared value z^2] over the last 20 points, difference vs the reference-segment value |
| `logabs_kurt_w20` | heavy-tailedness (kurtosis) of [log-magnitude] over the last 20 points, raw running value |
| `logabs_kurt_w20_vs_ref` | heavy-tailedness (kurtosis) of [log-magnitude] over the last 20 points, difference vs the reference-segment value |
| `cumsum_kurt_w20` | heavy-tailedness (kurtosis) of [running sum] over the last 20 points, raw running value |
| `cumsum_kurt_w20_vs_ref` | heavy-tailedness (kurtosis) of [running sum] over the last 20 points, difference vs the reference-segment value |
| `diff_kurt_w20` | heavy-tailedness (kurtosis) of [step-to-step change] over the last 20 points, raw running value |
| `diff_kurt_w20_vs_ref` | heavy-tailedness (kurtosis) of [step-to-step change] over the last 20 points, difference vs the reference-segment value |
| `diff2_kurt_w20` | heavy-tailedness (kurtosis) of [curvature (2nd difference)] over the last 20 points, raw running value |
| `diff2_kurt_w20_vs_ref` | heavy-tailedness (kurtosis) of [curvature (2nd difference)] over the last 20 points, difference vs the reference-segment value |
| `rank_kurt_w20` | heavy-tailedness (kurtosis) of [percentile rank vs history] over the last 20 points, raw running value |
| `rank_kurt_w20_vs_ref` | heavy-tailedness (kurtosis) of [percentile rank vs history] over the last 20 points, difference vs the reference-segment value |
| `rank_centered_kurt_w20` | heavy-tailedness (kurtosis) of [centered percentile rank] over the last 20 points, raw running value |
| `rank_centered_kurt_w20_vs_ref` | heavy-tailedness (kurtosis) of [centered percentile rank] over the last 20 points, difference vs the reference-segment value |
| `rollmean16_kurt_w20` | heavy-tailedness (kurtosis) of [local 16-point mean] over the last 20 points, raw running value |
| `rollmean16_kurt_w20_vs_ref` | heavy-tailedness (kurtosis) of [local 16-point mean] over the last 20 points, difference vs the reference-segment value |
| `rollstd16_kurt_w20` | heavy-tailedness (kurtosis) of [local 16-point volatility] over the last 20 points, raw running value |
| `rollstd16_kurt_w20_vs_ref` | heavy-tailedness (kurtosis) of [local 16-point volatility] over the last 20 points, difference vs the reference-segment value |
| `z_mean_w50` | average level of [standardized value] over the last 50 points, raw running value |
| `z_mean_w50_vs_ref` | average level of [standardized value] over the last 50 points, difference vs the reference-segment value |
| `abs_mean_w50` | average level of [magnitude |z|] over the last 50 points, raw running value |
| `abs_mean_w50_vs_ref` | average level of [magnitude |z|] over the last 50 points, difference vs the reference-segment value |
| `sign_mean_w50` | average level of [sign of z] over the last 50 points, raw running value |
| `sign_mean_w50_vs_ref` | average level of [sign of z] over the last 50 points, difference vs the reference-segment value |
| `square_mean_w50` | average level of [squared value z^2] over the last 50 points, raw running value |
| `square_mean_w50_vs_ref` | average level of [squared value z^2] over the last 50 points, difference vs the reference-segment value |
| `logabs_mean_w50` | average level of [log-magnitude] over the last 50 points, raw running value |
| `logabs_mean_w50_vs_ref` | average level of [log-magnitude] over the last 50 points, difference vs the reference-segment value |
| `cumsum_mean_w50` | average level of [running sum] over the last 50 points, raw running value |
| `cumsum_mean_w50_vs_ref` | average level of [running sum] over the last 50 points, difference vs the reference-segment value |
| `diff_mean_w50` | average level of [step-to-step change] over the last 50 points, raw running value |
| `diff_mean_w50_vs_ref` | average level of [step-to-step change] over the last 50 points, difference vs the reference-segment value |
| `diff2_mean_w50` | average level of [curvature (2nd difference)] over the last 50 points, raw running value |
| `diff2_mean_w50_vs_ref` | average level of [curvature (2nd difference)] over the last 50 points, difference vs the reference-segment value |
| `rank_mean_w50` | average level of [percentile rank vs history] over the last 50 points, raw running value |
| `rank_mean_w50_vs_ref` | average level of [percentile rank vs history] over the last 50 points, difference vs the reference-segment value |
| `rank_centered_mean_w50` | average level of [centered percentile rank] over the last 50 points, raw running value |
| `rank_centered_mean_w50_vs_ref` | average level of [centered percentile rank] over the last 50 points, difference vs the reference-segment value |
| `rollmean16_mean_w50` | average level of [local 16-point mean] over the last 50 points, raw running value |
| `rollmean16_mean_w50_vs_ref` | average level of [local 16-point mean] over the last 50 points, difference vs the reference-segment value |
| `rollstd16_mean_w50` | average level of [local 16-point volatility] over the last 50 points, raw running value |
| `rollstd16_mean_w50_vs_ref` | average level of [local 16-point volatility] over the last 50 points, difference vs the reference-segment value |
| `z_skew_w50` | lopsidedness (skew) of [standardized value] over the last 50 points, raw running value |
| `z_skew_w50_vs_ref` | lopsidedness (skew) of [standardized value] over the last 50 points, difference vs the reference-segment value |
| `abs_skew_w50` | lopsidedness (skew) of [magnitude |z|] over the last 50 points, raw running value |
| `abs_skew_w50_vs_ref` | lopsidedness (skew) of [magnitude |z|] over the last 50 points, difference vs the reference-segment value |
| `sign_skew_w50` | lopsidedness (skew) of [sign of z] over the last 50 points, raw running value |
| `sign_skew_w50_vs_ref` | lopsidedness (skew) of [sign of z] over the last 50 points, difference vs the reference-segment value |
| `square_skew_w50` | lopsidedness (skew) of [squared value z^2] over the last 50 points, raw running value |
| `square_skew_w50_vs_ref` | lopsidedness (skew) of [squared value z^2] over the last 50 points, difference vs the reference-segment value |
| `logabs_skew_w50` | lopsidedness (skew) of [log-magnitude] over the last 50 points, raw running value |
| `logabs_skew_w50_vs_ref` | lopsidedness (skew) of [log-magnitude] over the last 50 points, difference vs the reference-segment value |
| `cumsum_skew_w50` | lopsidedness (skew) of [running sum] over the last 50 points, raw running value |
| `cumsum_skew_w50_vs_ref` | lopsidedness (skew) of [running sum] over the last 50 points, difference vs the reference-segment value |
| `diff_skew_w50` | lopsidedness (skew) of [step-to-step change] over the last 50 points, raw running value |
| `diff_skew_w50_vs_ref` | lopsidedness (skew) of [step-to-step change] over the last 50 points, difference vs the reference-segment value |
| `diff2_skew_w50` | lopsidedness (skew) of [curvature (2nd difference)] over the last 50 points, raw running value |
| `diff2_skew_w50_vs_ref` | lopsidedness (skew) of [curvature (2nd difference)] over the last 50 points, difference vs the reference-segment value |
| `rank_skew_w50` | lopsidedness (skew) of [percentile rank vs history] over the last 50 points, raw running value |
| `rank_skew_w50_vs_ref` | lopsidedness (skew) of [percentile rank vs history] over the last 50 points, difference vs the reference-segment value |
| `rank_centered_skew_w50` | lopsidedness (skew) of [centered percentile rank] over the last 50 points, raw running value |
| `rank_centered_skew_w50_vs_ref` | lopsidedness (skew) of [centered percentile rank] over the last 50 points, difference vs the reference-segment value |
| `rollmean16_skew_w50` | lopsidedness (skew) of [local 16-point mean] over the last 50 points, raw running value |
| `rollmean16_skew_w50_vs_ref` | lopsidedness (skew) of [local 16-point mean] over the last 50 points, difference vs the reference-segment value |
| `rollstd16_skew_w50` | lopsidedness (skew) of [local 16-point volatility] over the last 50 points, raw running value |
| `rollstd16_skew_w50_vs_ref` | lopsidedness (skew) of [local 16-point volatility] over the last 50 points, difference vs the reference-segment value |
| `z_kurt_w50` | heavy-tailedness (kurtosis) of [standardized value] over the last 50 points, raw running value |
| `z_kurt_w50_vs_ref` | heavy-tailedness (kurtosis) of [standardized value] over the last 50 points, difference vs the reference-segment value |
| `abs_kurt_w50` | heavy-tailedness (kurtosis) of [magnitude |z|] over the last 50 points, raw running value |
| `abs_kurt_w50_vs_ref` | heavy-tailedness (kurtosis) of [magnitude |z|] over the last 50 points, difference vs the reference-segment value |
| `sign_kurt_w50` | heavy-tailedness (kurtosis) of [sign of z] over the last 50 points, raw running value |
| `sign_kurt_w50_vs_ref` | heavy-tailedness (kurtosis) of [sign of z] over the last 50 points, difference vs the reference-segment value |
| `square_kurt_w50` | heavy-tailedness (kurtosis) of [squared value z^2] over the last 50 points, raw running value |
| `square_kurt_w50_vs_ref` | heavy-tailedness (kurtosis) of [squared value z^2] over the last 50 points, difference vs the reference-segment value |
| `logabs_kurt_w50` | heavy-tailedness (kurtosis) of [log-magnitude] over the last 50 points, raw running value |
| `logabs_kurt_w50_vs_ref` | heavy-tailedness (kurtosis) of [log-magnitude] over the last 50 points, difference vs the reference-segment value |
| `cumsum_kurt_w50` | heavy-tailedness (kurtosis) of [running sum] over the last 50 points, raw running value |
| `cumsum_kurt_w50_vs_ref` | heavy-tailedness (kurtosis) of [running sum] over the last 50 points, difference vs the reference-segment value |
| `diff_kurt_w50` | heavy-tailedness (kurtosis) of [step-to-step change] over the last 50 points, raw running value |
| `diff_kurt_w50_vs_ref` | heavy-tailedness (kurtosis) of [step-to-step change] over the last 50 points, difference vs the reference-segment value |
| `diff2_kurt_w50` | heavy-tailedness (kurtosis) of [curvature (2nd difference)] over the last 50 points, raw running value |
| `diff2_kurt_w50_vs_ref` | heavy-tailedness (kurtosis) of [curvature (2nd difference)] over the last 50 points, difference vs the reference-segment value |
| `rank_kurt_w50` | heavy-tailedness (kurtosis) of [percentile rank vs history] over the last 50 points, raw running value |
| `rank_kurt_w50_vs_ref` | heavy-tailedness (kurtosis) of [percentile rank vs history] over the last 50 points, difference vs the reference-segment value |
| `rank_centered_kurt_w50` | heavy-tailedness (kurtosis) of [centered percentile rank] over the last 50 points, raw running value |
| `rank_centered_kurt_w50_vs_ref` | heavy-tailedness (kurtosis) of [centered percentile rank] over the last 50 points, difference vs the reference-segment value |
| `rollmean16_kurt_w50` | heavy-tailedness (kurtosis) of [local 16-point mean] over the last 50 points, raw running value |
| `rollmean16_kurt_w50_vs_ref` | heavy-tailedness (kurtosis) of [local 16-point mean] over the last 50 points, difference vs the reference-segment value |
| `rollstd16_kurt_w50` | heavy-tailedness (kurtosis) of [local 16-point volatility] over the last 50 points, raw running value |
| `rollstd16_kurt_w50_vs_ref` | heavy-tailedness (kurtosis) of [local 16-point volatility] over the last 50 points, difference vs the reference-segment value |
| `z_mean_w100` | average level of [standardized value] over the last 100 points, raw running value |
| `z_mean_w100_vs_ref` | average level of [standardized value] over the last 100 points, difference vs the reference-segment value |
| `abs_mean_w100` | average level of [magnitude |z|] over the last 100 points, raw running value |
| `abs_mean_w100_vs_ref` | average level of [magnitude |z|] over the last 100 points, difference vs the reference-segment value |
| `sign_mean_w100` | average level of [sign of z] over the last 100 points, raw running value |
| `sign_mean_w100_vs_ref` | average level of [sign of z] over the last 100 points, difference vs the reference-segment value |
| `square_mean_w100` | average level of [squared value z^2] over the last 100 points, raw running value |
| `square_mean_w100_vs_ref` | average level of [squared value z^2] over the last 100 points, difference vs the reference-segment value |
| `logabs_mean_w100` | average level of [log-magnitude] over the last 100 points, raw running value |
| `logabs_mean_w100_vs_ref` | average level of [log-magnitude] over the last 100 points, difference vs the reference-segment value |
| `cumsum_mean_w100` | average level of [running sum] over the last 100 points, raw running value |
| `cumsum_mean_w100_vs_ref` | average level of [running sum] over the last 100 points, difference vs the reference-segment value |
| `diff_mean_w100` | average level of [step-to-step change] over the last 100 points, raw running value |
| `diff_mean_w100_vs_ref` | average level of [step-to-step change] over the last 100 points, difference vs the reference-segment value |
| `diff2_mean_w100` | average level of [curvature (2nd difference)] over the last 100 points, raw running value |
| `diff2_mean_w100_vs_ref` | average level of [curvature (2nd difference)] over the last 100 points, difference vs the reference-segment value |
| `rank_mean_w100` | average level of [percentile rank vs history] over the last 100 points, raw running value |
| `rank_mean_w100_vs_ref` | average level of [percentile rank vs history] over the last 100 points, difference vs the reference-segment value |
| `rank_centered_mean_w100` | average level of [centered percentile rank] over the last 100 points, raw running value |
| `rank_centered_mean_w100_vs_ref` | average level of [centered percentile rank] over the last 100 points, difference vs the reference-segment value |
| `rollmean16_mean_w100` | average level of [local 16-point mean] over the last 100 points, raw running value |
| `rollmean16_mean_w100_vs_ref` | average level of [local 16-point mean] over the last 100 points, difference vs the reference-segment value |
| `rollstd16_mean_w100` | average level of [local 16-point volatility] over the last 100 points, raw running value |
| `rollstd16_mean_w100_vs_ref` | average level of [local 16-point volatility] over the last 100 points, difference vs the reference-segment value |
| `z_skew_w100` | lopsidedness (skew) of [standardized value] over the last 100 points, raw running value |
| `z_skew_w100_vs_ref` | lopsidedness (skew) of [standardized value] over the last 100 points, difference vs the reference-segment value |
| `abs_skew_w100` | lopsidedness (skew) of [magnitude |z|] over the last 100 points, raw running value |
| `abs_skew_w100_vs_ref` | lopsidedness (skew) of [magnitude |z|] over the last 100 points, difference vs the reference-segment value |
| `sign_skew_w100` | lopsidedness (skew) of [sign of z] over the last 100 points, raw running value |
| `sign_skew_w100_vs_ref` | lopsidedness (skew) of [sign of z] over the last 100 points, difference vs the reference-segment value |
| `square_skew_w100` | lopsidedness (skew) of [squared value z^2] over the last 100 points, raw running value |
| `square_skew_w100_vs_ref` | lopsidedness (skew) of [squared value z^2] over the last 100 points, difference vs the reference-segment value |
| `logabs_skew_w100` | lopsidedness (skew) of [log-magnitude] over the last 100 points, raw running value |
| `logabs_skew_w100_vs_ref` | lopsidedness (skew) of [log-magnitude] over the last 100 points, difference vs the reference-segment value |
| `cumsum_skew_w100` | lopsidedness (skew) of [running sum] over the last 100 points, raw running value |
| `cumsum_skew_w100_vs_ref` | lopsidedness (skew) of [running sum] over the last 100 points, difference vs the reference-segment value |
| `diff_skew_w100` | lopsidedness (skew) of [step-to-step change] over the last 100 points, raw running value |
| `diff_skew_w100_vs_ref` | lopsidedness (skew) of [step-to-step change] over the last 100 points, difference vs the reference-segment value |
| `diff2_skew_w100` | lopsidedness (skew) of [curvature (2nd difference)] over the last 100 points, raw running value |
| `diff2_skew_w100_vs_ref` | lopsidedness (skew) of [curvature (2nd difference)] over the last 100 points, difference vs the reference-segment value |
| `rank_skew_w100` | lopsidedness (skew) of [percentile rank vs history] over the last 100 points, raw running value |
| `rank_skew_w100_vs_ref` | lopsidedness (skew) of [percentile rank vs history] over the last 100 points, difference vs the reference-segment value |
| `rank_centered_skew_w100` | lopsidedness (skew) of [centered percentile rank] over the last 100 points, raw running value |
| `rank_centered_skew_w100_vs_ref` | lopsidedness (skew) of [centered percentile rank] over the last 100 points, difference vs the reference-segment value |
| `rollmean16_skew_w100` | lopsidedness (skew) of [local 16-point mean] over the last 100 points, raw running value |
| `rollmean16_skew_w100_vs_ref` | lopsidedness (skew) of [local 16-point mean] over the last 100 points, difference vs the reference-segment value |
| `rollstd16_skew_w100` | lopsidedness (skew) of [local 16-point volatility] over the last 100 points, raw running value |
| `rollstd16_skew_w100_vs_ref` | lopsidedness (skew) of [local 16-point volatility] over the last 100 points, difference vs the reference-segment value |
| `z_kurt_w100` | heavy-tailedness (kurtosis) of [standardized value] over the last 100 points, raw running value |
| `z_kurt_w100_vs_ref` | heavy-tailedness (kurtosis) of [standardized value] over the last 100 points, difference vs the reference-segment value |
| `abs_kurt_w100` | heavy-tailedness (kurtosis) of [magnitude |z|] over the last 100 points, raw running value |
| `abs_kurt_w100_vs_ref` | heavy-tailedness (kurtosis) of [magnitude |z|] over the last 100 points, difference vs the reference-segment value |
| `sign_kurt_w100` | heavy-tailedness (kurtosis) of [sign of z] over the last 100 points, raw running value |
| `sign_kurt_w100_vs_ref` | heavy-tailedness (kurtosis) of [sign of z] over the last 100 points, difference vs the reference-segment value |
| `square_kurt_w100` | heavy-tailedness (kurtosis) of [squared value z^2] over the last 100 points, raw running value |
| `square_kurt_w100_vs_ref` | heavy-tailedness (kurtosis) of [squared value z^2] over the last 100 points, difference vs the reference-segment value |
| `logabs_kurt_w100` | heavy-tailedness (kurtosis) of [log-magnitude] over the last 100 points, raw running value |
| `logabs_kurt_w100_vs_ref` | heavy-tailedness (kurtosis) of [log-magnitude] over the last 100 points, difference vs the reference-segment value |
| `cumsum_kurt_w100` | heavy-tailedness (kurtosis) of [running sum] over the last 100 points, raw running value |
| `cumsum_kurt_w100_vs_ref` | heavy-tailedness (kurtosis) of [running sum] over the last 100 points, difference vs the reference-segment value |
| `diff_kurt_w100` | heavy-tailedness (kurtosis) of [step-to-step change] over the last 100 points, raw running value |
| `diff_kurt_w100_vs_ref` | heavy-tailedness (kurtosis) of [step-to-step change] over the last 100 points, difference vs the reference-segment value |
| `diff2_kurt_w100` | heavy-tailedness (kurtosis) of [curvature (2nd difference)] over the last 100 points, raw running value |
| `diff2_kurt_w100_vs_ref` | heavy-tailedness (kurtosis) of [curvature (2nd difference)] over the last 100 points, difference vs the reference-segment value |
| `rank_kurt_w100` | heavy-tailedness (kurtosis) of [percentile rank vs history] over the last 100 points, raw running value |
| `rank_kurt_w100_vs_ref` | heavy-tailedness (kurtosis) of [percentile rank vs history] over the last 100 points, difference vs the reference-segment value |
| `rank_centered_kurt_w100` | heavy-tailedness (kurtosis) of [centered percentile rank] over the last 100 points, raw running value |
| `rank_centered_kurt_w100_vs_ref` | heavy-tailedness (kurtosis) of [centered percentile rank] over the last 100 points, difference vs the reference-segment value |
| `rollmean16_kurt_w100` | heavy-tailedness (kurtosis) of [local 16-point mean] over the last 100 points, raw running value |
| `rollmean16_kurt_w100_vs_ref` | heavy-tailedness (kurtosis) of [local 16-point mean] over the last 100 points, difference vs the reference-segment value |
| `rollstd16_kurt_w100` | heavy-tailedness (kurtosis) of [local 16-point volatility] over the last 100 points, raw running value |
| `rollstd16_kurt_w100_vs_ref` | heavy-tailedness (kurtosis) of [local 16-point volatility] over the last 100 points, difference vs the reference-segment value |
| `z_mean_w200` | average level of [standardized value] over the last 200 points, raw running value |
| `z_mean_w200_vs_ref` | average level of [standardized value] over the last 200 points, difference vs the reference-segment value |
| `abs_mean_w200` | average level of [magnitude |z|] over the last 200 points, raw running value |
| `abs_mean_w200_vs_ref` | average level of [magnitude |z|] over the last 200 points, difference vs the reference-segment value |
| `sign_mean_w200` | average level of [sign of z] over the last 200 points, raw running value |
| `sign_mean_w200_vs_ref` | average level of [sign of z] over the last 200 points, difference vs the reference-segment value |
| `square_mean_w200` | average level of [squared value z^2] over the last 200 points, raw running value |
| `square_mean_w200_vs_ref` | average level of [squared value z^2] over the last 200 points, difference vs the reference-segment value |
| `logabs_mean_w200` | average level of [log-magnitude] over the last 200 points, raw running value |
| `logabs_mean_w200_vs_ref` | average level of [log-magnitude] over the last 200 points, difference vs the reference-segment value |
| `cumsum_mean_w200` | average level of [running sum] over the last 200 points, raw running value |
| `cumsum_mean_w200_vs_ref` | average level of [running sum] over the last 200 points, difference vs the reference-segment value |
| `diff_mean_w200` | average level of [step-to-step change] over the last 200 points, raw running value |
| `diff_mean_w200_vs_ref` | average level of [step-to-step change] over the last 200 points, difference vs the reference-segment value |
| `diff2_mean_w200` | average level of [curvature (2nd difference)] over the last 200 points, raw running value |
| `diff2_mean_w200_vs_ref` | average level of [curvature (2nd difference)] over the last 200 points, difference vs the reference-segment value |
| `rank_mean_w200` | average level of [percentile rank vs history] over the last 200 points, raw running value |
| `rank_mean_w200_vs_ref` | average level of [percentile rank vs history] over the last 200 points, difference vs the reference-segment value |
| `rank_centered_mean_w200` | average level of [centered percentile rank] over the last 200 points, raw running value |
| `rank_centered_mean_w200_vs_ref` | average level of [centered percentile rank] over the last 200 points, difference vs the reference-segment value |
| `rollmean16_mean_w200` | average level of [local 16-point mean] over the last 200 points, raw running value |
| `rollmean16_mean_w200_vs_ref` | average level of [local 16-point mean] over the last 200 points, difference vs the reference-segment value |
| `rollstd16_mean_w200` | average level of [local 16-point volatility] over the last 200 points, raw running value |
| `rollstd16_mean_w200_vs_ref` | average level of [local 16-point volatility] over the last 200 points, difference vs the reference-segment value |
| `z_skew_w200` | lopsidedness (skew) of [standardized value] over the last 200 points, raw running value |
| `z_skew_w200_vs_ref` | lopsidedness (skew) of [standardized value] over the last 200 points, difference vs the reference-segment value |
| `abs_skew_w200` | lopsidedness (skew) of [magnitude |z|] over the last 200 points, raw running value |
| `abs_skew_w200_vs_ref` | lopsidedness (skew) of [magnitude |z|] over the last 200 points, difference vs the reference-segment value |
| `sign_skew_w200` | lopsidedness (skew) of [sign of z] over the last 200 points, raw running value |
| `sign_skew_w200_vs_ref` | lopsidedness (skew) of [sign of z] over the last 200 points, difference vs the reference-segment value |
| `square_skew_w200` | lopsidedness (skew) of [squared value z^2] over the last 200 points, raw running value |
| `square_skew_w200_vs_ref` | lopsidedness (skew) of [squared value z^2] over the last 200 points, difference vs the reference-segment value |
| `logabs_skew_w200` | lopsidedness (skew) of [log-magnitude] over the last 200 points, raw running value |
| `logabs_skew_w200_vs_ref` | lopsidedness (skew) of [log-magnitude] over the last 200 points, difference vs the reference-segment value |
| `cumsum_skew_w200` | lopsidedness (skew) of [running sum] over the last 200 points, raw running value |
| `cumsum_skew_w200_vs_ref` | lopsidedness (skew) of [running sum] over the last 200 points, difference vs the reference-segment value |
| `diff_skew_w200` | lopsidedness (skew) of [step-to-step change] over the last 200 points, raw running value |
| `diff_skew_w200_vs_ref` | lopsidedness (skew) of [step-to-step change] over the last 200 points, difference vs the reference-segment value |
| `diff2_skew_w200` | lopsidedness (skew) of [curvature (2nd difference)] over the last 200 points, raw running value |
| `diff2_skew_w200_vs_ref` | lopsidedness (skew) of [curvature (2nd difference)] over the last 200 points, difference vs the reference-segment value |
| `rank_skew_w200` | lopsidedness (skew) of [percentile rank vs history] over the last 200 points, raw running value |
| `rank_skew_w200_vs_ref` | lopsidedness (skew) of [percentile rank vs history] over the last 200 points, difference vs the reference-segment value |
| `rank_centered_skew_w200` | lopsidedness (skew) of [centered percentile rank] over the last 200 points, raw running value |
| `rank_centered_skew_w200_vs_ref` | lopsidedness (skew) of [centered percentile rank] over the last 200 points, difference vs the reference-segment value |
| `rollmean16_skew_w200` | lopsidedness (skew) of [local 16-point mean] over the last 200 points, raw running value |
| `rollmean16_skew_w200_vs_ref` | lopsidedness (skew) of [local 16-point mean] over the last 200 points, difference vs the reference-segment value |
| `rollstd16_skew_w200` | lopsidedness (skew) of [local 16-point volatility] over the last 200 points, raw running value |
| `rollstd16_skew_w200_vs_ref` | lopsidedness (skew) of [local 16-point volatility] over the last 200 points, difference vs the reference-segment value |
| `z_kurt_w200` | heavy-tailedness (kurtosis) of [standardized value] over the last 200 points, raw running value |
| `z_kurt_w200_vs_ref` | heavy-tailedness (kurtosis) of [standardized value] over the last 200 points, difference vs the reference-segment value |
| `abs_kurt_w200` | heavy-tailedness (kurtosis) of [magnitude |z|] over the last 200 points, raw running value |
| `abs_kurt_w200_vs_ref` | heavy-tailedness (kurtosis) of [magnitude |z|] over the last 200 points, difference vs the reference-segment value |
| `sign_kurt_w200` | heavy-tailedness (kurtosis) of [sign of z] over the last 200 points, raw running value |
| `sign_kurt_w200_vs_ref` | heavy-tailedness (kurtosis) of [sign of z] over the last 200 points, difference vs the reference-segment value |
| `square_kurt_w200` | heavy-tailedness (kurtosis) of [squared value z^2] over the last 200 points, raw running value |
| `square_kurt_w200_vs_ref` | heavy-tailedness (kurtosis) of [squared value z^2] over the last 200 points, difference vs the reference-segment value |
| `logabs_kurt_w200` | heavy-tailedness (kurtosis) of [log-magnitude] over the last 200 points, raw running value |
| `logabs_kurt_w200_vs_ref` | heavy-tailedness (kurtosis) of [log-magnitude] over the last 200 points, difference vs the reference-segment value |
| `cumsum_kurt_w200` | heavy-tailedness (kurtosis) of [running sum] over the last 200 points, raw running value |
| `cumsum_kurt_w200_vs_ref` | heavy-tailedness (kurtosis) of [running sum] over the last 200 points, difference vs the reference-segment value |
| `diff_kurt_w200` | heavy-tailedness (kurtosis) of [step-to-step change] over the last 200 points, raw running value |
| `diff_kurt_w200_vs_ref` | heavy-tailedness (kurtosis) of [step-to-step change] over the last 200 points, difference vs the reference-segment value |
| `diff2_kurt_w200` | heavy-tailedness (kurtosis) of [curvature (2nd difference)] over the last 200 points, raw running value |
| `diff2_kurt_w200_vs_ref` | heavy-tailedness (kurtosis) of [curvature (2nd difference)] over the last 200 points, difference vs the reference-segment value |
| `rank_kurt_w200` | heavy-tailedness (kurtosis) of [percentile rank vs history] over the last 200 points, raw running value |
| `rank_kurt_w200_vs_ref` | heavy-tailedness (kurtosis) of [percentile rank vs history] over the last 200 points, difference vs the reference-segment value |
| `rank_centered_kurt_w200` | heavy-tailedness (kurtosis) of [centered percentile rank] over the last 200 points, raw running value |
| `rank_centered_kurt_w200_vs_ref` | heavy-tailedness (kurtosis) of [centered percentile rank] over the last 200 points, difference vs the reference-segment value |
| `rollmean16_kurt_w200` | heavy-tailedness (kurtosis) of [local 16-point mean] over the last 200 points, raw running value |
| `rollmean16_kurt_w200_vs_ref` | heavy-tailedness (kurtosis) of [local 16-point mean] over the last 200 points, difference vs the reference-segment value |
| `rollstd16_kurt_w200` | heavy-tailedness (kurtosis) of [local 16-point volatility] over the last 200 points, raw running value |
| `rollstd16_kurt_w200_vs_ref` | heavy-tailedness (kurtosis) of [local 16-point volatility] over the last 200 points, difference vs the reference-segment value |
| `z_mean_w500` | average level of [standardized value] over the last 500 points, raw running value |
| `z_mean_w500_vs_ref` | average level of [standardized value] over the last 500 points, difference vs the reference-segment value |
| `abs_mean_w500` | average level of [magnitude |z|] over the last 500 points, raw running value |
| `abs_mean_w500_vs_ref` | average level of [magnitude |z|] over the last 500 points, difference vs the reference-segment value |
| `sign_mean_w500` | average level of [sign of z] over the last 500 points, raw running value |
| `sign_mean_w500_vs_ref` | average level of [sign of z] over the last 500 points, difference vs the reference-segment value |
| `square_mean_w500` | average level of [squared value z^2] over the last 500 points, raw running value |
| `square_mean_w500_vs_ref` | average level of [squared value z^2] over the last 500 points, difference vs the reference-segment value |
| `logabs_mean_w500` | average level of [log-magnitude] over the last 500 points, raw running value |
| `logabs_mean_w500_vs_ref` | average level of [log-magnitude] over the last 500 points, difference vs the reference-segment value |
| `cumsum_mean_w500` | average level of [running sum] over the last 500 points, raw running value |
| `cumsum_mean_w500_vs_ref` | average level of [running sum] over the last 500 points, difference vs the reference-segment value |
| `diff_mean_w500` | average level of [step-to-step change] over the last 500 points, raw running value |
| `diff_mean_w500_vs_ref` | average level of [step-to-step change] over the last 500 points, difference vs the reference-segment value |
| `diff2_mean_w500` | average level of [curvature (2nd difference)] over the last 500 points, raw running value |
| `diff2_mean_w500_vs_ref` | average level of [curvature (2nd difference)] over the last 500 points, difference vs the reference-segment value |
| `rank_mean_w500` | average level of [percentile rank vs history] over the last 500 points, raw running value |
| `rank_mean_w500_vs_ref` | average level of [percentile rank vs history] over the last 500 points, difference vs the reference-segment value |
| `rank_centered_mean_w500` | average level of [centered percentile rank] over the last 500 points, raw running value |
| `rank_centered_mean_w500_vs_ref` | average level of [centered percentile rank] over the last 500 points, difference vs the reference-segment value |
| `rollmean16_mean_w500` | average level of [local 16-point mean] over the last 500 points, raw running value |
| `rollmean16_mean_w500_vs_ref` | average level of [local 16-point mean] over the last 500 points, difference vs the reference-segment value |
| `rollstd16_mean_w500` | average level of [local 16-point volatility] over the last 500 points, raw running value |
| `rollstd16_mean_w500_vs_ref` | average level of [local 16-point volatility] over the last 500 points, difference vs the reference-segment value |
| `z_skew_w500` | lopsidedness (skew) of [standardized value] over the last 500 points, raw running value |
| `z_skew_w500_vs_ref` | lopsidedness (skew) of [standardized value] over the last 500 points, difference vs the reference-segment value |
| `abs_skew_w500` | lopsidedness (skew) of [magnitude |z|] over the last 500 points, raw running value |
| `abs_skew_w500_vs_ref` | lopsidedness (skew) of [magnitude |z|] over the last 500 points, difference vs the reference-segment value |
| `sign_skew_w500` | lopsidedness (skew) of [sign of z] over the last 500 points, raw running value |
| `sign_skew_w500_vs_ref` | lopsidedness (skew) of [sign of z] over the last 500 points, difference vs the reference-segment value |
| `square_skew_w500` | lopsidedness (skew) of [squared value z^2] over the last 500 points, raw running value |
| `square_skew_w500_vs_ref` | lopsidedness (skew) of [squared value z^2] over the last 500 points, difference vs the reference-segment value |
| `logabs_skew_w500` | lopsidedness (skew) of [log-magnitude] over the last 500 points, raw running value |
| `logabs_skew_w500_vs_ref` | lopsidedness (skew) of [log-magnitude] over the last 500 points, difference vs the reference-segment value |
| `cumsum_skew_w500` | lopsidedness (skew) of [running sum] over the last 500 points, raw running value |
| `cumsum_skew_w500_vs_ref` | lopsidedness (skew) of [running sum] over the last 500 points, difference vs the reference-segment value |
| `diff_skew_w500` | lopsidedness (skew) of [step-to-step change] over the last 500 points, raw running value |
| `diff_skew_w500_vs_ref` | lopsidedness (skew) of [step-to-step change] over the last 500 points, difference vs the reference-segment value |
| `diff2_skew_w500` | lopsidedness (skew) of [curvature (2nd difference)] over the last 500 points, raw running value |
| `diff2_skew_w500_vs_ref` | lopsidedness (skew) of [curvature (2nd difference)] over the last 500 points, difference vs the reference-segment value |
| `rank_skew_w500` | lopsidedness (skew) of [percentile rank vs history] over the last 500 points, raw running value |
| `rank_skew_w500_vs_ref` | lopsidedness (skew) of [percentile rank vs history] over the last 500 points, difference vs the reference-segment value |
| `rank_centered_skew_w500` | lopsidedness (skew) of [centered percentile rank] over the last 500 points, raw running value |
| `rank_centered_skew_w500_vs_ref` | lopsidedness (skew) of [centered percentile rank] over the last 500 points, difference vs the reference-segment value |
| `rollmean16_skew_w500` | lopsidedness (skew) of [local 16-point mean] over the last 500 points, raw running value |
| `rollmean16_skew_w500_vs_ref` | lopsidedness (skew) of [local 16-point mean] over the last 500 points, difference vs the reference-segment value |
| `rollstd16_skew_w500` | lopsidedness (skew) of [local 16-point volatility] over the last 500 points, raw running value |
| `rollstd16_skew_w500_vs_ref` | lopsidedness (skew) of [local 16-point volatility] over the last 500 points, difference vs the reference-segment value |
| `z_kurt_w500` | heavy-tailedness (kurtosis) of [standardized value] over the last 500 points, raw running value |
| `z_kurt_w500_vs_ref` | heavy-tailedness (kurtosis) of [standardized value] over the last 500 points, difference vs the reference-segment value |
| `abs_kurt_w500` | heavy-tailedness (kurtosis) of [magnitude |z|] over the last 500 points, raw running value |
| `abs_kurt_w500_vs_ref` | heavy-tailedness (kurtosis) of [magnitude |z|] over the last 500 points, difference vs the reference-segment value |
| `sign_kurt_w500` | heavy-tailedness (kurtosis) of [sign of z] over the last 500 points, raw running value |
| `sign_kurt_w500_vs_ref` | heavy-tailedness (kurtosis) of [sign of z] over the last 500 points, difference vs the reference-segment value |
| `square_kurt_w500` | heavy-tailedness (kurtosis) of [squared value z^2] over the last 500 points, raw running value |
| `square_kurt_w500_vs_ref` | heavy-tailedness (kurtosis) of [squared value z^2] over the last 500 points, difference vs the reference-segment value |
| `logabs_kurt_w500` | heavy-tailedness (kurtosis) of [log-magnitude] over the last 500 points, raw running value |
| `logabs_kurt_w500_vs_ref` | heavy-tailedness (kurtosis) of [log-magnitude] over the last 500 points, difference vs the reference-segment value |
| `cumsum_kurt_w500` | heavy-tailedness (kurtosis) of [running sum] over the last 500 points, raw running value |
| `cumsum_kurt_w500_vs_ref` | heavy-tailedness (kurtosis) of [running sum] over the last 500 points, difference vs the reference-segment value |
| `diff_kurt_w500` | heavy-tailedness (kurtosis) of [step-to-step change] over the last 500 points, raw running value |
| `diff_kurt_w500_vs_ref` | heavy-tailedness (kurtosis) of [step-to-step change] over the last 500 points, difference vs the reference-segment value |
| `diff2_kurt_w500` | heavy-tailedness (kurtosis) of [curvature (2nd difference)] over the last 500 points, raw running value |
| `diff2_kurt_w500_vs_ref` | heavy-tailedness (kurtosis) of [curvature (2nd difference)] over the last 500 points, difference vs the reference-segment value |
| `rank_kurt_w500` | heavy-tailedness (kurtosis) of [percentile rank vs history] over the last 500 points, raw running value |
| `rank_kurt_w500_vs_ref` | heavy-tailedness (kurtosis) of [percentile rank vs history] over the last 500 points, difference vs the reference-segment value |
| `rank_centered_kurt_w500` | heavy-tailedness (kurtosis) of [centered percentile rank] over the last 500 points, raw running value |
| `rank_centered_kurt_w500_vs_ref` | heavy-tailedness (kurtosis) of [centered percentile rank] over the last 500 points, difference vs the reference-segment value |
| `rollmean16_kurt_w500` | heavy-tailedness (kurtosis) of [local 16-point mean] over the last 500 points, raw running value |
| `rollmean16_kurt_w500_vs_ref` | heavy-tailedness (kurtosis) of [local 16-point mean] over the last 500 points, difference vs the reference-segment value |
| `rollstd16_kurt_w500` | heavy-tailedness (kurtosis) of [local 16-point volatility] over the last 500 points, raw running value |
| `rollstd16_kurt_w500_vs_ref` | heavy-tailedness (kurtosis) of [local 16-point volatility] over the last 500 points, difference vs the reference-segment value |

## distributional (72 features)

| feature | meaning |
|---|---|
| `z_q25` | 25th percentile of [standardized value] (expanding), raw running value |
| `z_q25_vs_ref` | 25th percentile of [standardized value] (expanding), difference vs the reference-segment value |
| `z_q25_ratio` | 25th percentile of [standardized value] (expanding), ratio to the reference-segment value |
| `abs_q25` | 25th percentile of [magnitude |z|] (expanding), raw running value |
| `abs_q25_vs_ref` | 25th percentile of [magnitude |z|] (expanding), difference vs the reference-segment value |
| `abs_q25_ratio` | 25th percentile of [magnitude |z|] (expanding), ratio to the reference-segment value |
| `sign_q25` | 25th percentile of [sign of z] (expanding), raw running value |
| `sign_q25_vs_ref` | 25th percentile of [sign of z] (expanding), difference vs the reference-segment value |
| `sign_q25_ratio` | 25th percentile of [sign of z] (expanding), ratio to the reference-segment value |
| `square_q25` | 25th percentile of [squared value z^2] (expanding), raw running value |
| `square_q25_vs_ref` | 25th percentile of [squared value z^2] (expanding), difference vs the reference-segment value |
| `square_q25_ratio` | 25th percentile of [squared value z^2] (expanding), ratio to the reference-segment value |
| `logabs_q25` | 25th percentile of [log-magnitude] (expanding), raw running value |
| `logabs_q25_vs_ref` | 25th percentile of [log-magnitude] (expanding), difference vs the reference-segment value |
| `logabs_q25_ratio` | 25th percentile of [log-magnitude] (expanding), ratio to the reference-segment value |
| `cumsum_q25` | 25th percentile of [running sum] (expanding), raw running value |
| `cumsum_q25_vs_ref` | 25th percentile of [running sum] (expanding), difference vs the reference-segment value |
| `cumsum_q25_ratio` | 25th percentile of [running sum] (expanding), ratio to the reference-segment value |
| `diff_q25` | 25th percentile of [step-to-step change] (expanding), raw running value |
| `diff_q25_vs_ref` | 25th percentile of [step-to-step change] (expanding), difference vs the reference-segment value |
| `diff_q25_ratio` | 25th percentile of [step-to-step change] (expanding), ratio to the reference-segment value |
| `diff2_q25` | 25th percentile of [curvature (2nd difference)] (expanding), raw running value |
| `diff2_q25_vs_ref` | 25th percentile of [curvature (2nd difference)] (expanding), difference vs the reference-segment value |
| `diff2_q25_ratio` | 25th percentile of [curvature (2nd difference)] (expanding), ratio to the reference-segment value |
| `rank_q25` | 25th percentile of [percentile rank vs history] (expanding), raw running value |
| `rank_q25_vs_ref` | 25th percentile of [percentile rank vs history] (expanding), difference vs the reference-segment value |
| `rank_q25_ratio` | 25th percentile of [percentile rank vs history] (expanding), ratio to the reference-segment value |
| `rank_centered_q25` | 25th percentile of [centered percentile rank] (expanding), raw running value |
| `rank_centered_q25_vs_ref` | 25th percentile of [centered percentile rank] (expanding), difference vs the reference-segment value |
| `rank_centered_q25_ratio` | 25th percentile of [centered percentile rank] (expanding), ratio to the reference-segment value |
| `rollmean16_q25` | 25th percentile of [local 16-point mean] (expanding), raw running value |
| `rollmean16_q25_vs_ref` | 25th percentile of [local 16-point mean] (expanding), difference vs the reference-segment value |
| `rollmean16_q25_ratio` | 25th percentile of [local 16-point mean] (expanding), ratio to the reference-segment value |
| `rollstd16_q25` | 25th percentile of [local 16-point volatility] (expanding), raw running value |
| `rollstd16_q25_vs_ref` | 25th percentile of [local 16-point volatility] (expanding), difference vs the reference-segment value |
| `rollstd16_q25_ratio` | 25th percentile of [local 16-point volatility] (expanding), ratio to the reference-segment value |
| `z_q75` | 75th percentile of [standardized value] (expanding), raw running value |
| `z_q75_vs_ref` | 75th percentile of [standardized value] (expanding), difference vs the reference-segment value |
| `z_q75_ratio` | 75th percentile of [standardized value] (expanding), ratio to the reference-segment value |
| `abs_q75` | 75th percentile of [magnitude |z|] (expanding), raw running value |
| `abs_q75_vs_ref` | 75th percentile of [magnitude |z|] (expanding), difference vs the reference-segment value |
| `abs_q75_ratio` | 75th percentile of [magnitude |z|] (expanding), ratio to the reference-segment value |
| `sign_q75` | 75th percentile of [sign of z] (expanding), raw running value |
| `sign_q75_vs_ref` | 75th percentile of [sign of z] (expanding), difference vs the reference-segment value |
| `sign_q75_ratio` | 75th percentile of [sign of z] (expanding), ratio to the reference-segment value |
| `square_q75` | 75th percentile of [squared value z^2] (expanding), raw running value |
| `square_q75_vs_ref` | 75th percentile of [squared value z^2] (expanding), difference vs the reference-segment value |
| `square_q75_ratio` | 75th percentile of [squared value z^2] (expanding), ratio to the reference-segment value |
| `logabs_q75` | 75th percentile of [log-magnitude] (expanding), raw running value |
| `logabs_q75_vs_ref` | 75th percentile of [log-magnitude] (expanding), difference vs the reference-segment value |
| `logabs_q75_ratio` | 75th percentile of [log-magnitude] (expanding), ratio to the reference-segment value |
| `cumsum_q75` | 75th percentile of [running sum] (expanding), raw running value |
| `cumsum_q75_vs_ref` | 75th percentile of [running sum] (expanding), difference vs the reference-segment value |
| `cumsum_q75_ratio` | 75th percentile of [running sum] (expanding), ratio to the reference-segment value |
| `diff_q75` | 75th percentile of [step-to-step change] (expanding), raw running value |
| `diff_q75_vs_ref` | 75th percentile of [step-to-step change] (expanding), difference vs the reference-segment value |
| `diff_q75_ratio` | 75th percentile of [step-to-step change] (expanding), ratio to the reference-segment value |
| `diff2_q75` | 75th percentile of [curvature (2nd difference)] (expanding), raw running value |
| `diff2_q75_vs_ref` | 75th percentile of [curvature (2nd difference)] (expanding), difference vs the reference-segment value |
| `diff2_q75_ratio` | 75th percentile of [curvature (2nd difference)] (expanding), ratio to the reference-segment value |
| `rank_q75` | 75th percentile of [percentile rank vs history] (expanding), raw running value |
| `rank_q75_vs_ref` | 75th percentile of [percentile rank vs history] (expanding), difference vs the reference-segment value |
| `rank_q75_ratio` | 75th percentile of [percentile rank vs history] (expanding), ratio to the reference-segment value |
| `rank_centered_q75` | 75th percentile of [centered percentile rank] (expanding), raw running value |
| `rank_centered_q75_vs_ref` | 75th percentile of [centered percentile rank] (expanding), difference vs the reference-segment value |
| `rank_centered_q75_ratio` | 75th percentile of [centered percentile rank] (expanding), ratio to the reference-segment value |
| `rollmean16_q75` | 75th percentile of [local 16-point mean] (expanding), raw running value |
| `rollmean16_q75_vs_ref` | 75th percentile of [local 16-point mean] (expanding), difference vs the reference-segment value |
| `rollmean16_q75_ratio` | 75th percentile of [local 16-point mean] (expanding), ratio to the reference-segment value |
| `rollstd16_q75` | 75th percentile of [local 16-point volatility] (expanding), raw running value |
| `rollstd16_q75_vs_ref` | 75th percentile of [local 16-point volatility] (expanding), difference vs the reference-segment value |
| `rollstd16_q75_ratio` | 75th percentile of [local 16-point volatility] (expanding), ratio to the reference-segment value |

## dependence (36 features)

| feature | meaning |
|---|---|
| `z_autocorr` | lag-1 autocorrelation of [standardized value] (expanding), raw running value |
| `z_autocorr_vs_ref` | lag-1 autocorrelation of [standardized value] (expanding), difference vs the reference-segment value |
| `z_autocorr_ratio` | lag-1 autocorrelation of [standardized value] (expanding), ratio to the reference-segment value |
| `abs_autocorr` | lag-1 autocorrelation of [magnitude |z|] (expanding), raw running value |
| `abs_autocorr_vs_ref` | lag-1 autocorrelation of [magnitude |z|] (expanding), difference vs the reference-segment value |
| `abs_autocorr_ratio` | lag-1 autocorrelation of [magnitude |z|] (expanding), ratio to the reference-segment value |
| `sign_autocorr` | lag-1 autocorrelation of [sign of z] (expanding), raw running value |
| `sign_autocorr_vs_ref` | lag-1 autocorrelation of [sign of z] (expanding), difference vs the reference-segment value |
| `sign_autocorr_ratio` | lag-1 autocorrelation of [sign of z] (expanding), ratio to the reference-segment value |
| `square_autocorr` | lag-1 autocorrelation of [squared value z^2] (expanding), raw running value |
| `square_autocorr_vs_ref` | lag-1 autocorrelation of [squared value z^2] (expanding), difference vs the reference-segment value |
| `square_autocorr_ratio` | lag-1 autocorrelation of [squared value z^2] (expanding), ratio to the reference-segment value |
| `logabs_autocorr` | lag-1 autocorrelation of [log-magnitude] (expanding), raw running value |
| `logabs_autocorr_vs_ref` | lag-1 autocorrelation of [log-magnitude] (expanding), difference vs the reference-segment value |
| `logabs_autocorr_ratio` | lag-1 autocorrelation of [log-magnitude] (expanding), ratio to the reference-segment value |
| `cumsum_autocorr` | lag-1 autocorrelation of [running sum] (expanding), raw running value |
| `cumsum_autocorr_vs_ref` | lag-1 autocorrelation of [running sum] (expanding), difference vs the reference-segment value |
| `cumsum_autocorr_ratio` | lag-1 autocorrelation of [running sum] (expanding), ratio to the reference-segment value |
| `diff_autocorr` | lag-1 autocorrelation of [step-to-step change] (expanding), raw running value |
| `diff_autocorr_vs_ref` | lag-1 autocorrelation of [step-to-step change] (expanding), difference vs the reference-segment value |
| `diff_autocorr_ratio` | lag-1 autocorrelation of [step-to-step change] (expanding), ratio to the reference-segment value |
| `diff2_autocorr` | lag-1 autocorrelation of [curvature (2nd difference)] (expanding), raw running value |
| `diff2_autocorr_vs_ref` | lag-1 autocorrelation of [curvature (2nd difference)] (expanding), difference vs the reference-segment value |
| `diff2_autocorr_ratio` | lag-1 autocorrelation of [curvature (2nd difference)] (expanding), ratio to the reference-segment value |
| `rank_autocorr` | lag-1 autocorrelation of [percentile rank vs history] (expanding), raw running value |
| `rank_autocorr_vs_ref` | lag-1 autocorrelation of [percentile rank vs history] (expanding), difference vs the reference-segment value |
| `rank_autocorr_ratio` | lag-1 autocorrelation of [percentile rank vs history] (expanding), ratio to the reference-segment value |
| `rank_centered_autocorr` | lag-1 autocorrelation of [centered percentile rank] (expanding), raw running value |
| `rank_centered_autocorr_vs_ref` | lag-1 autocorrelation of [centered percentile rank] (expanding), difference vs the reference-segment value |
| `rank_centered_autocorr_ratio` | lag-1 autocorrelation of [centered percentile rank] (expanding), ratio to the reference-segment value |
| `rollmean16_autocorr` | lag-1 autocorrelation of [local 16-point mean] (expanding), raw running value |
| `rollmean16_autocorr_vs_ref` | lag-1 autocorrelation of [local 16-point mean] (expanding), difference vs the reference-segment value |
| `rollmean16_autocorr_ratio` | lag-1 autocorrelation of [local 16-point mean] (expanding), ratio to the reference-segment value |
| `rollstd16_autocorr` | lag-1 autocorrelation of [local 16-point volatility] (expanding), raw running value |
| `rollstd16_autocorr_vs_ref` | lag-1 autocorrelation of [local 16-point volatility] (expanding), difference vs the reference-segment value |
| `rollstd16_autocorr_ratio` | lag-1 autocorrelation of [local 16-point volatility] (expanding), ratio to the reference-segment value |

## volatility (228 features)

| feature | meaning |
|---|---|
| `z_std` | spread / std of [standardized value] (expanding), raw running value |
| `z_std_vs_ref` | spread / std of [standardized value] (expanding), difference vs the reference-segment value |
| `z_std_ratio` | spread / std of [standardized value] (expanding), ratio to the reference-segment value |
| `abs_std` | spread / std of [magnitude |z|] (expanding), raw running value |
| `abs_std_vs_ref` | spread / std of [magnitude |z|] (expanding), difference vs the reference-segment value |
| `abs_std_ratio` | spread / std of [magnitude |z|] (expanding), ratio to the reference-segment value |
| `sign_std` | spread / std of [sign of z] (expanding), raw running value |
| `sign_std_vs_ref` | spread / std of [sign of z] (expanding), difference vs the reference-segment value |
| `sign_std_ratio` | spread / std of [sign of z] (expanding), ratio to the reference-segment value |
| `square_std` | spread / std of [squared value z^2] (expanding), raw running value |
| `square_std_vs_ref` | spread / std of [squared value z^2] (expanding), difference vs the reference-segment value |
| `square_std_ratio` | spread / std of [squared value z^2] (expanding), ratio to the reference-segment value |
| `logabs_std` | spread / std of [log-magnitude] (expanding), raw running value |
| `logabs_std_vs_ref` | spread / std of [log-magnitude] (expanding), difference vs the reference-segment value |
| `logabs_std_ratio` | spread / std of [log-magnitude] (expanding), ratio to the reference-segment value |
| `cumsum_std` | spread / std of [running sum] (expanding), raw running value |
| `cumsum_std_vs_ref` | spread / std of [running sum] (expanding), difference vs the reference-segment value |
| `cumsum_std_ratio` | spread / std of [running sum] (expanding), ratio to the reference-segment value |
| `diff_std` | spread / std of [step-to-step change] (expanding), raw running value |
| `diff_std_vs_ref` | spread / std of [step-to-step change] (expanding), difference vs the reference-segment value |
| `diff_std_ratio` | spread / std of [step-to-step change] (expanding), ratio to the reference-segment value |
| `diff2_std` | spread / std of [curvature (2nd difference)] (expanding), raw running value |
| `diff2_std_vs_ref` | spread / std of [curvature (2nd difference)] (expanding), difference vs the reference-segment value |
| `diff2_std_ratio` | spread / std of [curvature (2nd difference)] (expanding), ratio to the reference-segment value |
| `rank_std` | spread / std of [percentile rank vs history] (expanding), raw running value |
| `rank_std_vs_ref` | spread / std of [percentile rank vs history] (expanding), difference vs the reference-segment value |
| `rank_std_ratio` | spread / std of [percentile rank vs history] (expanding), ratio to the reference-segment value |
| `rank_centered_std` | spread / std of [centered percentile rank] (expanding), raw running value |
| `rank_centered_std_vs_ref` | spread / std of [centered percentile rank] (expanding), difference vs the reference-segment value |
| `rank_centered_std_ratio` | spread / std of [centered percentile rank] (expanding), ratio to the reference-segment value |
| `rollmean16_std` | spread / std of [local 16-point mean] (expanding), raw running value |
| `rollmean16_std_vs_ref` | spread / std of [local 16-point mean] (expanding), difference vs the reference-segment value |
| `rollmean16_std_ratio` | spread / std of [local 16-point mean] (expanding), ratio to the reference-segment value |
| `rollstd16_std` | spread / std of [local 16-point volatility] (expanding), raw running value |
| `rollstd16_std_vs_ref` | spread / std of [local 16-point volatility] (expanding), difference vs the reference-segment value |
| `rollstd16_std_ratio` | spread / std of [local 16-point volatility] (expanding), ratio to the reference-segment value |
| `z_range` | max-min range of [standardized value] (expanding), raw running value |
| `z_range_vs_ref` | max-min range of [standardized value] (expanding), difference vs the reference-segment value |
| `z_range_ratio` | max-min range of [standardized value] (expanding), ratio to the reference-segment value |
| `abs_range` | max-min range of [magnitude |z|] (expanding), raw running value |
| `abs_range_vs_ref` | max-min range of [magnitude |z|] (expanding), difference vs the reference-segment value |
| `abs_range_ratio` | max-min range of [magnitude |z|] (expanding), ratio to the reference-segment value |
| `sign_range` | max-min range of [sign of z] (expanding), raw running value |
| `sign_range_vs_ref` | max-min range of [sign of z] (expanding), difference vs the reference-segment value |
| `sign_range_ratio` | max-min range of [sign of z] (expanding), ratio to the reference-segment value |
| `square_range` | max-min range of [squared value z^2] (expanding), raw running value |
| `square_range_vs_ref` | max-min range of [squared value z^2] (expanding), difference vs the reference-segment value |
| `square_range_ratio` | max-min range of [squared value z^2] (expanding), ratio to the reference-segment value |
| `logabs_range` | max-min range of [log-magnitude] (expanding), raw running value |
| `logabs_range_vs_ref` | max-min range of [log-magnitude] (expanding), difference vs the reference-segment value |
| `logabs_range_ratio` | max-min range of [log-magnitude] (expanding), ratio to the reference-segment value |
| `cumsum_range` | max-min range of [running sum] (expanding), raw running value |
| `cumsum_range_vs_ref` | max-min range of [running sum] (expanding), difference vs the reference-segment value |
| `cumsum_range_ratio` | max-min range of [running sum] (expanding), ratio to the reference-segment value |
| `diff_range` | max-min range of [step-to-step change] (expanding), raw running value |
| `diff_range_vs_ref` | max-min range of [step-to-step change] (expanding), difference vs the reference-segment value |
| `diff_range_ratio` | max-min range of [step-to-step change] (expanding), ratio to the reference-segment value |
| `diff2_range` | max-min range of [curvature (2nd difference)] (expanding), raw running value |
| `diff2_range_vs_ref` | max-min range of [curvature (2nd difference)] (expanding), difference vs the reference-segment value |
| `diff2_range_ratio` | max-min range of [curvature (2nd difference)] (expanding), ratio to the reference-segment value |
| `rank_range` | max-min range of [percentile rank vs history] (expanding), raw running value |
| `rank_range_vs_ref` | max-min range of [percentile rank vs history] (expanding), difference vs the reference-segment value |
| `rank_range_ratio` | max-min range of [percentile rank vs history] (expanding), ratio to the reference-segment value |
| `rank_centered_range` | max-min range of [centered percentile rank] (expanding), raw running value |
| `rank_centered_range_vs_ref` | max-min range of [centered percentile rank] (expanding), difference vs the reference-segment value |
| `rank_centered_range_ratio` | max-min range of [centered percentile rank] (expanding), ratio to the reference-segment value |
| `rollmean16_range` | max-min range of [local 16-point mean] (expanding), raw running value |
| `rollmean16_range_vs_ref` | max-min range of [local 16-point mean] (expanding), difference vs the reference-segment value |
| `rollmean16_range_ratio` | max-min range of [local 16-point mean] (expanding), ratio to the reference-segment value |
| `rollstd16_range` | max-min range of [local 16-point volatility] (expanding), raw running value |
| `rollstd16_range_vs_ref` | max-min range of [local 16-point volatility] (expanding), difference vs the reference-segment value |
| `rollstd16_range_ratio` | max-min range of [local 16-point volatility] (expanding), ratio to the reference-segment value |
| `z_cov` | coefficient of variation of [standardized value] (expanding), raw running value |
| `z_cov_vs_ref` | coefficient of variation of [standardized value] (expanding), difference vs the reference-segment value |
| `z_cov_ratio` | coefficient of variation of [standardized value] (expanding), ratio to the reference-segment value |
| `abs_cov` | coefficient of variation of [magnitude |z|] (expanding), raw running value |
| `abs_cov_vs_ref` | coefficient of variation of [magnitude |z|] (expanding), difference vs the reference-segment value |
| `abs_cov_ratio` | coefficient of variation of [magnitude |z|] (expanding), ratio to the reference-segment value |
| `sign_cov` | coefficient of variation of [sign of z] (expanding), raw running value |
| `sign_cov_vs_ref` | coefficient of variation of [sign of z] (expanding), difference vs the reference-segment value |
| `sign_cov_ratio` | coefficient of variation of [sign of z] (expanding), ratio to the reference-segment value |
| `square_cov` | coefficient of variation of [squared value z^2] (expanding), raw running value |
| `square_cov_vs_ref` | coefficient of variation of [squared value z^2] (expanding), difference vs the reference-segment value |
| `square_cov_ratio` | coefficient of variation of [squared value z^2] (expanding), ratio to the reference-segment value |
| `logabs_cov` | coefficient of variation of [log-magnitude] (expanding), raw running value |
| `logabs_cov_vs_ref` | coefficient of variation of [log-magnitude] (expanding), difference vs the reference-segment value |
| `logabs_cov_ratio` | coefficient of variation of [log-magnitude] (expanding), ratio to the reference-segment value |
| `cumsum_cov` | coefficient of variation of [running sum] (expanding), raw running value |
| `cumsum_cov_vs_ref` | coefficient of variation of [running sum] (expanding), difference vs the reference-segment value |
| `cumsum_cov_ratio` | coefficient of variation of [running sum] (expanding), ratio to the reference-segment value |
| `diff_cov` | coefficient of variation of [step-to-step change] (expanding), raw running value |
| `diff_cov_vs_ref` | coefficient of variation of [step-to-step change] (expanding), difference vs the reference-segment value |
| `diff_cov_ratio` | coefficient of variation of [step-to-step change] (expanding), ratio to the reference-segment value |
| `diff2_cov` | coefficient of variation of [curvature (2nd difference)] (expanding), raw running value |
| `diff2_cov_vs_ref` | coefficient of variation of [curvature (2nd difference)] (expanding), difference vs the reference-segment value |
| `diff2_cov_ratio` | coefficient of variation of [curvature (2nd difference)] (expanding), ratio to the reference-segment value |
| `rank_cov` | coefficient of variation of [percentile rank vs history] (expanding), raw running value |
| `rank_cov_vs_ref` | coefficient of variation of [percentile rank vs history] (expanding), difference vs the reference-segment value |
| `rank_cov_ratio` | coefficient of variation of [percentile rank vs history] (expanding), ratio to the reference-segment value |
| `rank_centered_cov` | coefficient of variation of [centered percentile rank] (expanding), raw running value |
| `rank_centered_cov_vs_ref` | coefficient of variation of [centered percentile rank] (expanding), difference vs the reference-segment value |
| `rank_centered_cov_ratio` | coefficient of variation of [centered percentile rank] (expanding), ratio to the reference-segment value |
| `rollmean16_cov` | coefficient of variation of [local 16-point mean] (expanding), raw running value |
| `rollmean16_cov_vs_ref` | coefficient of variation of [local 16-point mean] (expanding), difference vs the reference-segment value |
| `rollmean16_cov_ratio` | coefficient of variation of [local 16-point mean] (expanding), ratio to the reference-segment value |
| `rollstd16_cov` | coefficient of variation of [local 16-point volatility] (expanding), raw running value |
| `rollstd16_cov_vs_ref` | coefficient of variation of [local 16-point volatility] (expanding), difference vs the reference-segment value |
| `rollstd16_cov_ratio` | coefficient of variation of [local 16-point volatility] (expanding), ratio to the reference-segment value |
| `z_std_w20` | spread / std of [standardized value] over the last 20 points, raw running value |
| `z_std_w20_vs_ref` | spread / std of [standardized value] over the last 20 points, difference vs the reference-segment value |
| `abs_std_w20` | spread / std of [magnitude |z|] over the last 20 points, raw running value |
| `abs_std_w20_vs_ref` | spread / std of [magnitude |z|] over the last 20 points, difference vs the reference-segment value |
| `sign_std_w20` | spread / std of [sign of z] over the last 20 points, raw running value |
| `sign_std_w20_vs_ref` | spread / std of [sign of z] over the last 20 points, difference vs the reference-segment value |
| `square_std_w20` | spread / std of [squared value z^2] over the last 20 points, raw running value |
| `square_std_w20_vs_ref` | spread / std of [squared value z^2] over the last 20 points, difference vs the reference-segment value |
| `logabs_std_w20` | spread / std of [log-magnitude] over the last 20 points, raw running value |
| `logabs_std_w20_vs_ref` | spread / std of [log-magnitude] over the last 20 points, difference vs the reference-segment value |
| `cumsum_std_w20` | spread / std of [running sum] over the last 20 points, raw running value |
| `cumsum_std_w20_vs_ref` | spread / std of [running sum] over the last 20 points, difference vs the reference-segment value |
| `diff_std_w20` | spread / std of [step-to-step change] over the last 20 points, raw running value |
| `diff_std_w20_vs_ref` | spread / std of [step-to-step change] over the last 20 points, difference vs the reference-segment value |
| `diff2_std_w20` | spread / std of [curvature (2nd difference)] over the last 20 points, raw running value |
| `diff2_std_w20_vs_ref` | spread / std of [curvature (2nd difference)] over the last 20 points, difference vs the reference-segment value |
| `rank_std_w20` | spread / std of [percentile rank vs history] over the last 20 points, raw running value |
| `rank_std_w20_vs_ref` | spread / std of [percentile rank vs history] over the last 20 points, difference vs the reference-segment value |
| `rank_centered_std_w20` | spread / std of [centered percentile rank] over the last 20 points, raw running value |
| `rank_centered_std_w20_vs_ref` | spread / std of [centered percentile rank] over the last 20 points, difference vs the reference-segment value |
| `rollmean16_std_w20` | spread / std of [local 16-point mean] over the last 20 points, raw running value |
| `rollmean16_std_w20_vs_ref` | spread / std of [local 16-point mean] over the last 20 points, difference vs the reference-segment value |
| `rollstd16_std_w20` | spread / std of [local 16-point volatility] over the last 20 points, raw running value |
| `rollstd16_std_w20_vs_ref` | spread / std of [local 16-point volatility] over the last 20 points, difference vs the reference-segment value |
| `z_std_w50` | spread / std of [standardized value] over the last 50 points, raw running value |
| `z_std_w50_vs_ref` | spread / std of [standardized value] over the last 50 points, difference vs the reference-segment value |
| `abs_std_w50` | spread / std of [magnitude |z|] over the last 50 points, raw running value |
| `abs_std_w50_vs_ref` | spread / std of [magnitude |z|] over the last 50 points, difference vs the reference-segment value |
| `sign_std_w50` | spread / std of [sign of z] over the last 50 points, raw running value |
| `sign_std_w50_vs_ref` | spread / std of [sign of z] over the last 50 points, difference vs the reference-segment value |
| `square_std_w50` | spread / std of [squared value z^2] over the last 50 points, raw running value |
| `square_std_w50_vs_ref` | spread / std of [squared value z^2] over the last 50 points, difference vs the reference-segment value |
| `logabs_std_w50` | spread / std of [log-magnitude] over the last 50 points, raw running value |
| `logabs_std_w50_vs_ref` | spread / std of [log-magnitude] over the last 50 points, difference vs the reference-segment value |
| `cumsum_std_w50` | spread / std of [running sum] over the last 50 points, raw running value |
| `cumsum_std_w50_vs_ref` | spread / std of [running sum] over the last 50 points, difference vs the reference-segment value |
| `diff_std_w50` | spread / std of [step-to-step change] over the last 50 points, raw running value |
| `diff_std_w50_vs_ref` | spread / std of [step-to-step change] over the last 50 points, difference vs the reference-segment value |
| `diff2_std_w50` | spread / std of [curvature (2nd difference)] over the last 50 points, raw running value |
| `diff2_std_w50_vs_ref` | spread / std of [curvature (2nd difference)] over the last 50 points, difference vs the reference-segment value |
| `rank_std_w50` | spread / std of [percentile rank vs history] over the last 50 points, raw running value |
| `rank_std_w50_vs_ref` | spread / std of [percentile rank vs history] over the last 50 points, difference vs the reference-segment value |
| `rank_centered_std_w50` | spread / std of [centered percentile rank] over the last 50 points, raw running value |
| `rank_centered_std_w50_vs_ref` | spread / std of [centered percentile rank] over the last 50 points, difference vs the reference-segment value |
| `rollmean16_std_w50` | spread / std of [local 16-point mean] over the last 50 points, raw running value |
| `rollmean16_std_w50_vs_ref` | spread / std of [local 16-point mean] over the last 50 points, difference vs the reference-segment value |
| `rollstd16_std_w50` | spread / std of [local 16-point volatility] over the last 50 points, raw running value |
| `rollstd16_std_w50_vs_ref` | spread / std of [local 16-point volatility] over the last 50 points, difference vs the reference-segment value |
| `z_std_w100` | spread / std of [standardized value] over the last 100 points, raw running value |
| `z_std_w100_vs_ref` | spread / std of [standardized value] over the last 100 points, difference vs the reference-segment value |
| `abs_std_w100` | spread / std of [magnitude |z|] over the last 100 points, raw running value |
| `abs_std_w100_vs_ref` | spread / std of [magnitude |z|] over the last 100 points, difference vs the reference-segment value |
| `sign_std_w100` | spread / std of [sign of z] over the last 100 points, raw running value |
| `sign_std_w100_vs_ref` | spread / std of [sign of z] over the last 100 points, difference vs the reference-segment value |
| `square_std_w100` | spread / std of [squared value z^2] over the last 100 points, raw running value |
| `square_std_w100_vs_ref` | spread / std of [squared value z^2] over the last 100 points, difference vs the reference-segment value |
| `logabs_std_w100` | spread / std of [log-magnitude] over the last 100 points, raw running value |
| `logabs_std_w100_vs_ref` | spread / std of [log-magnitude] over the last 100 points, difference vs the reference-segment value |
| `cumsum_std_w100` | spread / std of [running sum] over the last 100 points, raw running value |
| `cumsum_std_w100_vs_ref` | spread / std of [running sum] over the last 100 points, difference vs the reference-segment value |
| `diff_std_w100` | spread / std of [step-to-step change] over the last 100 points, raw running value |
| `diff_std_w100_vs_ref` | spread / std of [step-to-step change] over the last 100 points, difference vs the reference-segment value |
| `diff2_std_w100` | spread / std of [curvature (2nd difference)] over the last 100 points, raw running value |
| `diff2_std_w100_vs_ref` | spread / std of [curvature (2nd difference)] over the last 100 points, difference vs the reference-segment value |
| `rank_std_w100` | spread / std of [percentile rank vs history] over the last 100 points, raw running value |
| `rank_std_w100_vs_ref` | spread / std of [percentile rank vs history] over the last 100 points, difference vs the reference-segment value |
| `rank_centered_std_w100` | spread / std of [centered percentile rank] over the last 100 points, raw running value |
| `rank_centered_std_w100_vs_ref` | spread / std of [centered percentile rank] over the last 100 points, difference vs the reference-segment value |
| `rollmean16_std_w100` | spread / std of [local 16-point mean] over the last 100 points, raw running value |
| `rollmean16_std_w100_vs_ref` | spread / std of [local 16-point mean] over the last 100 points, difference vs the reference-segment value |
| `rollstd16_std_w100` | spread / std of [local 16-point volatility] over the last 100 points, raw running value |
| `rollstd16_std_w100_vs_ref` | spread / std of [local 16-point volatility] over the last 100 points, difference vs the reference-segment value |
| `z_std_w200` | spread / std of [standardized value] over the last 200 points, raw running value |
| `z_std_w200_vs_ref` | spread / std of [standardized value] over the last 200 points, difference vs the reference-segment value |
| `abs_std_w200` | spread / std of [magnitude |z|] over the last 200 points, raw running value |
| `abs_std_w200_vs_ref` | spread / std of [magnitude |z|] over the last 200 points, difference vs the reference-segment value |
| `sign_std_w200` | spread / std of [sign of z] over the last 200 points, raw running value |
| `sign_std_w200_vs_ref` | spread / std of [sign of z] over the last 200 points, difference vs the reference-segment value |
| `square_std_w200` | spread / std of [squared value z^2] over the last 200 points, raw running value |
| `square_std_w200_vs_ref` | spread / std of [squared value z^2] over the last 200 points, difference vs the reference-segment value |
| `logabs_std_w200` | spread / std of [log-magnitude] over the last 200 points, raw running value |
| `logabs_std_w200_vs_ref` | spread / std of [log-magnitude] over the last 200 points, difference vs the reference-segment value |
| `cumsum_std_w200` | spread / std of [running sum] over the last 200 points, raw running value |
| `cumsum_std_w200_vs_ref` | spread / std of [running sum] over the last 200 points, difference vs the reference-segment value |
| `diff_std_w200` | spread / std of [step-to-step change] over the last 200 points, raw running value |
| `diff_std_w200_vs_ref` | spread / std of [step-to-step change] over the last 200 points, difference vs the reference-segment value |
| `diff2_std_w200` | spread / std of [curvature (2nd difference)] over the last 200 points, raw running value |
| `diff2_std_w200_vs_ref` | spread / std of [curvature (2nd difference)] over the last 200 points, difference vs the reference-segment value |
| `rank_std_w200` | spread / std of [percentile rank vs history] over the last 200 points, raw running value |
| `rank_std_w200_vs_ref` | spread / std of [percentile rank vs history] over the last 200 points, difference vs the reference-segment value |
| `rank_centered_std_w200` | spread / std of [centered percentile rank] over the last 200 points, raw running value |
| `rank_centered_std_w200_vs_ref` | spread / std of [centered percentile rank] over the last 200 points, difference vs the reference-segment value |
| `rollmean16_std_w200` | spread / std of [local 16-point mean] over the last 200 points, raw running value |
| `rollmean16_std_w200_vs_ref` | spread / std of [local 16-point mean] over the last 200 points, difference vs the reference-segment value |
| `rollstd16_std_w200` | spread / std of [local 16-point volatility] over the last 200 points, raw running value |
| `rollstd16_std_w200_vs_ref` | spread / std of [local 16-point volatility] over the last 200 points, difference vs the reference-segment value |
| `z_std_w500` | spread / std of [standardized value] over the last 500 points, raw running value |
| `z_std_w500_vs_ref` | spread / std of [standardized value] over the last 500 points, difference vs the reference-segment value |
| `abs_std_w500` | spread / std of [magnitude |z|] over the last 500 points, raw running value |
| `abs_std_w500_vs_ref` | spread / std of [magnitude |z|] over the last 500 points, difference vs the reference-segment value |
| `sign_std_w500` | spread / std of [sign of z] over the last 500 points, raw running value |
| `sign_std_w500_vs_ref` | spread / std of [sign of z] over the last 500 points, difference vs the reference-segment value |
| `square_std_w500` | spread / std of [squared value z^2] over the last 500 points, raw running value |
| `square_std_w500_vs_ref` | spread / std of [squared value z^2] over the last 500 points, difference vs the reference-segment value |
| `logabs_std_w500` | spread / std of [log-magnitude] over the last 500 points, raw running value |
| `logabs_std_w500_vs_ref` | spread / std of [log-magnitude] over the last 500 points, difference vs the reference-segment value |
| `cumsum_std_w500` | spread / std of [running sum] over the last 500 points, raw running value |
| `cumsum_std_w500_vs_ref` | spread / std of [running sum] over the last 500 points, difference vs the reference-segment value |
| `diff_std_w500` | spread / std of [step-to-step change] over the last 500 points, raw running value |
| `diff_std_w500_vs_ref` | spread / std of [step-to-step change] over the last 500 points, difference vs the reference-segment value |
| `diff2_std_w500` | spread / std of [curvature (2nd difference)] over the last 500 points, raw running value |
| `diff2_std_w500_vs_ref` | spread / std of [curvature (2nd difference)] over the last 500 points, difference vs the reference-segment value |
| `rank_std_w500` | spread / std of [percentile rank vs history] over the last 500 points, raw running value |
| `rank_std_w500_vs_ref` | spread / std of [percentile rank vs history] over the last 500 points, difference vs the reference-segment value |
| `rank_centered_std_w500` | spread / std of [centered percentile rank] over the last 500 points, raw running value |
| `rank_centered_std_w500_vs_ref` | spread / std of [centered percentile rank] over the last 500 points, difference vs the reference-segment value |
| `rollmean16_std_w500` | spread / std of [local 16-point mean] over the last 500 points, raw running value |
| `rollmean16_std_w500_vs_ref` | spread / std of [local 16-point mean] over the last 500 points, difference vs the reference-segment value |
| `rollstd16_std_w500` | spread / std of [local 16-point volatility] over the last 500 points, raw running value |
| `rollstd16_std_w500_vs_ref` | spread / std of [local 16-point volatility] over the last 500 points, difference vs the reference-segment value |

## changepoint (144 features)

| feature | meaning |
|---|---|
| `cal_mean_exp` | [calibrated running mean-z] of [mean shift] |
| `cal_mean_scan` | [GLR change-point scan] of [mean shift] |
| `cal_mean_scanpeak` | [strongest scan evidence so far] of [mean shift] |
| `cal_mean_roll50` | [recent-50 calibrated mean] of [mean shift] |
| `cal_mean_roll200` | [recent-200 calibrated mean] of [mean shift] |
| `cal_mean_frac` | [fraction of steps over threshold] of [mean shift] |
| `cal_var_exp` | [calibrated running mean-z] of [variance shift] |
| `cal_var_scan` | [GLR change-point scan] of [variance shift] |
| `cal_var_scanpeak` | [strongest scan evidence so far] of [variance shift] |
| `cal_var_roll50` | [recent-50 calibrated mean] of [variance shift] |
| `cal_var_roll200` | [recent-200 calibrated mean] of [variance shift] |
| `cal_var_frac` | [fraction of steps over threshold] of [variance shift] |
| `cal_absdev_exp` | [calibrated running mean-z] of [robust spread shift] |
| `cal_absdev_scan` | [GLR change-point scan] of [robust spread shift] |
| `cal_absdev_scanpeak` | [strongest scan evidence so far] of [robust spread shift] |
| `cal_absdev_roll50` | [recent-50 calibrated mean] of [robust spread shift] |
| `cal_absdev_roll200` | [recent-200 calibrated mean] of [robust spread shift] |
| `cal_absdev_frac` | [fraction of steps over threshold] of [robust spread shift] |
| `cal_rank_exp` | [calibrated running mean-z] of [rank/location shift] |
| `cal_rank_scan` | [GLR change-point scan] of [rank/location shift] |
| `cal_rank_scanpeak` | [strongest scan evidence so far] of [rank/location shift] |
| `cal_rank_roll50` | [recent-50 calibrated mean] of [rank/location shift] |
| `cal_rank_roll200` | [recent-200 calibrated mean] of [rank/location shift] |
| `cal_rank_frac` | [fraction of steps over threshold] of [rank/location shift] |
| `cal_rank_disp_exp` | [calibrated running mean-z] of [rank dispersion] |
| `cal_rank_disp_scan` | [GLR change-point scan] of [rank dispersion] |
| `cal_rank_disp_scanpeak` | [strongest scan evidence so far] of [rank dispersion] |
| `cal_rank_disp_roll50` | [recent-50 calibrated mean] of [rank dispersion] |
| `cal_rank_disp_roll200` | [recent-200 calibrated mean] of [rank dispersion] |
| `cal_rank_disp_frac` | [fraction of steps over threshold] of [rank dispersion] |
| `cal_tail_exp` | [calibrated running mean-z] of [tail occupancy] |
| `cal_tail_scan` | [GLR change-point scan] of [tail occupancy] |
| `cal_tail_scanpeak` | [strongest scan evidence so far] of [tail occupancy] |
| `cal_tail_roll50` | [recent-50 calibrated mean] of [tail occupancy] |
| `cal_tail_roll200` | [recent-200 calibrated mean] of [tail occupancy] |
| `cal_tail_frac` | [fraction of steps over threshold] of [tail occupancy] |
| `cal_autocorr1_exp` | [calibrated running mean-z] of [lag-1 dependence] |
| `cal_autocorr1_scan` | [GLR change-point scan] of [lag-1 dependence] |
| `cal_autocorr1_scanpeak` | [strongest scan evidence so far] of [lag-1 dependence] |
| `cal_autocorr1_roll50` | [recent-50 calibrated mean] of [lag-1 dependence] |
| `cal_autocorr1_roll200` | [recent-200 calibrated mean] of [lag-1 dependence] |
| `cal_autocorr1_frac` | [fraction of steps over threshold] of [lag-1 dependence] |
| `cal_autocorr2_exp` | [calibrated running mean-z] of [lag-2 dependence] |
| `cal_autocorr2_scan` | [GLR change-point scan] of [lag-2 dependence] |
| `cal_autocorr2_scanpeak` | [strongest scan evidence so far] of [lag-2 dependence] |
| `cal_autocorr2_roll50` | [recent-50 calibrated mean] of [lag-2 dependence] |
| `cal_autocorr2_roll200` | [recent-200 calibrated mean] of [lag-2 dependence] |
| `cal_autocorr2_frac` | [fraction of steps over threshold] of [lag-2 dependence] |
| `cal_sign_runs_exp` | [calibrated running mean-z] of [sign persistence] |
| `cal_sign_runs_scan` | [GLR change-point scan] of [sign persistence] |
| `cal_sign_runs_scanpeak` | [strongest scan evidence so far] of [sign persistence] |
| `cal_sign_runs_roll50` | [recent-50 calibrated mean] of [sign persistence] |
| `cal_sign_runs_roll200` | [recent-200 calibrated mean] of [sign persistence] |
| `cal_sign_runs_frac` | [fraction of steps over threshold] of [sign persistence] |
| `cal_roughness_exp` | [calibrated running mean-z] of [step-change bumpiness] |
| `cal_roughness_scan` | [GLR change-point scan] of [step-change bumpiness] |
| `cal_roughness_scanpeak` | [strongest scan evidence so far] of [step-change bumpiness] |
| `cal_roughness_roll50` | [recent-50 calibrated mean] of [step-change bumpiness] |
| `cal_roughness_roll200` | [recent-200 calibrated mean] of [step-change bumpiness] |
| `cal_roughness_frac` | [fraction of steps over threshold] of [step-change bumpiness] |
| `cal_absdz_exp` | [calibrated running mean-z] of [size of step changes] |
| `cal_absdz_scan` | [GLR change-point scan] of [size of step changes] |
| `cal_absdz_scanpeak` | [strongest scan evidence so far] of [size of step changes] |
| `cal_absdz_roll50` | [recent-50 calibrated mean] of [size of step changes] |
| `cal_absdz_roll200` | [recent-200 calibrated mean] of [size of step changes] |
| `cal_absdz_frac` | [fraction of steps over threshold] of [size of step changes] |
| `cal_jump_exp` | [calibrated running mean-z] of [big-jump frequency] |
| `cal_jump_scan` | [GLR change-point scan] of [big-jump frequency] |
| `cal_jump_scanpeak` | [strongest scan evidence so far] of [big-jump frequency] |
| `cal_jump_roll50` | [recent-50 calibrated mean] of [big-jump frequency] |
| `cal_jump_roll200` | [recent-200 calibrated mean] of [big-jump frequency] |
| `cal_jump_frac` | [fraction of steps over threshold] of [big-jump frequency] |
| `cal_ring0_exp` | [calibrated running mean-z] of [quiet-band occupancy] |
| `cal_ring0_scan` | [GLR change-point scan] of [quiet-band occupancy] |
| `cal_ring0_scanpeak` | [strongest scan evidence so far] of [quiet-band occupancy] |
| `cal_ring0_roll50` | [recent-50 calibrated mean] of [quiet-band occupancy] |
| `cal_ring0_roll200` | [recent-200 calibrated mean] of [quiet-band occupancy] |
| `cal_ring0_frac` | [fraction of steps over threshold] of [quiet-band occupancy] |
| `cal_ring12_exp` | [calibrated running mean-z] of [active-band occupancy] |
| `cal_ring12_scan` | [GLR change-point scan] of [active-band occupancy] |
| `cal_ring12_scanpeak` | [strongest scan evidence so far] of [active-band occupancy] |
| `cal_ring12_roll50` | [recent-50 calibrated mean] of [active-band occupancy] |
| `cal_ring12_roll200` | [recent-200 calibrated mean] of [active-band occupancy] |
| `cal_ring12_frac` | [fraction of steps over threshold] of [active-band occupancy] |
| `cal_ar5_var_exp` | [calibrated running mean-z] of [AR(5) residual variance] |
| `cal_ar5_var_scan` | [GLR change-point scan] of [AR(5) residual variance] |
| `cal_ar5_var_scanpeak` | [strongest scan evidence so far] of [AR(5) residual variance] |
| `cal_ar5_var_roll50` | [recent-50 calibrated mean] of [AR(5) residual variance] |
| `cal_ar5_var_roll200` | [recent-200 calibrated mean] of [AR(5) residual variance] |
| `cal_ar5_var_frac` | [fraction of steps over threshold] of [AR(5) residual variance] |
| `cal_cdf0.1_exp` | [calibrated running mean-z] of [CDF at 10th pctile] |
| `cal_cdf0.1_scan` | [GLR change-point scan] of [CDF at 10th pctile] |
| `cal_cdf0.1_scanpeak` | [strongest scan evidence so far] of [CDF at 10th pctile] |
| `cal_cdf0.1_roll50` | [recent-50 calibrated mean] of [CDF at 10th pctile] |
| `cal_cdf0.1_roll200` | [recent-200 calibrated mean] of [CDF at 10th pctile] |
| `cal_cdf0.1_frac` | [fraction of steps over threshold] of [CDF at 10th pctile] |
| `cal_cdf0.25_exp` | [calibrated running mean-z] of [CDF at 25th pctile] |
| `cal_cdf0.25_scan` | [GLR change-point scan] of [CDF at 25th pctile] |
| `cal_cdf0.25_scanpeak` | [strongest scan evidence so far] of [CDF at 25th pctile] |
| `cal_cdf0.25_roll50` | [recent-50 calibrated mean] of [CDF at 25th pctile] |
| `cal_cdf0.25_roll200` | [recent-200 calibrated mean] of [CDF at 25th pctile] |
| `cal_cdf0.25_frac` | [fraction of steps over threshold] of [CDF at 25th pctile] |
| `cal_cdf0.5_exp` | [calibrated running mean-z] of [CDF at median] |
| `cal_cdf0.5_scan` | [GLR change-point scan] of [CDF at median] |
| `cal_cdf0.5_scanpeak` | [strongest scan evidence so far] of [CDF at median] |
| `cal_cdf0.5_roll50` | [recent-50 calibrated mean] of [CDF at median] |
| `cal_cdf0.5_roll200` | [recent-200 calibrated mean] of [CDF at median] |
| `cal_cdf0.5_frac` | [fraction of steps over threshold] of [CDF at median] |
| `cal_cdf0.75_exp` | [calibrated running mean-z] of [CDF at 75th pctile] |
| `cal_cdf0.75_scan` | [GLR change-point scan] of [CDF at 75th pctile] |
| `cal_cdf0.75_scanpeak` | [strongest scan evidence so far] of [CDF at 75th pctile] |
| `cal_cdf0.75_roll50` | [recent-50 calibrated mean] of [CDF at 75th pctile] |
| `cal_cdf0.75_roll200` | [recent-200 calibrated mean] of [CDF at 75th pctile] |
| `cal_cdf0.75_frac` | [fraction of steps over threshold] of [CDF at 75th pctile] |
| `cal_cdf0.9_exp` | [calibrated running mean-z] of [CDF at 90th pctile] |
| `cal_cdf0.9_scan` | [GLR change-point scan] of [CDF at 90th pctile] |
| `cal_cdf0.9_scanpeak` | [strongest scan evidence so far] of [CDF at 90th pctile] |
| `cal_cdf0.9_roll50` | [recent-50 calibrated mean] of [CDF at 90th pctile] |
| `cal_cdf0.9_roll200` | [recent-200 calibrated mean] of [CDF at 90th pctile] |
| `cal_cdf0.9_frac` | [fraction of steps over threshold] of [CDF at 90th pctile] |
| `cal_seg_mean` | mean measured on the estimated post-break segment |
| `cal_seg_var` | var measured on the estimated post-break segment |
| `cal_seg_absdev` | absdev measured on the estimated post-break segment |
| `cal_seg_rank` | rank measured on the estimated post-break segment |
| `cal_seg_rank_disp` | rank_disp measured on the estimated post-break segment |
| `cal_seg_tail` | tail measured on the estimated post-break segment |
| `cal_seg_autocorr1` | autocorr1 measured on the estimated post-break segment |
| `cal_seg_autocorr2` | autocorr2 measured on the estimated post-break segment |
| `cal_seg_sign_runs` | sign_runs measured on the estimated post-break segment |
| `cal_seg_roughness` | roughness measured on the estimated post-break segment |
| `cal_seg_absdz` | absdz measured on the estimated post-break segment |
| `cal_seg_jump` | jump measured on the estimated post-break segment |
| `cal_seg_ring0` | ring0 measured on the estimated post-break segment |
| `cal_seg_ring12` | ring12 measured on the estimated post-break segment |
| `cal_seg_ar5_var` | ar5_var measured on the estimated post-break segment |
| `cal_seg_cdf0.1` | cdf0.1 measured on the estimated post-break segment |
| `cal_seg_cdf0.25` | cdf0.25 measured on the estimated post-break segment |
| `cal_seg_cdf0.5` | cdf0.5 measured on the estimated post-break segment |
| `cal_seg_cdf0.75` | cdf0.75 measured on the estimated post-break segment |
| `cal_seg_cdf0.9` | cdf0.9 measured on the estimated post-break segment |
| `cal_glr_strength` | pooled 'did it break' detector strength |
| `cal_glr_peak` | strongest pooled break evidence so far |
| `cal_glr_seg_len` | log length of estimated post-break segment |
| `cal_glr_seg_frac` | fraction of stream estimated post-break |

## hypothesis (7 features)

| feature | meaning |
|---|---|
| `ftest_stat` | F-test statistic (log variance ratio) |
| `ftest_pval` | F-test p-value for equal variance |
| `levene_t` | Levene robust equal-spread test |
| `ks_stat` | KS distance between live and history distributions |
| `ks_pval` | KS test p-value |
| `mannwhitney_z` | Mann-Whitney location test (z-score) |
| `mannwhitney_pval` | Mann-Whitney p-value |

## spectral (4 features)

| feature | meaning |
|---|---|
| `spectral_entropy` | spread of the frequency content (spectral entropy) |
| `spectral_centroid` | center-of-mass frequency |
| `spectral_domfreq` | dominant frequency |
| `spectral_lowband` | low-frequency power fraction |

## context (5 features)

| feature        | meaning                                                    |
| -------------- | ---------------------------------------------------------- |
| `ref_log_n`    | fixed history fact: log_n (context, not break evidence)    |
| `ref_kurt`     | fixed history fact: kurt (context, not break evidence)     |
| `ref_skew`     | fixed history fact: skew (context, not break evidence)     |
| `ref_autocorr` | fixed history fact: autocorr (context, not break evidence) |
| `ref_std_raw`  | fixed history fact: std_raw (context, not break evidence)  |
