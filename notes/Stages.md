
## feature engineering
- Full features bank (the big 1072 generated features)
	- statistical
	- distributional
	- hypothesis testing scores
	- etc.
- however the full feature bank is unrealistic to stream in cloud run env with compute time budget
	- selected ~50 streaming features
	- with O(1) or O(window)
- 
- Auxiliary targets

## Validation harness rebuild
- previously been using a fixed 8000 unique series as train & 2000 series as validation
	- hard to assess gain/noise comparison
- newly built the *K-fold CV by series* harness
	- per fold score; report mean + std
	- use t=mean/std as judge for real improvements?

## modelling
- weighting & sampling
	- Log training step
	- for speed
	- been using log subsampling 
		- e.g.
- Objectives
	- log-loss vs. rank
- Models
	- LightGBM
	- CatBoost
	- XGBoost

