
feature engineering
- Full features bank (the big 1072 generated features)
	- statistical
	- distributional
	- hypothesis testing scores
	- etc.

modelling
- LightGBM
- CatBoost
- XGBoost


**key models and results so far**

||Model|Result|Source|Reproduce?|
|---|---|---|---|
|Value detector|0.53|`causal_baseline()` in [pipeline.ipynb](vscode-webview://1kr4mkfg55fq7a1paf4ditma24f8p826amutjscgfesp5u08mnob/pipeline.ipynb)|✅ yes|
|Old pipeline (7 feats)|0.53|`build_features()` + `evidence_cummax` in [pipeline.ipynb](vscode-webview://1kr4mkfg55fq7a1paf4ditma24f8p826amutjscgfesp5u08mnob/pipeline.ipynb)|✅ yes|
|Batch bank (1072 feats)|0.606|[sb_bank.py](vscode-webview://1kr4mkfg55fq7a1paf4ditma24f8p826amutjscgfesp5u08mnob/experiments/sb_bank.py) + [run_final.py](vscode-webview://1kr4mkfg55fq7a1paf4ditma24f8p826amutjscgfesp5u08mnob/experiments/run_final.py)|⚠️ partial|
|**Streaming (submitted)**|0.603|[sb_stream.py](vscode-webview://1kr4mkfg55fq7a1paf4ditma24f8p826amutjscgfesp5u08mnob/experiments/sb_stream.py) + `_dev_stream.py`/`_verify_sub.py`|✅ yes|
|Oracle ceiling|0.72|`oracle_v2.py` — **DELETED**|❌ no|     |     |



worth exploring
- **Learned sequence model** (small TCN/Transformer over the online stream), trained directly for the ranking metric. Highest upside — learns break signatures instead of fixed test statistics.
- TabPFN
