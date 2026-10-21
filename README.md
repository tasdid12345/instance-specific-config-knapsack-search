# Knapsack local-search setting selection

Question: can a cross-validation-tuned model choose a better per-instance local-search setting
than a standard MLP configuration predictor, judged by held-out normalised solution quality
(value / DP optimum) on a shifted test family of larger knapsack instances?

## Method
- Instances are generated with a fixed seed (42). Training: 700 small/mixed instances. Test: 300 larger instances with shifted correlation mix.
- Each of 6 local-search settings (restarts x step size) is run with a fixed evaluation budget and three search seeds.
- Baseline: standard `MLPClassifier` (64-32, scaled base features) predicting the best setting.
- Improved: an equal-weight ensemble of per-setting LightGBM regressors and an ExtraTrees regressor. It uses the base features plus scale-free greedy and LP-bound features, row-centred quality targets, and hyperparameters tuned by 3-fold CV on the training set only. The chosen setting is the one with the highest predicted quality, averaged over seeds.
- The test set is used only for the final evaluation.

## Run
```
pip install -r requirements.txt
python analysis.py
```
Writes `results.json` and `figures/quality_comparison.png`.

## Results (latest run)

On the generated 0/1 knapsack test set (300 shifted-family instances), a standard MLP configuration predictor reached 0.970 mean normalised solution quality, while my LightGBM+ExtraTrees ensemble reached 0.970, a negligible difference (0.00001), and its top-1 accuracy fell from 57% to 43%.

| Metric | Value |
|---|---|
| baseline_score | 0.97 |
| improved_score | 0.97 |
| improvement | 0.000009955 |
| baseline_score_std | 0.0002229 |
| improved_score_std | 0.0001082 |
| baseline_top1_accuracy | 0.5733 |
| improved_top1_accuracy | 0.4256 |
| baseline_within1pct | 0.9756 |
| improved_within1pct | 0.9744 |
| baseline_f1 | 0.1215 |
| improved_f1 | 0.2065 |
| oracle_score | 0.9712 |
| lgbm_cv_mean_quality | 0.9594 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Labels are noisy (single small grid, budget-limited search). The improved model uses the full per-setting quality information from training rather than only the argmax label, so it is not tied to label noise. Improvements over the previous version: scale-free greedy/LP-bound features (improved model only), row-centred regression targets (argmax unchanged), and an LightGBM+ExtraTrees ensemble. Test family is shifted in size and correlation, so tree models cannot extrapolate in size; the scale-free features are meant to help with this. Test set used once for both models; tuning used only training CV. Results depend on the generator and budget and are not a replication of OPTICAT's deep RL results.

## Reference

This project is motivated by the following paper:

> E Schede, MV Seiler, C Mensendiek (2026). *OPTICAT: A Deep Reinforcement Learning Framework for Instance-Specific Algorithm Configuration*. ACM Transactions on. https://dl.acm.org/doi/abs/10.1145/3844954

Paper author: Prof. Heike Trautmann, Universität Paderborn (heike.trautmann@uni-paderborn.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
