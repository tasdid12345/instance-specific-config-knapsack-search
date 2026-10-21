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

## Reference

This project is motivated by the following paper:

> E Schede, MV Seiler, C Mensendiek (2026). *OPTICAT: A Deep Reinforcement Learning Framework for Instance-Specific Algorithm Configuration*. ACM Transactions on. https://dl.acm.org/doi/abs/10.1145/3844954

Paper author: Prof. Heike Trautmann, Universität Paderborn (heike.trautmann@uni-paderborn.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
