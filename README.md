# Knapsack local-search setting selection

Question: can a CV-tuned LightGBM selector choose a better per-instance local-search setting than a standard MLP predictor, judged by held-out normalised solution quality (value / DP optimum)?

## Data
Generated 0/1 knapsack instances (numpy seed 42). Training set: 700 small/mixed instances (20–50 items). Held-out test set: 300 shifted, larger instances (60–100 items) with different value-weight correlations.

## Methods
- Labels: the local-search setting (restarts × step size, 6 candidates) with the highest mean quality over three search seeds, computed on training instances only.
- Baseline (unchanged): `MLPClassifier((64, 32))` on standardised instance features, predicting the setting; averaged over seeds 42–44.
- Improved: one LightGBM regressor per setting, predicting its training-set mean quality from the same features. The chosen setting is the one with the highest predicted quality. Hyperparameters (num_leaves, learning_rate, n_estimators, min_child_samples) are selected by 3-fold CV on the training set only, scored by true training quality of the chosen setting. Final predictions use seeds 42–44.

The test set is used once for the final comparison; it is never used for tuning.

## Run
```
pip install -r requirements.txt
python analysis.py
```
Outputs `results.json` and `figures/quality_comparison.png`. Runtime is well under 20 minutes on CPU.

## Caveats
Labels come from a single small grid and a fixed evaluation budget. The test family is shifted in size and correlation, so tree models cannot extrapolate in instance size. Results are a simulation and do not replicate OPTICAT's deep RL results.

## Reference

This project is motivated by the following paper:

> E Schede, MV Seiler, C Mensendiek (2026). *OPTICAT: A Deep Reinforcement Learning Framework for Instance-Specific Algorithm Configuration*. ACM Transactions on. https://dl.acm.org/doi/abs/10.1145/3844954

Paper author: Prof. Heike Trautmann, Universität Paderborn (heike.trautmann@uni-paderborn.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
