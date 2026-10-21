# Instance-specific local-search tuning for knapsack: MLP baseline vs CV-tuned LightGBM

This experiment is motivated by "OPTICAT: A Deep Reinforcement Learning Framework for Instance-Specific Algorithm Configuration" (2026), which proposes choosing algorithm parameter settings separately for each problem instance. Here we ask a simpler, supervised version of that question: given cheap features of a 0/1 knapsack instance, can a predictor pick the local-search setting that yields the best solution within a fixed evaluation budget? The paper's deep RL policy is replaced by a standard MLP baseline.

**Question:** Can a cross-validation-tuned, class-balanced LightGBM classifier choose a better per-instance local-search parameter setting than a standard MLP configuration predictor on generated knapsack instances, when both are judged by the same held-out solution quality?

**Dataset:** Generated 0/1 knapsack instances created in code with a fixed numpy seed (a simulation, no download). Training instances come from one family (smaller, mixed correlations); the held-out test set comes from a shifted family (larger, different correlation mix). This is not the paper's own dataset.

**Baseline vs improved:** The baseline is a standard MLPClassifier on instance features. The improved model is a class-balanced LightGBM with hyperparameters chosen by cross-validation on the training set, using the same features, labels and evaluation.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: results are written to results.json (and a chart to figures/).

## Reference

This project is motivated by the following paper:

> E Schede, MV Seiler, C Mensendiek (2026). *OPTICAT: A Deep Reinforcement Learning Framework for Instance-Specific Algorithm Configuration*. ACM Transactions on. https://dl.acm.org/doi/abs/10.1145/3844954

Paper author: Prof. Heike Trautmann, Universität Paderborn (heike.trautmann@uni-paderborn.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
