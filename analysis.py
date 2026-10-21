"""Instance-specific parameter selection for a 0/1 knapsack local search.
Baseline: standard MLPClassifier on instance features (unchanged).
Improved: per-setting LightGBM quality regressors (one per local-search setting, trained on
the training-set mean normalised quality of every setting), hyperparameters tuned by 3-fold CV
on the TRAIN set only, and predictions averaged over model seeds. The chosen setting is the one
with the highest predicted quality. Both models share data, features, evaluation table and seeds."""
import json, os, sys, time
from itertools import product
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
SEEDS = [42, 43, 44]               # search and model seeds
N_TRAIN, N_TEST = 700, 300         # train and held-out test sizes
BUDGET = 400                       # fixed evaluation budget per local-search run
# Local-search grid: (restarts, step size) -> 6 candidate settings
SETTINGS = [(r, s) for r in (1, 5) for s in (1, 3, 6)]
T0 = time.time()

def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)

def make_instances(rng, n_inst, n_range, corr_probs, cap_range):
    """0/1 knapsack instances: weights 1..100, value-weight correlation type drawn from corr_probs."""
    out = []
    for _ in range(n_inst):
        n = int(rng.integers(*n_range))
        w = rng.integers(1, 101, n)
        kind = rng.choice(4, p=corr_probs)  # 0 uncorrelated, 1 weak, 2 strong, 3 subset-sum
        if kind == 0:
            v = rng.integers(1, 101, n)
        elif kind == 1:
            v = np.clip(w + rng.integers(-10, 11, n), 1, None)
        elif kind == 2:
            v = w + 10
        else:
            v = w.copy()
        C = max(1, int(rng.uniform(*cap_range) * w.sum()))
        out.append((w.astype(int), v.astype(int), C))
    return out

def features(w, v, C):
    """Simple instance features shared by both models."""
    r = v / w
    corr = np.nan_to_num(np.corrcoef(v, w)[0, 1])
    return [len(w), C / w.sum(), w.mean(), w.std(), v.mean(), v.std(), corr,
            r.mean(), r.max(), C / w.mean(), C / w.max()]

def knapsack_opt(w, v, C):
    """Exact 0/1 knapsack optimum via dynamic programming over capacity."""
    dp = np.zeros(C + 1, dtype=np.int64)
    for wi, vi in zip(w, v):
        if wi <= C:
            dp[wi:] = np.maximum(dp[wi:], dp[:C + 1 - wi] + vi)
    return int(dp[C])

def local_search(w, v, C, restarts, step, rng):
    """Bit-flip hill climbing: flip `step` random items per move, accept feasible non-worsening moves.
    The budget is split evenly across random-greedy restarts. Returns the best value found."""
    n, per, best = len(w), BUDGET // restarts, 0
    for _ in range(restarts):
        x = np.zeros(n, dtype=bool)
        tw = tv = 0
        for i in rng.permutation(n):          # random greedy start
            if tw + w[i] <= C:
                x[i] = True
                tw += w[i]
                tv += v[i]
        for _ in range(per):
            idx = rng.choice(n, step, replace=False)
            s = np.where(x[idx], -1, 1)        # +1 add item, -1 remove item
            dw, dv = int(s @ w[idx]), int(s @ v[idx])
            if tw + dw <= C and dv >= 0:
                x[idx] = ~x[idx]
                tw += dw
                tv += dv
        best = max(best, tv)
    return best

def quality_table(insts, seeds):
    """Normalised quality (value / DP optimum) for every instance, setting and search seed."""
    Q = np.zeros((len(insts), len(SETTINGS), len(seeds)))
    for i, (w, v, C) in enumerate(insts):
        opt = knapsack_opt(w, v, C)
        for k, (r, s) in enumerate(SETTINGS):
            for j, sd in enumerate(seeds):
                val = local_search(w, v, C, r, s, np.random.default_rng([sd, i, k]))
                Q[i, k, j] = val / opt if opt > 0 else 1.0
    return Q

def evaluate(model_preds, Qte, yte):
    """Main score: mean quality of predicted setting; secondary: top-1 accuracy, within-1% rate, macro F1."""
    Qm = Qte.mean(2)
    rows = np.arange(len(yte))
    out = {"score": [], "top1": [], "within1": [], "f1": []}
    for j in range(len(SEEDS)):
        pred = model_preds[j]
        out["score"].append(Qte[rows, pred, j].mean())
        out["top1"].append((pred == yte).mean())
        out["within1"].append((Qm[rows, pred] >= Qm.max(1) - 0.01).mean())
        out["f1"].append(f1_score(yte, pred, average="macro", zero_division=0))
    return {k: np.array(v) for k, v in out.items()}

def fit_quality_regressors(X, Qm, params, seed):
    """One LightGBM regressor per setting, predicting that setting's mean normalised quality."""
    return [LGBMRegressor(**params, subsample=0.8, subsample_freq=1, verbose=-1,
                          random_state=seed).fit(X, Qm[:, k])
            for k in range(Qm.shape[1])]

def pick_setting(models, X):
    """Choose the setting with the highest predicted quality."""
    P = np.column_stack([m.predict(X) for m in models])
    return P.argmax(1)

def main():
    rng = np.random.default_rng(SEED)
    log("Generating instances (train: small/mixed; test: shifted larger/other correlations)")
    train = make_instances(rng, N_TRAIN, (20, 51), [.4, .3, .2, .1], (.2, .8))
    test = make_instances(rng, N_TEST, (60, 101), [.1, .2, .3, .4], (.3, .7))
    log("Computing exact optima and local-search quality tables")
    Qtr, Qte = quality_table(train, SEEDS), quality_table(test, SEEDS)
    # Labels: best setting by mean quality over search seeds, on TRAIN only
    Qm_tr = Qtr.mean(2)
    ytr = Qm_tr.argmax(1)
    yte = Qte.mean(2).argmax(1)              # oracle labels, used only for secondary metrics
    if len(np.unique(ytr)) < 2:
        log("ERROR: fewer than two setting classes in training labels; no meaningful model")
        sys.exit(1)
    Xtr = np.array([features(*t) for t in train])
    Xte = np.array([features(*t) for t in test])
    log(f"Label distribution (train): {np.bincount(ytr, minlength=len(SETTINGS)).tolist()}")

    # Baseline: standard MLP (scaled inputs, no tuning, no class balancing) -- unchanged
    log("Fitting baseline MLP")
    base_preds = []
    for sd in SEEDS:
        mlp = make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=1000,
                                                            early_stopping=True, random_state=sd))
        base_preds.append(mlp.fit(Xtr, ytr).predict(Xte).astype(int))

    # Improved: quality-regression selector, hyperparameters tuned by 3-fold CV on TRAIN only.
    # CV objective = mean true training quality of the setting chosen on the held-out fold.
    log("Tuning quality-regression LightGBM by cross-validation on training set")
    grid = {"num_leaves": [7, 15], "learning_rate": [0.05, 0.1],
            "n_estimators": [100, 300], "min_child_samples": [5, 15]}
    kf = KFold(3, shuffle=True, random_state=SEED)
    best_cv, best_params = -np.inf, None
    for vals in product(*grid.values()):
        params = dict(zip(grid.keys(), vals))
        fold_scores = []
        for tr_idx, va_idx in kf.split(Xtr):
            ms = fit_quality_regressors(Xtr[tr_idx], Qm_tr[tr_idx], params, SEED)
            pick = pick_setting(ms, Xtr[va_idx])
            fold_scores.append(Qm_tr[va_idx, pick].mean())
        cv = float(np.mean(fold_scores))
        if cv > best_cv:
            best_cv, best_params = cv, params
    log(f"Best CV params {best_params}, CV mean quality {best_cv:.4f}")

    imp_preds = []
    for sd in SEEDS:
        ms = fit_quality_regressors(Xtr, Qm_tr, best_params, sd)
        imp_preds.append(pick_setting(ms, Xte).astype(int))

    log("Evaluating both models on held-out shifted test set")
    b, im = evaluate(base_preds, Qte, yte), evaluate(imp_preds, Qte, yte)
    oracle = Qte.mean(2).max(1).mean()
    res = {
        "question": "Can a cross-validation-tuned, class-balanced LightGBM classifier choose a better "
                    "per-instance local-search parameter setting than a standard MLP configuration "
                    "predictor on generated knapsack instances, judged by the same held-out solution quality?",
        "method": "Generated 0/1 knapsack instances (fixed seed) with per-setting local-search quality "
                  "measured under a fixed evaluation budget; MLP baseline (classifier) vs CV-tuned "
                  "per-setting LightGBM quality regressors choosing the highest predicted setting; "
                  "test set from a shifted instance family; evaluated by mean normalised solution quality.",
        "dataset": {"name": "Generated 0/1 knapsack instances (simulation, numpy seed 42)",
                    "source_url": "https://numpy.org", "n_rows": N_TRAIN + N_TEST,
                    "n_train": N_TRAIN, "n_test": N_TEST, "same_as_paper": False,
                    "baseline_is_papers_method": False},
        "metric_name": "mean normalised solution quality (value / optimum) on held-out test instances",
        "higher_is_better": True,
        "baseline": "Standard MLPClassifier (64-32 hidden, scaled features) predicting the local-search setting",
        "improved": "Per-setting LightGBM quality regressors (3-fold CV-tuned on training quality, "
                    "seed-averaged models) on the same features; choose the setting with highest predicted quality",
        "metrics": {
            "baseline_score": float(b["score"].mean()),
            "improved_score": float(im["score"].mean()),
            "improvement": float(im["score"].mean() - b["score"].mean()),
            "baseline_score_std": float(b["score"].std()),
            "improved_score_std": float(im["score"].std()),
            "baseline_top1_accuracy": float(b["top1"].mean()),
            "improved_top1_accuracy": float(im["top1"].mean()),
            "baseline_within1pct": float(b["within1"].mean()),
            "improved_within1pct": float(im["within1"].mean()),
            "baseline_f1": float(b["f1"].mean()),
            "improved_f1": float(im["f1"].mean()),
            "oracle_score": float(oracle),
            "lgbm_cv_mean_quality": float(best_cv),
        },
        "notes": "Labels are noisy (single small grid, budget-limited search). The improved model uses the "
                 "full per-setting quality information from training rather than only the argmax label, "
                 "so it is not tied to label noise. Test family is shifted in size and correlation, so tree "
                 "models cannot extrapolate in size. Test set used once for both models; tuning used only "
                 "training CV. Results depend on the generator and budget and are not a replication of "
                 "OPTICAT's deep RL results.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["MLP baseline", "LightGBM improved"], [b["score"].mean(), im["score"].mean()],
           yerr=[b["score"].std(), im["score"].std()], color=["grey", "steelblue"], capsize=5)
    ax.set_ylabel("Mean normalised quality (test)")
    ax.set_title("Setting selection on shifted knapsack test set")
    fig.tight_layout()
    fig.savefig("figures/quality_comparison.png", dpi=120)
    log("Done. Wrote results.json and figures/quality_comparison.png")

if __name__ == "__main__":
    main()
