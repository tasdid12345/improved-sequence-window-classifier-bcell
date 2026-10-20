"""Stand-in experiment: class-balanced LightGBM (+ isotonic calibration) vs a standard
random forest on one-hot DNA window features (UCI splice-junction data).
This is NOT epitope data; it only mimics a fixed-length residue-window classification task."""
import itertools, json, sys, time
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")  # headless plotting on CI
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from ucimlrepo import fetch_ucirepo

SEED = 42
SEEDS = [42, 43, 44]          # used for the randomized models
MAX_ROWS = 200_000            # subsample guard (dataset is ~3k rows, so not triggered)
URL = "https://archive.ics.uci.edu/dataset/69/molecular+biology+splice+junction+gene+sequences"
T0 = time.time()

def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)

def get_with_retry(url, tries=5):
    """GET with timeout=60; retry with exponential backoff on 429/5xx and network errors."""
    for i in range(tries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if i == tries - 1:
                raise
            log(f"download failed ({e}); retrying in {2 ** i}s")
            time.sleep(2 ** i)

def load_data():
    """Load UCI dataset 69 via ucimlrepo; pre-check the landing page with timeout/retry first."""
    get_with_retry(URL)
    for i in range(5):  # ucimlrepo's own HTTP calls are not configurable, so retry around them
        try:
            ds = fetch_ucirepo(id=69)
            break
        except Exception as e:
            if i == 4:
                raise
            log(f"ucimlrepo fetch failed ({e}); retrying in {2 ** i}s")
            time.sleep(2 ** i)
    X = ds.data.features.fillna("missing").astype(str)
    y = ds.data.targets.iloc[:, 0].astype(str)
    if len(X) > MAX_ROWS:  # subsample very large data (not needed here)
        idx = X.sample(MAX_ROWS, random_state=SEED).index
        X, y = X.loc[idx], y.loc[idx]
    return X, y

def evaluate(proba, y_true, k):
    """Shared evaluation: macro-F1 (main), accuracy and multiclass Brier (secondary)."""
    pred = proba.argmax(1)
    onehot = np.eye(k)[y_true]
    return {"f1": f1_score(y_true, pred, average="macro"),
            "acc": accuracy_score(y_true, pred),
            "brier": float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))}

def run_baseline(seed, Xtr, ytr, Xte):
    """Standard 300-tree random forest on one-hot window features, raw probabilities."""
    rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    rf.fit(Xtr, ytr)
    return rf.predict_proba(Xte)

def run_improved(seed, best, Xtr, ytr, Xte):
    """Class-balanced LightGBM with CV-chosen learning rate / leaves, isotonic calibration
    fitted on an inner stratified split of the training data (never on the test set)."""
    fi, ci = train_test_split(np.arange(len(ytr)), test_size=0.2, stratify=ytr, random_state=SEED)
    m = LGBMClassifier(n_estimators=200, class_weight="balanced", random_state=seed, verbose=-1, **best)
    m.fit(Xtr[fi], ytr[fi])
    pc, pt = m.predict_proba(Xtr[ci]), m.predict_proba(Xte)
    k = pc.shape[1]
    iso = [IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
           .fit(pc[:, j], (ytr[ci] == j).astype(float)) for j in range(k)]
    out = np.column_stack([iso[j].predict(pt[:, j]) for j in range(k)]) + 1e-9
    return out / out.sum(1, keepdims=True)  # renormalise rows to valid probabilities

def main():
    X, y = load_data()
    le = LabelEncoder()
    y = le.fit_transform(y)
    k = len(le.classes_)
    log(f"loaded {len(X)} rows, {X.shape[1]} positions, {k} classes")

    # Single stratified 75/25 split, fixed seed
    Xtr_df, Xte_df, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    # One-hot encode window positions; encoder fitted on training rows only
    enc = OneHotEncoder(handle_unknown="ignore").fit(Xtr_df)
    Xtr, Xte = enc.transform(Xtr_df).toarray(), enc.transform(Xte_df).toarray()
    log(f"train {len(ytr)}, test {len(yte)}, one-hot features {Xtr.shape[1]}")

    # Small CV grid on the TRAINING set only (3-fold stratified, macro-F1)
    grid = {"learning_rate": [0.05, 0.1], "num_leaves": [15, 31]}
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    skf = StratifiedKFold(3, shuffle=True, random_state=SEED)

    def cv_score(params):
        s = []
        for a, b in skf.split(Xtr, ytr):
            m = LGBMClassifier(n_estimators=200, class_weight="balanced", random_state=SEED,
                               verbose=-1, **params).fit(Xtr[a], ytr[a])
            s.append(f1_score(ytr[b], m.predict(Xtr[b]), average="macro"))
        return float(np.mean(s))

    cv_scores = [cv_score(c) for c in combos]
    best_i = int(np.argmax(cv_scores))
    best = combos[best_i]
    log(f"CV grid {cv_scores} -> best {best}")

    # Baseline and improved, each with seeds 42/43/44, evaluated once on the test set
    base = [evaluate(run_baseline(s, Xtr, ytr, Xte), yte, k) for s in SEEDS]
    log("baseline done")
    impr = [evaluate(run_improved(s, best, Xtr, ytr, Xte), yte, k) for s in SEEDS]
    log("improved done")

    agg = lambda runs, key: np.array([r[key] for r in runs], dtype=float)
    bf, if_ = agg(base, "f1"), agg(impr, "f1")
    metrics = {
        "baseline_score": float(bf.mean()),
        "improved_score": float(if_.mean()),
        "improvement": float(if_.mean() - bf.mean()),
        "baseline_score_std": float(bf.std()),
        "improved_score_std": float(if_.std()),
        "baseline_accuracy": float(agg(base, "acc").mean()),
        "improved_accuracy": float(agg(impr, "acc").mean()),
        "baseline_brier": float(agg(base, "brier").mean()),
        "improved_brier": float(agg(impr, "brier").mean()),
        "brier_improvement": float(agg(base, "brier").mean() - agg(impr, "brier").mean()),
        "cv_best_macro_f1": float(max(cv_scores)),
        "best_learning_rate": float(best["learning_rate"]),
        "best_num_leaves": float(best["num_leaves"]),
    }
    if not all(np.isfinite(v) for v in metrics.values()):
        log("non-finite metrics; failing")
        sys.exit(1)

    # Optional chart: baseline vs improved macro-F1
    import os
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.bar(["baseline RF", "balanced LGBM+iso"], [bf.mean(), if_.mean()],
            yerr=[bf.std(), if_.std()], color=["grey", "steelblue"], capsize=5)
    plt.ylabel("test macro-F1 (mean over seeds)")
    plt.tight_layout()
    plt.savefig("figures/macro_f1.png", dpi=120)
    plt.close()

    results = {
        "question": "Can a class-balanced LightGBM with one-hot window features beat a standard random forest "
                    "on a public residue-window sequence classification task that stands in for linear B-cell "
                    "epitope prediction?",
        "method": "Single stratified 75/25 split (seed 42). Baseline: 300-tree random forest on one-hot window "
                  "features. Improved: class-balanced LightGBM with 3-fold CV grid over learning rate and leaves, "
                  "plus isotonic calibration on an inner split. Test set used once; seeds 42-44.",
        "dataset": {
            "name": "UCI Molecular Biology Splice-junction Gene Sequences (stand-in, not epitope data)",
            "source_url": URL,
            "n_rows": int(len(y)),
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "macro-F1 (test)",
        "higher_is_better": True,
        "baseline": "Standard random forest (300 trees) on one-hot encoded window positions, raw class probabilities.",
        "improved": "Class-balanced LightGBM on the same one-hot features, small CV grid over learning rate and "
                    "num_leaves, isotonic calibration fitted on a 20% inner split of the training data.",
        "metrics": metrics,
        "notes": "Stand-in task: DNA splice-junction windows, not linear B-cell epitopes; results say nothing "
                 "about epitope prediction. Single split, so variance across splits is not measured. "
                 "The improved model's final fit uses 80% of training rows (the rest calibrates isotonic). "
                 "The baseline uses all training rows. Seeds vary model randomness only. The ucimlrepo "
                 "internal HTTP calls cannot take a custom timeout; only the pre-flight request does.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"baseline macro-F1 {metrics['baseline_score']:.4f} vs improved {metrics['improved_score']:.4f}")
    log("wrote results.json")

if __name__ == "__main__":
    main()
