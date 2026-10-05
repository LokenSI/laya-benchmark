"""Metrics and policies kept separate from inference to make correctness testable."""
import math
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from scipy.stats import binomtest

def probabilities(values):
    p = np.asarray(values, dtype=float)
    if p.ndim != 2 or p.shape[1] < 2 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Probabilities must be finite, nonnegative N x K with K >= 2")
    sums = p.sum(axis=1, keepdims=True)
    if (sums <= 0).any() or (np.abs(sums - 1) > .01).any():
        raise ValueError("Probability rows must sum to one (allowing SDK rounding)")
    return p / sums

def wilson(successes, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = successes / n
    den = 1 + z*z/n
    centre = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return [max(0., centre-half), min(1., centre+half)]

def cluster_interval(values, groups, seed=20260926, repeats=2000):
    """Resample source documents, retaining all their sampled sentences."""
    grouped = defaultdict(list)
    for value, group in zip(values, groups):
        grouped[group].append(float(value))
    sums = np.array([sum(v) for v in grouped.values()])
    sizes = np.array([len(v) for v in grouped.values()])
    if len(sums) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        ix = rng.integers(0, len(sums), len(sums))
        estimates.append(float(sums[ix].sum() / sizes[ix].sum()))
    return np.quantile(estimates, [.025, .975]).tolist()

def classification(y, pred, labels, groups=None):
    # Aggregate decision reports use only Wilson intervals from this module.
    # Avoid importing optional pandas extensions through sklearn at startup;
    # Windows application-control policy may reject those unrelated binaries.
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    y, pred = np.asarray(y), np.asarray(pred)
    if len(y) == 0 or len(y) != len(pred):
        raise ValueError("A nonempty aligned test set is required")
    if not set(y).issubset(labels) or not set(pred).issubset(labels):
        raise ValueError("Unknown classification label")
    correct = y == pred
    return {"n": len(y), "correct": int(correct.sum()), "accuracy": float(correct.mean()),
            "accuracy_wilson95": wilson(int(correct.sum()), len(y)),
            "accuracy_cluster95": cluster_interval(correct, groups) if groups is not None else None,
            "clusters": len(set(groups)) if groups is not None else len(y),
            "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
            "balanced_accuracy": float(np.mean([np.mean(pred[y == c] == c) for c in labels if np.any(y == c)])),
            "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(), "labels": labels,
            "per_class": classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)}

def calibration(y_indices, p, pred_indices=None, bins=10):
    p = probabilities(p)
    y = np.asarray(y_indices, dtype=int)
    pred = p.argmax(1) if pred_indices is None else np.asarray(pred_indices, dtype=int)
    confidence = p[np.arange(len(p)), pred]
    correct = pred == y
    bin_id = np.minimum((confidence * bins).astype(int), bins-1)
    reliability, ece = [], 0.
    for i in range(bins):
        mask = bin_id == i
        n = int(mask.sum())
        acc = float(correct[mask].mean()) if n else None
        conf = float(confidence[mask].mean()) if n else None
        if n:
            ece += n / len(y) * abs(acc-conf)
        reliability.append({"low": i/bins, "high": (i+1)/bins, "n": n, "accuracy": acc, "confidence": conf})
    onehot = np.eye(p.shape[1])[y]
    high = confidence >= .9
    return {"ece": float(ece), "brier": float(np.mean(np.sum((p-onehot)**2, axis=1))),
            "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-8, 1)).mean()),
            "mean_confidence": float(confidence.mean()), "reliability": reliability,
            "confident_errors": int((high & ~correct).sum()), "confident_count": int(high.sum())}

def rescale(p, temperature):
    return softmax(np.log(np.clip(probabilities(p), 1e-8, 1)) / temperature, axis=1)

def fit_temperature(p, y):
    p, y = probabilities(p), np.asarray(y, dtype=int)
    def loss(log_t):
        q = rescale(p, math.exp(log_t))
        return -np.log(np.clip(q[np.arange(len(y)), y], 1e-8, 1)).mean()
    fit = minimize_scalar(loss, bounds=(math.log(.25), math.log(10)), method="bounded")
    return float(math.exp(fit.x))

def select_threshold(p, y, pred=None, target=.95, min_n=30):
    p = probabilities(p)
    pred = p.argmax(1) if pred is None else np.asarray(pred)
    confidence = p[np.arange(len(p)), pred]
    correct = pred == np.asarray(y)
    candidates = []
    for t in [.5, .6, .7, .8, .9, .95, .99]:
        use = confidence >= t
        n = int(use.sum())
        interval = wilson(int(correct[use].sum()), n)
        if n >= min_n and interval[0] >= target:
            candidates.append({"threshold": t, "validation_n": n, "validation_accuracy": float(correct[use].mean()),
                               "validation_wilson95": interval})
    return max(candidates, key=lambda r: r["validation_n"]) if candidates else None

def selective(p, y, pred, threshold):
    p = probabilities(p)
    pred, y = np.asarray(pred), np.asarray(y)
    use = np.zeros(len(y), dtype=bool) if threshold is None else p[np.arange(len(p)), pred] >= threshold
    n = int(use.sum())
    errors = int((pred[use] != y[use]).sum())
    return {"threshold": threshold, "accepted": n, "coverage": n/len(y), "errors": errors,
            "accuracy": 1-errors/n if n else None, "accuracy_wilson95": wilson(n-errors, n),
            "errors_per_1000_total": errors/len(y)*1000, "review_per_1000": (1-n/len(y))*1000}

def paired_comparison(a, b, seed=20260926):
    """Positive delta means B is better; align on IDs and verify identical gold labels."""
    left, right = {r["id"]: r for r in a}, {r["id"]: r for r in b}
    if len(left) != len(a) or len(right) != len(b) or left.keys() != right.keys():
        raise ValueError("Paired comparison requires identical unique example IDs")
    ids = sorted(left)
    if any(left[i]["label"] != right[i]["label"] for i in ids):
        raise ValueError("Gold labels differ in paired comparison")
    ac = np.array([left[i]["prediction"] == left[i]["label"] for i in ids], dtype=int)
    bc = np.array([right[i]["prediction"] == right[i]["label"] for i in ids], dtype=int)
    win_b, win_a = int(((bc == 1) & (ac == 0)).sum()), int(((ac == 1) & (bc == 0)).sum())
    return {"n": len(ids), "delta_accuracy_b_minus_a": float((bc-ac).mean()),
            "delta_cluster95": cluster_interval(bc-ac, [left[i]["group"] for i in ids], seed),
            "b_only_correct": win_b, "a_only_correct": win_a,
            "mcnemar_exact_p": float(binomtest(win_b, win_a+win_b, .5).pvalue) if win_a+win_b else 1.,
            "note": "Exploratory unadjusted p-value; clustered CI takes priority for multiple sentences per review."}
