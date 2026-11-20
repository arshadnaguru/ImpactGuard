"""Classification metrics and validation-set threshold selection."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def best_threshold(ys, probs):
    """Decision threshold that maximises F1 on the VALIDATION set (never tuned on test)."""
    grid = np.linspace(0.2, 0.8, 25)
    return float(grid[int(np.argmax([f1_score(ys, (probs >= t).astype(int), zero_division=0) for t in grid]))])


def cls_metrics(y, prob, thr=0.5):
    """AUROC (threshold-free), F1 on the rumour class, and accuracy."""
    y, prob = np.asarray(y), np.asarray(prob)
    pred = (prob >= thr).astype(int)
    return dict(auroc=float(roc_auc_score(y, prob)) if len(set(y)) > 1 and len(set(prob)) > 1 else 0.5,
                f1=float(f1_score(y, pred, zero_division=0)), acc=float(accuracy_score(y, pred)))
