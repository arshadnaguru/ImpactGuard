"""Score a model on a set of cascades, optionally seeing only the first `deadline` minutes of each."""
import numpy as np
import torch.nn.functional as F
import torch
from sklearn.metrics import confusion_matrix

from src.data import truncate, DEADLINES
from src.models import batch_graphs
from src.utils import cls_metrics


@torch.no_grad()
def predict(model, graphs, device, deadline=None, bs=64):
    """Rumour probability and predicted log-size for every cascade, seen up to `deadline` minutes."""
    model.eval()
    probs, sizes = [], []
    for i in range(0, len(graphs), bs):
        b = batch_graphs([truncate(g, deadline) for g in graphs[i:i + bs]], device)
        logits, size = model(b)
        probs += F.softmax(logits, 1)[:, 1].cpu().tolist()
        sizes += (size.cpu().tolist() if size is not None else [float("nan")] * len(b["lengths"]))
    return np.array(probs), np.array(sizes)


def evaluate(model, graphs, device, deadline=None, thr=0.5):
    ys = np.array([g["y"] for g in graphs])
    probs, sizes = predict(model, graphs, device, deadline)
    m = cls_metrics(ys, probs, thr)
    m["cm"] = confusion_matrix(ys, (probs >= thr).astype(int), labels=[0, 1]).tolist()
    if not np.isnan(sizes).any():
        m["size_mae"] = float(np.mean(np.abs(sizes - np.log1p([g["n_full"] for g in graphs]))))
    return m


def full_report(model, graphs, device, thr=0.5):
    rep = {"full": evaluate(model, graphs, device, thr=thr)}
    for d in DEADLINES:
        rep[f"{d}min"] = evaluate(model, graphs, device, d, thr)
    return rep
