"""Run a trained model on NEW threads (any folder in the PHEME layout), e.g. the bundled real samples.

    python -m src.predict --checkpoint models/impactguard_seed42.pt --data data/samples/pheme_samples --deadline 30

Uses the featurizer saved with the checkpoint, so no retraining and no full dataset are needed. If the folder
has rumours/ and non-rumours/ subfolders, accuracy is reported too.
"""
import argparse
import os
import warnings

import numpy as np
import torch

from src.data import load_threads, truncate, Featurizer
from src.eval import predict
from src.models import build_model
from src.utils import cls_metrics

warnings.filterwarnings("ignore", message="Sparse invariant checks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/impactguard_seed42.pt")
    ap.add_argument("--featurizer", default=None, help="default: <checkpoint name>_featurizer.pkl, or featurizer.pkl beside it")
    ap.add_argument("--data", default="data/samples/pheme_samples")
    ap.add_argument("--deadline", type=float, default=None, help="minutes after the source tweet (default: whole thread)")
    ap.add_argument("--show", type=int, default=15, help="how many threads to print")
    c = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(c.checkpoint, map_location=device, weights_only=False)
    fz_path = c.featurizer or next(p for p in [os.path.splitext(c.checkpoint)[0] + "_featurizer.pkl",
                                              os.path.join(os.path.dirname(c.checkpoint), "featurizer.pkl")] if os.path.exists(p))
    fz = Featurizer.load(fz_path)
    if os.path.isdir(c.data) is False and os.path.exists(c.data + ".tar.gz"):
        import tarfile
        with tarfile.open(c.data + ".tar.gz") as tar:
            try:
                tar.extractall(os.path.dirname(c.data) or ".", filter="data")
            except TypeError:                      # Python < 3.12
                tar.extractall(os.path.dirname(c.data) or ".")
    threads = load_threads(c.data, cache=False)
    if not threads:
        raise SystemExit(f"No threads found under {c.data}")
    graphs = fz.transform(threads)
    model = build_model(ck["args"]["model"], graphs[0]["x"].shape[1], ck["args"]["hidden"]).to(device)
    model.load_state_dict(ck["state"])
    thr = ck.get("threshold", 0.5)
    probs, sizes = predict(model, graphs, device, c.deadline)
    view = "whole thread" if c.deadline is None else f"first {c.deadline:g} minutes"
    print(f"{len(graphs)} threads from {c.data}, seen for the {view} | decision threshold {thr:.2f}\n")
    print(f"{'event':<18}{'tweets seen':>11}{'P(rumour)':>11}  {'prediction':<11}{'true label':<11}{'pred. final size':>16}  source tweet")
    for g, p, s in list(zip(graphs, probs, sizes))[:c.show]:
        seen = truncate(g, c.deadline)["x"].shape[0]
        size = "-" if np.isnan(s) else f"{max(seen, float(np.expm1(s))):.0f}"
        print(f"{g['event'][:17]:<18}{seen:>11}{p:>11.2f}  {('rumour' if p >= thr else 'non-rumour'):<11}"
              f"{('rumour' if g['y'] else 'non-rumour'):<11}{size:>16}  {g['source_text'][:55]!r}")
    m = cls_metrics([g["y"] for g in graphs], probs, thr)
    print(f"\nOn these {len(graphs)} threads: AUROC {m['auroc']:.3f} | F1 {m['f1']:.3f} | accuracy {m['acc']:.3f}")


if __name__ == "__main__":
    main()
