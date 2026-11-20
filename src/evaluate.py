"""Evaluate a saved checkpoint on its test split (re-creates the same split and features from the checkpoint).

    python -m src.evaluate --checkpoint models/impactguard_seed42.pt --data data/pheme
"""
import argparse
import json
import warnings

import numpy as np
import torch

from src.eval import full_report
from src.models import build_model
from src.train import get_args, prepare

warnings.filterwarnings("ignore", message="Sparse invariant checks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default=None, help="override the data folder stored in the checkpoint")
    ap.add_argument("--out", default=None, help="optional path to save the metrics as JSON")
    c = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(c.checkpoint, map_location=device, weights_only=False)
    a = get_args([]); vars(a).update(ck["args"])
    if c.data:
        a.data = c.data
    _, _, test_g, _ = prepare(a)
    model = build_model(a.model, test_g[0]["x"].shape[1], a.hidden).to(device)
    model.load_state_dict(ck["state"])
    thr = ck.get("threshold", 0.5)
    rep = full_report(model, test_g, device, thr)
    print(f"Checkpoint {c.checkpoint} | model {a.model} | epoch {ck['epoch']} | threshold {thr:.2f} | {len(test_g)} test threads")
    for view, m in rep.items():
        extra = f" | size MAE(log) {m['size_mae']:.3f}" if "size_mae" in m else ""
        print(f"{view:>7}: AUROC {m['auroc']:.4f} | F1 {m['f1']:.4f} | Acc {m['acc']:.4f}{extra}")
    print(f"Confusion matrix (rows = true non-rumour/rumour, cols = predicted): {rep['full']['cm']}")
    if c.out:
        json.dump(rep, open(c.out, "w"), indent=1)


if __name__ == "__main__":
    main()
