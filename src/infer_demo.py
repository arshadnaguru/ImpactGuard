"""Inference demo: load a saved checkpoint and run it on one batch of unseen test cascades.

    python -m src.infer_demo --ckpt runs/impactguard_seed42/best.pt --n 8
"""
import argparse
import warnings

import numpy as np
import torch

from src.data import truncate, DEADLINES
from src.eval import predict
from src.models import build_model
from src.train import get_args, prepare

warnings.filterwarnings("ignore", message="Sparse invariant checks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/impactguard_seed42/best.pt")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--data", default=None)
    c = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(c.ckpt, map_location=device)
    a = get_args([]); vars(a).update(ck["args"])
    if c.data:
        a.data = c.data
    thr = ck.get("threshold", 0.5)
    print(f"Loaded {c.ckpt} (model {a.model}, saved at epoch {ck['epoch']}, decision threshold {thr:.2f})")
    _, _, test_g, _ = prepare(a, verbose=False)
    model = build_model(a.model, test_g[0]["x"].shape[1], a.hidden).to(device)
    model.load_state_dict(ck["state"])

    rng = np.random.RandomState(0)
    batch = [test_g[i] for i in rng.choice(len(test_g), min(c.n, len(test_g)), replace=False)]
    views = [(f"{d}m", d) for d in DEADLINES] + [("full", None)]
    P = {v: predict(model, batch, device, d) for v, d in views}

    print(f"\nBatch of {len(batch)} test cascades. P(rumour) as the cascade unfolds:")
    print(f"{'#':>2} {'event':<16} {'true':<10} " + " ".join(f"{v:>6}" for v, _ in views) + "  pred       ok  source tweet")
    correct = 0
    for i, g in enumerate(batch):
        p_full = P["full"][0][i]; pred = int(p_full >= thr); correct += pred == g["y"]
        lab = lambda y: "rumour" if y else "non-rumour"
        print(f"{i:>2} {g['event'][:16]:<16} {lab(g['y']):<10} " + " ".join(f"{P[v][0][i]:6.2f}" for v, _ in views)
              + f"  {lab(pred):<10} {'yes' if pred == g['y'] else 'NO ':<3} {g['source_text'][:60]!r}")
    print(f"Batch accuracy on full cascades: {correct}/{len(batch)}")

    if not np.isnan(P["30m"][1]).any():
        print("\nImpact estimate at 30 minutes (size head): tweets seen so far -> predicted final size -> actual")
        for i, g in enumerate(batch):
            seen = truncate(g, 30)["x"].shape[0]
            pred_n = max(seen, float(np.expm1(P["30m"][1][i])))
            print(f"{i:>2}  seen {seen:>3} -> predicted {pred_n:6.1f} -> actual {g['n_full']:>3}"
                  f"   (reach still to come if nothing is done: about {pred_n - seen:5.1f} tweets)")


if __name__ == "__main__":
    main()
