"""Train + test one model. Saves a training log, checkpoints and test metrics under runs/<name>/.

    python -m src.train --model impactguard          # proposed core model
    python -m src.train --model gcn                  # Ckpt 2 baseline
    python -m src.train --model gcn --prefix-aug     # ablation: baseline trained on time-prefixes too
    python -m src.train --model mlp                  # no-graph baseline
    python -m src.train --model gat | gin | bigcn    # recent graph baselines
"""
import argparse
import csv
import json
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from src.data import load_threads, split_indices, featurize, truncate
from src.eval import predict, evaluate, full_report
from src.models import build_model, batch_graphs
from src.utils import best_threshold, set_seed

warnings.filterwarnings("ignore", message="Sparse invariant checks")
TRAIN_WINDOWS = [5, 10, 20, 30, 60, 120, 360, None, None, None]   # None = whole cascade


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pheme")
    ap.add_argument("--model", choices=["mlp", "gcn", "gat", "gin", "bigcn", "impactguard"], default="impactguard")
    ap.add_argument("--prefix-aug", action="store_true", help="train on random time-prefixes (always on for impactguard)")
    ap.add_argument("--split", choices=["stratified", "event"], default="stratified")
    ap.add_argument("--test-event", default=None)
    ap.add_argument("--max-threads", type=int, default=0, help="0 = use every thread")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-seed", type=int, default=42, help="fixes the subsample and the split for every model")
    ap.add_argument("--out", default="runs")
    a = ap.parse_args(argv)
    if a.model == "impactguard":
        a.prefix_aug = True
    return a


def run_name(a):
    return f"{a.model}{'_aug' if a.prefix_aug and a.model != 'impactguard' else ''}_seed{a.seed}"


def prepare(a, verbose=True):
    """Same data, split and features for every model (controlled by --data-seed)."""
    threads = load_threads(a.data, a.max_threads, a.data_seed)
    if not threads:
        raise SystemExit(f"No PHEME threads found under {a.data}. Run: python -m src.data.download")
    tr, va, te = split_indices(threads, a.split, a.data_seed, a.test_event)
    graphs, fz = featurize(threads, tr, seed=a.data_seed, return_featurizer=True)
    if verbose:
        n = [g["n_full"] for g in graphs]
        print(f"Cascades: {len(graphs)} | rumour {sum(g['y'] for g in graphs)} | non-rumour {sum(1 - g['y'] for g in graphs)}"
              f" | tweets {sum(n)} (mean {np.mean(n):.1f})")
        print(f"Split ({a.split}): train {len(tr)} | val {len(va)} | test {len(te)} | node features {graphs[0]['x'].shape[1]}-d")
    pick = lambda ids: [graphs[i] for i in ids]
    return pick(tr), pick(va), pick(te), fz


def main():
    a = get_args()
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = os.path.join(a.out, run_name(a)); os.makedirs(out, exist_ok=True)
    log_f = open(os.path.join(out, "train.log"), "w")

    def log(msg):
        print(msg); log_f.write(msg + "\n"); log_f.flush()

    log(f"=== {run_name(a)} | device {device} | args {vars(a)}")
    train_g, val_g, test_g, fz = prepare(a)
    fz.save(os.path.join(out, "featurizer.pkl"))          # needed to score new threads later
    model = build_model(a.model, train_g[0]["x"].shape[1], a.hidden).to(device)
    log(f"Model: {model.__class__.__name__} | parameters: {sum(p.numel() for p in model.parameters()):,}"
        f" | prefix training: {a.prefix_aug}")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
    pos = sum(g["y"] for g in train_g)
    w = torch.tensor([1.0, (len(train_g) - pos) / max(pos, 1)], device=device)   # rumour = minority class
    rng = random.Random(a.seed)

    rows, best, t0 = [], -1.0, time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); rng.shuffle(train_g); tot = 0.0
        for i in range(0, len(train_g), a.bs):
            chunk = train_g[i:i + a.bs]
            if a.prefix_aug:                                   # show the cascade at a random moment in time
                chunk = [truncate(g, rng.choice(TRAIN_WINDOWS)) for g in chunk]
            b = batch_graphs(chunk, device)
            logits, size = model(b)
            loss = F.cross_entropy(logits, b["y"], weight=w)
            if size is not None:                               # impact head: final cascade size
                loss = loss + 0.5 * F.smooth_l1_loss(size, b["size"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step(); tot += loss.item() * len(chunk)
        v_full, v_30 = evaluate(model, val_g, device), evaluate(model, val_g, device, 30)
        score = (v_full["auroc"] + v_30["auroc"]) / 2          # model selection: full view + 30-minute view
        rows.append(dict(epoch=ep, train_loss=round(tot / len(train_g), 4), val_auroc_full=round(v_full["auroc"], 4),
                         val_f1_full=round(v_full["f1"], 4), val_auroc_30min=round(v_30["auroc"], 4)))
        flag = ""
        if score >= best:
            best = score
            torch.save(dict(state=model.state_dict(), args=vars(a), epoch=ep, val_score=best), os.path.join(out, "best.pt"))
            flag = "  <- saved best.pt"
        log(f"epoch {ep:02d}/{a.epochs} | train loss {rows[-1]['train_loss']:.4f} | val AUROC full {v_full['auroc']:.4f}"
            f" | val AUROC@30min {v_30['auroc']:.4f} | val F1 {v_full['f1']:.4f}{flag}")
    torch.save(dict(state=model.state_dict(), args=vars(a), epoch=a.epochs), os.path.join(out, "last.pt"))
    with open(os.path.join(out, "train_log.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    log(f"Training time: {time.time() - t0:.1f}s | best val score {best:.4f}")

    ck = torch.load(os.path.join(out, "best.pt"), map_location=device)
    model.load_state_dict(ck["state"])
    thr = best_threshold(np.array([g["y"] for g in val_g]), predict(model, val_g, device)[0])
    ck["threshold"] = thr; torch.save(ck, os.path.join(out, "best.pt"))
    rep = full_report(model, test_g, device, thr)
    log(f"--- TEST (best.pt from epoch {ck['epoch']}, decision threshold {thr:.2f} chosen on validation) ---")
    for view, m in rep.items():
        extra = f" | size MAE(log) {m['size_mae']:.3f}" if "size_mae" in m else ""
        log(f"{view:>7}: AUROC {m['auroc']:.4f} | F1 {m['f1']:.4f} | Acc {m['acc']:.4f}{extra}")
    log(f"Confusion matrix on full cascades [rows=true non-rumour/rumour, cols=pred]: {rep['full']['cm']}")
    json.dump(dict(name=run_name(a), model=a.model, prefix_aug=a.prefix_aug, seed=a.seed, best_epoch=ck["epoch"], threshold=thr,
                   n_test=len(test_g), test=rep), open(os.path.join(out, "metrics.json"), "w"), indent=1)
    log(f"Saved: {out}/train.log, train_log.csv, best.pt, last.pt, featurizer.pkl, metrics.json")


if __name__ == "__main__":
    main()
