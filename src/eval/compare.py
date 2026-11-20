"""Collect every runs/*/metrics.json into one comparison table (mean +- std over seeds) and a chart.

    python -m src.eval.compare runs
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ORDER = ["majority", "tfidf_logreg", "mlp", "gcn", "gat", "gin", "bigcn", "bigcn_aug", "gcn_aug", "impactguard"]
LABEL = {"majority": "Majority class", "tfidf_logreg": "TF-IDF + LogReg", "mlp": "Mean-pool MLP",
         "gcn": "Static GCN (Ckpt 2)", "gat": "GAT (2018)", "gin": "GIN (2019)", "bigcn": "Bi-GCN (2020)",
         "bigcn_aug": "Bi-GCN + prefix training", "gcn_aug": "Static GCN + prefix training", "impactguard": "ImpactGuard-T (ours)"}
COLS = [("AUROC", "full", "auroc"), ("F1", "full", "f1"), ("Acc", "full", "acc"),
        ("AUROC@10m", "10min", "auroc"), ("AUROC@30m", "30min", "auroc"), ("AUROC@60m", "60min", "auroc"),
        ("SizeMAE@30m", "30min", "size_mae")]


def main(runs="runs"):
    groups = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(runs, "*", "metrics.json"))):
        m = json.load(open(p))
        groups[m["name"].rsplit("_seed", 1)[0]].append(m)
    if not groups:
        raise SystemExit("No runs found. Train something first.")
    table = {}
    for key, ms in groups.items():
        table[key] = {c: ([m["test"][view][k] for m in ms if k in m["test"].get(view, {})]) for c, view, k in COLS}
    names = [k for k in ORDER if k in table] + [k for k in table if k not in ORDER]
    fmt = lambda v: "-" if not v else (f"{np.mean(v):.3f}" + (f" ± {np.std(v):.3f}" if len(v) > 1 else ""))
    head = "| Method | seeds | " + " | ".join(c for c, _, _ in COLS) + " |"
    lines = [head, "|" + "---|" * (len(COLS) + 2)]
    for k in names:
        lines.append(f"| {LABEL.get(k, k)} | {len(groups[k])} | " + " | ".join(fmt(table[k][c]) for c, _, _ in COLS) + " |")
    md = "\n".join(lines)
    print(f"\nTEST-SET COMPARISON ({groups[names[0]][0]['n_test']} test cascades; mean ± std over seeds; SizeMAE lower is better)\n")
    print(md)
    open(os.path.join(runs, "comparison.md"), "w", encoding="utf-8").write(md + "\n")
    for base in ("gcn", "bigcn"):
        if base in table and "impactguard" in table:
            print(f"\nImpactGuard-T minus {LABEL[base]}:")
            for c, _, _ in COLS[:6]:
                if table[base][c] and table["impactguard"][c]:
                    print(f"  {c:<10} {np.mean(table['impactguard'][c]) - np.mean(table[base][c]):+.3f}")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = ["AUROC@10m", "AUROC@30m", "AUROC@60m", "AUROC"]
        fig, ax = plt.subplots(figsize=(10, 3.8)); wdt = 0.8 / len(names)
        for i, k in enumerate(names):
            ax.bar(np.arange(len(cols)) + i * wdt, [np.mean(table[k][c]) if table[k][c] else 0 for c in cols], wdt,
                   yerr=[np.std(table[k][c]) if len(table[k][c]) > 1 else 0 for c in cols], label=LABEL.get(k, k))
        ax.set_xticks(np.arange(len(cols)) + 0.4 - wdt / 2, ["first 10 min", "first 30 min", "first 60 min", "full cascade"])
        ax.set_ylabel("Test AUROC"); ax.set_ylim(0.4, 1.0); ax.legend(fontsize=7, ncol=2); ax.set_title("Rumour detection: how early can each model tell?")
        plt.tight_layout(); plt.savefig(os.path.join(runs, "comparison.png"), dpi=150)
        print(f"\nSaved {runs}/comparison.md and {runs}/comparison.png")
    except Exception as e:
        print("(chart skipped:", e, ")")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs")
