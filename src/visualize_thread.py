"""Draw one concrete cascade graph per class (addresses Ckpt 2 feedback: show a real thread).

Nodes = tweets, edges = reply-to links, colour = minutes after the source tweet.
Tweets that arrive after the 30-minute cut are drawn hollow: that is what the model does NOT see early.
    python -m src.visualize_thread --data data/pheme --out runs/thread_example.png
"""
import argparse
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import load_threads


def tree_layout(parent):
    """x = position among leaves (depth-first), y = -depth. Pure Python, no networkx needed."""
    n = len(parent); kids = [[] for _ in range(n)]
    for c in range(1, n):
        kids[parent[c]].append(c)
    x, y, nxt = np.zeros(n), np.zeros(n), [0]
    stack, order = [(0, 0)], []
    while stack:                                   # iterative DFS (pre-order), then place bottom-up
        v, d = stack.pop(); order.append(v); y[v] = -d
        stack += [(k, d + 1) for k in reversed(kids[v])]
    for v in reversed(order):
        if kids[v]:
            x[v] = np.mean([x[k] for k in kids[v]])
        else:
            x[v] = nxt[0]; nxt[0] += 1
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pheme"); ap.add_argument("--out", default="thread_example.png")
    ap.add_argument("--max-threads", type=int, default=0); ap.add_argument("--data-seed", type=int, default=42)
    ap.add_argument("--min-nodes", type=int, default=18); ap.add_argument("--max-nodes", type=int, default=40)
    ap.add_argument("--cut", type=float, default=30.0)
    a = ap.parse_args()
    threads = load_threads(a.data, a.max_threads, a.data_seed)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, lab, name in zip(axes, (1, 0), ("RUMOUR", "NON-RUMOUR")):
        cand = [t for t in threads if t["label"] == lab and a.min_nodes <= len(t["texts"]) <= a.max_nodes
                and t["depth"].max() >= 2] or [t for t in threads if t["label"] == lab]
        th = sorted(cand, key=lambda t: -t["depth"].max())[0]
        x, y = tree_layout(th["parent"]); t = th["t"]; seen = t <= a.cut
        for c in range(1, len(t)):
            ax.plot([x[th["parent"][c]], x[c]], [y[th["parent"][c]], y[c]], color="#9AA3B2", lw=1, zorder=1)
        col = np.log1p(t)
        sc = ax.scatter(x[seen], y[seen], c=col[seen], cmap="viridis", vmin=0, vmax=col.max(), s=260, edgecolors="k", zorder=2)
        ax.scatter(x[~seen], y[~seen], facecolors="white", edgecolors=plt.cm.viridis(col[~seen] / max(col.max(), 1e-9)),
                   linewidths=2.2, s=260, zorder=2)
        for i in range(len(t)):
            ax.text(x[i], y[i], str(i), ha="center", va="center", fontsize=7, color="white" if seen[i] and col[i] < 0.6 * col.max() else "black", zorder=3)
        cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02); ticks = [0, 1, 5, 30, 120, 600]
        ticks = [k for k in ticks if np.log1p(k) <= col.max()]; cb.set_ticks(np.log1p(ticks)); cb.set_ticklabels(ticks)
        cb.set_label("minutes after source tweet")
        ax.set_title(f"{name} | event: {th['event']} | {len(t)} tweets, depth {th['depth'].max()}, "
                     f"{int(seen.sum())} seen in first {a.cut:.0f} min", fontsize=10)
        ax.set_xlabel("node 0 = source tweet: \"" + "\n".join(textwrap.wrap(th["texts"][0][:150], 75)) + "\"", fontsize=8)
        ax.set_xticks([]); ax.set_yticks(sorted(set(y))); ax.set_yticklabels([f"depth {int(-v)}" for v in sorted(set(y))], fontsize=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("One concrete cascade per class. Filled = visible at the 30-minute cut; hollow = arrives later. Numbers = arrival order.", fontsize=10)
    plt.tight_layout(); plt.savefig(a.out, dpi=150); print("saved", a.out)


if __name__ == "__main__":
    main()
