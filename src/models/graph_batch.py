"""Batching variable-size cascades into one block-diagonal graph, plus pooling helpers. Pure PyTorch."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_graphs(graphs, device):
    """Block-diagonal batching of (possibly truncated) cascades.

    Returns a dict with:
      x      [N, 67]   node features (text + user)
      tfeat  [N, 3]    log minutes since source, log minutes since parent, log depth
      adj    [N, N]    symmetric-normalised sparse adjacency with self loops
      gid    [N]       which graph each node belongs to
      lengths[B]       nodes per graph (nodes are time-ordered inside each graph)
      e_par, e_chd     directed reply edges (parent -> child), used by GAT / GIN / Bi-GCN
      root   [N]       index of the source tweet of each node's cascade
      gstat  [B, 2]    log observed size, log observation window (minutes)
      y      [B]       label;   size [B] = log(1 + final cascade size), the impact target
    """
    xs, tf, rows, cols, gid, ys, lens, gstat, size = [], [], [], [], [], [], [], [], []
    e_par, e_chd, root = [], [], []
    offset = 0
    for g_i, g in enumerate(graphs):
        n = g["x"].shape[0]
        xs.append(g["x"])
        par = g["parent"]
        gap = g["t"] - g["t"][np.maximum(par, 0)]
        tf.append(np.stack([np.log1p(g["t"]), np.log1p(np.maximum(gap, 0)), np.log1p(g["depth"])], 1))
        for c in range(1, n):
            rows += [c + offset, par[c] + offset]
            e_par.append(par[c] + offset); e_chd.append(c + offset)
            cols += [par[c] + offset, c + offset]
        rows += list(range(offset, offset + n))            # self loops
        cols += list(range(offset, offset + n))
        gid += [g_i] * n
        root += [offset] * n
        ys.append(g["y"]); lens.append(n)
        gstat.append([np.log1p(n), np.log1p(g.get("deadline", float(g["t"][-1])))])
        size.append(np.log1p(g["n_full"]))
        offset += n
    idx = torch.tensor([rows, cols], dtype=torch.long)
    adj = torch.sparse_coo_tensor(idx, torch.ones(idx.shape[1]), (offset, offset)).coalesce()
    dinv = torch.sparse.sum(adj, dim=1).to_dense().pow(-0.5)
    i, j = adj.indices()
    adj = torch.sparse_coo_tensor(adj.indices(), dinv[i] * adj.values() * dinv[j], (offset, offset))
    return dict(x=torch.from_numpy(np.concatenate(xs)).float().to(device),
                tfeat=torch.from_numpy(np.concatenate(tf)).float().to(device),
                adj=adj.to(device), gid=torch.tensor(gid, device=device), lengths=lens,
                e_par=torch.tensor(e_par, dtype=torch.long, device=device),
                e_chd=torch.tensor(e_chd, dtype=torch.long, device=device),
                root=torch.tensor(root, device=device), n_nodes=offset,
                gstat=torch.tensor(gstat, dtype=torch.float32, device=device),
                y=torch.tensor(ys, device=device),
                size=torch.tensor(size, dtype=torch.float32, device=device))


def mean_pool(h, gid, n_graphs):
    out = torch.zeros(n_graphs, h.shape[1], device=h.device).index_add_(0, gid, h)
    cnt = torch.zeros(n_graphs, device=h.device).index_add_(0, gid, torch.ones(h.shape[0], device=h.device))
    return out / cnt.clamp(min=1).unsqueeze(1)


def _both_ways(b):
    """Undirected edge list (src -> dst) with self loops."""
    loops = torch.arange(b["n_nodes"], device=b["x"].device)
    return torch.cat([b["e_par"], b["e_chd"], loops]), torch.cat([b["e_chd"], b["e_par"], loops])
