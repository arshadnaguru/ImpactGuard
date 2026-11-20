"""Neural baselines: MeanMLP (no graph), StaticGCN (Ckpt 2), GAT, GIN and Bi-GCN."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_batch import mean_pool, _both_ways


class MeanMLP(nn.Module):
    """Baseline: no graph, no time. Average the tweet features and classify."""
    def __init__(self, in_dim, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.out = nn.Linear(hidden, 2)

    def forward(self, b):
        n = len(b["lengths"])
        return self.out(self.net(mean_pool(b["x"], b["gid"], n))), None


class StaticGCN(nn.Module):
    """Baseline from Ckpt 2: two GCN layers on the reply graph + mean pooling. No notion of time."""
    def __init__(self, in_dim, hidden=64, dropout=0.3):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, b):
        h = F.relu(torch.sparse.mm(b["adj"], self.l1(b["x"])))
        h = F.dropout(h, self.dropout, self.training)
        h = F.relu(torch.sparse.mm(b["adj"], self.l2(h)))
        pooled = mean_pool(h, b["gid"], len(b["lengths"]))
        return self.out(F.dropout(pooled, self.dropout, self.training)), None


class GATLayer(nn.Module):
    """Graph attention layer (Velickovic et al., ICLR 2018), multi-head, heads concatenated."""
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.h, self.d = heads, out_dim // heads
        self.lin = nn.Linear(in_dim, self.h * self.d, bias=False)
        self.a_src = nn.Parameter(torch.randn(self.h, self.d) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(self.h, self.d) * 0.1)

    def forward(self, x, src, dst):
        n = x.shape[0]
        z = self.lin(x).view(n, self.h, self.d)
        e = F.leaky_relu((z * self.a_src).sum(-1)[src] + (z * self.a_dst).sum(-1)[dst], 0.2)      # [E, heads]
        e_max = torch.full((n, self.h), -1e9, device=x.device).scatter_reduce(0, dst[:, None].expand_as(e), e, "amax")
        w = torch.exp(e - e_max[dst])
        w = w / torch.zeros(n, self.h, device=x.device).index_add_(0, dst, w)[dst].clamp(min=1e-9)   # softmax over neighbours
        out = torch.zeros(n, self.h, self.d, device=x.device).index_add_(0, dst, w.unsqueeze(-1) * z[src])
        return out.reshape(n, self.h * self.d)


class GAT(nn.Module):
    """Baseline: 2-layer graph attention network + mean pooling."""
    def __init__(self, in_dim, hidden=64, dropout=0.3):
        super().__init__()
        self.g1, self.g2 = GATLayer(in_dim, hidden), GATLayer(hidden, hidden)
        self.out = nn.Linear(hidden, 2); self.dropout = dropout

    def forward(self, b):
        src, dst = _both_ways(b)
        h = F.dropout(F.elu(self.g1(b["x"], src, dst)), self.dropout, self.training)
        h = F.elu(self.g2(h, src, dst))
        return self.out(F.dropout(mean_pool(h, b["gid"], len(b["lengths"])), self.dropout, self.training)), None


class GIN(nn.Module):
    """Baseline: Graph Isomorphism Network (Xu et al., ICLR 2019), sum aggregation + mean pooling."""
    def __init__(self, in_dim, hidden=64, dropout=0.3):
        super().__init__()
        mlp = lambda i: nn.Sequential(nn.Linear(i, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.m1, self.m2 = mlp(in_dim), mlp(hidden)
        self.eps = nn.Parameter(torch.zeros(2))
        self.out = nn.Linear(hidden, 2); self.dropout = dropout

    def _agg(self, h, b):
        s = torch.zeros_like(h)
        s = s.index_add(0, b["e_chd"], h[b["e_par"]]).index_add(0, b["e_par"], h[b["e_chd"]])
        return s

    def forward(self, b):
        h = F.relu(self.m1((1 + self.eps[0]) * b["x"] + self._agg(b["x"], b)))
        h = F.dropout(h, self.dropout, self.training)
        h = F.relu(self.m2((1 + self.eps[1]) * h + self._agg(h, b)))
        return self.out(F.dropout(mean_pool(h, b["gid"], len(b["lengths"])), self.dropout, self.training)), None


class BiGCN(nn.Module):
    """Baseline: Bi-Directional GCN for rumour detection (Bian et al., AAAI 2020).

    One GCN follows the reply tree top-down (propagation), one bottom-up (dispersion); both use
    root feature enhancement (the source tweet is concatenated to every node). Pools are concatenated.
    """
    def __init__(self, in_dim, hidden=64, dropout=0.3):
        super().__init__()
        mk = lambda: nn.ModuleList([nn.Linear(in_dim, hidden), nn.Linear(hidden + in_dim, hidden)])
        self.td, self.bu = mk(), mk()
        self.out = nn.Linear(4 * hidden, 2); self.dropout = dropout

    @staticmethod
    def _prop(h, src, dst, n):
        """Row-normalised directed propagation with self loops: mean over {self} + in-neighbours."""
        loops = torch.arange(n, device=h.device)
        s, d = torch.cat([src, loops]), torch.cat([dst, loops])
        deg = torch.zeros(n, device=h.device).index_add_(0, d, torch.ones(d.shape[0], device=h.device))
        return torch.zeros_like(h).index_add_(0, d, h[s]) / deg.unsqueeze(1)

    def _branch(self, layers, b, src, dst):
        n, x = b["n_nodes"], b["x"]
        h1 = F.relu(self._prop(layers[0](x), src, dst, n))
        h1 = F.dropout(torch.cat([h1, x[b["root"]]], 1), self.dropout, self.training)        # root enhancement
        h2 = F.relu(self._prop(layers[1](h1), src, dst, n))
        h2 = torch.cat([h2, h1[b["root"]][:, :h2.shape[1]]], 1)                               # root enhancement
        return mean_pool(h2, b["gid"], len(b["lengths"]))

    def forward(self, b):
        z = torch.cat([self._branch(self.td, b, b["e_par"], b["e_chd"]),
                       self._branch(self.bu, b, b["e_chd"], b["e_par"])], 1)
        return self.out(z), None
