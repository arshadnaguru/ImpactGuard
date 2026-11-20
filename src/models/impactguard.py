"""ImpactGuard-T: the proposed time-aware cascade model (core implementation)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_batch import mean_pool


class ImpactGuardT(nn.Module):
    """Proposed core model: a time-aware cascade encoder with two heads.

    1. Time encoding  : each tweet gets a learnable cosine encoding of its arrival time, plus
                        its delay after the parent tweet and its depth in the reply tree.
    2. Graph encoder  : two GCN layers on the reply graph (who replied to whom).
    3. Temporal encoder: a GRU reads the tweets in arrival order (how the cascade unfolded).
    4. Attention pool : learns which moments of the cascade matter most.
    5. Heads          : (a) rumour vs non-rumour, (b) final cascade size = impact.
    Trained on random time-prefixes of each cascade, so it can be asked at any early deadline.
    """
    def __init__(self, in_dim, hidden=64, dropout=0.3, time_dim=16):
        super().__init__()
        self.time_w = nn.Linear(1, time_dim)
        self.inp = nn.Linear(in_dim + time_dim + 3, hidden)
        self.g1 = nn.Linear(hidden, hidden)
        self.g2 = nn.Linear(hidden, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.att = nn.Linear(hidden, 1)
        self.stat = nn.Linear(2, 16)
        self.mlp = nn.Sequential(nn.Linear(3 * hidden + 16, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.cls_head = nn.Linear(hidden, 2)
        self.size_head = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, b):
        n_graphs, lengths = len(b["lengths"]), b["lengths"]
        t_enc = torch.cos(self.time_w(b["tfeat"][:, :1]))                    # 1. time encoding
        h0 = F.relu(self.inp(torch.cat([b["x"], t_enc, b["tfeat"]], 1)))
        h = F.relu(torch.sparse.mm(b["adj"], self.g1(h0)))                   # 2. graph encoder
        h = F.dropout(h, self.dropout, self.training)
        h = F.relu(torch.sparse.mm(b["adj"], self.g2(h))) + h0               #    (residual)
        seq = nn.utils.rnn.pad_sequence(torch.split(h, lengths), batch_first=True)
        packed = nn.utils.rnn.pack_padded_sequence(seq, torch.tensor(lengths), batch_first=True,
                                                   enforce_sorted=False)
        out, last = self.gru(packed)                                         # 3. temporal encoder
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        mask = torch.arange(out.shape[1], device=h.device)[None, :] < torch.tensor(lengths, device=h.device)[:, None]
        score = self.att(out).squeeze(-1).masked_fill(~mask, -1e9)
        att = (torch.softmax(score, 1).unsqueeze(-1) * out).sum(1)           # 4. attention pooling
        z = torch.cat([att, last.squeeze(0), mean_pool(h, b["gid"], n_graphs), F.relu(self.stat(b["gstat"]))], 1)
        z = self.mlp(z)
        return self.cls_head(z), self.size_head(z).squeeze(-1)               # 5. two heads
