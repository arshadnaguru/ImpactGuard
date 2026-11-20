"""All models share one interface: forward(batch) -> (class_logits [B, 2], size_pred [B] or None)."""
from .graph_batch import batch_graphs, mean_pool
from .baselines import MeanMLP, StaticGCN, GAT, GIN, BiGCN
from .impactguard import ImpactGuardT

MODELS = {"mlp": MeanMLP, "gcn": StaticGCN, "gat": GAT, "gin": GIN, "bigcn": BiGCN, "impactguard": ImpactGuardT}


def build_model(name, in_dim, hidden=64):
    return MODELS[name](in_dim, hidden)
