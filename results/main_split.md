# Main split: stratified 70/15/15, 871 test cascades

Mean ± std over 3 training seeds. AUROC@Xm = only the first X minutes of each cascade are visible. SizeMAE@30m = mean absolute error of log(1 + final size) at 30 minutes (lower is better); for the two classic rows it is the train-mean predictor and a linear fit on the observed size.

| Method | seeds | AUROC | F1 | Acc | AUROC@10m | AUROC@30m | AUROC@60m | SizeMAE@30m |
|---|---|---|---|---|---|---|---|---|
| Majority class | 1 | 0.500 | 0.000 | 0.660 | 0.500 | 0.500 | 0.500 | 0.686 |
| TF-IDF + LogReg | 1 | 0.887 | 0.726 | 0.781 | 0.899 | 0.896 | 0.894 | 0.418 |
| Mean-pool MLP | 3 | 0.835 ± 0.001 | 0.676 ± 0.007 | 0.754 ± 0.008 | 0.837 ± 0.001 | 0.833 ± 0.000 | 0.837 ± 0.001 | - |
| Static GCN (Ckpt 2) | 3 | 0.838 ± 0.002 | 0.684 ± 0.005 | 0.761 ± 0.011 | 0.842 ± 0.002 | 0.835 ± 0.001 | 0.837 ± 0.001 | - |
| GAT (2018) | 3 | 0.852 ± 0.000 | 0.702 ± 0.007 | 0.768 ± 0.016 | 0.851 ± 0.001 | 0.848 ± 0.000 | 0.850 ± 0.000 | - |
| GIN (2019) | 3 | 0.847 ± 0.001 | 0.711 ± 0.005 | 0.783 ± 0.004 | 0.851 ± 0.000 | 0.841 ± 0.001 | 0.843 ± 0.001 | - |
| Bi-GCN (2020) | 3 | 0.862 ± 0.002 | 0.714 ± 0.007 | 0.801 ± 0.007 | 0.858 ± 0.003 | 0.859 ± 0.002 | 0.861 ± 0.002 | - |
| Bi-GCN + prefix training | 3 | 0.865 ± 0.002 | 0.715 ± 0.004 | 0.799 ± 0.007 | 0.864 ± 0.001 | 0.864 ± 0.001 | 0.864 ± 0.001 | - |
| Static GCN + prefix training | 3 | 0.838 ± 0.001 | 0.692 ± 0.007 | 0.769 ± 0.004 | 0.841 ± 0.002 | 0.837 ± 0.002 | 0.839 ± 0.001 | - |
| ImpactGuard-T (ours) | 3 | 0.861 ± 0.002 | 0.711 ± 0.010 | 0.776 ± 0.011 | 0.859 ± 0.001 | 0.860 ± 0.003 | 0.860 ± 0.002 | 0.381 ± 0.009 |

## Difference of ImpactGuard-T to two key baselines

| Metric | vs Static GCN (Ckpt 2) | vs Bi-GCN (2020) |
|---|---|---|
| AUROC | +0.022 | -0.002 |
| F1 | +0.027 | -0.003 |
| Accuracy | +0.015 | -0.025 |
| AUROC@10m | +0.017 | +0.001 |
| AUROC@30m | +0.025 | +0.000 |
| AUROC@60m | +0.022 | -0.001 |
