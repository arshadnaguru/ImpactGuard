#!/usr/bin/env bash
# Full experiment: 2 classic baselines + 5 neural baselines + 2 ablations + ImpactGuard-T, 3 seeds each,
# then the comparison table. Run from the repo root.
#   bash scripts/run_all.sh                         # main split (stratified 70/15/15)
#   bash scripts/run_all.sh data/pheme runs_event --split event --test-event ferguson
set -e
DATA=${1:-data/pheme}; OUT=${2:-runs}; EXTRA="${@:3}"
python -m src.classic_baselines --data $DATA --out $OUT $EXTRA
for SEED in 42 43 44; do
  for M in "mlp" "gcn" "gat" "gin" "bigcn" "bigcn --prefix-aug" "gcn --prefix-aug" "impactguard"; do
    N=$(echo $M | sed 's/ --prefix-aug/_aug/')
    # skip runs that already finished, so the script can be re-run after an interruption
    [ -f $OUT/${N}_seed$SEED/metrics.json ] || python -m src.train --data $DATA --model $M --seed $SEED --out $OUT $EXTRA
  done
done
python -m src.eval.compare $OUT
