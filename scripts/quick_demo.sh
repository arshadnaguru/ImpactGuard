#!/usr/bin/env bash
# 2-minute smoke test on the bundled real samples (data/samples/pheme_samples.tar.gz), CPU is fine.
# Numbers from 120 threads are NOT meaningful; this only shows the pipeline runs end to end.
set -e
DATA=${1:-data/samples/pheme_samples}; OUT=runs_demo
if [ ! -d "$DATA" ] && [ -f "$DATA.tar.gz" ]; then tar -xzf "$DATA.tar.gz" -C "$(dirname "$DATA")"; fi
python -m src.train --data $DATA --model gcn --epochs 5 --out $OUT
python -m src.train --data $DATA --model impactguard --epochs 5 --out $OUT
python -m src.eval.compare $OUT
python -m src.infer_demo --ckpt $OUT/impactguard_seed42/best.pt --data $DATA --n 6
