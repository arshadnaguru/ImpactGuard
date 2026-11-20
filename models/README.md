# Trained model checkpoints

| File | What it is |
|---|---|
| `impactguard_seed42.pt` | ImpactGuard-T trained on the main split (seed 42): weights, training arguments, best epoch and the decision threshold chosen on validation |
| `impactguard_seed42_featurizer.pkl` | the TF-IDF + SVD text featurizer fitted on that run's training threads; needed to score new threads |

Use them without training:
```bash
python -m src.predict  --checkpoint models/impactguard_seed42.pt --data data/samples/pheme_samples --deadline 30
python -m src.evaluate --checkpoint models/impactguard_seed42.pt --data data/pheme      # needs the full corpus
```

Every new training run also writes `best.pt`, `last.pt` and `featurizer.pkl` to `runs/<model>_seed<k>/`
(not tracked by git). To publish another checkpoint, copy it here with the same naming.
