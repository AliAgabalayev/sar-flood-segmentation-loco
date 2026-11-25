# Supporting analyses

## Current dataset audit

`label_inventory.json` counts water and nonwater on finite two-band SAR pixels with hand labels 0 or 1. The official hand training split has 5,446,678 water labels among 57,279,257 valid pixels (9.51%). The official test split has 2,515,008 water labels among 20,443,959 valid pixels (12.30%). The 10.6% value in the historical study notes described the 446-chip corpus. The historical notebook used a label-only mask for that count.

The June SAR exploratory analysis sampled 80 images from the entire hand catalog. Sampling the local catalog in the notebook's recorded order with its seed gives 44 official training images, 17 validation images, 16 test images and three Bolivia holdout images spanning all 11 event groups. The notebook's clipping ranges are data-derived. The contemporaneous configuration also annotates the selected ranges with full-catalog percentiles. The historical ranges consequently incorporate SAR covariates from outside the training splits, including the eventual excluded event groups. This remains a limitation of the fixed-checkpoint comparison. The archived loss sweep's 8.4 BCE weighting also used the full hand-label corpus. The selected LOCO Dice configuration has no such weight. [SAR notebook and config commit](https://github.com/AliAgabalayev/Flood-Damage-Detection/commit/5f0c2e94e53f2a5d6be8cef26db8a07170288e80), [historical loss configuration](https://github.com/AliAgabalayev/Flood-Damage-Detection/blob/b531ebf015abc278f7cc6868c3468e0d90315ee1/config/experiments/loss_study.yaml).

Recreate the local inventory with Python, NumPy and GDAL Python bindings.

```sh
python scripts/audit_label_inventory.py /path/to/Flood-Damage-Detection
```

## Historical logs and weights

`historical_runs.json` extracts a subset of the historical local MLflow database, checks the completed single-stage run against its final text log, and hashes the available checkpoints. No model was retrained or scored in this extraction. Checkpoint best-validation values and epoch metadata match their corresponding run records when inspected with CPU PyTorch. The logged source trees and exact historical input membership remain incompletely documented.

The seed-42 DeepLabV3+/ResNet-50 Dice runs log official test IoU of 0.676564 with weak pretraining and 0.673172 with hand-only training. Their recorded fine-tuning configurations agree on optimization and data splits and differ in data-loader workers, checkpoint destination, and inference fields. The runs use different source revisions with recorded working-tree changes. The 0.003392 test difference is descriptive for this pair, with no estimate of between-seed variance. The selected SegFormer-B4 configuration has a separate historical log and checkpoint. Validation IoU selected the checkpoint within each run. The default model was chosen using test IoU, including comparisons with the two B4 variants that had higher validation IoU. Its raw test result is descriptive after selection and gives no transfer evidence for the application prototype. [Default-model change](https://github.com/AliAgabalayev/Flood-Damage-Detection/commit/5e727aaed7b5995c6edeec33e90c4e8a87718c8f).

The tracked short loss sweep gives validation IoU 0.642526 for Dice and 0.643373 for focal Tversky, from one seed per variant. The distinct local 50-epoch finalist runs place Dice first among those three variants. The short and long schedules provide exploratory model-selection records without a replicated effect estimate. [Tracked loss sweep](https://github.com/AliAgabalayev/Flood-Damage-Detection/blob/b531ebf015abc278f7cc6868c3468e0d90315ee1/experiments/loss_study/results.csv).

Extract the same fields with Python 3.11 or later. CPU PyTorch is required for `--inspect-checkpoints`, which adds Lightning metadata to the weight hashes. Run against the archived local database and its corresponding checkpoints. The upstream DVC pointer and a public Git clone may lack the full local database and later single-stage artifacts.

```sh
python scripts/audit_historical_runs.py --inspect-checkpoints /path/to/Flood-Damage-Detection
```
