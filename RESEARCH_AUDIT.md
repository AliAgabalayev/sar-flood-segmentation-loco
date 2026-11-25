# Research audit

The benchmark is a retrospective, partial comparison of water segmentation on
fixed Sen1Floods11 test chips. The separate Azerbaijan prototype produces
unlabeled maps.

The historical implementation is pinned to
`867a4f3e3dae9a935578e401f5c5f970e3d0adac`. The October 2026 reconstruction
selected completed runs by validation IoU before running paired inference
on the official test chips. The November audit checked model and data
lineage, recomputed the released results and examined chip sensitivity,
spatial overlap and class frequencies.

## Official-test comparison

| Issue | Finding and disposition |
| --- | --- |
| Historical all-chip generalist comparator | Among 431 standard chips used in the historical comparison, 252 had supplied hand training data and 89 had supplied validation data for the generalist. Same-chip comparison alone did not prevent training or selection reuse. Retired the 11-event mean 0.575, the supposedly uniform negative delta and mean −0.053, and the associated rankings and intervals as headline findings. |
| Reconstructed official test | 90 official chips in ten events, with 15 separate Bolivia holdout chips outside this comparison. The held-out architecture is DeepLabV3+/ResNet-50. Fixed sigmoid >0.5, no TTA or fusion, water-class pooled TP/FP/FN, identical valid masks. Two chips with no valid pixels are preserved in the manifest and yield undefined individual scores. |
| Checkpoint and run evidence | Recomputed ten file hashes and verified completed MLflow runs, 50 validation scores per selected fine-tune, and validation maxima. Artifact-directory Nigeria s43 has a direct run link. The remaining model-directory mappings are supported by hashes and callback/MLflow evidence. Current committed manifests and run paths support reconstructed historical membership. Exact historical reads, original pretraining linkage and the USA retry initializer remain unresolved. USA uses completed best-v1 from `46bcbbc82c68448a95b73ac39feb5725`. The original evaluator used interrupted best.ckpt. |
| Retained evidence | Nine seed-specific pairs, 51/90 unique test chips, seven events. The generalist covers 90/90 chips and pools to IoU 0.676563854 and F1 0.807083909. Across the nine pairs five deltas are positive, four negative, range −0.033583 to +0.025312. Mekong and Pakistan reverse sign across the available seeds. Ghana, India and Paraguay lack LOCO evidence, and 21 of 30 planned pairs are unavailable. |
| Statistical stability | Recomputed all 63 one-chip omissions from paired counts. Seven of nine pairs reverse sign in at least one omission. Pakistan seed 44 spans −0.2220 to +0.0963. These are deterministic intervals on the observed sample. The one generalist seed and uneven surviving LOCO runs preclude reliable event-level uncertainty, ranking or a population estimate. |
| Train and validation roles | Generalist 252/89 hand and 3,946/438 weak. Each official LOCO manifest removes exactly the named group from each of the four stages. The weak manifests contain 12 filename groups including Bolivia and Colombia. The prior reconstruction counted 11. LOCO hand training has 211–242 entries, LOCO hand validation 75–85. |
| Leakage and spatial dependence | No canonical test IDs overlap the known hand/weak training or validation manifests. GDAL found zero positive-area interior crossings above 1e−8° across 425,250 test-to-role tile pairs. Seventy-three test tiles touch weak train tiles and 35 touch hand train tiles within the stated tolerance. This checks current grids. Scene-level, temporal and undocumented derived-sample independence remain unresolved. |
| Data-dependent preprocessing | Historical clipping constants were justified with full-catalog exploratory SAR summaries. The reconstructed 80-chip exploratory sample contains 16 official test chips, three Bolivia chips and all eleven hand groups. Clipping therefore used benchmark covariates. A historical weighted BCE candidate also used a class weight based on full-catalog labels. The selected Dice LOCO recipe has no BCE class weight. |
| Size-matching control | Twelve logged official-test evaluations across Somalia, Spain, Sri Lanka and USA, one seed and one replacement draw each. Ctrl minus LOCO is −0.018847, −0.004847, −0.023419 and −0.267960, respectively. Full minus LOCO has mixed signs. The older validation-plus-test shortcut compares to a generalist that used validation for selection. The retained pilot is descriptive. Draws, budgets and training variability remain confounded. |

Source checks, IDs, row-level score definitions and commands are in the
[LOCO evidence audit](reproducibility/LOCO.md). Chip deletion outputs are in
[chip_deletion.csv](sections/tables/chip_deletion.csv).

## Other training and ablation evidence

- Across finite valid SAR pixels, water occupies 9.509% of official training
  labels and 12.302% of official test labels. The entire 446-chip inventory
  gives 10.558%. The manuscript avoids treating the full-catalog proportion
  as a training-only statistic.
- The archived two-stage and single-stage DeepLabV3+ runs log test IoU
  0.676564 and 0.673172, respectively. The +0.003392 difference is a single
  seed under closely matched logged settings and different source snapshots.
  It does not quantify a repeatable pretraining advantage.
- The short loss sweep and the 50-epoch finalist runs are distinct. The
  short sweep's Dice and focal Tversky results were within 0.000847 IoU.
  The finalist ranking differs. The final manuscript drops the generalized
  loss-superiority and architecture-superiority claims.
- The reported SegFormer-B4 test IoU of 0.702703 is a historical benchmark
  result from a different run and architecture. The default variant was
  historically selected using test comparisons, so this score is
  descriptive. The historical fused evaluation metrics have insufficient protocol provenance
  for an accuracy comparison.

[Supporting analyses and commands](reproducibility/SUPPORTING_ANALYSES.md)

## Deployment and post-processing

The offline viewer archives are historical products. Their recorded counts
and reference grids can be compared with fresh SegFormer-B4 inference. The
corrected November 25, 2026 run applies the same JRC recurrence rule and
coverage policy with the corrected signed terrain incidence in the API
and offline CLI. The deployment reproduction record compares those
masks. No co-temporal inundation labels were available. The area
changes quantify the filtering effect on these maps.

[Deployment audit and runnable checks](reproducibility/deployment/README.md)

## Remaining evidence gaps

The reported benchmark compares the surviving weights on fixed chips.
A prospective geographic-transfer estimate would require training-only or
externally fixed preprocessing, geographic and scene-aware splitting,
run-time split and initializer hashes, matched training and validation designs,
and repeated seeds for both arms across additional independent events.
The released benchmark numbers can be verified from the frozen confusion
counts. Recovery of missing historical checkpoints could expand the
retrospective results. Co-temporal inundation labels would be required
to validate the prototype.
