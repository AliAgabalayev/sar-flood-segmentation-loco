# LOCO benchmark reproduction

## Recompute the released scores

From the manuscript repository root, run the following with Python 3.11 or
newer. These checks need the released confusion counts and provenance files
under `reproducibility/benchmark-results`.

```sh
python3 scripts/verify_loco_evidence.py
```

The script verifies the evidence checksums, model/run identity, paired image,
label and valid-mask hashes, coverage, and arithmetic in the summary CSV and
JSON. It sums flood-class counts within each event and checkpoint. IoU is
TP/(TP+FP+FN), and F1 is 2TP/(2TP+FP+FN). All-ignored chips keep their manifest
entries and contribute no counts. The official test chips were evaluated with
the same threshold and valid-pixel policy for both arms. Delta is LOCO IoU
minus generalist IoU.

For a deeper local audit with the companion repository, provision its archived
`mlflow.db`, selected checkpoint files, and processed data at the paths in the
frozen selection. Ensure its Git object database includes the official, weak and
LOCO split commits in the footprint JSON. The GDAL Python bindings are needed
only for the footprint audit. The checked output used GDAL 3.12.4.

```sh
python3 scripts/verify_loco_evidence.py --ml-root ../Flood-Damage-Detection
python3 scripts/audit_loco_inputs.py ../Flood-Damage-Detection
python3 scripts/audit_size_control.py ../Flood-Damage-Detection
```

The first command with `--ml-root` checks the exact official test IDs and all
ten checkpoint files, including the archived MLflow records for completed
fine-tuning runs and validation maxima. The commands read the companion
repository without modifying it. Their released outputs are in
[evidence](evidence/). The size-control command also requires the original
September CSV and generated control manifests from the research workspace, which
were separate from the tracked companion repository. It exports the logged
metrics and run IDs without rerunning model inference.

## Evaluation design and limitations

- The released scores come from an October 2026 re-evaluation of existing
  checkpoints. The current audit independently confirms the nine paired rows from
  per-chip counts. Seven of ten official event groups and 51 of 90 official test
  chips have a paired result. The generalist covers all 90 chips. Twenty-one
  event/seed pairs lack surviving selected weights or equivalent test-only
  prediction evidence. A single generalist seed is reused in every pair.
- The historical all-chip comparison put all 431 official train, validation and
  test rows in the event-specific sets. Of those, 252 belonged to generalist
  training and 89 to generalist validation. The official-test reconstruction
  uses the 90 test rows. The separately listed 15 Bolivia hand chips are outside
  that test set.
- Current official training is 252 hand chips and current official validation is
  89. In the LOCO Git manifests those counts range from 211 to 242 and 75 to 85
  respectively. The weak pool contains 3,946 training and 438 validation chips
  and covers **12** filename groups, including the ten official hand groups plus
  Bolivia and Colombia. With an official group excluded, eleven weak groups
  remain.
- The frozen run/weight mapping is supported by file hashes, validation callback
  values, recorded run statuses, logged stage paths and compatible step counts.
  The historical process-read manifests and initialization lineage remain
  reconstructed. The USA comparison uses the completed validation-selected
  retry, while the older all-chip evaluation used the interrupted attempt via
  the default `best.ckpt` filename. The completed fine-tune initialization
  checkpoint remains unresolved.
- The immutable manifests have zero canonical-ID overlap between official test
  and hand/weak training or validation. They also show that the 40 LOCO
  training/validation manifests consist of the corresponding generalist
  membership with the target group removed. The GeoTIFF header audit compares
  90 official-test SAR footprints against the 4,725 known training/validation
  SAR footprints. None intersects on both axes beyond a tolerance of
  `1e−8` degrees. Contacts on tile boundaries or within that tolerance
  affect 73 test chips against weak train (291 pairs) and 35 against hand train
  (60 pairs). Acquisition, scene and spatial independence are consequently
  unestablished. The audit checks available local TIFF headers and Git
  manifests and leaves undocumented derivations and process-read exposure open.
- The clipping policy also carries information from the test pool. Companion
  commit `5f0c2e94e53f2a5d6be8cef26db8a07170288e80` introduced preprocessing
  constants alongside a SAR exploratory analysis and annotated those constants
  with percentiles of all 446 hand SAR chips. The historical test comparison
  uses the common fixed policy with this design limitation. A prospective
  evaluation should declare a fixed external policy or obtain constants from
  training imagery alone.

## Historical size-matched control

The companion CSV and MLflow records establish 12 finished evaluations, one arm
each for four groups at seed 42. The run for subsequent seed-43 Somalia
pretraining failed. Current manifest rows implement an equal-sized
`ctrl`/LOCO contrast by replacing one draw of non-target training chips with
the same number of target-event training chips. The third, `full`, arm
uses all 252 hand training rows. The control recipe uses target-excluded weak
pretraining and hand validation for the three arms within a fold. It reuses
the historical LOCO checkpoint in three folds and refits all three arms in USA.
Exact fresh fine-tuning checkpoint files and process-read split manifests are
unavailable.

| Group | Test chips | LOCO | Ctrl | Full | Ctrl − LOCO | Full − LOCO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Somalia | 6 | 0.515104 | 0.496257 | 0.549553 | −0.018847 | +0.034449 |
| Spain | 6 | 0.721537 | 0.716690 | 0.710674 | −0.004847 | −0.010863 |
| Sri Lanka | 9 | 0.837732 | 0.814313 | 0.845993 | −0.023419 | +0.008261 |
| USA | 14 | 0.541676 | 0.273716 | 0.523093 | −0.267960 | −0.018582 |

The table extracts official-test scores already logged by the historical
control runs. The old document emphasized validation-plus-test. That composite
can describe the three control arms with target-excluded validation, but the
corresponding generalist scores include chips used in its model selection. The
control comparison here is descriptive. With one seed, one swap per group and
only four groups, the arm contrasts cannot isolate geographic exposure from
sample composition and training variability. The `full` arm is a separate
fit from the October generalist checkpoint. We report individual point
estimates and omit inferential intervals and population-level conclusions.
