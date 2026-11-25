# Reproducibility

| Result | Evidence and instructions | Inputs |
| --- | --- | --- |
| Paired official-test IoU and F1, coverage, selected-run checks | [LOCO](LOCO.md) | Included counts. Extended verification uses local MLflow, checkpoints, and Git objects. |
| Figure and chip-deletion analysis | `scripts/make_results.py` and `sections/tables/chip_deletion.csv` | Included counts. Matplotlib 3.10.8 for figure generation. |
| Manifest membership and spatial overlap | [LOCO](LOCO.md) and `evidence/loco_input_audit.json` | Historical companion Git history and local processed GeoTIFF headers, GDAL 3.12.4. |
| Class inventory, preprocessing exposure, and historical ablations | [Supporting analyses](SUPPORTING_ANALYSES.md) | Local rasters, logs, weights and Git objects as specified. |
| Historical Azerbaijan prototype products | [Deployment audit](deployment/README.md) | Companion Git products and local inference artifacts. |
| Current upload endpoint | [Upload replay](deployment/README.md#current-upload-replay) | Reviewed ML/API commit, original scene bytes, checkpoint, and JRC/DEM files. |
| Historical inference | [Frozen recipe](historical-inference/README.md) | Selected weights and original Sen1Floods11 rasters with pinned Python dependencies. |

## Timeline of results

The training runs and their original all-chip metrics predate this release.
The corrected nine comparisons come from an October 2026 frozen re-evaluation.
The November 2026 audit independently verifies those scores from chip counts,
checks run and raster provenance, and adds chip deletion, footprint, class
inventory and deployment analyses. The November 25 release carries those
results forward and adds a multipart HTTP and offline CLI replay with
corrected terrain geometry.

Definitions and numeric values in the current [research audit](../RESEARCH_AUDIT.md)
and manuscript supersede the earlier reconstruction's weak-group count
and its open footprint question. The public evidence preserves the
numerical records, checkpoint/run IDs and per-chip CSVs. Its integrity
manifest is computed after removing machine-specific paths and unrelated
workspace inventory from the exported metadata.

## Environment

The dependency-free checks were run with Python 3.14.10. Geospatial checks
used Python GDAL 3.12.4. The TeX build used TeX Live 2026, pdfTeX 1.40.29,
latexmk 4.89, and BibTeX. The plotted PDF uses embedded DejaVu Sans.
Archived model inference environment details are in
[`runtime_and_sources.json`](../results/a1_reconstruction/runtime_and_sources.json).
Model resources listed in the frozen selection were available locally. Public
availability of all selected weight files and MLflow records is unverified.
