# Interpolation vs. Generalization in SAR Water Segmentation

**An Event-Held-Out Study on Sen1Floods11 with an Azerbaijan Mapping Prototype**

Ali Agabalazade, Nigar Rustamova, Jala Suleymanova, and Isgandar Panahov

[Read the paper (PDF)](manuscript.pdf) · [Citation](CITATION.cff) · [Reproducibility](reproducibility/README.md)

## Study

How does a model trained without a particular flood event perform on that event?
We compared event-held-out DeepLabV3+/ResNet-50 checkpoints with a fixed
generalist on the same official Sen1Floods11 test chips. We also built a
SegFormer-B4 mapping prototype for Baku and Sabirabad that filters predicted
water using JRC water occurrence and terrain masks.

## Key findings

- The benchmark covers nine recoverable event–seed comparisons across seven
  events and 51 unique paired test chips. Water IoU favored the event-held-out
  model in five comparisons and the generalist in four.
- In seven of the nine comparisons, leaving out a single test chip could reverse
  which model scored higher. The reported differences describe the surviving
  checkpoints and observed test chips.
- The Azerbaijan case studies show how filtering changes the water masks.
  Neither scene has time-matched inundation labels, so flood-map accuracy
  remains unmeasured.

## Reproduce the tables and figures

Run `make verify` to recompute the paired results and all 63 single-chip
omissions from the published counts. Run `make results` with Matplotlib to
regenerate the result tables and figure. Run `make manuscript` with TeX Live,
BibTeX, and latexmk to build the PDF.

The [reproducibility guide](reproducibility/README.md) documents the evaluation
inputs and replay procedures. The
[companion ML implementation](https://github.com/AliAgabalayev/Flood-Damage-Detection/commit/23563a7bebf28e508b394cf6858649718fd148a3)
contains the terrain-corrected upload API and offline CLI.
