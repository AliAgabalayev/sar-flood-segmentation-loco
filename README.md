# Interpolation vs. Generalization in SAR Water Segmentation

A retrospective Sen1Floods11 study and an Azerbaijan prototype, by Ali
Agabalazade, Nigar Rustamova, Jala Suleymanova and Isgandar Panahov.

[Manuscript PDF](manuscript.pdf) · [Research audit](RESEARCH_AUDIT.md) ·
[Reproducibility](reproducibility/README.md) ·
[ML implementation repository](https://github.com/AliAgabalayev/Flood-Damage-Detection)

The original event comparison scored the generalist on some of its own
training chips. The paired official-test reconstruction covers nine
event–seed combinations across seven events and yields both positive and
negative water-IoU differences. Omitting one paired test chip can reverse
the difference in seven comparisons. The Azerbaijan results describe
unlabeled map products, with inundation accuracy unmeasured.

## Reproduce the results

From this repository, verify the frozen official-test counts and
all 63 single-chip omissions without imagery, weights, or a GPU.

```sh
make verify
```

Run `make results` with Matplotlib installed to generate the paired table,
deletion CSV, vector figure and supporting tables. Figure 1 is drawn
from its TikZ source during the LaTeX build. The frozen checkpoint
selection and confusion counts are in `results/a1_reconstruction/`.
The [reproducibility index](reproducibility/README.md) documents the original
split and source audits, the offline archive and the new upload replay.

The historical benchmark uses ML commit
[`867a4f3`](https://github.com/AliAgabalayev/Flood-Damage-Detection/commit/867a4f3e3dae9a935578e401f5c5f970e3d0adac).
The upload and offline replay use the
[terrain-corrected ML commit](https://github.com/AliAgabalayev/Flood-Damage-Detection/commit/23563a7bebf28e508b394cf6858649718fd148a3).
[Deployment reproduction](reproducibility/deployment/README.md) records the
historical and corrected outputs by scene.

## Build the manuscript

With TeX Live, pdfLaTeX, BibTeX and latexmk installed:

```sh
make manuscript
```

The entry point is `main.tex`, with text and tables under `sections/` and
figures under `figures/`. Run `make arxiv` to create a source archive
under `.build/`. The repository includes the final PDF and the generated
bibliography.

Cite the work using [CITATION.cff](CITATION.cff). No DOI or arXiv identifier
has been assigned. The [Sen1Floods11 dataset repository](https://github.com/cloudtostreet/Sen1Floods11)
provides its availability and terms. The upstream implementation and datasets
retain their respective terms.
