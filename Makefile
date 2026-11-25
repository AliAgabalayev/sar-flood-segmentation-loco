LATEXMK ?= latexmk
BUILD_DIR = .build
TEX_SOURCES = main.tex main.bib llncs.cls splncs04.bst $(wildcard sections/*.tex) $(wildcard sections/tables/*.tex) $(wildcard figures/*.pdf) $(wildcard figures/*.tex)

# latexmk can use the tracked bibliography without writing an output copy.
BIBLIOGRAPHY_DIR = $(if $(wildcard $(BUILD_DIR)/main.bbl),$(BUILD_DIR),.)

.PHONY: manuscript arxiv results verify

manuscript: $(BUILD_DIR)/main.pdf

$(BUILD_DIR)/main.pdf: $(TEX_SOURCES) main.bbl
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILD_DIR) main.tex

arxiv: manuscript
	tar -czf $(BUILD_DIR)/arxiv-source.tar.gz $(TEX_SOURCES) -C $(BIBLIOGRAPHY_DIR) main.bbl

verify:
	python3 scripts/verify_loco_evidence.py

results:
	python3 scripts/make_results.py --figure
	python3 scripts/make_supporting_tables.py
