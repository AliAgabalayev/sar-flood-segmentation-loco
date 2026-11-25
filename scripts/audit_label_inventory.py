#!/usr/bin/env python3
"""Count label classes under the reconstructed evaluation mask and audit EDA membership."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from osgeo import gdal


SPLITS = ("train", "val", "test", "bolivia_holdout")


def eda_quantiles(image_dir, sample):
    vv = []
    vh = []
    ratio = []
    for image in sample:
        raw = gdal.Open(str(image_dir / image)).ReadAsArray().astype(np.float32)
        x, y = raw[:, np.isfinite(raw).all(axis=0)]
        vv.append(x)
        vh.append(y)
        ratio.append(x - y)
    return {
        "finite_sar_pixels": sum(len(chip) for chip in vv),
        "p1_p99_db": {
            label: np.percentile(np.concatenate(chips), [1, 99], overwrite_input=True).tolist()
            for label, chips in (("VV", vv), ("VH", vh), ("VV_minus_VH", ratio))
        },
    }


def audit(root):
    gdal.UseExceptions()
    data = root / "data/processed/sen1floods11"
    membership = {}
    manifests = {}
    totals = {}
    for split in SPLITS:
        path = root / f"data/splits/official/{split}.csv"
        counts = Counter()
        with path.open(newline="") as stream:
            for image, label in csv.reader(stream):
                if image in membership:
                    raise ValueError(f"Duplicate hand-labeled image in split manifests: {image}")
                membership[image] = split
                raw = gdal.Open(str(data / "S1Hand" / image)).ReadAsArray()
                target = gdal.Open(str(data / "LabelHand" / label)).ReadAsArray()
                if raw.shape != (2, 512, 512) or target.shape != (512, 512):
                    raise ValueError(f"Unexpected shape for {image}")
                if not np.isin(target, (-1, 0, 1)).all():
                    raise ValueError(f"Unexpected target value for {label}")
                valid = (target != -1) & np.isfinite(raw).all(axis=0)
                counts["chips"] += 1
                counts["chips_with_valid_pixels"] += int(valid.any())
                counts["total_pixels"] += target.size
                counts["label_ignored_pixels"] += int((target == -1).sum())
                counts["valid_pixels"] += int(valid.sum())
                counts["water_pixels"] += int((valid & (target == 1)).sum())
                counts["nonwater_pixels"] += int((valid & (target == 0)).sum())
        manifests[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        totals[split] = dict(counts)

    # The archived notebook catalogs all hand images with a matching label,
    # sorts by path name, then calls DataFrame.sample(n=80, random_state=42).
    catalog = sorted(
        path.name
        for path in (data / "S1Hand").glob("*.tif")
        if (data / "LabelHand" / path.name.replace("_S1Hand", "_LabelHand")).is_file()
    )
    if set(catalog) != membership.keys():
        raise ValueError("Hand catalog and split membership differ")
    sample = [
        catalog[index]
        for index in np.random.RandomState(42).choice(len(catalog), 80, replace=False)
    ]
    all_counts = Counter()
    for counts in totals.values():
        all_counts.update(counts)
    totals["all_hand"] = dict(all_counts)
    for counts in totals.values():
        counts["ignored_or_nonfinite_pixels"] = counts["total_pixels"] - counts["valid_pixels"]
        counts["water_fraction_valid"] = counts["water_pixels"] / counts["valid_pixels"]
    return {
        "method": "Labels 0/1 counted where label != -1 and both raw SAR bands are finite.",
        "manifest_sha256": manifests,
        "current_hand_pixel_inventory": totals,
        "historical_eda_sample_membership_reconstruction": {
            "notebook": "5f0c2e94e53f2a5d6be8cef26db8a07170288e80:notebooks/01_sen1floods11_sar_eda.ipynb",
            "catalog_size": len(catalog),
            "sample_size": len(sample),
            "sample_seed": 42,
            "by_split": dict(Counter(membership[image] for image in sample)),
            "by_event": dict(Counter(image.split("_", 1)[0] for image in sample)),
            **eda_quantiles(data / "S1Hand", sample),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ml_repo", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.ml_repo), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
