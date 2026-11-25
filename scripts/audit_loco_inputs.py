#!/usr/bin/env python3
"""Audit immutable split membership and local GeoTIFF footprints (requires GDAL)."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

from osgeo import gdal

from verify_loco_evidence import A1, require

HAND = "d42023bea530fcf12e7bee379cd36150aa81e5a7"
WEAK = "1d2199b762b5c7690ddb878af8d78d5d74729ddd"
LOCO = "f5c34320a10df9ffb18b45c2dd06b176e6d4c7d5"
TOLERANCE_DEGREES = 1e-8
ROOTS = {
    "hand": Path("data/processed/sen1floods11/S1Hand"),
    "weak": Path("data/processed/sen1floods11_weak/S1Weak"),
}


def group(name):
    return name.split("_", maxsplit=1)[0]


def canonical(name):
    return re.sub(r"_(S1|Label|S2IndexLabel)(Hand|Weak)$", "", Path(name).stem).casefold()


def from_git(root, ref, path):
    data = subprocess.check_output(["git", "-C", str(root), "show", f"{ref}:{path}"])
    rows = [tuple(row) for row in csv.reader(data.decode().splitlines())]
    require(all(len(row) == 2 and canonical(row[0]) == canonical(row[1]) for row in rows),
            f"Invalid image/label record in {ref}:{path}")
    require(len({canonical(row[0]) for row in rows}) == len(rows),
            f"Duplicated chip ID in {ref}:{path}")
    return rows, hashlib.sha256(data).hexdigest()


def footprint(path):
    dataset = gdal.Open(str(path))
    spatial_ref = dataset.GetSpatialRef()
    affine = dataset.GetGeoTransform(can_return_null=True)
    require(spatial_ref is not None and spatial_ref.GetAuthorityCode(None) == "4326"
            and affine is not None, f"Expected EPSG:4326 affine footprint for {path.name}")
    west, dx, xrot, north, yrot, dy = affine
    require(xrot == yrot == 0 and dx > 0 and dy < 0, f"Expected north-up raster: {path.name}")
    east = west + dx * dataset.RasterXSize
    south = north + dy * dataset.RasterYSize
    require(-180 <= west < east <= 180 and -90 <= south < north <= 90,
            f"Invalid or dateline-crossing footprint: {path.name}")
    return (west, south, east, north)


def compare(test, membership):
    results = Counter()
    interior_contacts = 0
    boundary_contacts = 0
    for test_name, (a, b, c, d) in test.items():
        for other_name, (w, s, e, n) in membership.items():
            width = min(c, e) - max(a, w)
            height = min(d, n) - max(b, s)
            if min(width, height) > TOLERANCE_DEGREES:
                results["interior_intersection_pairs"] += 1
                results["cross_group_intersection_pairs"] += group(test_name) != group(other_name)
                interior_contacts += 1
            elif min(width, height) >= -TOLERANCE_DEGREES:
                results["boundary_or_sub_tolerance_pairs"] += 1
                boundary_contacts += 1
        results["test_tiles_with_interior_intersections"] += interior_contacts > 0
        results["test_tiles_with_boundary_or_sub_tolerance_contact"] += boundary_contacts > 0
        interior_contacts = boundary_contacts = 0
    keys = ("interior_intersection_pairs", "cross_group_intersection_pairs",
            "boundary_or_sub_tolerance_pairs", "test_tiles_with_interior_intersections",
            "test_tiles_with_boundary_or_sub_tolerance_contact")
    return {key: results[key] for key in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ml_root", type=Path)
    args = parser.parse_args()
    gdal.UseExceptions()
    split = {}
    for kind, directory, ref in (("hand", "official", HAND), ("weak", "weak", WEAK)):
        for role in ("train", "val", "test") if kind == "hand" else ("train", "val"):
            path = f"data/splits/{directory}/{role}.csv"
            records, sha = from_git(args.ml_root, ref, path)
            split[kind, role] = (records, sha, f"{ref}:{path}")

    test_rows = split["hand", "test"][0]
    official_groups = sorted({group(image) for image, _ in test_rows})
    all_test_ids = {canonical(image) for image, _ in test_rows}
    test = {image: footprint(args.ml_root / ROOTS["hand"] / image) for image, _ in test_rows}
    results = {}
    for (kind, role), (rows, sha, ref) in split.items():
        key = f"{kind}_{role}"
        results[key] = dict(git_reference=ref, sha256=sha, tiles=len(rows),
                            groups=dict(sorted(Counter(group(image) for image, _ in rows).items())))
        if role == "test":
            continue
        other = {image: footprint(args.ml_root / ROOTS[kind] / image) for image, _ in rows}
        results[key].update(
            canonical_id_overlaps=sorted(all_test_ids & {canonical(image) for image in other}),
            footprint_pair_comparisons=len(test) * len(other),
            **compare(test, other),
        )

    evidence = json.loads((A1 / "selection_frozen.json").read_text())
    frozen = {("hand" if item["stage"] == "finetune" else "weak", item["role"]):
              item["sha256"] for item in evidence["generalist"]["manifests"]}
    for (kind, role), (_, sha, _) in split.items():
        if role != "test":
            require(frozen[kind, role] == sha, f"Frozen {kind} {role} hash mismatch")

    folds = {}
    for event in official_groups:
        folds[event] = {}
        for (kind, role), (all_rows, _, _) in split.items():
            if role == "test":
                continue
            locodir = "train9" if kind == "hand" else "weak9"
            path = f"data/splits/loco/{event}/{locodir}/{role}.csv"
            fold_rows, sha = from_git(args.ml_root, LOCO, path)
            expected = {row for row in all_rows if group(row[0]) != event}
            require(set(fold_rows) == expected, f"LOCO membership differs from {kind} minus {event}")
            selected = [run for run in evidence["selected"] if run["event"] == event]
            stage = "finetune" if kind == "hand" else "pretrain"
            for run in selected:
                manifest = next(item for item in run["manifests"]
                                if (item["stage"], item["role"]) == (stage, role))
                require(sha == manifest["sha256"],
                        f"Frozen {event} {kind} {role} hash mismatch")
            folds[event][f"{kind}_{role}"] = len(fold_rows)

    print(json.dumps({
        "gdal_version": gdal.VersionInfo("RELEASE_NAME"),
        "crs": "EPSG:4326",
        "axis_tolerance_degrees": TOLERANCE_DEGREES,
        "method": "Compare the entire north-up SAR raster bounds, including ignored/nodata pixels. "
                  "An interior intersection requires overlap on both axes exceeding the tolerance.",
        "generalist_membership": results,
        "loco_membership": {"git_commit": LOCO, "all_folds_equal_generalist_minus_event": True,
                            "counts": folds},
        "scope": "Local TIFF footprint evidence for the referenced Git manifests. Historical "
                 "process-read membership, adjacent tiles, shared source scenes, undocumented "
                 "derivations and preprocessing design remain separate concerns.",
    }, indent=2))


if __name__ == "__main__":
    main()
