#!/usr/bin/env python3
"""Recompute the published LOCO metrics from the frozen confusion-count exports."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import subprocess

from make_results import comparisons, read_rows

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_RESULTS = ROOT / "reproducibility/benchmark-results"
COUNTS = ("tp", "fp", "fn", "tn")
MATCHED = ("valid_pixels", "image_sha256", "label_sha256", "mask_sha256")
SEEDS = {42, 43, 44}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def pooled(counts):
    tp, fp, fn = (counts[k] for k in COUNTS[:3])
    union = tp + fp + fn
    return (tp / union, 2 * tp / (union + tp)) if union else (0.0, 0.0)


def load_counts(stem, model, event):
    path = BENCHMARK_RESULTS / "counts" / (stem + ".csv")
    chips = {}
    totals = dict.fromkeys((*COUNTS, "valid_pixels"), 0)
    for row in read_csv(path):
        chip = row["chip_id"]
        require(chip not in chips, f"{stem}: duplicated chip {chip}")
        require(row["event"] == event and chip.startswith(event.casefold() + "_"),
                f"{stem}: mismatched event for {chip}")
        require(row["checkpoint_sha256"] == model["sha256"] and row["run_id"] == model["run_id"],
                f"{stem}: mismatched selection for {chip}")
        counts = {key: int(row[key]) for key in COUNTS}
        valid = int(row["valid_pixels"])
        require(min(counts.values()) >= 0 and sum(counts.values()) == valid <= 512**2,
                f"{stem}: invalid confusion counts for {chip}")
        if valid:
            require(row["status"] == "scored", f"{stem}: invalid status for {chip}")
            for name, metric in zip(("iou", "f1"), pooled(counts)):
                close(row[name], metric, f"{stem}: invalid chip {name} for {chip}")
        else:
            require(row["status"] == "all_ignored" and not row["iou"] and not row["f1"],
                    f"{stem}: invalid ignored chip {chip}")
        chips[chip] = row
        for key, value in counts.items():
            totals[key] += value
        totals["valid_pixels"] += valid
    require(chips and totals["valid_pixels"] > 0, f"{stem}: no usable counts")
    return chips, totals


def close(observed, expected, message):
    require(math.isclose(float(observed), expected, rel_tol=0, abs_tol=1e-12), message)


def official_test_ids(ml_root, fold, manifest, evidence):
    test_csv = subprocess.check_output(
        ["git", "-C", str(ml_root), "show", "d42023b:data/splits/official/test.csv"],
        text=True,
    )
    records = sorted(row for row in csv.reader(test_csv.splitlines())
                     if row[0].startswith(fold + "_"))
    text = "".join(",".join(row) + "\r\n" for row in records)
    require(hashlib.sha256(text.encode()).hexdigest() == manifest["sha256"],
            f"{fold}: official test manifest differs from frozen hash")
    ids = {Path(image).stem.removesuffix("_S1Hand").casefold() for image, _ in records}
    require(ids == set(evidence), f"{fold}: official test membership mismatch")


def local_checkpoints(root, models):
    db_address = (root.resolve() / "mlflow.db").as_uri() + "?mode=ro"
    with sqlite3.connect(db_address, uri=True) as db:
        for model in models:
            require(digest(root / model["path"]) == model["sha256"],
                    f"Checkpoint hash mismatch: {model['path']}")
            run = (model["run_id"],)
            status = db.execute("SELECT status FROM runs WHERE run_uuid = ?", run).fetchone()
            best, views = db.execute(
                "SELECT MAX(value), COUNT(*) FROM metrics "
                "WHERE run_uuid = ? AND key = 'val_iou' AND NOT is_nan", run
            ).fetchone()
            summary = dict(db.execute(
                "SELECT key, value FROM latest_metrics WHERE run_uuid = ? "
                "AND key IN ('best_epoch', 'epochs_run')", run
            ))
            require(status == ("FINISHED",) and views == summary["epochs_run"],
                    f"Run/status mismatch: {model['run_id']}")
            close(best, model["best_validation_score"], f"Validation maximum differs: {run[0]}")
            require(summary["best_epoch"] == model["epoch"],
                    f"Best epoch mismatch: {model['run_id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml-root", type=Path,
                        help="Verify test IDs, local checkpoint hashes and MLflow validation records")
    args = parser.parse_args()
    for name, expected in json.loads((BENCHMARK_RESULTS / "checksums.json").read_text()).items():
        require(digest(BENCHMARK_RESULTS / name) == expected, f"Evidence hash mismatch: {name}")
    selection = json.loads((BENCHMARK_RESULTS / "selection_frozen.json").read_text())
    summary = json.loads((BENCHMARK_RESULTS / "results.json").read_text())
    require(digest(BENCHMARK_RESULTS / "selection_frozen.json") == summary["selection_sha256"],
            "Selection hash mismatch")
    checkpoints = {(m["event"], m["seed"]): m for m in selection["selected"]}
    pairs = {(r["event"], r["seed"]): r for r in summary["individual"]}
    require(pairs.keys() == checkpoints.keys() and
            len(pairs) == len(summary["individual"]) == len(selection["selected"]),
            "Incomplete or duplicated event/seed results")
    tables = {(r["event"], int(r["seed"])): r for r in read_csv(BENCHMARK_RESULTS / "individual_seeds.csv")}
    require(tables.keys() == pairs.keys(), "CSV result coverage differs from JSON")
    coverage = {r["event"]: r for r in summary["coverage"]}
    releases = read_csv(BENCHMARK_RESULTS / "coverage.csv")
    require(coverage.keys() == {r["event"] for r in releases} ==
            selection["manifests"].keys() == summary["generalist"].keys(),
            "Generalist coverage differs from frozen events")

    if args.ml_root:
        local_checkpoints(args.ml_root, [selection["generalist"], *checkpoints.values()])

    print("event,seed,chips,generalist_iou,loco_iou,delta_loco_minus_generalist")
    paired_chips = chip_seed_pairs = all_chips = 0
    ignored = []
    baseline_totals = dict.fromkeys(COUNTS, 0)
    for fold, manifest in selection["manifests"].items():
        baseline, totals = load_counts(f"generalist_{fold}", selection["generalist"], fold)
        giou, gf1 = pooled(totals)
        for key in COUNTS:
            baseline_totals[key] += totals[key]
        for key, value in (*totals.items(), ("iou", giou), ("f1", gf1)):
            close(summary["generalist"][fold][key], value, f"{fold}: generalist {key} mismatch")
        n_chips = len(baseline)
        ignored_here = [chip for chip, row in baseline.items() if not int(row["valid_pixels"])]
        ignored.extend(ignored_here)
        all_chips += n_chips
        available = sorted(seed for event, seed in checkpoints if event == fold)
        missing = sorted(SEEDS - set(available))
        expected_coverage = (n_chips, available, len(available), missing)
        c = coverage[fold]
        require(expected_coverage ==
                (c["n_official_test"], c["available_seeds"], c["n_available"], c["missing_seeds"]),
                f"{fold}: coverage mismatch")
        require(n_chips == manifest["n"], f"{fold}: counts differ from frozen coverage")
        rc = next(row for row in releases if row["event"] == fold)
        require(expected_coverage == (int(rc["n_official_test"]), json.loads(rc["available_seeds"]),
                                      int(rc["n_available"]), json.loads(rc["missing_seeds"])),
                f"{fold}: coverage CSV mismatch")
        if args.ml_root:
            official_test_ids(args.ml_root, fold, manifest, baseline)
        if available:
            paired_chips += n_chips
        for seed in available:
            chip_seed_pairs += n_chips
            loco, loco_totals = load_counts(f"{fold}_s{seed}", checkpoints[fold, seed], fold)
            require(loco.keys() == baseline.keys(), f"{fold} {seed}: chip IDs differ")
            for chip, row in loco.items():
                require(all(row[k] == baseline[chip][k] for k in MATCHED),
                        f"{fold} {seed}: chip or valid-pixel mask differs for {chip}")
            liou, lf1 = pooled(loco_totals)
            delta = liou - giou
            expected = dict(n_chips=n_chips, n_valid_chips=n_chips - len(ignored_here),
                            n_all_ignored=len(ignored_here), generalist_iou=giou,
                            generalist_f1=gf1, loco_iou=liou, loco_f1=lf1, delta=delta)
            for key, value in expected.items():
                close(pairs[fold, seed][key], value, f"{fold} {seed}: JSON {key} mismatch")
                close(tables[fold, seed][key], value, f"{fold} {seed}: CSV {key} mismatch")
            print(f"{fold},{seed},{n_chips},{giou:.9f},{liou:.9f},{delta:+.9f}")
    frozen_ignored = sorted(Path(s).stem.removesuffix("_S1Hand").casefold()
                            for s in selection["all_ignored_test"])
    require(sorted(ignored) == frozen_ignored, "All-ignored chip list mismatch")
    baseline_iou, baseline_f1 = pooled(baseline_totals)
    print(f"Verified {len(pairs)} pairs across {len({event for event, _ in pairs})} groups, "
          f"{paired_chips}/{all_chips} paired chips, {chip_seed_pairs} chip–seed pairs, "
          f"{len(ignored)} all-ignored official-test chips.")
    print(f"Generalist pooled across all {all_chips} official-test chips: "
          f"IoU {baseline_iou:.9f}, F1 {baseline_f1:.9f}.")

    require((len(pairs), len(coverage) * len(SEEDS), paired_chips, all_chips) == (9, 30, 51, 90),
            "Unexpected benchmark coverage")
    require(len({event for event, _ in pairs}) == 7, "Unexpected paired event count")
    require(round(baseline_iou, 6) == 0.676564, "Unexpected baseline IoU")
    recalculated = comparisons()
    deltas = [pair[2] for pair in recalculated.values()]
    require((sum(delta > 0 for delta in deltas), sum(delta < 0 for delta in deltas)) == (5, 4),
            "Unexpected delta sign counts")
    require((round(min(deltas), 6), round(max(deltas), 6)) == (-0.033583, 0.025312),
            "Unexpected delta extrema")
    sensitivity = [
        (event, seed, chip, delta, omitted)
        for (event, seed), (_, _, delta, omissions) in recalculated.items()
        for chip, omitted in omissions.items()
    ]
    reversals = sum(
        any(delta * omitted < 0 for omitted in omissions.values())
        for _, _, delta, omissions in recalculated.values()
    )
    require(len(sensitivity) == 63 and reversals == 7, "Unexpected chip sensitivity summary")
    recorded = read_csv(ROOT / "sections/tables/chip_deletion.csv")
    require(sensitivity == [
        (row["event"], int(row["seed"]), row["omitted_chip"],
         float(row["full_delta"]), float(row["deletion_delta"])) for row in recorded
    ], "Chip-deletion export differs from the per-chip source counts")
    water_labels = {
        row["chip_id"]: int(row["tp"]) + int(row["fn"])
        for row in read_rows(BENCHMARK_RESULTS / "counts/Pakistan_s44.csv")
    }
    dominant_chip = max(water_labels, key=water_labels.get)
    _, _, full, omissions = recalculated["Pakistan", 44]
    require((sum(water_labels.values()), water_labels[dominant_chip],
             round(full, 4), round(omissions[dominant_chip], 4))
            == (198897, 171908, 0.0253, -0.2220), "Pakistan dominant chip does not match")
    print(f"Verified {len(sensitivity)} omissions, {reversals} sign reversals. "
          f"Pakistan seed 44 without {dominant_chip}: {omissions[dominant_chip]:+.6f}.")



if __name__ == "__main__":
    main()
