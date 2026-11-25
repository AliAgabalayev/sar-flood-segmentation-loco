#!/usr/bin/env python3
"""Verify the historical size-control CSV against MLflow, then report test-only contrasts."""
import argparse
import csv
import json
from pathlib import Path
import random
import sqlite3
import subprocess

from verify_loco_evidence import digest, require

ARMS = ("loco", "ctrl", "full")
METRICS = ("iou_unseen", "f1_unseen", "iou_test", "f1_test", "n_test", "n_unseen")
FOLDS = ("Somalia", "Spain", "Sri-Lanka", "USA")
SEED = 42


def csv_rows(text):
    return [tuple(row) for row in csv.reader(text.splitlines())]


def manifest(path, expected, hashes, root):
    observed = csv_rows(path.read_text())
    require(observed == expected, f"Membership or order differs: {path}")
    hashes[str(path.relative_to(root))] = digest(path)


def review_manifests(root, fold, official):
    hashes = {}
    target = {role: [row for row in official[role] if row[0].startswith(fold + "_")]
              for role in ("train", "val", "test")}
    excluded = {role: [row for row in official[role] if not row[0].startswith(fold + "_")]
                for role in ("train", "val")}
    base = root / "data/splits/control" / fold
    manifest(base / "eval_unseen/test.csv", target["val"] + target["test"], hashes, root)
    manifest(base / "eval_test/test.csv", target["test"], hashes, root)
    swap_out = set(random.Random(f"{fold}-{SEED}").sample(sorted(excluded["train"]),
                                                          len(target["train"])))
    retained = [row for row in excluded["train"] if row not in swap_out]
    training = {"loco": excluded["train"], "ctrl": retained + target["train"],
                "full": official["train"]}
    sizes = {}
    for arm in ARMS:
        arm_path = base / f"seed{SEED}" / arm
        manifest(arm_path / "train.csv", training[arm], hashes, root)
        manifest(arm_path / "val.csv", excluded["val"], hashes, root)
        sizes[arm] = len(training[arm])
    return len(target["test"]), len(target["test"] + target["val"]), sizes, hashes


def read_run(db, name, fold, status="FINISHED"):
    records = db.execute(
        "SELECT r.run_uuid, r.status FROM runs r JOIN experiments e USING (experiment_id) "
        "WHERE r.name = ? AND e.name = ?",
        (name, f"flood-water-seg/control/{fold}"),
    ).fetchall()
    require(len(records) == 1 and records[0][1] == status, f"Unexpected run record for {name}")
    return records[0][0]


def evaluate_audit(db, fold, arm, published_row, n_test, n_unseen, n_train):
    eval_run = read_run(db, f"eval_{arm}_{fold}_seed{SEED}", fold)
    metrics = dict(db.execute("SELECT key, value FROM latest_metrics WHERE run_uuid=?",
                              (eval_run,)))
    for metric in METRICS:
        require(abs(float(published_row[metric]) - metrics[metric]) <= 5.01e-7,
                f"MLflow vs CSV mismatch: {fold} {arm} {metric}")
    params = dict(db.execute("SELECT key, value FROM params WHERE run_uuid=?", (eval_run,)))
    require(metrics["n_unseen"] == n_unseen and metrics["n_test"] == n_test
            and int(params["n_train"]) == int(published_row["n_train"]) == n_train,
            f"Size mismatch: {fold} {arm}")
    require(params["arm"] == arm and params["fold"] == fold and int(params["seed"]) == SEED
            and params["reused_ckpt"] == published_row["reused_ckpt"],
            f"Run identity mismatch: {fold} {arm}")
    reused = params["reused_ckpt"] == "1"
    train_run = None
    if not reused:
        train_run = read_run(db, f"finetune_{arm}_{fold}_seed{SEED}", fold)
        train_params = dict(db.execute("SELECT key, value FROM params WHERE run_uuid=?",
                                       (train_run,)))
        require(train_params["data/split_dir"].endswith(f"/{fold}/seed{SEED}/{arm}"),
                f"Fine-tune split path mismatch: {fold} {arm}")
    return dict(evaluation_run_id=eval_run, finetune_run_id=train_run, reused_checkpoint=reused,
                n_train=n_train, **{metric: metrics[metric] for metric in METRICS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ml_root", type=Path)
    root = parser.parse_args().ml_root.resolve()
    original = root / "data/reference/size_matched_control.csv"
    with original.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_key = {(r["fold"], r["arm"]): r for r in rows if int(r["seed"]) == SEED}
    require(len(rows) == len(by_key) == len(FOLDS) * len(ARMS), "Unexpected control CSV coverage")
    official = {
        role: csv_rows(subprocess.check_output(
            ["git", "-C", str(root), "show", f"d42023b:data/splits/official/{role}.csv"], text=True))
        for role in ("train", "val", "test")
    }

    evidence = {}
    db_uri = (root / "mlflow.db").as_uri() + "?mode=ro"
    with sqlite3.connect(db_uri, uri=True) as db:
        for fold in FOLDS:
            n_test, n_unseen, sizes, hashes = review_manifests(root, fold, official)
            scores = {arm: evaluate_audit(db, fold, arm, by_key[fold, arm], n_test, n_unseen, sizes[arm])
                      for arm in ARMS}
            loco, ctrl, full = (scores[arm]["iou_test"] for arm in ARMS)
            evidence[fold] = dict(
                arms=scores,
                official_test_iou_differences=dict(ctrl_minus_loco=ctrl - loco,
                                                   full_minus_ctrl=full - ctrl,
                                                   full_minus_loco=full - loco),
                current_manifest_sha256=hashes,
            )
        interrupted = read_run(db, "pretrain_Somalia_seed43", "Somalia", status="FAILED")

    print(json.dumps({
        "seed": SEED,
        "database_sha256": digest(root / "mlflow.db"),
        "historical_control_csv_sha256": digest(original),
        "interrupted_seed43_pretraining_run_id": interrupted,
        "folds": evidence,
        "scope": "September 2026 retrospective audit of logged metrics. Current manifests "
                 "match the recorded algorithm. Checkpoint/source bytes for most model fits "
                 "and historical process-read split membership are not established.",
    }, indent=2))


if __name__ == "__main__":
    main()
