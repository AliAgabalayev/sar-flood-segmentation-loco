#!/usr/bin/env python3
"""Extract a public-safe subset of historical ablation evidence from local artifacts."""

import argparse
import ast
from contextlib import closing
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import subprocess


RUNS = {
    "deeplab_two_stage_pretrain": "7639f2cdd9044527aaeb7c7d4045bc4d",
    "deeplab_two_stage_finetune": "c0ec384f3a0e46d59ab8caa51011395e",
    "deeplab_single_stage": "6538d99d08b74edd81ed267af06a842e",
    "segformer_b4_two_stage_pretrain": "7081e1388b394806a73dd4abaeb2f4f8",
    "segformer_b4_two_stage_finetune": "fe3de4c6dc7c41eb8da72d0917361036",
    "segformer_b4_tuned": "5f2fb67ac9d84d8ead1238e788fdac3d",
    "segformer_b4_aggressive": "a109976a226844e7b1af9cee788521e9",
    "loss_finalist_dice": "79e09d0690944fc687e00a06f9c5199c",
    "loss_finalist_tversky": "7bcae2f0d26c4d818d89c03e9486b533",
    "loss_finalist_focal_tversky": "bc1e0553b6bc42568d4d0f1c5eb5b529",
}
CHECKPOINTS = {
    "deeplab_two_stage_pretrain": "models/weak_pretrain_finetune/pretrain/best.ckpt",
    "deeplab_two_stage_finetune": "models/weak_pretrain_finetune/finetune/best.ckpt",
    "deeplab_single_stage": "models/weak_pretrain_finetune/single_stage_baseline/best.ckpt",
    "segformer_b4_two_stage_pretrain": "models/segformer_b4/pretrain/best.ckpt",
    "segformer_b4_two_stage_finetune": "models/segformer_b4/finetune/best.ckpt",
}
METRICS = {
    "best_epoch", "epochs_run", "best_val_iou", "best_val_f1",
    "test_iou", "test_f1", "test_iou_fused", "test_f1_fused",
}
PARAM_KEYS = {
    "seed", "data/augment", "data/split_dir", "data/num_workers", "data/image_size",
    "data/vv_clip", "data/vh_clip", "data/ratio_clip", "model/arch", "model/backbone",
    "model/pretrained", "training/batch_size", "training/accumulate_grad_batches",
    "training/optimizer", "training/lr", "training/epochs", "training/loss",
    "training/early_stopping_patience", "training/precision", "training/pos_weight",
    "training/tversky_alpha", "training/tversky_beta", "training/tversky_gamma",
    "inference/threshold", "inference/tta", "training/checkpoint_dir",
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract_run(db, identifier):
    name, status = db.execute(
        "SELECT name, status FROM runs WHERE run_uuid = ?", (identifier,)
    ).fetchone()
    if status != "FINISHED":
        raise ValueError(f"Selected historical run has status {status}: {identifier}")
    params = {
        key: val
        for key, val in db.execute("SELECT key, value FROM params WHERE run_uuid = ?", (identifier,))
        if key in PARAM_KEYS
    }
    metrics = {
        key: val
        for key, val in db.execute("SELECT key, value FROM latest_metrics WHERE run_uuid = ?", (identifier,))
        if key in METRICS
    }
    validation_max, validation_metric_count = db.execute(
        "SELECT MAX(value), COUNT(*) FROM metrics "
        "WHERE run_uuid = ? AND key = 'val_iou' AND is_nan = 0",
        (identifier,),
    ).fetchone()
    if "best_val_iou" in metrics and not math.isclose(metrics["best_val_iou"], validation_max):
        raise ValueError(f"Logged best value disagrees with validation curve: {identifier}")
    version = db.execute(
        "SELECT value FROM tags WHERE run_uuid = ? AND key = 'git_sha'", (identifier,)
    ).fetchone()
    return {
        "run_id": identifier,
        "run_name": name,
        "status": status,
        "git_sha_tag": version[0] if version else None,
        "recorded_parameters": params,
        "recorded_metrics": metrics,
        "validation_curve_max_iou": validation_max,
        "validation_curve_points": validation_metric_count,
    }


def checkpoint_metadata(path, run):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    selection = [
        value for value in checkpoint["callbacks"].values()
        if isinstance(value, dict) and "best_model_score" in value
    ]
    if len(selection) != 1:
        raise ValueError(f"Expected a single model-selection callback in {path.name}")
    score = float(selection[0]["best_model_score"])
    if not math.isclose(score, run["validation_curve_max_iou"]):
        raise ValueError(f"Checkpoint and validation curve disagree: {path.name}")
    best_epoch = run["recorded_metrics"].get("best_epoch")
    if best_epoch is not None and best_epoch != checkpoint["epoch"]:
        raise ValueError(f"Checkpoint and logged epoch disagree: {path.name}")
    return {
        "epoch_zero_based": checkpoint["epoch"],
        "global_step": checkpoint["global_step"],
        "lightning_version": checkpoint["pytorch-lightning_version"],
        "stored_best_validation_iou": score,
    }


def audit(root, inspect_checkpoints):
    db_path = root / "mlflow.db"
    with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        runs = {label: extract_run(db, identifier) for label, identifier in RUNS.items()}
        attempts = dict(db.execute(
            "SELECT status, COUNT(*) FROM runs WHERE name = ? GROUP BY status",
            ("single_stage_resnet50_dice",),
        ))

    checkpoints = {
        key: {"path": path, "sha256": sha256(root / path)}
        for key, path in CHECKPOINTS.items()
    }
    if inspect_checkpoints:
        for label, fields in checkpoints.items():
            fields.update(checkpoint_metadata(root / fields["path"], runs[label]))
    one = runs["deeplab_single_stage"]
    two = runs["deeplab_two_stage_finetune"]
    log_path = root / "single_stage_baseline.log"
    log_result = ast.literal_eval(log_path.read_text().rsplit("result: ", 1)[1])
    for metric in ("best_epoch", "epochs_run", "best_val_iou", "best_val_f1", "test_iou", "test_f1"):
        if log_result[metric] != one["recorded_metrics"][metric]:
            raise ValueError(f"Single-stage log and database disagree for {metric}")
    records = root / "experiments/loss_study/results.csv"
    origin = "b531ebf015abc278f7cc6868c3468e0d90315ee1:experiments/loss_study/results.csv"
    original = subprocess.check_output(["git", "show", origin], cwd=root)
    if records.read_bytes() != original:
        raise ValueError("Current loss-sweep CSV differs from the historical tracked version")
    with records.open(newline="") as stream:
        sweep = list(csv.DictReader(stream))
    return {
        "mlflow_snapshot_sha256": sha256(db_path),
        "runs": runs,
        "local_checkpoint_file_hashes": checkpoints,
        "single_stage_attempts_by_status": attempts,
        "single_stage_log": {"path": log_path.name, "sha256": sha256(log_path), "summary": log_result},
        "matched_seed_deeplab_two_minus_single": {
            metric: two["recorded_metrics"][metric] - one["recorded_metrics"][metric]
            for metric in ("best_val_iou", "best_val_f1", "test_iou", "test_f1")
        },
        "short_schedule_loss_sweep": {
            "git_reference": origin,
            "sha256": sha256(records),
            "rows": sweep,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ml_repo", type=Path)
    parser.add_argument(
        "--inspect-checkpoints", action="store_true",
        help="check epoch and validation score fields against MLflow using CPU PyTorch",
    )
    args = parser.parse_args()
    print(json.dumps(audit(args.ml_repo, args.inspect_checkpoints), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
