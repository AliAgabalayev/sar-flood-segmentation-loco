"""Generate supporting manuscript tables from the archived source records."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reproducibility/evidence"
DEPLOYMENT = ROOT / "reproducibility/deployment"
TABLES = ROOT / "sections/tables"


def read_json(path):
    return json.loads(path.read_text())


def write_table(name, alignment, header, rows):
    row_end = r" \\"
    lines = [
        r"\begin{tabular}{@{}" + alignment + r"@{}}",
        r"\toprule", " & ".join(header) + row_end, r"\midrule",
        *(" & ".join(row) + row_end for row in rows),
        r"\bottomrule", r"\end{tabular}",
    ]
    (TABLES / f"{name}.tex").write_text("\n".join(lines) + "\n")


def make_historical_tables():
    records = read_json(EVIDENCE / "historical_runs.json")["runs"]
    single = records["deeplab_single_stage"]["recorded_metrics"]
    staged = records["deeplab_two_stage_finetune"]["recorded_metrics"]
    segformer = records["segformer_b4_two_stage_finetune"]["recorded_metrics"]
    write_table("two_stage", "lrr", ["Training", "IoU", "F1"], [
        ("Single-stage (hand labels only)", f"{single['test_iou']:.4f}", f"{single['test_f1']:.4f}"),
        (r"Two-stage (weak $\to$ hand)", f"{staged['test_iou']:.4f}", f"{staged['test_f1']:.4f}"),
        ("Difference", f"{staged['test_iou'] - single['test_iou']:+.4f}",
         f"{staged['test_f1'] - single['test_f1']:+.4f}"),
    ])
    write_table("model_comparison", "lrr", ["Model", "Val. IoU", "Test IoU"], [
        (label, f"{record['best_val_iou']:.4f}", f"{record['test_iou']:.4f}")
        for label, record in (("DeepLabV3+/ResNet-50", staged), ("SegFormer-B4", segformer))
    ])

    with (EVIDENCE / "loss_sweep.csv").open(newline="") as stream:
        runs = sorted(csv.DictReader(stream), key=lambda row: -float(row["best_val_iou"]))
    labels = {
        "focal_tversky_g1.33": r"Focal-Tversky ($\gamma=1.33$)",
        "tversky_a0.3_b0.7": r"Tversky ($\alpha=0.3$, $\beta=0.7$)",
        "dice": "Dice", "dice_focal": "Dice + Focal", "dice_bce_pw8.4": "Dice + BCE (weight 8.4)",
        "bce_pw4": "BCE, positive weight 4", "bce_pw8.4": "BCE, positive weight 8.4",
        "focal_a0.25": r"Focal ($\alpha=0.25$)", "lovasz": r"Lov\'asz",
        "bce_pw13": "BCE, positive weight 13", "bce_pw1": "BCE, positive weight 1",
    }
    write_table("loss_sweep", "lrr", ["Loss", "Val. IoU", "Val. F1"], [
        (labels[row["run_name"]], f"{float(row['best_val_iou']):.4f}", f"{float(row['best_val_f1']):.4f}")
        for row in runs
    ])


def make_scene_tables():
    archive = {row["site"]: row for row in read_json(DEPLOYMENT / "archived_mask_counts.json")}
    rows = []
    for replay in read_json(DEPLOYMENT / "scene_replay_counts.json")["scenes"]:
        site = replay["site"]
        archived = archive[site]
        final = archived["positive_pixels"]
        if replay["published_positive_pixels"] != final:
            raise ValueError(f"{site}: offline replay cites a different archived final count")
        rows.append((
            site.capitalize(), archived["acquisition_date_as_recorded"],
            f"{replay['raw_positive_pixels']:,}", f"{replay['after_jrc_positive_pixels']:,}",
            f"{replay['after_jrc_and_dem_positive_pixels']:,}", f"{final:,}",
        ))
    write_table("historical_scene_counts", "llrrrr",
                ["Site", "Date", "Raw", "After JRC", "Replay final", "Archived final"], rows)


    upload = read_json(DEPLOYMENT / "api_replay.json")
    write_table(
        "api_replay", "lrrrrr",
        ["Site", "HTTP water", "CLI water", "API--CLI diff.", "Terrain flips", "Water flips"],
        [
            (row["site"].capitalize(), f"{row['api_positive_pixels']:,}",
             f"{row['offline_positive_pixels']:,}", f"{row['api_vs_offline_differing_pixels']:,}",
             f"{row['terrain_vs_previous_api']['different_pixels']:,}",
             f"{row['final_water_vs_previous_api']['different_pixels']:,}")
            for row in upload["scenes"]
        ],
    )



if __name__ == "__main__":
    make_historical_tables()
    make_scene_tables()
