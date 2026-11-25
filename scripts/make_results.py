"""Render the manuscript table and a paired leave-one-chip-out sensitivity plot."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reproducibility/benchmark-results"


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def pooled_iou(rows):
    tp = sum(int(row["tp"]) for row in rows)
    union = sum(sum(int(row[key]) for key in ("tp", "fp", "fn")) for row in rows)
    if union == 0:
        raise ValueError("IoU requires a nonempty positive-class union")
    return tp / union


def comparisons():
    pairs = {}
    for row in read_rows(EVIDENCE / "individual_seeds.csv"):
        event, seed = row["event"], int(row["seed"])
        generalist = read_rows(EVIDENCE / "counts" / f"generalist_{event}.csv")
        excluded = read_rows(EVIDENCE / "counts" / f"{event}_s{seed}.csv")
        generalist_by_id = {entry["chip_id"]: entry for entry in generalist}
        excluded_by_id = {entry["chip_id"]: entry for entry in excluded}
        if set(generalist_by_id) != set(excluded_by_id):
            raise ValueError(f"Mismatched chips for {event} seed {seed}")
        for chip, baseline in generalist_by_id.items():
            model = excluded_by_id[chip]
            keys = ("valid_pixels", "mask_sha256", "image_sha256", "label_sha256")
            if any(baseline[key] != model[key] for key in keys):
                raise ValueError(f"Mismatched evaluation inputs for {chip}")
        baseline = pooled_iou(generalist)
        score = pooled_iou(excluded)
        delta = score - baseline
        if abs(delta - float(row["delta"])) > 1e-12:
            raise ValueError(f"Delta changed for {event} seed {seed}")
        deletions = {}
        for chip in generalist_by_id:
            deletions[chip] = (
                pooled_iou([entry for entry in excluded if entry["chip_id"] != chip])
                - pooled_iou([entry for entry in generalist if entry["chip_id"] != chip])
            )
        pairs[event, seed] = (baseline, score, delta, deletions)
    return pairs


def render_table(pairs):
    lines = [
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Event & Test chips & Seed & Generalist & LOCO & $\Delta$ \\",
        r"\midrule",
    ]
    for row in read_rows(EVIDENCE / "coverage.csv"):
        event, n = row["event"], row["n_official_test"]
        event_pairs = [(seed, pair) for (group, seed), pair in pairs.items() if group == event]
        generalist = pooled_iou(read_rows(EVIDENCE / "counts" / f"generalist_{event}.csv"))
        label = event.replace("Sri-Lanka", "Sri Lanka")
        if not event_pairs:
            lines.append(f"{label} & {n} & -- & {generalist:.4f} & -- & --" + r" \\")
        for seed, (base, score, delta, _) in event_pairs:
            lines.append(f"{label} & {n} & {seed} & {base:.4f} & {score:.4f} & ${delta:+.4f}$" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "sections/tables/test_only.tex").write_text("\n".join(lines) + "\n")


def render_sensitivity(pairs, figure):
    output = ROOT / "sections/tables/chip_deletion.csv"
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("event", "seed", "omitted_chip", "full_delta", "deletion_delta"))
        for (event, seed), (_, _, delta, deletions) in pairs.items():
            writer.writerows((event, seed, chip, delta, score) for chip, score in deletions.items())
    if not figure:
        return

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
    labels = []
    for y, ((event, seed), (_, _, delta, deletions)) in enumerate(pairs.items()):
        labels.append(f'{event.replace("Sri-Lanka", "Sri Lanka")} · {seed}')
        low, high = min(deletions.values()), max(deletions.values())
        ax.hlines(y, low * 100, high * 100, color="#6882a1", linewidth=2)
        ax.scatter([delta * 100], [y], color="#18385c", s=36, zorder=3)
    ax.axvline(0, color="#434b55", lw=0.9, linestyle="--")
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.set_xlabel("LOCO minus generalist IoU (percentage points)")
    ax.grid(axis="x", color="#dfe5eb", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", color="#18385c", label="Full test fold"),
            Line2D([], [], linewidth=2, color="#6882a1", label="Range when one chip is omitted"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False,
    )
    fig.savefig(ROOT / "figures/chip_sensitivity.pdf", metadata={"CreationDate": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", action="store_true", help="render the PDF plot with Matplotlib")
    pairs = comparisons()
    render_table(pairs)
    render_sensitivity(pairs, parser.parse_args().figure)
