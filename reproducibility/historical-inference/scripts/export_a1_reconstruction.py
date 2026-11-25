"""Generate lightweight tables/figure/report from frozen partial A1 results."""
import csv
import json
from pathlib import Path
from reconstruct_a1 import OUT


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=json.loads((OUT/'results.json').read_text())
    plan=json.loads((OUT/'selection_frozen.json').read_text())
    supplement=json.loads((OUT/'provenance_supplement.json').read_text())
    individuals=result['individual'];coverage=result['coverage']
    latex=[r'\begin{table}[tb]',r'\centering\small',r'\caption{Partial test-only reconstruction (R-strong membership, not fully proven). Individual surviving seeds; $\Delta=\mathrm{IoU}_{LOCO}-\mathrm{IoU}_{gen}$. No complete-experiment mean. USA uses the completed retry; its pretraining initialization is unresolved.}',r'\label{tab:a1test}',r'\begin{tabular}{@{}lrrrrr@{}}',r'\toprule',r'Event & Seed & Test chips & Generalist & LOCO & $\Delta$ \\',r'\midrule']
    for c in coverage:
        entries=[x for x in individuals if x['event']==c['event']]
        for x in entries:
            latex.append(f"{x['event']} & {x['seed']} & {x['n_chips']} & {x['generalist_iou']:.4f} & {x['loco_iou']:.4f} & ${x['delta']:+.4f}$ \\\\")
        if not entries:latex.append(f"{c['event']} & -- & {c['n_official_test']} & -- & -- & pending \\\\")
    latex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (OUT/'test_only_table.tex').write_text('\n'.join(latex)+'\n')
    budget=[r'\begin{table}[tb]',r'\centering\small',r'\caption{Current fold manifests, byte-matched to committed pre-run manifests. Historical usage is a strong reconstruction. Counts are unique chip entries, not augmented examples or optimizer steps.}',r'\label{tab:a1budgets}',r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',r'Event & Hand train & Hand val & Weak train & Weak val \\',r'\midrule']
    for x in supplement['fold_counts']:budget.append(f"{x['event']} & {x['hand_train']} & {x['hand_val']} & {x['weak_train']} & {x['weak_val']} \\\\")
    budget += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (OUT/'training_counts_table.tex').write_text('\n'.join(budget)+'\n')
    fig,ax=plt.subplots(figsize=(7,4.8),layout='constrained')
    for i,c in enumerate(coverage):
        entries=[x for x in individuals if x['event']==c['event']]
        for j,x in enumerate(entries):
            y=i+(.34*(j-(len(entries)-1)/2))
            ax.plot(x['delta'],y,'o',color={42:'#1f77b4',43:'#e1812c',44:'#35964a'}[x['seed']])
            ax.annotate(f"s{x['seed']}: {x['delta']:+.4f}",(x['delta'],y),xytext=(7,0),textcoords='offset points',va='center',fontsize=8)
        if not entries:ax.text(-.048,i,'No surviving defensible LOCO output',va='center',fontsize=8,color='gray')
    ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(len(coverage)),[f"{x['event']} (n seeds={x['n_available']})" for x in coverage]);ax.set_ylim(len(coverage)-.5,-.5);ax.set_xlim(-.05,.065)
    ax.set_xlabel('Test-only IoU delta: LOCO − generalist');ax.set_title('Partial reconstruction: 9/30 event–seed combinations')
    fig.savefig(OUT/'test_only_deltas.png',dpi=180);plt.close(fig)
    text='''# A1 corrected test-only partial reconstruction

This supersedes the initial blocked A1 handoff for the surviving artifacts.
**R-strong membership reconstruction, not fully proven provenance or the complete
original ten-event, three-seed experiment.** No training was launched.

Selection was frozen in `selection_frozen.json` and its SHA file before new
inference: existing completed runs, validation-max checkpoints; no test ranking.
USA seed 42 uses completed run 46bcbbc8... `best-v1.ckpt`, not interrupted run
9e351da9... `best.ckpt`. The latter remains on disk as a historical artifact.
Completed fine-tuning may have initialized from attempt-1 pretrain `best.ckpt`;
the completed retry pretrain `best-v1` is not established as its initializer.
This uncertainty applies to scientific interpretation, not the identity of the
fine-tuned weights just evaluated. Nigeria seed 43 was recovered in MLflow's
artifact directory; the supplied note's assertion that all Nigeria checkpoints
were absent was too strong.

## Individual seed results

Delta = pooled flood IoU(LOCO) − pooled flood IoU(generalist) on exactly the same
official test chip IDs and valid-pixel masks. No tuning. The same generalist
seed-42 checkpoint is used throughout. F1 and full precision are in the CSV.

| Event | Seed | Test chips | Generalist IoU | LOCO IoU | Delta |
|---|---:|---:|---:|---:|---:|
'''
    for x in individuals:text+=f"| {x['event']} | {x['seed']} | {x['n_chips']} | {x['generalist_iou']:.6f} | {x['loco_iou']:.6f} | {x['delta']:+.6f} |\n"
    text+='''
## Coverage (all ten official event groups retained)

| Event | Official test chips | Available seeds | Missing seeds |
|---|---:|---|---|
'''
    for x in coverage:text+=f"| {x['event']} | {x['n_official_test']} | {', '.join(map(str,x['available_seeds'])) or 'none'} | {', '.join(map(str,x['missing_seeds']))} |\n"
    text+='''
Generalist coverage: 90/90 chips across ten events. LOCO paired coverage: 51/90
unique chips across seven events; 9/30 event/seed combinations (63 chip–seed
pairs). No ten-event mean, uniform three-seed mean, confidence interval or
significance claim is reported. Missing Ghana, India and Paraguay are explicit.
Bolivia's 15 separate holdout chips are outside the official test comparison.

The two all-ignored official test chips are Ghana_83483 and Paraguay_34417.
They remain in both manifest and generalist coverage. Their TP/FP/FN/TN counts
are zero and individual IoU/F1 are null, not fabricated zero scores. They add
no pixels to pooled metrics. Neither event has a defensible LOCO checkpoint;
no LOCO prediction is invented for them. Their events' test counts remain 11
and 14, with 10 and 13 chips containing any valid pixels respectively.

## Provenance and budget limits

D: observed checkpoint SHA-256 values, stored MLflow parameters/status/validation
history, checkpoint epoch/global_step/callback values, committed manifest bytes.
R-strong: the historical process read those manifests and the checkpoint/run
association for model-directory files. Membership is not fully proven by git or
compatible optimizer-step counts. U: exact pretraining initialization (especially
USA), process-read membership beyond reconstruction, spatial/derived overlap.
The artifact-directory Nigeria run identity is direct stored evidence; init
and historical membership still require reconstruction. Full hashes, exact run
IDs, commit:path manifest references and per-field evidence classes are in
`selection_frozen.json` and `provenance_supplement.json`.

Current committed manifests have 211–242 hand-training entries versus 252 for
the generalist; selection uses the official validation split excluding X.
These entries include all-ignored chips: official train has 251 nonempty-valid
entries among 252. They are not augmented examples, epochs or optimizer steps.
Weak pretraining uses 3,946 optimizer-training chips and 438 validation chips
(4,384 total); each LOCO weak pool removes X from both roles. Per-fold counts
are in `training_counts_table.tex`. Generalist hand labels cover ten official
events; hand LOCO covers the other nine. The weak pool covers eleven events,
including Bolivia, so excluding an official event leaves ten weak event groups.
446 hand chips = 431 official + 15 separate Bolivia, verified from unique IDs
and 446 label files. Spatial overlap beyond exact IDs/byte duplicates remains
unresolved; test-only membership does not ensure spatial independence.

All paired test IDs and masks match; reconstructed hand/weak training and
validation have zero canonical-ID overlap with the event test subset.
Generalist sees the held-out event's OTHER training/validation chips by design.
Fixed preprocessing: VV [-25,0], VH [-35,0], VV−VH [0,18], clip/scale;
sigmoid >0.5; no TTA or deployment postprocessing. Valid pixels exclude label
-1 and nonfinite raw SAR. Metrics pool flood-class TP/FP/FN over valid pixels.
Per-chip confusion sums reproduce the existing evaluator within 1e-6 for each
checkpoint/event. Counts include image, label, mask and checkpoint hashes and
run IDs, enabling audited future reuse under this frozen policy.

No old cache was reused. `cache_recovery_audit.json` inventories candidate
artifacts and CSV schemas; `cache_database_audit.json` checks current and backup
MLflow databases (aggregate metrics, zero evaluation-dataset records). The
truncated archive contains no recoverable chip predictions before truncation.
Existing deployment TIFF/PNG products lack matching LOCO chip/run provenance.
Aggregate historical event scores cannot produce official-test confusion counts.
Artifact recovery should precede any future training decision.

## Conclusions and minimum additional work

| Unsupported claim | Minimum additional evidence/experiment |
|---|---|
| Negative delta for every event | Withdraw: observed surviving seeds already include positive deltas. No additional run can validate that universal claim for this reconstruction. |
| Complete original ten-event, three-seed result/mean | Recover the 21 missing selected checkpoints or complete provenance-linked per-chip predictions/counts; then evaluate official test only. If recovery fails, new runs under a prospectively documented protocol constitute a new experiment, not recovered original weights. No training authorized/launched. |
| Event-level stable direction or ranking | Recover/evaluate missing seeds per coverage.csv; at minimum obtain independent additional seeds for single-seed events. More independent events are needed beyond one flood per group; two seeds already change sign for Mekong/Pakistan. |
| Delta exclusively measures geographic exclusion | Separate matched-budget/composition experiments with repeated seeds and training-only/validation-only design choices; retain the same untouched test sets. Test matching alone is insufficient. |
| Spatially independent geographic generalization | First audit georeferenced hand/weak footprints and derived samples; if overlap exists, construct a separately declared spatially disjoint evaluation/training protocol and train only after approval. |
| Completed USA retry used its completed pretrain | Recover initialization trace or weights lineage. Without it, only a newly documented run with explicit init hash could establish that recipe; this would not prove historical initialization. |
| Field flood accuracy in Azerbaijan | Independent real-flood ground truth and evaluation; LOCO reconstruction cannot establish deployment accuracy. |

## Reproduce

From repository root, on the existing environment:

```sh
# Already executed once, before inference; refuses to overwrite the selection:
.venv/bin/python scripts/reconstruct_a1.py freeze
.venv/bin/python scripts/reconstruct_a1.py run --smoke
.venv/bin/python scripts/reconstruct_a1.py run
.venv/bin/python -m unittest discover -s tests -p 'test_a1*.py'
.venv/bin/python scripts/export_a1_reconstruction.py
```

The smoke pass checked the first generalist event batch; full paired inference
then used the frozen mapping and checked every score against the shared evaluator.
Nine focused checks passed. Existing software versions are retained from the
initial audit; reconstruction runtime/source hashes and command logs accompany
these results. No original weights were overwritten; historical outputs remain.
'''
    (OUT/'REPORT.md').write_text(text)

if __name__=='__main__':main()
