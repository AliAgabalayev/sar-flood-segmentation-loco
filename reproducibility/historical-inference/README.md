# Frozen historical inference recipe

This overlay preserves the source bytes used for the October reconstruction
where recorded SHA-256 values are available. `SHA256SUMS` also covers the
transferred evaluator dependencies, archived YAML configurations, and the ten
frozen per-event test manifests. All other inference modules come from
companion Git commit `867a4f3e3dae9a935578e401f5c5f970e3d0adac`.

Prepare an isolated tree from a companion checkout containing that commit.
Use a new output directory. The setup script archives the tracked source and
config at that revision, applies the reviewed overlay, and copies the frozen
selection and archived count fixtures. It uses symlinks for the source
checkout's rasters and checkpoints, which the archived inference command
reads. Outputs go into the isolated tree.

```sh
scratch_dir=$(mktemp -d)
python3 scripts/prepare_historical_inference.py ../Flood-Damage-Detection "$scratch_dir/inference"
cd "$scratch_dir/inference"
python3 -m unittest discover -s tests -p test_a1_reconstruction.py
```

For model inference, use Python 3.13 and install the companion's pinned
`requirements.txt` in a dedicated environment with the PyTorch build for
your hardware. The historical run used PyTorch 2.12.1+cu130 on an RTX 3060
Laptop GPU. Use `config/experiments/weak_finetune.yaml` to identify the
processed dataset layout. Provision all ten files by the relative names and
SHA-256 values in the frozen selection. Only use trusted weight artifacts,
since the original runner loads Lightning checkpoints with Python pickle
enabled. Then, from the scratch directory:

```sh
python3 scripts/reconstruct_a1.py run --smoke
python3 scripts/reconstruct_a1.py run
```

Use the existing freeze from October. The `freeze` command is retained for
provenance and requires the original run database and source context.
The full `run` command regenerates evidence beneath
`data/reference/a1_reconstruction` in the scratch tree. Hardware and library
differences can move pixels close to the threshold. Keep regenerated outputs
with their own environment record.

In November the scratch preparation and all three archived reconstruction
tests passed against the released fixtures. The benchmark model smoke run
reached the model import and stopped at the missing
`segmentation_models_pytorch` dependency. The available CPU environment
did not have that package and could not download it. This leaves the
October inference record and the newly verified aggregate arithmetic as
separate evidence.
