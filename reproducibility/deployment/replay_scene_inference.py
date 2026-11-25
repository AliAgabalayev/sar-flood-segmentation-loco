#!/usr/bin/env python3
"""Replay full-scene SegFormer inference on CPU and apply the archived masks."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Callable, List, NamedTuple, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import interpolate
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import rename_source_key


ML_COMMIT = "867a4f3e3dae9a935578e401f5c5f970e3d0adac"
CHECKPOINT_SHA256 = "96d634b95a7ac53a77dba4f5ccc4f1ebbec7779a4103cd6b26bd7544d5c956f6"
SOURCE_DEFINITIONS = {
    "data/preprocessing.py": {
        "build_vv", "build_vh", "build_ratio", "Preprocessor", "default_preprocessor", "mask_layover_shadow",
    },
    "inference/tiling.py": {"TileRecord", "_compute_starts", "_pad_tile", "generate_tiles"},
    "inference/stitching.py": {"MaskTile", "_to_2d_numpy", "_feather_window", "stitch_tiles", "binarize"},
    "inference/permanent_water.py": {"subtract_permanent_water"},
}


def load_pipeline(repository):
    # Extract the pure-array operations from the pinned ML sources. The staged
    # arrays replace raster reads, so the CPU environment needs no GDAL binding.
    namespace = dict(globals())
    namespace.update(
        ArrayLike=np.ndarray,
        ChannelBuilder=Callable[[np.ndarray], np.ndarray],
        ClipRange=Tuple[float, float],
    )
    for source, names in SOURCE_DEFINITIONS.items():
        object_name = f"{ML_COMMIT}:src/{source}"
        blob = subprocess.check_output(["git", "-C", str(repository), "cat-file", "blob", object_name])
        source_tree = ast.parse(blob, filename=source)
        definitions = [node for node in source_tree.body if getattr(node, "name", None) in names]
        if len(definitions) != len(names):
            raise ValueError(f"Missing or repeated source definitions in {source}")
        module = ast.Module(body=definitions, type_ignores=[])
        exec(compile(module, source, "exec"), namespace)
    return namespace


def load_model(path):
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != CHECKPOINT_SHA256:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    transforms = (
        get_checkpoint_conversion_mapping("SegformerModel")
        + get_checkpoint_conversion_mapping("SegformerForSemanticSegmentation")
    )
    weights = {
        rename_source_key(name.removeprefix("net.net."), transforms, [])[0]: value
        for name, value in checkpoint["state_dict"].items()
        if name.startswith("net.net.")
    }
    config = SegformerConfig(
        num_channels=3,
        num_labels=1,
        depths=[3, 8, 27, 3],
        hidden_sizes=[64, 128, 320, 512],
        decoder_hidden_size=768,
    )
    model = SegformerForSemanticSegmentation(config).eval()
    model.load_state_dict(weights, strict=True)
    return model, {"sha256": checksum, "epoch_index": checkpoint["epoch"], "global_step": checkpoint["global_step"], "tensor_count": len(weights)}


def replay_site(site, file, model, pipeline):
    with np.load(file, allow_pickle=False) as data:
        image, published, jrc, invalid = (data[key] for key in ("scene", "published", "jrc", "invalid"))
    pipeline["_load_image"] = lambda _: image
    data_config = SimpleNamespace(vv_clip=(-25, 0), vh_clip=(-35, 0), ratio_clip=(0, 18))
    preprocessor = pipeline["default_preprocessor"](SimpleNamespace(data=data_config))
    tiles, shape = pipeline["generate_tiles"](None, preprocessor, tile_size=512, overlap=64)
    mask_tiles = []
    with torch.inference_mode():
        for tile in tiles:
            logits = model(pixel_values=tile.tile.unsqueeze(0)).logits
            probability = interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False).sigmoid()[0, 0]
            mask_tiles.append(pipeline["MaskTile"](probability, tile.row_start, tile.col_start))
    raw = pipeline["binarize"](pipeline["stitch_tiles"](mask_tiles, shape), threshold=0.5)
    after_jrc = pipeline["subtract_permanent_water"](raw, jrc)
    after_both = pipeline["mask_layover_shadow"](after_jrc, invalid)
    raw_count, final_count = int(raw.sum()), int(after_both.sum())
    return {
        "site": site,
        "tile_count": len(tiles),
        "full_raster_pixels": raw.size,
        "raw_positive_pixels": raw_count,
        "raw_positive_pct_full_raster": 100 * raw_count / raw.size,
        "after_jrc_positive_pixels": int(after_jrc.sum()),
        "after_jrc_and_dem_positive_pixels": final_count,
        "after_jrc_and_dem_positive_pct_full_raster": 100 * final_count / raw.size,
        "raw_positives_removed_pct": 100 * (raw_count - final_count) / raw_count,
        "published_positive_pixels": int(published.sum()),
        "pixel_differences_vs_published_mask": int(np.count_nonzero(after_both != published)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ml_repository", type=Path)
    parser.add_argument("staged_scenes", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args()
    torch.set_num_threads(arguments.threads)
    pipeline = load_pipeline(arguments.ml_repository)
    model, checkpoint = load_model(arguments.ml_repository / "models/segformer_b4/finetune/best.ckpt")
    rows = [replay_site(site, arguments.staged_scenes / f"{site}.npz", model, pipeline) for site in ("baku", "sabirabad")]
    print(json.dumps({"ml_commit": ML_COMMIT, "checkpoint": checkpoint, "scenes": rows}, indent=2))


if __name__ == "__main__":
    main()
