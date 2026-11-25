#!/usr/bin/env python3
"""Compare real HTTP uploads with fresh offline GeoTIFFs and archived masks."""

import argparse
import base64
import hashlib
import io
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import httpx
import numpy as np
from PIL import Image
import rasterio
from rasterio.io import MemoryFile
import yaml


HERE = Path(__file__).resolve().parent
CHECKPOINT_SHA256 = "96d634b95a7ac53a77dba4f5ccc4f1ebbec7779a4103cd6b26bd7544d5c956f6"


def digest(path, algorithm="sha256"):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def git(repository, *arguments):
    return subprocess.check_output(["git", "-C", str(repository), *arguments])


def mask_grid(raster):
    return raster.shape, raster.transform, raster.crs


def read_archived_masks(ml, record, grid):
    masks = {}
    for name, checksum in record["asset_sha256"].items():
        data = git(ml, "show", f"{record['asset_commit']}:{record['asset_directory']}/{name}")
        assert hashlib.sha256(data).hexdigest() == checksum
        with MemoryFile(data) as memory, memory.open() as raster:
            assert mask_grid(raster) == grid
            masks[name] = raster.read(1) > 0
    return masks


def compare_masks(before, after):
    added = int(np.count_nonzero(after & ~before))
    removed = int(np.count_nonzero(before & ~after))
    return {
        "before_positive_pixels": int(before.sum()),
        "after_positive_pixels": int(after.sum()),
        "added_positive_pixels": added,
        "removed_positive_pixels": removed,
        "different_pixels": added + removed,
    }


def mask_digest(mask):
    return hashlib.sha256(mask.tobytes()).hexdigest()


def upload_scenes(ml, work, env, scenes):
    with socket.socket() as listener, (work / "uvicorn.log").open("w+") as log:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        address = f"http://127.0.0.1:{listener.getsockname()[1]}"
        args = [sys.executable, "-m", "uvicorn", "api.app:app", "--fd", str(listener.fileno())]
        with subprocess.Popen(
            args, cwd=ml, env=env, pass_fds=(listener.fileno(),),
            stdout=log, stderr=subprocess.STDOUT,
        ) as server:
            try:
                with httpx.Client(base_url=address, trust_env=False, timeout=600) as client:
                    for _ in range(120):
                        if server.poll() is not None:
                            raise RuntimeError(f"API startup exited with {server.returncode}")
                        try:
                            client.get("/openapi.json", timeout=1).raise_for_status()
                            break
                        except (httpx.RequestError, httpx.HTTPStatusError):
                            time.sleep(0.5)
                    else:
                        raise TimeoutError("Timed out waiting for API startup")

                    uploaded = {}
                    for name, scene in scenes.items():
                        with scene.open("rb") as stream:
                            response = client.post("/predict", files={"file": (scene.name, stream, "image/tiff")})
                        response.raise_for_status()
                        payload = response.json()
                        image_data = io.BytesIO(base64.b64decode(payload["mask_png_base64"], validate=True))
                        with Image.open(image_data) as image:
                            assert image.mode == "RGBA"
                            mask = np.asarray(image.getchannel("A")) > 0
                        uploaded[name] = (response.status_code, payload, mask)
                        print(f"{name}: HTTP {response.status_code}, {int(mask.sum())} positive pixels",
                              file=sys.stderr, flush=True)
                    return uploaded
            except Exception:
                log.seek(0)
                print(log.read(), file=sys.stderr)
                raise
            finally:
                server.terminate()
                server.wait(timeout=30)


def offline_scene(ml, work, env, scene, threshold, archive, previous):
    paths = {name: work / name for name in (
        "flood_mask.tif", "permanent_water.tif", "layover_shadow.tif", "probability.tif",
    )}
    command = [
        sys.executable, "-m", "inference.predict", "--input", str(scene),
        "--output", str(paths["flood_mask.tif"]),
        "--prob-output", str(paths["probability.tif"]),
        "--permanent-water-output", str(paths["permanent_water.tif"]),
        "--layover-shadow-output", str(paths["layover_shadow.tif"]),
    ]
    run = subprocess.run(command, cwd=ml, env=env, text=True, capture_output=True)
    if run.returncode:
        raise RuntimeError(run.stderr)
    with rasterio.open(scene) as raster:
        grid = mask_grid(raster)
    products = {}
    for key, path in paths.items():
        with rasterio.open(path) as raster:
            assert mask_grid(raster) == grid
            image = raster.read(1)
        if key == "probability.tif":
            raw = image >= threshold
        else:
            products[key] = image > 0
    archived_masks = read_archived_masks(ml, archive, grid)
    archived_differences = {
        name: compare_masks(archived_masks[name], mask) for name, mask in products.items()
    }
    # The previous replay established exact terrain and JRC parity with the
    # archived layers. Recreate its final water mask on the fresh post-JRC
    # pixels, then require its recorded SHA-256 before counting a change.
    for name in ("layover_shadow.tif", "permanent_water.tif"):
        assert previous["current_offline_vs_archived"][name]["different_pixels"] == 0
    assert archived_differences["permanent_water.tif"]["different_pixels"] == 0
    after_jrc = raw & ~products["permanent_water.tif"]
    final = products["flood_mask.tif"]
    assert np.array_equal(final, after_jrc & ~products["layover_shadow.tif"])
    legacy_final = after_jrc & ~archived_masks["layover_shadow.tif"]
    assert mask_digest(legacy_final) == previous["offline_mask_sha256"]
    return final, {
        "offline_raw_positive_pixels": int(raw.sum()),
        "offline_after_jrc_positive_pixels": int(after_jrc.sum()),
        "offline_positive_pixels": int(final.sum()),
        "terrain_vs_previous_api": archived_differences["layover_shadow.tif"],
        "final_water_vs_previous_api": compare_masks(legacy_final, final),
        "current_offline_vs_archived": archived_differences,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ml_repository", type=Path, help="Checkout with the reviewed API commit and local scene artifacts")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    ml = args.ml_repository.resolve()
    config_path = ml / "config/default.yaml"
    config = yaml.safe_load(config_path.read_text())
    inference = config["inference"]
    checkpoint_hash = digest(ml / inference["checkpoint"])
    assert checkpoint_hash == CHECKPOINT_SHA256
    prior = json.loads((HERE / "api_replay_before_dem_fix.json").read_text())
    assert prior["checkpoint_sha256"] == checkpoint_hash
    assert prior["config_sha256"] == digest(config_path)
    previous_scenes = {row["site"]: row for row in prior["scenes"]}
    archives = json.loads((HERE / "archived_mask_counts.json").read_text())
    refs = json.loads((HERE / "reference_mask_checks.json").read_text())
    jrc = ml / inference["permanent_water"]["gsw_dir"] / "occurrence_40E_50Nv1_4_2021.tif"
    jrc_hash = digest(jrc)
    for row in refs:
        assert jrc_hash == row["jrc_source_tile_sha256"]
        dem = ml / inference["layover_shadow"]["dem_dir"] / row["dem_source_tile"]
        assert digest(dem) == row["dem_source_tile_sha256"]
    scenes = {row["site"]: ml / f"data/demo_scenes/{row['site']}.tif" for row in archives}
    for scene in scenes.values():
        source = yaml.safe_load(scene.with_suffix(".tif.dvc").read_text())["outs"][0]
        assert scene.stat().st_size == source["size"] and digest(scene, "md5") == source["md5"]

    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads), CUDA_VISIBLE_DEVICES="",
        NO_ALBUMENTATIONS_UPDATE="1", HF_HUB_OFFLINE="1", GDAL_CACHEMAX="128",
        PYTHONPATH=os.pathsep.join(filter(None, [str(ml / "src"), env.get("PYTHONPATH")])),
    )
    rows = []
    with tempfile.TemporaryDirectory(prefix="flood-api-replay-") as temporary:
        work = Path(temporary)
        uploaded = upload_scenes(ml, work, env, scenes)
        for archived in archives:
            name = archived["site"]
            scene = scenes[name]
            status, api, api_mask = uploaded.pop(name)
            previous = previous_scenes[name]
            final, diagnostics = offline_scene(
                ml, work, env, scene, inference["threshold"], archived, previous,
            )
            scene_hash = digest(scene)
            assert scene_hash == previous["scene_sha256"]
            assert api["fusion"] == previous["fusion"]
            assert api_mask.shape == final.shape
            differences = int(np.count_nonzero(api_mask != final))
            with rasterio.open(scene) as raster:
                left, bottom, right, top = raster.bounds
                assert api["bounds"] == [[round(bottom, 6), round(left, 6)], [round(top, 6), round(right, 6)]]
                transform = raster.transform
                pixel_km2 = abs(transform.a * transform.e) * 111.32**2 * math.cos(math.radians((bottom + top) / 2))
            count = int(api_mask.sum())
            assert api["flooded_pct"] == round(100 * count / api_mask.size, 2)
            assert api["flooded_area_km2"] == round(count * pixel_km2, 2)
            assert api["fusion"]["raster_pixels"] == api_mask.size
            rows.append({
                "site": name, "scene_sha256": scene_hash, "http_status": status,
                "api_positive_pixels": count, "api_flooded_pct": api["flooded_pct"],
                "api_flooded_area_km2": api["flooded_area_km2"], "fusion": api["fusion"],
                "api_mask_sha256": mask_digest(api_mask),
                "offline_mask_sha256": mask_digest(final),
                "api_vs_offline_differing_pixels": differences,
                **diagnostics,
            })
            print(f"{name}: {differences} differing pixels vs offline CLI", file=sys.stderr, flush=True)

    print(json.dumps({
        "ml_commit": git(ml, "rev-parse", "HEAD").decode().strip(),
        "previous_api_commit": prior["ml_commit"],
        "checkpoint_sha256": checkpoint_hash,
        "config_sha256": digest(config_path),
        "reference_hashes": {
            jrc.name: jrc_hash,
            **{row["dem_source_tile"]: row["dem_source_tile_sha256"] for row in refs},
        },
        "runtime": {
            "python": sys.version.split()[0], "threads": args.threads, "device": "cpu",
            **{name: version(name) for name in ("torch", "transformers", "rasterio", "fastapi", "uvicorn")},
            "gdal": rasterio.__gdal_version__,
        },
        "scenes": rows,
    }, indent=2))
    if any(row["api_vs_offline_differing_pixels"] for row in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
