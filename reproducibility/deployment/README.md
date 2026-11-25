# Azerbaijan prototype artifact audit

The two acquisitions exercise an offline SegFormer-B4 mapping pipeline. The archive provides no independent, time-matched flood labels. The measurements below describe model masks and exclusions on full raster grids. They cannot estimate accuracy, false alarms, or true flooded area.

## Pixel counts from the committed products

| Site and recorded date | Positive pixels | Share of full raster | Approximate positive area | JRC exclusion pixels | Terrain or missing DEM exclusion pixels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baku, 7 April 2026 | 41,320 / 6,202,752 | 0.6662% | 3.145 km² | 902,191 | 1,241,953 (20.02%) |
| Sabirabad, 30 June 2026 | 26,669 / 5,806,168 | 0.4593% | 2.042 km² | 98,108 | 1,826,806 (31.46%) |

The dates are recorded by the project. The original Sentinel product IDs and overpass times are absent from the archived GeoTIFF tags. Baku was acquired during a documented rainfall and flooding episode. The city reported suburban flooding on 6 April. The Institute of Geography documented a rain-triggered wall failure near Lake Bulbula on 7 April. The records lack spatial inundation extent at overpass time. Sabirabad is an unlabeled acquisition with no documented matching flood event in the inspected project records. [Baku City Executive Power](https://baku-ih.gov.az/en/news/heavy-rains-have-caused-complications-in-baku-institutions-are-working-in-an-enhanced-mode.html), [Institute of Geography](https://igaz.az/en/news/2043), [original scene acquisition record](https://github.com/AliAgabalayev/Flood-Damage-Detection/commit/3ebbb18221dfa979ffbbe61afe769f4f7f266963).

The final positive rasters are disjoint from both exclusion rasters on their shared grid. The area estimate follows the project's mid-latitude pixel-area approximation. Its denominator and the positive percentage cover the entire raster, including all excluded pixels. The GeoTIFFs store those exclusions as zeros in the final layer, so a zero may mean either an unsuppressed negative prediction or an excluded pixel. [Baku archive](https://github.com/AliAgabalayev/Flood-Damage-Detection/tree/74d6227388bb11b936a7819d8b552e4360dcfef6/web/public/data/baku/2026-04-07), [Sabirabad archive](https://github.com/AliAgabalayev/Flood-Damage-Detection/tree/ad1e350e68dde39c664ddc924d9cd0a61919a030/web/public/data/sabirabad/2026-06-30).

```bash
python reproducibility/deployment/audit_archived_masks.py ../Flood-Damage-Detection
```

This requires Git, GDAL Python bindings and NumPy. The recorded audit used GDAL 3.12.4 and NumPy 2.4.6. It reads the GeoTIFFs from the cited Git objects. A full clone with those commits suffices and no DVC data, credentials or GPU are needed. The artifact [archived_mask_counts.json](archived_mask_counts.json) records hashes, overlap counts, bounds and unrounded areas.

## Historical CPU replay with archived filters

| Site | Raw positive pixels | After JRC | After JRC and DEM | Raw positives removed | Pixel differences versus original final product |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baku | 960,106 | 165,158 | 41,319 | 95.70% | 1 |
| Sabirabad | 116,713 | 44,785 | 26,668 | 77.15% | 1 |

The replay used the original scene bytes and the configured SegFormer-B4 fine-tune checkpoint at `models/segformer_b4/finetune/best.ckpt` (SHA-256 `96d634b95a7ac53a77dba4f5ccc4f1ebbec7779a4103cd6b26bd7544d5c956f6`, epoch index 46, global step 1504). The code assigns the checkpoint named by the deployment configuration and performs no checkpoint or threshold search. It loads all 930 model state tensors strictly after Hugging Face's checkpoint key renames. It uses 512-pixel tiles, 64-pixel overlap, feathered probability stitching, sigmoid and a ≥0.5 threshold without TTA. The two filters are applied using the archived exclusion rasters. The raw-to-filtered positive areas round to 73.08 to 3.15 km² in Baku and 8.94 to 2.04 km² in Sabirabad. These reductions measure the sensitivity of map output to masking. The source of the single per-scene pixel disagreement with the original final products is unresolved. [ML configuration](https://github.com/AliAgabalayev/Flood-Damage-Detection/blob/867a4f3e3dae9a935578e401f5c5f970e3d0adac/config/default.yaml), [SegFormer-B4 architecture configuration](https://huggingface.co/nvidia/mit-b4/blob/refs%2Fpr%2F2/config.json).

Use a checkout containing ML commit `867a4f3e3dae9a935578e401f5c5f970e3d0adac` with `data/demo_scenes/{baku,sabirabad}.tif` and `models/segformer_b4/finetune/best.ckpt` restored from local artifacts or the original DVC remote if authorized. The staging step verifies each SAR scene against the DVC MD5 and verifies grid alignment with the archived masks. In the commands below, run the first step with a Python interpreter providing GDAL 3.12.4 and NumPy 2.4.6. The second step used Python 3.13, NumPy 2.3.5, PyTorch 2.14.0+cu130 and Transformers 5.17.0, on CPU with four threads. The steps can use different interpreters. The staged arrays require roughly 130 MB of scratch space.

```bash
work_dir=$(mktemp -d)
PYTHONDONTWRITEBYTECODE=1 python reproducibility/deployment/prepare_scene_replay.py ../Flood-Damage-Detection "$work_dir"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python3.13 reproducibility/deployment/replay_scene_inference.py ../Flood-Damage-Detection "$work_dir"
```

The packaged commands reproduce the values in [scene_replay_counts.json](scene_replay_counts.json).

The historical MLflow run `fe3de4c6dc7c41eb8da72d0917361036` for the seed-42 SegFormer-B4 fine-tune logs best validation epoch 46 and test IoU/F1 of 0.702703/0.825397. It also logs `test_iou_fused` and `test_f1_fused` values of 0.300595 and 0.462242 in the same metric batch. The historical fusion evaluation command, masks and per-chip results were not recovered. The logged fused values cannot establish the effect of the deployment filters under a controlled evaluation, and none of these scores measure accuracy in Azerbaijan.

## Historical terrain mask and coverage

The archived DEM mask stores a union of local-angle threshold flags and
dilated DEM nodata. This audit intentionally uses the historical sign
when projecting the downhill aspect along the sensor look direction. Reprojection of the local SRTM1 tiles `N40E049` (Baku) and `N40E048` (Sabirabad) with bilinear GDAL resampling and the project's slope/aspect and range-slope formula reproduced both committed invalid masks pixel for pixel at the configured 37.55° nominal incidence and eastward look approximation. Baku contained 1,240,996 dilated missing-DEM pixels and 957 additional angle flags. Sabirabad contained 1,826,806 dilated missing-DEM pixels and zero additional angle flags. These gaps track the source tile edges at 50° E and 40° N, respectively. Incidence values of 29.1° and 46.0° produced 2,623 and 324 Baku angle flags outside the dilated missing region, respectively, and zero in Sabirabad. The sensitivity check uses a fixed orbit-direction assumption and leaves the missing coverage unchanged. [Terrain mask implementation](https://github.com/AliAgabalayev/Flood-Damage-Detection/blob/867a4f3e3dae9a935578e401f5c5f970e3d0adac/src/data/preprocessing.py).

The JRC filter excludes any inferred positive pixel where bilinearly resampled historical water occurrence reaches 20% and is valid. This rule can remove recurring or transient water. The 20% setting and terrain threshold were not validated on these scenes. The older SRTM1-versus-GLO-90 slope graphic, SHA-256 `f3ab7c9102846f82b79e8411281f349143903126608d193eefa60119971647aa`, specifies chosen 512-pixel crops and depicts sensitivity to input DEM. It lacks an independent terrain or flood reference. Release access to a GLO-30 tile through a particular download route does not establish absence of global product coverage. [JRC filter implementation](https://github.com/AliAgabalayev/Flood-Damage-Detection/blob/867a4f3e3dae9a935578e401f5c5f970e3d0adac/src/inference/permanent_water.py).

Both JRC exclusion rasters also reproduced pixel for pixel from the local Global Surface Water v1.4 occurrence tile. That tile left 1,243,224 Baku and 1,829,412 Sabirabad pixels with missing occurrence values, at the same respective tile edges as the missing DEM. The JRC rule treats missing occurrence as no JRC exclusion. The independent reference-layer audit uses the source tiles listed in [reference_mask_checks.json](reference_mask_checks.json), restored under `data/reference/dem/` and `data/reference/jrc_gsw/occurrence/` in the ML checkout.

```bash
PYTHONDONTWRITEBYTECODE=1 python reproducibility/deployment/audit_reference_masks.py ../Flood-Damage-Detection
```

## Earlier Baku images and label suitability

An untracked `data/raw/baku/` collection with October-2024-like names has two binary masks on a different grid (2,227 by 1,671 pixels, southwest corner near 49.7500° E, 40.3000° N). Their counts were 1,051,187 before and 993,197 after, with 28,135 gained and 86,125 lost positive pixels. The `baku_after_oct25.tif` VV band was entirely nonfinite. The masks lack known producer, acquisition timestamps and a documented reference labeling method. The difference in mask area cannot establish event absence or ground-truth accuracy, and these rasters cannot validate April 2026 predictions. Comparison would require an independently derived flood extent aligned in time and grid with the evaluated Sentinel-1 scene and explicit labels for unobserved or ambiguous regions.

## Historical offline products and the old upload endpoint

The five-site viewer uses precomputed assets. The separate FastAPI upload endpoint returns rendered PNGs, bounds and area statistics in JSON and explicitly disables the JRC and DEM exclusion stages. It checks the file extension, size, raster readability and minimum band count. This audit did not measure uptime, latency, operating-point accuracy or load handling. [Viewer asset index](https://github.com/AliAgabalayev/Flood-Damage-Detection/blob/867a4f3e3dae9a935578e401f5c5f970e3d0adac/web/public/data/locations.json), [upload API](https://github.com/AliAgabalayev/Flood-Damage-Detection/blob/867a4f3e3dae9a935578e401f5c5f970e3d0adac/src/api/app.py).

## Current upload replay

The [terrain-corrected ML commit](https://github.com/AliAgabalayev/Flood-Damage-Detection/commit/23563a7bebf28e508b394cf6858649718fd148a3)
corrects the signed local incidence used by both the upload API and offline
CLI. The sensor-to-ground look is nominally east on ascent and west on
descent. Aspect is downhill and clockwise from north. Terrain rising
along the look direction reduces local incidence. Both filters still use
the configured reference tiles and geometry. Missing DEM and its one-pixel
boundary are excluded. Missing JRC has no subtraction of its own. A request
without jointly usable JRC and DEM coverage receives HTTP 503.

Run the replay from the manuscript checkout with the corrected commit
checked out in the ML repository. Restore the original checkpoint and
two scene GeoTIFFs listed above and the reference tiles listed in
`reference_mask_checks.json`. The command reads archived masks from the
specified ML Git objects and checks the input digests. Use the ML API
environment with `httpx` available.

```sh
python reproducibility/deployment/verify_api_replay.py ../Flood-Damage-Detection
```

The script launches Uvicorn on loopback, sends both real multipart uploads,
and runs offline inference separately with the same checkpoint and
configuration. It compares each PNG alpha mask against the CLI GeoTIFF,
checks the bounds and recomputes the area and percentage. The
[November 25, 2026 replay](api_replay.json) records coverage, mask hashes,
and differences. The [earlier upload replay](api_replay_before_dem_fix.json)
records the prior sign convention. Both archived exclusion layers matched
that prior replay exactly. The script verifies the prior final-water mask
hash from the fresh post-JRC pixels and the archived terrain mask before
counting the effect of the correction.

The terrain difference is measured on the exclusion layer. The water
difference is measured on the final binary mask after both filters.
Comparisons against the archived viewer products remain separate from the
differences against the previous API output. Those archived products have
their own historical one-pixel difference from the previous API output.

| Scene | Terrain flips vs. prior | Final water flips vs. prior | Final positive pixels | HTTP vs. CLI flips | Current vs. archived water flips |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baku | 892 | 2 | 41,321 | 0 | 3 |
| Sabirabad | 0 | 0 | 26,668 | 0 | 1 |

In Baku the corrected terrain rule newly excludes 47 pixels and removes
845 previous exclusions. Two previously excluded raw positives survive
JRC and are restored to the final water mask. Both uploads returned
HTTP 200 with partial reference coverage. Mask digests and coverage
counts are in the replay record.
