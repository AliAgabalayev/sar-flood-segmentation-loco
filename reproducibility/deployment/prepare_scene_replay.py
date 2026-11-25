#!/usr/bin/env python3
"""Stage exact SAR scenes and archived exclusions for a CPU model replay."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from osgeo import gdal

from audit_archived_masks import SITES, read_mask


SCENE_MD5 = {
    "baku": "aa0ea89fb2e0a230c1215da22fc3f7c8",
    "sabirabad": "8327c3c92be35af88fcca46254f51a37",
}


def prepare_site(repository, output_dir, site, date, commit):
    path = repository / "data" / "demo_scenes" / f"{site}.tif"
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "md5").hexdigest()
    if checksum != SCENE_MD5[site]:
        raise ValueError(f"DVC scene hash mismatch for {path}")
    assets = Path("web/public/data") / site / date
    with gdal.Open(str(path)) as raster:
        if raster.RasterCount != 2 or raster.GetSpatialRef().GetAuthorityCode(None) != "4326":
            raise ValueError(f"Expected VV/VH on an EPSG:4326 raster: {path}")
        grid = raster.GetGeoTransform()
        image = raster.ReadAsArray().astype(np.float32)

    masks = {}
    for key, layer in (("published", "flood_mask"), ("jrc", "permanent_water"), ("invalid", "layover_shadow")):
        masks[key], mask_grid, _ = read_mask(repository, commit, assets / f"{layer}.tif")
        if grid != mask_grid or masks[key].shape != image.shape[1:]:
            raise ValueError(f"Scene and archived layer grids differ for {site}/{layer}")
    np.savez(output_dir / f"{site}.npz", scene=image, **masks)
    return {"site": site, "dvc_scene_md5": checksum, "nonfinite_sar_pixel_count": int((~np.isfinite(image).all(axis=0)).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ml_repository", type=Path)
    parser.add_argument("output_dir", type=Path)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    gdal.UseExceptions()
    print(json.dumps([prepare_site(arguments.ml_repository, arguments.output_dir, *site) for site in SITES], indent=2))


if __name__ == "__main__":
    main()
