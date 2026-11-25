#!/usr/bin/env python3
"""Recount the committed Azerbaijan prototype masks from their original Git blobs."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from osgeo import gdal


SITES = (
    ("baku", "2026-04-07", "74d6227388bb11b936a7819d8b552e4360dcfef6"),
    ("sabirabad", "2026-06-30", "ad1e350e68dde39c664ddc924d9cd0a61919a030"),
)


def read_mask(repository, commit, asset):
    blob = subprocess.check_output(["git", "-C", str(repository), "cat-file", "blob", f"{commit}:{asset}"])
    filename = f"/vsimem/{asset.name}"
    gdal.FileFromMemBuffer(filename, blob)
    try:
        with gdal.Open(filename) as raster:
            if raster.RasterCount != 1 or raster.GetSpatialRef().GetAuthorityCode(None) != "4326":
                raise ValueError(f"Expected a single-band EPSG:4326 raster: {asset}")
            pixels = raster.ReadAsArray()
            grid = raster.GetGeoTransform()
        if not np.all((pixels == 0) | (pixels == 1)):
            raise ValueError(f"Expected a binary raster: {asset}")
        return pixels.astype(bool), grid, hashlib.sha256(blob).hexdigest()
    finally:
        gdal.Unlink(filename)


def audit_site(repository, site, date, commit):
    assets = Path("web/public/data") / site / date
    predicted, grid, predicted_hash = read_mask(repository, commit, assets / "flood_mask.tif")
    jrc, jrc_grid, jrc_hash = read_mask(repository, commit, assets / "permanent_water.tif")
    invalid, invalid_grid, invalid_hash = read_mask(repository, commit, assets / "layover_shadow.tif")
    if grid != jrc_grid or grid != invalid_grid or predicted.shape != jrc.shape or predicted.shape != invalid.shape:
        raise ValueError(f"Raster grids differ for {site}")

    left, dx, x_rotation, top, y_rotation, dy = grid
    if x_rotation or y_rotation or dx <= 0 or dy >= 0:
        raise ValueError(f"Expected north-up unrotated raster for {site}")
    rows, cols = predicted.shape
    center_latitude = top + rows * dy / 2
    approximate_pixel_km2 = abs(dx * dy) * 111.32**2 * np.cos(np.deg2rad(center_latitude))
    longitudes = left + (np.arange(cols) + 0.5) * dx
    latitudes = top + (np.arange(rows) + 0.5) * dy
    if site == "baku":
        gap_boundary = "longitude >= 50 degrees east"
        gap = np.broadcast_to(longitudes[None, :] >= 50, predicted.shape)
    else:
        gap_boundary = "latitude < 40 degrees north"
        gap = np.broadcast_to(latitudes[:, None] < 40, predicted.shape)

    predicted_count = int(predicted.sum())
    invalid_count = int(invalid.sum())
    return {
        "site": site,
        "acquisition_date_as_recorded": date,
        "asset_commit": commit,
        "asset_directory": assets.as_posix(),
        "asset_sha256": {
            "flood_mask.tif": predicted_hash,
            "permanent_water.tif": jrc_hash,
            "layover_shadow.tif": invalid_hash,
        },
        "raster_height": rows,
        "raster_width": cols,
        "raster_pixel_count": predicted.size,
        "bounds_wgs84": [left, top + rows * dy, left + cols * dx, top],
        "positive_pixels": predicted_count,
        "positive_pct_full_raster": 100 * predicted_count / predicted.size,
        "positive_approx_km2": predicted_count * approximate_pixel_km2,
        "jrc_exclusion_pixels": int(jrc.sum()),
        "terrain_or_missing_dem_exclusion_pixels": invalid_count,
        "terrain_or_missing_dem_exclusion_pct_full_raster": 100 * invalid_count / predicted.size,
        "either_exclusion_pixels": int((jrc | invalid).sum()),
        "positive_overlap_with_jrc_exclusion_pixels": int((predicted & jrc).sum()),
        "positive_overlap_with_terrain_or_missing_dem_exclusion_pixels": int((predicted & invalid).sum()),
        "dem_tile_boundary_for_gap_check": gap_boundary,
        "pixels_on_gap_side_of_boundary": int(gap.sum()),
        "terrain_or_missing_dem_exclusion_on_gap_side_pixels": int((invalid & gap).sum()),
        "terrain_or_missing_dem_exclusion_on_gap_side_pct_of_flags": 100 * int((invalid & gap).sum()) / invalid_count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ml_repository", type=Path, help="Checkout with the two public asset commits")
    arguments = parser.parse_args()
    gdal.UseExceptions()
    print(json.dumps([audit_site(arguments.ml_repository, *site) for site in SITES], indent=2))


if __name__ == "__main__":
    main()
