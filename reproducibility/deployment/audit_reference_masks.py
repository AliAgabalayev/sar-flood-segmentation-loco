#!/usr/bin/env python3
"""Reconstruct historical masks using the archived terrain sign convention."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from osgeo import gdal

from audit_archived_masks import SITES, read_mask


DEM_TILES = {"baku": "SRTM1_N40E049.tif", "sabirabad": "SRTM1_N40E048.tif"}
OCCURRENCE_TILE = "occurrence_40E_50Nv1_4_2021.tif"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resample(tile, shape, grid, nodata):
    with gdal.Translate("/vsimem/reference.vrt", str(tile), format="VRT", noData=nodata) as source:
        rows, cols = shape
        aligned = gdal.GetDriverByName("MEM").Create("", cols, rows, 1, gdal.GDT_Float32)
        aligned.SetGeoTransform(grid)
        aligned.SetProjection(source.GetProjection())
        aligned.GetRasterBand(1).SetNoDataValue(nodata)
        aligned.GetRasterBand(1).Fill(nodata)
        gdal.ReprojectImage(source, aligned, source.GetProjection(), source.GetProjection(), gdal.GRA_Bilinear)
        pixels = aligned.ReadAsArray()
    gdal.Unlink("/vsimem/reference.vrt")
    return pixels


def dilate_four_neighbors(mask):
    expanded = mask.copy()
    expanded[1:, :] |= mask[:-1, :]
    expanded[:-1, :] |= mask[1:, :]
    expanded[:, 1:] |= mask[:, :-1]
    expanded[:, :-1] |= mask[:, 1:]
    return expanded


def audit_site(repository, site, date, commit):
    assets = Path("web/public/data") / site / date
    archived_invalid, grid, _ = read_mask(repository, commit, assets / "layover_shadow.tif")
    archived_jrc, jrc_grid, _ = read_mask(repository, commit, assets / "permanent_water.tif")
    if grid != jrc_grid or archived_invalid.shape != archived_jrc.shape:
        raise ValueError(f"Archived JRC and DEM grids are different for {site}")
    shape = archived_invalid.shape
    occurrence_tile = repository / "data/reference/jrc_gsw/occurrence" / OCCURRENCE_TILE
    dem_tile = repository / "data/reference/dem" / DEM_TILES[site]
    occurrence = resample(occurrence_tile, shape, grid, 255)
    jrc_nodata = occurrence == 255
    jrc = (occurrence >= 20) & ~jrc_nodata
    elevation = resample(dem_tile, shape, grid, -32768)
    dem_nodata = elevation == -32768
    missing = dilate_four_neighbors(dem_nodata)

    _, dx_deg, _, top, _, dy_deg = grid
    center_latitude = top + shape[0] * dy_deg / 2
    dx_m = abs(dx_deg) * 111_320 * np.cos(np.deg2rad(center_latitude))
    dy_m = abs(dy_deg) * 110_540
    grad_y, grad_x = np.gradient(elevation, dy_m, dx_m)
    slope = np.arctan(np.hypot(grad_x, grad_y))
    aspect = np.arctan2(-grad_x, grad_y) % (2 * np.pi)
    range_slope = np.arctan(np.tan(slope) * np.cos(np.deg2rad(90) - aspect))
    incidence_rows = []
    for nominal_angle in (29.1, 37.55, 46.0):
        angle = np.deg2rad(nominal_angle) - range_slope
        terrain = ((angle <= 0) | (angle >= np.pi / 2)) & ~missing
        incidence_rows.append({
            "nominal_incidence_degrees": nominal_angle,
            "angle_flags_outside_dilated_missing_dem": int(terrain.sum()),
            "pixel_differences_vs_archived_invalid_mask": int(((terrain | missing) ^ archived_invalid).sum()),
        })
    return {
        "site": site,
        "jrc_source_tile_sha256": digest(occurrence_tile),
        "jrc_nodata_pixels": int(jrc_nodata.sum()),
        "jrc_pixel_differences_vs_archived_exclusion_mask": int((jrc ^ archived_jrc).sum()),
        "dem_source_tile": dem_tile.name,
        "dem_source_tile_sha256": digest(dem_tile),
        "dem_nodata_pixels_before_dilation": int(dem_nodata.sum()),
        "dem_nodata_pixels_after_dilation": int(missing.sum()),
        "incidence_sensitivity": incidence_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ml_repository", type=Path)
    arguments = parser.parse_args()
    gdal.UseExceptions()
    print(json.dumps([audit_site(arguments.ml_repository, *site) for site in SITES], indent=2))


if __name__ == "__main__":
    main()
