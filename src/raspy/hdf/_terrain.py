"""Terrain HDF reading and GeoTIFF export.

Provides :func:`export_terrain` — reads a RasMapper terrain HDF5 file,
mosaics the source GeoTIFFs by priority, applies any Levee-type ground-line
modifications stored in the same HDF, and writes the result to a GeoTIFF.

Derived from analysis of:
  archive/DLLs/RasMapperLib/RasMapperLib/TerrainLayer.cs
  archive/DLLs/RasMapperLib/RasMapperLib.Terrain/RasterFileInfo.cs
  archive/DLLs/RasMapperLib/RasMapperLib/GroundLineModificationLayer.cs
  archive/DLLs/RasMapperLib/RasMapperLib/ElevationModificationGroup.cs

See docs/terrain_export.md for a full description of the HDF structure and
the modification-rasterisation algorithm.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("raspy.hdf")

__all__ = ["export_terrain"]

# NoData sentinel used by RasMapper terrain TIFFs.
_NODATA = -9999.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_terrain(hdf_path: str | Path, raster_path: str | Path) -> Path:
    """Export a RasMapper terrain HDF to a GeoTIFF.

    Reads all source GeoTIFFs from the ``Terrain/`` group of *hdf_path*,
    mosaics them in priority order (lowest ``@Priority`` value wins), applies
    any ``Levee``-type ground-line modifications from the ``Modifications/``
    group, and writes the result to *raster_path*.

    Parameters
    ----------
    hdf_path:
        Path to a RasMapper terrain HDF file (``File Type = "HEC Terrain"``).
    raster_path:
        Destination path for the output GeoTIFF.  Parent directories are
        created automatically.

    Returns
    -------
    Path
        Resolved absolute path of the written GeoTIFF.

    Raises
    ------
    FileNotFoundError
        If *hdf_path* does not exist, or any source TIFF referenced in the
        HDF is missing.
    KeyError
        If the HDF contains no ``Terrain/`` group or no source TIFF entries.
    """
    import h5py
    import rasterio
    from rasterio.merge import merge

    hdf_path = Path(hdf_path)
    raster_path = Path(raster_path)

    if not hdf_path.exists():
        raise FileNotFoundError(f"Terrain HDF not found: {hdf_path}")

    hdf_dir = hdf_path.parent

    # ------------------------------------------------------------------
    # 1. Collect source TIFFs and modification data from HDF
    # ------------------------------------------------------------------
    tiff_entries: list[tuple[int, Path]] = []
    modifications: list[dict[str, Any]] = []

    with h5py.File(hdf_path, "r") as f:
        terrain_grp = f.get("Terrain")
        if terrain_grp is None:
            raise KeyError(f"No 'Terrain' group in {hdf_path}")

        for name in terrain_grp:
            child = terrain_grp[name]
            if not isinstance(child, h5py.Group):
                continue
            if "File" not in child.attrs:
                continue
            rel = child.attrs["File"]
            if isinstance(rel, bytes):
                rel = rel.decode()
            priority = int(child.attrs.get("Priority", 999))
            tiff_path = (hdf_dir / rel).resolve()
            if not tiff_path.exists():
                raise FileNotFoundError(
                    f"Source TIFF not found: {tiff_path}\n"
                    f"  (referenced from {hdf_path})"
                )
            tiff_entries.append((priority, tiff_path))

        if not tiff_entries:
            raise KeyError(
                f"No source TIFF entries found in Terrain/ group of {hdf_path}"
            )

        mod_grp = f.get("Modifications")
        if mod_grp is not None:
            modifications = _read_modifications(mod_grp)

    # ------------------------------------------------------------------
    # 2. Mosaic source TIFFs  (lowest Priority number = highest priority
    #    = first in list = wins overlapping pixels in rasterio.merge)
    # ------------------------------------------------------------------
    tiff_entries.sort(key=lambda e: e[0])
    tiff_paths = [p for _, p in tiff_entries]

    datasets = [rasterio.open(p) for p in tiff_paths]
    try:
        mosaic, transform = merge(datasets)
        profile = datasets[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            compress="deflate",
        )
    finally:
        for ds in datasets:
            ds.close()

    # ------------------------------------------------------------------
    # 3. Apply modifications (if any)
    # ------------------------------------------------------------------
    if modifications:
        nodata = float(profile.get("nodata") or _NODATA)
        mosaic = _apply_modifications(mosaic, transform, modifications, nodata)

    # ------------------------------------------------------------------
    # 4. Write output
    # ------------------------------------------------------------------
    raster_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(raster_path, "w", **profile) as dst:
        dst.write(mosaic)

    logger.info("export_terrain: wrote %s", raster_path)
    return raster_path.resolve()


# ---------------------------------------------------------------------------
# Modification reading
# ---------------------------------------------------------------------------


def _read_modifications(mod_grp: Any) -> list[dict[str, Any]]:
    """Return a list of parsed modification dicts from an HDF Modifications/ group.

    Only ``Levee`` subtypes are parsed; others emit a warning and are skipped.
    The returned list is sorted so lower-priority modifications come first
    (they are applied first and can be overwritten by higher-priority ones).
    """
    import h5py

    mods: list[dict[str, Any]] = []
    for name, grp in mod_grp.items():
        if not isinstance(grp, h5py.Group):
            continue
        subtype = grp.attrs.get("Subtype", b"")
        if isinstance(subtype, bytes):
            subtype = subtype.decode()
        subtype = subtype.strip("\x00")

        if subtype != "Levee":
            logger.warning(
                "Modification %r has unsupported subtype %r — skipped "
                "(only 'Levee' is currently implemented)",
                name,
                subtype,
            )
            continue

        attrs_ds = grp.get("Attributes")
        if attrs_ds is None:
            logger.warning("Modification %r has no Attributes dataset — skipped", name)
            continue

        attrs = attrs_ds[0]

        elev_type = attrs["Elevation Type"]
        if isinstance(elev_type, bytes):
            elev_type = elev_type.decode()
        elev_type = elev_type.strip("\x00")

        top_width = float(attrs["Top Width"])
        left_slope = float(attrs["Left Slope"])
        right_slope = float(attrs["Right Slope"])
        max_reach = float(attrs["Max Reach"])

        # Slope normalisation: NaN → 3.0 (archive default); 0 → inf (vertical wall).
        if np.isnan(left_slope):
            left_slope = 3.0
        if np.isnan(right_slope):
            right_slope = 3.0
        if left_slope == 0.0:
            left_slope = float("inf")
        if right_slope == 0.0:
            right_slope = float("inf")

        pts_ds = grp.get("Polyline Points")
        if pts_ds is None:
            logger.warning("Modification %r has no Polyline Points — skipped", name)
            continue
        pts = pts_ds[:, :2].astype(np.float64)  # (N, 2) XY only

        profile_ds = grp.get("Profile Values")
        if profile_ds is None or len(profile_ds) < 2:
            logger.warning(
                "Modification %r has no usable Profile Values — skipped", name
            )
            continue
        profile = profile_ds[:].astype(np.float64)  # (M, 2): [station, elevation]

        priority = int(grp.attrs.get("Priority", 0))

        mods.append(
            {
                "name": name,
                "elev_type": elev_type,
                "top_width": top_width,
                "left_slope": left_slope,
                "right_slope": right_slope,
                "max_reach": max_reach,
                "pts": pts,
                "profile": profile,
                "priority": priority,
            }
        )

    # Lower priority value = applied last = wins.  Apply high-number (low-priority)
    # modifications first so low-number (high-priority) ones overwrite them.
    mods.sort(key=lambda m: -m["priority"])
    return mods


# ---------------------------------------------------------------------------
# Modification application
# ---------------------------------------------------------------------------


def _apply_modifications(
    mosaic: np.ndarray,
    transform: Any,
    modifications: list[dict[str, Any]],
    nodata: float,
) -> np.ndarray:
    """Apply levee modifications to *mosaic* and return the updated array.

    Parameters
    ----------
    mosaic:
        Float32 array of shape ``(1, H, W)`` from ``rasterio.merge``.
    transform:
        Affine transform mapping pixel coordinates to map coordinates.
    modifications:
        Parsed modification dicts from :func:`_read_modifications`.
    nodata:
        NoData sentinel value in the mosaic.

    Returns
    -------
    numpy.ndarray
        Updated mosaic array of the same shape and dtype.
    """
    from rasterio.transform import rowcol, xy

    data = mosaic[0].copy()  # (H, W)
    H, W = data.shape

    for mod in modifications:
        pts = mod["pts"]
        profile = mod["profile"]
        max_reach = mod["max_reach"]
        top_width = mod["top_width"]
        left_slope = mod["left_slope"]
        right_slope = mod["right_slope"]
        elev_type = mod["elev_type"]

        if max_reach <= 0:
            continue

        # -- Bounding box of modification footprint in map coordinates ------
        minx = pts[:, 0].min() - max_reach
        maxx = pts[:, 0].max() + max_reach
        miny = pts[:, 1].min() - max_reach
        maxy = pts[:, 1].max() + max_reach

        # Convert to pixel row/col (rowcol expects xs then ys)
        (r0, r1), (c0, c1) = rowcol(
            transform,
            [minx, maxx],
            [maxy, miny],  # maxy → smaller row (north-up), miny → larger row
        )

        r0 = max(0, int(r0))
        c0 = max(0, int(c0))
        r1 = min(H - 1, int(r1))
        c1 = min(W - 1, int(c1))

        if r0 > r1 or c0 > c1:
            continue  # footprint outside raster extent

        # -- Build pixel coordinate grid for the bounding box ---------------
        rows = np.arange(r0, r1 + 1)
        cols = np.arange(c0, c1 + 1)
        col_grid, row_grid = np.meshgrid(cols, rows)  # each (nrows, ncols)

        px, py = xy(transform, row_grid.ravel(), col_grid.ravel(), offset="center")
        px = np.asarray(px, dtype=np.float64)
        py = np.asarray(py, dtype=np.float64)

        # -- Project pixels onto polyline centerline ------------------------
        signed_perp, stations = _project_onto_polyline(px, py, pts)

        # -- Interpolate crest elevation from profile -----------------------
        crest = np.interp(stations, profile[:, 0], profile[:, 1])

        # -- Cross-section elevation ----------------------------------------
        abs_perp = np.abs(signed_perp)
        half_w = top_width / 2.0

        mod_elev = np.full(len(px), np.nan, dtype=np.float64)

        # Flat crest: |perp| <= half_width
        on_top = abs_perp <= half_w
        mod_elev[on_top] = crest[on_top]

        # Left side (signed_perp > 0) sloping zone
        left_zone = (signed_perp > half_w) & (abs_perp <= max_reach)
        if np.any(left_zone):
            drop = (abs_perp[left_zone] - half_w) / left_slope
            mod_elev[left_zone] = crest[left_zone] - drop

        # Right side (signed_perp < 0) sloping zone
        right_zone = (signed_perp < -half_w) & (abs_perp <= max_reach)
        if np.any(right_zone):
            drop = (abs_perp[right_zone] - half_w) / right_slope
            mod_elev[right_zone] = crest[right_zone] - drop

        # -- Apply elevation type -------------------------------------------
        valid = ~np.isnan(mod_elev)
        if not np.any(valid):
            continue

        vr = row_grid.ravel()[valid]
        vc = col_grid.ravel()[valid]
        orig = data[vr, vc].astype(np.float64)
        mval = mod_elev[valid]

        if elev_type in ("SetIfHigher", "TakeHigher"):
            new = np.maximum(orig, mval)
        elif elev_type in ("SetIfLower", "TakeLower"):
            new = np.minimum(orig, mval)
        elif elev_type in ("SetValue", "FixedElevation"):
            new = mval
        elif elev_type == "AddValue":
            new = orig + mval
        else:
            logger.warning(
                "Modification %r has unknown elevation type %r — skipped",
                mod["name"],
                elev_type,
            )
            continue

        # NoData pixels in original terrain always receive the modification value.
        nodata_mask = orig == nodata
        new[nodata_mask] = mval[nodata_mask]

        data[vr, vc] = new.astype(data.dtype)
        logger.debug(
            "Applied modification %r (%s) to %d pixels",
            mod["name"],
            elev_type,
            int(valid.sum()),
        )

    result = mosaic.copy()
    result[0] = data
    return result


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _project_onto_polyline(
    px: np.ndarray,
    py: np.ndarray,
    pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project query points onto a polyline, returning signed perpendicular
    distance and cumulative station for each point.

    Parameters
    ----------
    px, py:
        1-D arrays of query point X and Y coordinates.
    pts:
        ``(N, 2)`` array of polyline vertex coordinates.

    Returns
    -------
    signed_perp:
        Signed perpendicular distance from the centerline.
        Positive = left side of the line in the direction of travel.
    stations:
        Cumulative distance along the polyline to the closest point.
    """
    seg_dx = np.diff(pts[:, 0])
    seg_dy = np.diff(pts[:, 1])
    seg_len = np.sqrt(seg_dx**2 + seg_dy**2)

    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])

    n_pts = len(px)
    best_dist = np.full(n_pts, np.inf)
    best_station = np.zeros(n_pts)
    best_signed = np.zeros(n_pts)

    for i, (length, dx, dy) in enumerate(zip(seg_len, seg_dx, seg_dy)):
        if length < 1e-10:
            continue

        ax, ay = pts[i]

        # Parameter t for closest point on this segment, clamped to [0, 1]
        t = ((px - ax) * dx + (py - ay) * dy) / (length**2)
        t = np.clip(t, 0.0, 1.0)

        cx = ax + t * dx
        cy = ay + t * dy

        dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)

        # Signed distance: cross product of unit-direction and offset vector.
        # Positive → left of the line in direction of travel.
        ux, uy = dx / length, dy / length
        signed = ux * (py - cy) - uy * (px - cx)

        station = cum_len[i] + t * length

        closer = dist < best_dist
        best_dist[closer] = dist[closer]
        best_station[closer] = station[closer]
        best_signed[closer] = signed[closer]

    return best_signed, best_station
