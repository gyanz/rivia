"""Terrain HDF reading and GeoTIFF/VRT export.

Provides :func:`export_terrain` — reads a RasMapper terrain HDF5 file,
mosaics the source GeoTIFFs by priority, applies any Levee-type ground-line
modifications stored in the same HDF, and writes the result to a GeoTIFF or
GDAL VRT.

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
import shutil
import xml.dom.minidom
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np

logger = logging.getLogger("raspy.hdf")

__all__ = ["export_terrain"]

# NoData sentinel used by RasMapper terrain TIFFs.
_NODATA = -9999.0

# Mapping from rasterio dtype strings to GDAL VRT dataType attribute values.
_GDAL_DTYPE: dict[str, str] = {
    "float32": "Float32",
    "float64": "Float64",
    "int16": "Int16",
    "int32": "Int32",
    "uint8": "Byte",
    "uint16": "UInt16",
    "uint32": "UInt32",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_terrain(
    hdf_path: str | Path,
    raster_path: str | Path,
    copy: bool = False,
) -> Path:
    """Export a RasMapper terrain HDF to a GeoTIFF or GDAL VRT.

    Reads all source GeoTIFFs from the ``Terrain/`` group of *hdf_path*,
    mosaics them in priority order (lowest ``@Priority`` value wins), applies
    any ``Levee``-type ground-line modifications from the ``Modifications/``
    group, and writes the result to *raster_path*.

    When *raster_path* has a ``.vrt`` extension the output is a GDAL VRT that
    references the original source TIFFs directly rather than re-encoding
    pixels.  If modifications are present a sidecar
    ``<stem>_mods.tif`` is written beside the VRT and included as the
    top-most layer.

    Parameters
    ----------
    hdf_path:
        Path to a RasMapper terrain HDF file (``File Type = "HEC Terrain"``).
    raster_path:
        Destination path.  Use a ``.tif`` extension for a merged GeoTIFF or a
        ``.vrt`` extension for a GDAL VRT.  Parent directories are created
        automatically.
    copy:
        Only relevant when *raster_path* is a ``.vrt``.  When ``True`` each
        source TIFF is copied into the VRT's parent directory and the VRT uses
        relative paths.  When ``False`` (default) the VRT uses absolute paths
        and source files are not copied.

    Returns
    -------
    Path
        Resolved absolute path of the written file.

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

    is_vrt = raster_path.suffix.lower() == ".vrt"

    # ------------------------------------------------------------------
    # 3. Apply modifications (if any)
    # ------------------------------------------------------------------
    mod_sidecar_path: Path | None = None
    if modifications:
        nodata = float(profile.get("nodata") or _NODATA)
        merged_unmodified = mosaic.copy()
        mosaic = _apply_modifications(mosaic, transform, modifications, nodata)

        if is_vrt:
            # Write a sidecar TIF containing only the pixels that changed so
            # that the VRT can reference it as a top-most layer.
            changed = mosaic[0] != merged_unmodified[0]
            mod_only = np.full_like(mosaic, nodata)
            mod_only[0][changed] = mosaic[0][changed]

            mod_sidecar_path = raster_path.with_name(raster_path.stem + "_mods.tif")
            mod_profile = profile.copy()
            mod_profile["nodata"] = nodata
            raster_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(mod_sidecar_path, "w", **mod_profile) as dst:
                dst.write(mod_only)
            logger.debug("export_terrain: wrote mod sidecar %s", mod_sidecar_path)

    # ------------------------------------------------------------------
    # 4. Write output
    # ------------------------------------------------------------------
    raster_path.parent.mkdir(parents=True, exist_ok=True)

    if is_vrt:
        # VRT: sources in descending priority order so that the lowest
        # Priority number (highest precedence) is listed last and wins
        # overlapping pixels (GDAL VRT last-source-wins semantics).
        vrt_sources = list(reversed(tiff_paths))
        if mod_sidecar_path is not None:
            vrt_sources.append(mod_sidecar_path)
        _write_vrt(raster_path, vrt_sources, profile, transform, copy=copy)
    else:
        with rasterio.open(raster_path, "w", **profile) as dst:
            dst.write(mosaic)

    logger.info("export_terrain: wrote %s", raster_path)
    return raster_path.resolve()


# ---------------------------------------------------------------------------
# VRT writer
# ---------------------------------------------------------------------------


def _write_vrt(
    vrt_path: Path,
    source_paths: list[Path],
    vrt_profile: dict[str, Any],
    vrt_transform: Any,
    copy: bool,
) -> None:
    """Write a GDAL VRT mosaic referencing *source_paths*.

    Parameters
    ----------
    vrt_path:
        Destination ``.vrt`` file.
    source_paths:
        Ordered list of source TIFFs.  Sources listed later overwrite earlier
        ones in overlapping pixels (GDAL last-source-wins semantics), so the
        highest-precedence source should be last.
    vrt_profile:
        rasterio profile for the VRT canvas (must include ``width``,
        ``height``, ``crs``, ``dtype``).
    vrt_transform:
        Affine transform for the VRT canvas.
    copy:
        When ``True``, copy each source into the VRT's parent directory and
        use relative ``SourceFilename`` paths.  When ``False``, use absolute
        paths and leave source files in place.
    """
    import rasterio

    vrt_dir = vrt_path.parent
    W = vrt_profile["width"]
    H = vrt_profile["height"]
    nodata = float(vrt_profile.get("nodata") or _NODATA)
    dtype = str(vrt_profile.get("dtype", "float32")).lower()
    gdal_dtype = _GDAL_DTYPE.get(dtype, "Float32")

    crs = vrt_profile.get("crs")
    crs_wkt = crs.to_wkt() if crs else ""

    gt = vrt_transform
    gt_str = f"  {gt.c}, {gt.a}, {gt.b}, {gt.f}, {gt.d}, {gt.e}"

    root = Element("VRTDataset", rasterXSize=str(W), rasterYSize=str(H))
    if crs_wkt:
        SubElement(root, "SRS").text = crs_wkt
    SubElement(root, "GeoTransform").text = gt_str

    band_el = SubElement(root, "VRTRasterBand", dataType=gdal_dtype, band="1")
    SubElement(band_el, "NoDataValue").text = str(nodata)

    for src_path in source_paths:
        with rasterio.open(src_path) as src:
            src_w = src.width
            src_h = src.height
            src_t = src.transform
            block_shapes = src.block_shapes
            block_w, block_h = block_shapes[0] if block_shapes else (256, 256)

        # Pixel offset of this source within the VRT canvas.
        dst_x_off = round((src_t.c - gt.c) / gt.a)
        dst_y_off = round((src_t.f - gt.f) / gt.e)
        # Destination size scaled for any resolution difference.
        dst_x_size = round(src_w * src_t.a / gt.a)
        dst_y_size = round(src_h * abs(src_t.e) / abs(gt.e))

        if copy:
            if src_path.parent != vrt_dir:
                dest = vrt_dir / src_path.name
                shutil.copy2(src_path, dest)
            file_ref = src_path.name
            relative_to_vrt = "1"
        else:
            file_ref = src_path.as_posix()
            relative_to_vrt = "0"

        # ComplexSource (vs SimpleSource) is required so that nodata pixels in
        # the source are masked (transparent) when compositing.  With
        # SimpleSource, nodata values are copied as-is and would overwrite
        # valid pixels from underlying sources — critically wrong for the
        # modification sidecar TIF which is nodata everywhere except where
        # modifications were applied.
        src_el = SubElement(band_el, "ComplexSource")
        SubElement(
            src_el, "SourceFilename", relativeToVRT=relative_to_vrt
        ).text = file_ref
        SubElement(src_el, "SourceBand").text = "1"
        SubElement(
            src_el,
            "SourceProperties",
            RasterXSize=str(src_w),
            RasterYSize=str(src_h),
            DataType=gdal_dtype,
            BlockXSize=str(block_w),
            BlockYSize=str(block_h),
        )
        SubElement(
            src_el,
            "SrcRect",
            xOff="0",
            yOff="0",
            xSize=str(src_w),
            ySize=str(src_h),
        )
        SubElement(
            src_el,
            "DstRect",
            xOff=str(dst_x_off),
            yOff=str(dst_y_off),
            xSize=str(dst_x_size),
            ySize=str(dst_y_size),
        )
        SubElement(src_el, "NODATA").text = str(nodata)

    xml_str = xml.dom.minidom.parseString(tostring(root)).toprettyxml(indent="  ")
    vrt_path.write_text(xml_str, encoding="utf-8")


# ---------------------------------------------------------------------------
# Modification reading
# ---------------------------------------------------------------------------


def _read_modifications(mod_grp: Any) -> list[dict[str, Any]]:
    """Return a list of parsed modification dicts from an HDF Modifications/ group.

    ``Levee`` and ``Channel`` subtypes are parsed; others emit a warning and
    are skipped.  The returned list is sorted so lower-priority modifications
    come first (they are applied first and can be overwritten by
    higher-priority ones).
    """
    import h5py

    _SUPPORTED_SUBTYPES = {"Levee", "Channel"}

    mods: list[dict[str, Any]] = []
    for name, grp in mod_grp.items():
        if not isinstance(grp, h5py.Group):
            continue
        subtype = grp.attrs.get("Subtype", b"")
        if isinstance(subtype, bytes):
            subtype = subtype.decode()
        subtype = subtype.strip("\x00")

        if subtype not in _SUPPORTED_SUBTYPES:
            logger.warning(
                "Modification %r has unsupported subtype %r — skipped "
                "(only 'Levee' and 'Channel' are currently implemented)",
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

        # Slope normalisation: NaN → 3.0 (archive default); 0 → 0.001 (near-
        # vertical wall, matching RasMapper's GetUsableSlope behaviour).
        if np.isnan(left_slope):
            left_slope = 3.0
        if np.isnan(right_slope):
            right_slope = 3.0
        if left_slope == 0.0:
            left_slope = 0.001
        if right_slope == 0.0:
            right_slope = 0.001

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
                "subtype": subtype,
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
    """Apply levee/channel modifications to *mosaic* and return the updated array.

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
        subtype = mod["subtype"]

        if max_reach <= 0:
            continue

        # Max Reach in the HDF is the total edge-to-edge width of the
        # modification footprint; the distance from the centerline to the
        # outer edge is therefore half that value.
        half_reach = max_reach / 2.0

        # -- Bounding box of modification footprint in map coordinates ------
        minx = pts[:, 0].min() - half_reach
        maxx = pts[:, 0].max() + half_reach
        miny = pts[:, 1].min() - half_reach
        maxy = pts[:, 1].max() + half_reach

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
        # For a Levee the flat crest is the high point; sides slope *down*
        # away from it (crest − drop).  For a Channel the flat bottom is the
        # low point; sides slope *up* away from it (crest + drop).
        sign = 1.0 if subtype == "Channel" else -1.0

        abs_perp = np.abs(signed_perp)
        half_w = top_width / 2.0

        mod_elev = np.full(len(px), np.nan, dtype=np.float64)

        # Flat bottom/crest: |perp| <= half_width
        on_top = abs_perp <= half_w
        mod_elev[on_top] = crest[on_top]

        # Left side (signed_perp > 0) sloping zone
        left_zone = (signed_perp > half_w) & (abs_perp <= half_reach)
        if np.any(left_zone):
            drop = (abs_perp[left_zone] - half_w) / left_slope
            mod_elev[left_zone] = crest[left_zone] + sign * drop

        # Right side (signed_perp < 0) sloping zone
        right_zone = (signed_perp < -half_w) & (abs_perp <= half_reach)
        if np.any(right_zone):
            drop = (abs_perp[right_zone] - half_w) / right_slope
            mod_elev[right_zone] = crest[right_zone] + sign * drop

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
