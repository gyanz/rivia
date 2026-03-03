"""Raster creation from scattered HEC-RAS mesh-point data.

The single public function :func:`points_to_raster` interpolates values
defined at scattered cell-centre coordinates onto a regular grid and writes
a GeoTIFF or returns an in-memory rasterio dataset.

All ``rasterio`` and ``scipy`` imports are **deferred inside the function
body** so that ``import raspy.geo`` succeeds even when those libraries are
not installed.  A clear ``ImportError`` is raised only if
:func:`points_to_raster` is actually called without them.

Install optional dependencies with::

    pip install raspy[geo]

Grid alignment
--------------
Three mutually exclusive options control the output pixel grid:

1. *reference_raster* (path) — read the affine transform and CRS from an
   existing GeoTIFF; the output pixels snap to the same grid.
2. *transform* (``rasterio.transform.Affine``) — snap to an explicit
   transform supplied by the caller.
3. *cell_size* (float) — build a new grid whose origin is rounded to the
   nearest multiple of *cell_size*, matching RasMapper's convention.

Priority: ``reference_raster > transform > cell_size``.
Supplying both *reference_raster* and *transform* raises ``ValueError``.

Band layout
-----------
+----------------------+---------+--------------------------------------------+
| values shape         | bands   | band names                                 |
+======================+=========+============================================+
| ``(n,)`` — scalar   | 1       | variable name (e.g. ``"Water Surface"``)   |
+----------------------+---------+--------------------------------------------+
| ``(n, 2)`` — vector  | 4       | Vx, Vy, Speed, Direction_deg_from_N        |
+----------------------+---------+--------------------------------------------+

Derived from archive/ras_tools/r2d/ras2d_cell_velocity.py
``RAS2DCellVelocity.export_velocity_raster``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import rasterio.io


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def points_to_raster(
    points: np.ndarray,
    values: np.ndarray,
    output_path: str | Path | None = None,
    *,
    cell_size: float | None = None,
    transform: Any | None = None,
    reference_raster: str | Path | None = None,
    crs: Any | None = None,
    nodata: float = -9999.0,
    interp_method: str = "linear",
    min_value: float | None = None,
) -> Path | rasterio.io.DatasetReader:
    """Interpolate scattered point values to a GeoTIFF raster.

    Parameters
    ----------
    points : ndarray, shape ``(n, 2)``
        Source x, y coordinates (e.g. cell-centre coordinates from
        ``FlowArea.cell_centers``).
    values : ndarray, shape ``(n,)`` or ``(n, 2)``
        Scalar field — shape ``(n,)`` — or 2-component vector field —
        shape ``(n, 2)`` with columns ``[Vx, Vy]``.
    output_path : str, Path, or None
        Destination GeoTIFF file path (e.g. ``"depth.tif"``).  When
        ``None`` (default), the raster is written to an in-memory buffer
        and an open ``rasterio.DatasetReader`` is returned.  The caller
        is responsible for closing the returned dataset.
    cell_size : float, optional
        Output pixel size in the same coordinate units as *points*.
        Ignored when *transform* or *reference_raster* is supplied.
        Callers should pre-compute a sensible default (e.g. median face
        length) before calling this function.
    transform : rasterio.transform.Affine, optional
        Reference affine transform.  The output grid is snapped to this
        pixel grid so results align pixel-for-pixel with existing rasters.
        Only axis-aligned (north-up) transforms are supported.
        Mutually exclusive with *reference_raster*.
    reference_raster : str or Path, optional
        Path to an existing GeoTIFF.  Its transform is read and used for
        grid alignment; its CRS is inherited unless *crs* overrides it.
        Mutually exclusive with *transform*.
    crs : str, int, or rasterio.crs.CRS, optional
        Output coordinate reference system (e.g. ``"EPSG:26910"`` or
        ``26910``).  When *reference_raster* is given and *crs* is
        ``None``, the reference raster's CRS is used.
    nodata : float
        Fill value for pixels outside the convex hull of *points*.
    interp_method : str
        SciPy ``griddata`` method: ``"linear"`` (default),
        ``"nearest"``, or ``"cubic"``.
    min_value : float, optional
        Source points whose value (or speed, for vector fields) is below
        this threshold are excluded from interpolation and set to *nodata*.
        Useful for masking near-dry cells (e.g. ``min_value=0.01``).

    Returns
    -------
    Path
        Absolute path to the written GeoTIFF (when *output_path* is given).
    rasterio.io.DatasetReader
        Open in-memory dataset (when *output_path* is ``None``).  The
        caller must close it when done.

    Raises
    ------
    ImportError
        If ``scipy`` or ``rasterio`` are not installed.
    ValueError
        If both *reference_raster* and *transform* are provided, or if
        fewer than three source points remain after applying *min_value*.
    """
    # ── Deferred imports (raise clearly if geo deps missing) ──────────────
    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import Affine, from_origin
    except ImportError as exc:
        raise ImportError(
            "Raster export requires rasterio. Install it with: pip install raspy[geo]"
        ) from exc

    try:
        from scipy.interpolate import griddata
    except ImportError as exc:
        raise ImportError(
            "Raster export requires scipy. Install it with: pip install raspy[geo]"
        ) from exc

    # ── Validate inputs ───────────────────────────────────────────────────
    if reference_raster is not None and transform is not None:
        raise ValueError("Specify either reference_raster or transform, not both.")

    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    is_vector = values.ndim == 2 and values.shape[1] == 2

    # ── Apply min_value mask ──────────────────────────────────────────────
    if min_value is not None:
        if is_vector:
            magnitude = np.sqrt(values[:, 0] ** 2 + values[:, 1] ** 2)
            mask = magnitude >= min_value
        else:
            mask = values >= min_value
        points = points[mask]
        values = values[mask]

    if points.shape[0] < 3:
        raise ValueError(
            f"Fewer than 3 source points remain after applying "
            f"min_value={min_value}. Cannot triangulate."
        )

    # ── Resolve transform and CRS ─────────────────────────────────────────
    ref_width: int | None = None
    ref_height: int | None = None
    if reference_raster is not None:
        with rasterio.open(reference_raster) as src:
            transform = src.transform
            ref_width = src.width
            ref_height = src.height
            if crs is None:
                crs = src.crs

    if crs is not None and not isinstance(crs, CRS):
        crs = CRS.from_user_input(crs)

    # ── Build output pixel grid ───────────────────────────────────────────
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)

    if transform is not None:
        # Snap output grid to the reference transform's pixel grid.
        # Pixel size from the transform (always positive).
        dx = abs(transform.a)  # column pixel width
        dy = abs(transform.e)  # row pixel height (transform.e < 0 for north-up)

        # Column range covering [x_min, x_max] on the reference grid
        col_min = int(np.floor((x_min - transform.c) / dx))
        col_max = int(np.ceil((x_max - transform.c) / dx))

        # Row range covering [y_min, y_max] (rows increase southward)
        row_min = int(np.floor((transform.f - y_max) / dy))
        row_max = int(np.ceil((transform.f - y_min) / dy))

        # Clamp to the reference raster extent (only when one was given)
        if ref_width is not None:
            col_min = max(col_min, 0)
            col_max = min(col_max, ref_width)
            row_min = max(row_min, 0)
            row_max = min(row_max, ref_height)  # type: ignore[arg-type]

        # Pixel-centre coordinates aligned to the reference grid
        xi = transform.c + (np.arange(col_min, col_max) + 0.5) * dx
        yi = transform.f - (np.arange(row_min, row_max) + 0.5) * dy

        # Top-left corner of the output sub-grid
        out_transform = Affine(
            dx,
            0.0,
            transform.c + col_min * dx,
            0.0,
            -dy,
            transform.f - row_min * dy,
        )
    else:
        # Build a new grid snapped to round multiples of cell_size.
        if cell_size is None or cell_size <= 0:
            raise ValueError(
                "cell_size must be a positive number when transform and "
                "reference_raster are both None."
            )
        west = np.floor(x_min / cell_size) * cell_size
        north = np.ceil(y_max / cell_size) * cell_size

        xi = np.arange(west + cell_size / 2, x_max + cell_size, cell_size)
        yi = np.arange(north - cell_size / 2, y_min - cell_size, -cell_size)

        out_transform = from_origin(west, north, cell_size, cell_size)

    xi_grid, yi_grid = np.meshgrid(xi, yi)
    grid_pts = np.column_stack([xi_grid.ravel(), yi_grid.ravel()])
    n_rows, n_cols = xi_grid.shape

    # ── Interpolate ───────────────────────────────────────────────────────
    if is_vector:
        vx_src = values[:, 0]
        vy_src = values[:, 1]

        vx = griddata(
            points, vx_src, grid_pts, method=interp_method, fill_value=np.nan
        ).reshape(n_rows, n_cols)
        vy = griddata(
            points, vy_src, grid_pts, method=interp_method, fill_value=np.nan
        ).reshape(n_rows, n_cols)

        speed = np.sqrt(vx**2 + vy**2)
        # Direction: degrees clockwise from north, flow-going-to convention
        #   atan2(vy, vx) is CCW from east; transform to CW from north:
        direction = (90.0 - np.degrees(np.arctan2(vy, vx))) % 360.0
        direction = np.where(np.isnan(vx), np.nan, direction)

        band_arrays = [vx, vy, speed, direction]
        band_names = ["Vx", "Vy", "Speed", "Direction_deg_from_N"]
    else:
        scalar = griddata(
            points, values, grid_pts, method=interp_method, fill_value=np.nan
        ).reshape(n_rows, n_cols)
        band_arrays = [scalar]
        band_names = ["value"]

    # ── Write GeoTIFF ─────────────────────────────────────────────────────
    profile: dict = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": n_cols,
        "height": n_rows,
        "count": len(band_arrays),
        "nodata": nodata,
        "transform": out_transform,
        "compress": "lzw",
    }
    if crs is not None:
        profile["crs"] = crs

    def _write_bands(dst: Any) -> None:
        for band_idx, (arr, bname) in enumerate(zip(band_arrays, band_names), start=1):
            data = arr.astype(np.float32)
            data[np.isnan(data)] = nodata
            dst.write(data, band_idx)
            dst.update_tags(band_idx, name=bname)

    if output_path is None:
        memfile = rasterio.MemoryFile()
        with memfile.open(**profile) as dst:
            _write_bands(dst)
        return memfile.open()

    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        _write_bands(dst)
    return out_path
