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
2. *reference_transform* (``rasterio.transform.Affine``) — snap to an
   explicit transform supplied by the caller.
3. *cell_size* (float) — build a new grid whose origin is rounded to the
   nearest multiple of *cell_size*, matching RasMapper's convention.

Priority: ``reference_raster > reference_transform > cell_size``.
Supplying both *reference_raster* and *reference_transform* raises ``ValueError``.

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

import logging
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
    reference_transform: Any | None = None,
    reference_raster: str | Path | None = None,
    crs: Any | None = None,
    nodata: float = -9999.0,
    interp_method: str = "linear",
    min_value: float | None = None,
    snap_to_reference_extent: bool = True,
    adjacency: np.ndarray | None = None,
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
        Ignored when *reference_transform* or *reference_raster* is supplied.
        Callers should pre-compute a sensible default (e.g. median face
        length) before calling this function.
    reference_transform : rasterio.transform.Affine, optional
        Reference affine transform.  The output grid is snapped to this
        pixel grid so results align pixel-for-pixel with existing rasters.
        Only axis-aligned (north-up) transforms are supported.
        Mutually exclusive with *reference_raster*.
    reference_raster : str or Path, optional
        Path to an existing GeoTIFF.  Its transform is read and used for
        grid alignment; its CRS is inherited unless *crs* overrides it.
        Mutually exclusive with *reference_transform*.
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
        Two-stage threshold applied to the value field (or speed for vector
        fields).  Source points below the threshold are excluded from
        interpolation (*pre-filter*); output pixels whose interpolated value
        is < the threshold are then set to *nodata* (*post-mask*).  When
        ``None`` (default), no masking is applied.  Use ``min_value=0.01``
        to mask near-dry cells.
    snap_to_reference_extent : bool
        Only relevant when *reference_raster* is provided.  When ``True``
        (default), the output covers the **full extent** of the reference
        raster (col=0…width, row=0…height); pixels outside the convex
        hull of *points* are filled with *nodata*.  Set to ``False`` to
        produce a smaller raster cropped to the point cloud extent.
        When *reference_transform* is given instead, the output is always
        cropped to the point cloud extent (pixel-aligned to the
        transform); this flag has no effect.
    adjacency : ndarray, shape ``(n_edges, 2)``, optional
        Mesh face-to-cell connectivity array (e.g. ``FlowArea.face_cell_indexes``).
        Each row is a pair of cell indices sharing a face; ``-1`` indicates
        a boundary face.  When provided, only Delaunay triangles whose
        three edges all appear in *adjacency* are interpolated, preventing
        spurious fill across gaps between disconnected wet areas.  The
        interpolation method is always ``"linear"`` (``interp_method`` is
        ignored).  Indices refer to rows of the *original* (pre-filter)
        *points* array.

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
        If both *reference_raster* and *reference_transform* are provided,
        or if fewer than three source points remain after applying
        *min_value*.
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
        from scipy.interpolate import LinearNDInterpolator, griddata
        from scipy.spatial import Delaunay
    except ImportError as exc:
        raise ImportError(
            "Raster export requires scipy. Install it with: pip install raspy[geo]"
        ) from exc

    # ── Validate inputs ───────────────────────────────────────────────────
    if reference_raster is not None and reference_transform is not None:
        raise ValueError(
            "Specify either reference_raster or reference_transform, not both."
        )

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
        orig_idx = np.where(mask)[0]
        points = points[mask]
        values = values[mask]
    else:
        orig_idx = np.arange(len(points))

    if points.shape[0] < 3:
        raise ValueError(
            f"Fewer than 3 source points remain after applying "
            f"min_value={min_value}. Cannot triangulate."
        )

    # ── Resolve transform and CRS ─────────────────────────────────────────
    ref_width: int | None = None
    ref_height: int | None = None
    transform = reference_transform
    if reference_raster is not None:
        with rasterio.open(reference_raster) as src:
            transform = src.transform
            ref_width = src.width
            ref_height = src.height
            logging.debug(
                "Read transform from reference raster %s: %s",
                reference_raster,
                transform,
            )
            if crs is None:
                crs = src.crs

    if crs is not None and not isinstance(crs, CRS):
        crs = CRS.from_user_input(crs)

    # ── Build output pixel grid ───────────────────────────────────────────
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    logging.debug("Point cloud extent: x=[%f, %f], y=[%f, %f]", x_min, x_max, y_min, y_max)

    if transform is not None:
        # Snap output grid to the reference transform's pixel grid.
        # Pixel size from the transform (always positive).
        dx = abs(transform.a)  # column pixel width
        dy = abs(transform.e)  # row pixel height (transform.e < 0 for north-up)

        if snap_to_reference_extent and ref_width is not None:
            # Use the full reference raster extent so that rasters produced
            # from different point clouds (e.g. depth vs velocity) are
            # identical in size and can be multiplied pixel-for-pixel.
            col_min, col_max = 0, ref_width
            row_min, row_max = 0, ref_height  # type: ignore[assignment]
        else:
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
    # Build the per-pixel interpolator.  When mesh adjacency is supplied we
    # use a topology-aware path (Delaunay + LinearNDInterpolator) that masks
    # out grid pixels falling inside Delaunay triangles that span gaps between
    # disconnected wet areas.  Otherwise we fall back to scipy.griddata.
    if adjacency is not None:
        adj = np.asarray(adjacency, dtype=np.int64)
        valid_pairs = adj[(adj[:, 0] >= 0) & (adj[:, 1] >= 0)]

        if len(valid_pairs) > 0:
            # Encode each undirected edge as one int64 for fast np.isin lookup.
            max_cell = int(valid_pairs.max()) + 1
            sorted_pairs = np.sort(valid_pairs, axis=1)
            edge_keys = sorted_pairs[:, 0] * max_cell + sorted_pairs[:, 1]

            tri = Delaunay(points)
            # Map Delaunay local vertex indices → original (pre-filter) cell indices.
            simplices_orig = orig_idx[tri.simplices]  # (n_tri, 3)

            def _edge_valid(col_a: int, col_b: int) -> np.ndarray:
                s = np.sort(simplices_orig[:, [col_a, col_b]], axis=1)
                return np.isin(s[:, 0] * max_cell + s[:, 1], edge_keys)

            tri_valid = _edge_valid(0, 1) & _edge_valid(1, 2) & _edge_valid(0, 2)

            simplex_idx = tri.find_simplex(grid_pts)
            gap_mask = np.ones(len(grid_pts), dtype=bool)
            inside = simplex_idx >= 0
            gap_mask[inside] = ~tri_valid[simplex_idx[inside]]

            def _interp(vals_1d: np.ndarray) -> np.ndarray:
                fn = LinearNDInterpolator(tri, vals_1d, fill_value=np.nan)
                out = fn(grid_pts)
                out[gap_mask] = np.nan
                return out.reshape(n_rows, n_cols)

        else:
            # No valid adjacency pairs — fall back to griddata.
            def _interp(vals_1d: np.ndarray) -> np.ndarray:
                return griddata(
                    points, vals_1d, grid_pts,
                    method=interp_method, fill_value=np.nan,
                ).reshape(n_rows, n_cols)
    else:
        def _interp(vals_1d: np.ndarray) -> np.ndarray:
            return griddata(
                points, vals_1d, grid_pts,
                method=interp_method, fill_value=np.nan,
            ).reshape(n_rows, n_cols)

    if is_vector:
        vx = _interp(values[:, 0])
        vy = _interp(values[:, 1])

        speed = np.sqrt(vx**2 + vy**2)
        # Direction: degrees clockwise from north, flow-going-to convention
        #   atan2(vy, vx) is CCW from east; transform to CW from north:
        direction = (90.0 - np.degrees(np.arctan2(vy, vx))) % 360.0
        direction = np.where(np.isnan(vx), np.nan, direction)

        # Post-interpolation mask: speeds below min_value → nodata.
        if min_value is not None:
            _low = speed < min_value
            vx[_low] = np.nan
            vy[_low] = np.nan
            speed[_low] = np.nan
            direction[_low] = np.nan

        band_arrays = [vx, vy, speed, direction]
        band_names = ["Vx", "Vy", "Speed", "Direction_deg_from_N"]
    else:
        scalar = _interp(values)

        # Post-interpolation mask: values below min_value → nodata.
        if min_value is not None:
            scalar[scalar < min_value] = np.nan

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


def _compute_facepoint_values(
    cell_vals: np.ndarray,
    cell_centers: np.ndarray,
    facepoint_coordinates: np.ndarray,
    face_facepoint_indexes: np.ndarray,
    face_cell_indexes: np.ndarray,
    n_cells: int,
    n_facepoints: int,
) -> np.ndarray:
    """Assign distance-weighted interpolated values at mesh facepoints.

    For each interior face the value is computed by linearly interpolating
    between the two adjacent cell-centre values, weighted by the distance
    from each cell centre to the face centre (midpoint of the face's two
    bounding facepoints).  This correctly accounts for irregular cell
    geometry: when the face lies closer to one cell centre its value is
    pulled toward that cell, rather than being a simple average.

    Boundary faces (one adjacent cell only) and faces adjacent to a dry
    cell (``NaN`` value) use the single available wet-cell value directly.

    Each facepoint value is then the simple mean of the values from all
    adjacent valid (wet) faces.

    Parameters
    ----------
    cell_vals : ndarray, shape ``(n_cells,)``
        Cell-centre scalar values; ``NaN`` marks dry / excluded cells.
    cell_centers : ndarray, shape ``(n_cells, 2)``
        Cell-centre x, y coordinates.
    facepoint_coordinates : ndarray, shape ``(n_facepoints, 2)``
        Facepoint x, y coordinates.
    face_facepoint_indexes : ndarray, shape ``(n_faces, 2)``
        Start and end facepoint index for each face.
    face_cell_indexes : ndarray, shape ``(n_faces, 2)``
        Left and right cell indices; ``-1`` for boundary faces.
    n_cells, n_facepoints : int

    Returns
    -------
    ndarray, shape ``(n_facepoints,)``
        ``NaN`` where every adjacent face is dry or out-of-domain.
    """
    n_faces = len(face_facepoint_indexes)
    fp0 = face_facepoint_indexes[:, 0]
    fp1 = face_facepoint_indexes[:, 1]
    left_cells = face_cell_indexes[:, 0]
    right_cells = face_cell_indexes[:, 1]

    # Cell values at each face's adjacent cells (NaN for boundary / dry).
    left_vals = np.full(n_faces, np.nan)
    right_vals = np.full(n_faces, np.nan)
    valid_left = (left_cells >= 0) & (left_cells < n_cells)
    valid_right = (right_cells >= 0) & (right_cells < n_cells)
    left_vals[valid_left] = cell_vals[left_cells[valid_left]]
    right_vals[valid_right] = cell_vals[right_cells[valid_right]]

    # Face centre = midpoint of its two bounding facepoints.
    face_cx = (facepoint_coordinates[fp0, 0] + facepoint_coordinates[fp1, 0]) * 0.5
    face_cy = (facepoint_coordinates[fp0, 1] + facepoint_coordinates[fp1, 1]) * 0.5

    face_vals = np.full(n_faces, np.nan)

    # Interior faces where both adjacent cells are wet: distance-weighted.
    both_wet = ~np.isnan(left_vals) & ~np.isnan(right_vals)
    if both_wet.any():
        lc = left_cells[both_wet]
        rc = right_cells[both_wet]
        dl = np.hypot(
            face_cx[both_wet] - cell_centers[lc, 0],
            face_cy[both_wet] - cell_centers[lc, 1],
        )
        dr = np.hypot(
            face_cx[both_wet] - cell_centers[rc, 0],
            face_cy[both_wet] - cell_centers[rc, 1],
        )
        d_sum = dl + dr
        # t = 0 → value at left cell centre; t = 1 → value at right cell centre.
        t = np.where(d_sum > 0.0, dl / d_sum, 0.5)
        face_vals[both_wet] = (1.0 - t) * left_vals[both_wet] + t * right_vals[both_wet]

    # Boundary or dry-neighbour: use whichever cell value is available.
    left_only = ~np.isnan(left_vals) & np.isnan(right_vals)
    face_vals[left_only] = left_vals[left_only]
    right_only = np.isnan(left_vals) & ~np.isnan(right_vals)
    face_vals[right_only] = right_vals[right_only]

    # Accumulate face values at each bounding facepoint (simple mean).
    fp_sum = np.zeros(n_facepoints)
    fp_count = np.zeros(n_facepoints, dtype=np.int64)
    valid_faces = ~np.isnan(face_vals)
    np.add.at(fp_sum, fp0[valid_faces], face_vals[valid_faces])
    np.add.at(fp_count, fp0[valid_faces], 1)
    np.add.at(fp_sum, fp1[valid_faces], face_vals[valid_faces])
    np.add.at(fp_count, fp1[valid_faces], 1)

    with np.errstate(all="ignore"):
        return np.where(fp_count > 0, fp_sum / fp_count, np.nan)


# ---------------------------------------------------------------------------
# Rendering helpers — called from mesh_to_raster
# ---------------------------------------------------------------------------


def _horizontal_from_trifind(
    tri_idx_flat: np.ndarray,
    tri_to_cell: np.ndarray,
    cell_cv: np.ndarray,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """Map a flat trifinder result to per-cell constant values.

    Pixels whose triangle index is -1 (outside the mesh) are set to NaN.

    Parameters
    ----------
    tri_idx_flat : ndarray, shape ``(n_pixels,)``
        Triangle index per pixel from ``Triangulation.get_trifinder()(x, y)``.
        -1 indicates the pixel is outside all unmasked triangles.
    tri_to_cell : ndarray, shape ``(n_triangles,)``
        Maps each triangle index to its originating cell index.
    cell_cv : ndarray, shape ``(n_cells,)``
        Cell-centre scalar values; NaN for dry cells.
    grid_shape : (n_rows, n_cols)

    Returns
    -------
    ndarray, shape ``(n_rows, n_cols)``
    """
    n_out = len(tri_idx_flat)
    out: np.ndarray = np.full(n_out, np.nan)
    inside = tri_idx_flat >= 0
    out[inside] = cell_cv[tri_to_cell[tri_idx_flat[inside]]]
    return out.reshape(grid_shape)


def _render_horizontal(
    triang: Any,
    tri_to_cell: np.ndarray,
    cell_cv: np.ndarray,
    xi_grid: np.ndarray,
    yi_grid: np.ndarray,
) -> np.ndarray:
    """Assign each output pixel the constant value of the cell it falls inside.

    Wraps ``_horizontal_from_trifind`` for the common case where no
    ``tri_idx`` array is available yet.

    Returns
    -------
    ndarray, shape ``(n_rows, n_cols)`` or ``(n_rows, n_cols, 2)``
    """
    trifinder_fn = triang.get_trifinder()
    tri_idx_flat = trifinder_fn(xi_grid.ravel(), yi_grid.ravel())
    return _horizontal_from_trifind(tri_idx_flat, tri_to_cell, cell_cv, xi_grid.shape)


def _classify_horizontal_cells(
    face_cell_indexes: np.ndarray,
    dry_mask: np.ndarray,
    face_active: np.ndarray,
) -> np.ndarray:
    """Return a bool mask: True where a cell should use horizontal rendering.

    A wet cell is flagged when it has at least one wet neighbour that is
    separated from it by a *dry* (inactive) face.  Both cells in such a
    pair are flagged.  This matches HEC-RAS RASMapper's hybrid mode: sloping
    is used where the water surface is hydraulically connected; horizontal is
    used across dry-face boundaries between otherwise-wet cells.

    Parameters
    ----------
    face_cell_indexes : ndarray, shape ``(n_faces, 2)``
        Left/right cell index per face; -1 = boundary.
    dry_mask : ndarray, shape ``(n_cells,)``, bool
        True where a cell is dry (excluded from interpolation).
    face_active : ndarray, shape ``(n_faces,)``, bool
        True where the face is hydraulically active (wet).

    Returns
    -------
    ndarray, shape ``(n_cells,)``, bool
    """
    n_cells = len(dry_mask)
    use_horizontal = np.zeros(n_cells, dtype=bool)

    left = face_cell_indexes[:, 0]
    right = face_cell_indexes[:, 1]
    interior = (left >= 0) & (left < n_cells) & (right >= 0) & (right < n_cells)
    lc, rc = left[interior], right[interior]
    active = face_active[interior]

    both_wet = ~dry_mask[lc] & ~dry_mask[rc]
    disconnected = both_wet & ~active
    np.logical_or.at(use_horizontal, lc[disconnected], True)
    np.logical_or.at(use_horizontal, rc[disconnected], True)
    use_horizontal[dry_mask] = False
    return use_horizontal


def mesh_to_raster(
    cell_centers: np.ndarray,
    facepoint_coordinates: np.ndarray,
    face_facepoint_indexes: np.ndarray,
    face_cell_indexes: np.ndarray,
    cell_face_info: np.ndarray,
    cell_face_values: np.ndarray,
    cell_values: np.ndarray,
    output_path: str | Path | None = None,
    *,
    cell_size: float | None = None,
    reference_transform: Any | None = None,
    reference_raster: str | Path | None = None,
    crs: Any | None = None,
    nodata: float = -9999.0,
    min_value: float | None = None,
    min_above_ref: float | None = None,
    snap_to_reference_extent: bool = True,
    render_mode: str = "sloping",
    face_active: np.ndarray | None = None,
) -> Path | rasterio.io.DatasetReader:
    """Interpolate HEC-RAS mesh results to a raster using mesh-conforming triangulation.

    This function builds a triangulation that exactly conforms to the
    cell polygons — mirroring how HEC-RAS RASMapper creates its result maps.

    **Algorithm**

    1. Dry cells (value < *min_value*) are masked to ``NaN``.
    2. A value is assigned to every facepoint by averaging the face-midpoint
       values of all adjacent faces (face midpoint = NaN-aware mean of the two
       bordering cell values).
    3. Each cell is sub-divided into one triangle per bounding face:
       ``[cell_centre, facepoint_start, facepoint_end]``.
    4. Triangles whose cell-centre vertex is masked (dry cell) are excluded.
    5. ``matplotlib.tri.LinearTriInterpolator`` performs linear (barycentric)
       interpolation on this topology-conforming triangulation.
    6. Pixels outside the wet mesh domain (outside all unmasked triangles)
       are set to *nodata*.

    Parameters
    ----------
    cell_centers : ndarray, shape ``(n_cells, 2)``
        Cell-centre x, y coordinates (``FlowArea.cell_centers``).
    facepoint_coordinates : ndarray, shape ``(n_facepoints, 2)``
        Polygon-vertex x, y coordinates (``FlowArea.facepoint_coordinates``).
    face_facepoint_indexes : ndarray, shape ``(n_faces, 2)``
        Start/end facepoint index for each face
        (``FlowArea.face_facepoint_indexes``).
    face_cell_indexes : ndarray, shape ``(n_faces, 2)``
        Left/right cell indices per face; ``-1`` = boundary
        (``FlowArea.face_cell_indexes``).
    cell_face_info : ndarray, shape ``(n_cells, 2)``
        ``[start_idx, count]`` into *cell_face_values* for each cell
        (``FlowArea.cell_face_info``).
    cell_face_values : ndarray, shape ``(total_entries, 2)``
        ``[face_idx, orientation]`` for each cell-face association
        (``FlowArea.cell_face_values``).
    cell_values : ndarray, shape ``(n_cells,)``
        Scalar field (e.g. WSE) at cell centres.
    output_path : str, Path, or None
        Destination GeoTIFF.  ``None`` returns an open in-memory
        ``rasterio.DatasetReader``; the caller must close it.
    cell_size : float, optional
        Output pixel size in the same coordinate units as the point data.
        Ignored when *reference_transform* or *reference_raster* is given.
    reference_transform : rasterio.transform.Affine, optional
        Affine transform for pixel-perfect grid alignment with other rasters.
        Mutually exclusive with *reference_raster*.
    reference_raster : str or Path, optional
        Existing GeoTIFF whose transform and CRS are inherited.
        Mutually exclusive with *reference_transform*.
    crs : str, int, or rasterio.crs.CRS, optional
        Output CRS (overrides *reference_raster* CRS when given).
    nodata : float
        Fill value written to pixels outside the wet mesh polygon.
    min_value : float, optional
        Scalar threshold: cells whose value is below this are treated as dry
        and excluded before interpolation.
    min_above_ref : float, optional
        Minimum amount by which the interpolated scalar must exceed the
        reference raster value.  Only used when *reference_raster* is given.
        Pixels where ``scalar - reference_value < min_above_ref`` are masked
        to ``NaN``.  When ``None`` (default), pixels where the scalar is at
        or below the reference value are masked (equivalent to
        ``min_above_ref=0``).
    snap_to_reference_extent : bool
        When *reference_raster* is given, extend the output to its full
        extent (default ``True``).
    render_mode : str
        Water-surface rendering mode, matching HEC-RAS RASMapper options:

        ``"sloping"`` *(default)* — interpolates WSE between cell centres
        using distance-weighted face values and linear triangular
        interpolation, producing a smooth, continuous inundation surface.

        ``"horizontal"`` — assigns each output pixel the flat cell-centre
        WSE of the cell it falls inside.  Produces a stepped "patchwork"
        appearance at cell boundaries; useful for checking raw model output.

        ``"hybrid"`` — uses sloping rendering where adjacent wet cells are
        hydraulically connected (active face between them) and horizontal
        rendering where they are separated by a dry face.  Requires
        *face_active*.
    face_active : ndarray, shape ``(n_faces,)``, bool, optional
        Per-face hydraulic activity flag: ``True`` = face is wet/active.
        Required when *render_mode* is ``"hybrid"``; ignored otherwise.
        Typically derived from face velocity: ``abs(face_vel) > threshold``.

    Returns
    -------
    Path
        Written GeoTIFF path (when *output_path* is given).
    rasterio.io.DatasetReader
        Open in-memory dataset (when *output_path* is ``None``).

    Raises
    ------
    ImportError
        If ``rasterio`` or ``matplotlib`` are not installed.
    ValueError
        If both *reference_raster* and *reference_transform* are supplied,
        or if *cell_size* is missing when no transform is provided.

    Notes
    -----
    Always writes a single band named ``"value"``.  For velocity rasters use
    :func:`mesh_to_velocity_raster` instead.
    """
    # ── Deferred imports ───────────────────────────────────────────────────
    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import Affine, from_origin
        from rasterio.windows import Window
    except ImportError as exc:
        raise ImportError(
            "mesh_to_raster requires rasterio. "
            "Install it with: pip install raspy[geo]"
        ) from exc

    try:
        import matplotlib.tri as mtri
    except ImportError as exc:
        raise ImportError(
            "mesh_to_raster requires matplotlib. "
            "Install it with: pip install raspy[geo]"
        ) from exc

    # ── Validate ───────────────────────────────────────────────────────────
    if reference_raster is not None and reference_transform is not None:
        raise ValueError(
            "Specify either reference_raster or reference_transform, not both."
        )
    if render_mode not in ("sloping", "horizontal", "hybrid"):
        raise ValueError(
            f"render_mode must be 'sloping', 'horizontal', or 'hybrid';"
            f" got {render_mode!r}."
        )
    cell_values = np.asarray(cell_values, dtype=np.float64)
    if cell_values.ndim != 1:
        raise ValueError(
            "mesh_to_raster only accepts scalar cell_values (shape (n_cells,)). "
            "For velocity rasters use mesh_to_velocity_raster instead."
        )
    if render_mode == "hybrid" and face_active is None:
        raise ValueError("face_active is required when render_mode='hybrid'.")

    cell_centers = np.asarray(cell_centers, dtype=np.float64)
    facepoint_coordinates = np.asarray(facepoint_coordinates, dtype=np.float64)
    face_facepoint_indexes = np.asarray(face_facepoint_indexes, dtype=np.int64)
    face_cell_indexes = np.asarray(face_cell_indexes, dtype=np.int64)
    cell_face_values_arr = np.asarray(cell_face_values, dtype=np.int64)
    # cell_values already converted and validated above the validate block.

    n_cells = len(cell_centers)
    # Slice to real cells only — HDF may append ghost/padding rows beyond n_cells.
    cell_face_info = np.asarray(cell_face_info, dtype=np.int64)[:n_cells]
    n_facepoints = len(facepoint_coordinates)

    # ── Dry-cell masking ───────────────────────────────────────────────────
    dry_mask = (
        cell_values < min_value
        if min_value is not None
        else np.zeros(n_cells, dtype=bool)
    )
    cv = cell_values.copy()
    cv[dry_mask] = np.nan

    # ── Facepoint values (sloping / hybrid only) ───────────────────────────
    # Horizontal mode assigns each pixel its cell-centre value directly, so
    # facepoint interpolation is unnecessary.
    if render_mode in ("sloping", "hybrid"):
        _fpkw = dict(
            cell_centers=cell_centers,
            facepoint_coordinates=facepoint_coordinates,
            face_facepoint_indexes=face_facepoint_indexes,
            face_cell_indexes=face_cell_indexes,
            n_cells=n_cells,
            n_facepoints=n_facepoints,
        )
        fp_vals = _compute_facepoint_values(cv, **_fpkw)
        all_vals = np.concatenate([cv, fp_vals])
        vertex_nan = np.isnan(all_vals)

    # ── Build unified point set ────────────────────────────────────────────
    # Cell centres occupy indices [0, n_cells);
    # facepoints occupy [n_cells, n_cells + n_facepoints).
    all_pts = np.vstack([cell_centers, facepoint_coordinates])

    # ── Build mesh-conforming triangles (vectorised, no Python loop) ───────
    # For every cell-face association produce triangle
    # [cell_idx, n_cells + fp0, n_cells + fp1].
    counts = cell_face_info[:, 1]
    starts = cell_face_info[:, 0]
    total = int(counts.sum())

    # Generate the flat index array into cell_face_values using the cumsum trick
    # so that non-contiguous start offsets (if any) are handled correctly.
    idx_step = np.ones(total, dtype=np.int64)
    if n_cells > 1:
        boundary_pos = np.cumsum(counts[:-1])
        idx_step[boundary_pos] = starts[1:] - starts[:-1] - counts[:-1] + 1
    idx_step[0] = starts[0]
    entry_indices = np.cumsum(idx_step)

    face_idx_arr = cell_face_values_arr[entry_indices, 0]
    cell_for_entry = np.repeat(np.arange(n_cells, dtype=np.int64), counts)

    fp0_per_entry = face_facepoint_indexes[face_idx_arr, 0]
    fp1_per_entry = face_facepoint_indexes[face_idx_arr, 1]

    # Discard degenerate faces (identical facepoints on both ends).
    valid = fp0_per_entry != fp1_per_entry
    triangles = np.column_stack([
        cell_for_entry[valid],
        n_cells + fp0_per_entry[valid],
        n_cells + fp1_per_entry[valid],
    ])

    # Cell index for each triangle — needed by horizontal and hybrid renders.
    tri_to_cell = cell_for_entry[valid]

    # Mask triangles based on render mode.
    # Horizontal: only the cell-centre vertex matters (facepoint NaN irrelevant).
    # Sloping / hybrid: any NaN vertex (cell or facepoint) masks the triangle.
    if render_mode == "horizontal":
        tri_mask = dry_mask[tri_to_cell]
    else:
        tri_mask = np.any(vertex_nan[triangles], axis=1)

    # ── Resolve transform and CRS ──────────────────────────────────────────
    ref_width: int | None = None
    ref_height: int | None = None
    transform = reference_transform
    if reference_raster is not None:
        with rasterio.open(reference_raster) as src:
            transform = src.transform
            ref_width = src.width
            ref_height = src.height
            if crs is None:
                crs = src.crs

    if crs is not None and not isinstance(crs, CRS):
        crs = CRS.from_user_input(crs)

    # ── Build output pixel grid ────────────────────────────────────────────
    # Use the full point set (cell centres + facepoints) for extent so the
    # grid covers the mesh boundary rather than just the cell-centre cloud.
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)

    if transform is not None:
        dx = abs(transform.a)
        dy = abs(transform.e)

        if snap_to_reference_extent and ref_width is not None:
            col_min, col_max = 0, ref_width
            row_min, row_max = 0, ref_height  # type: ignore[assignment]
        else:
            col_min = int(np.floor((x_min - transform.c) / dx))
            col_max = int(np.ceil((x_max - transform.c) / dx))
            row_min = int(np.floor((transform.f - y_max) / dy))
            row_max = int(np.ceil((transform.f - y_min) / dy))

            if ref_width is not None:
                col_min = max(col_min, 0)
                col_max = min(col_max, ref_width)
                row_min = max(row_min, 0)
                row_max = min(row_max, ref_height)  # type: ignore[arg-type]

        xi = transform.c + (np.arange(col_min, col_max) + 0.5) * dx
        yi = transform.f - (np.arange(row_min, row_max) + 0.5) * dy
        out_transform = Affine(
            dx, 0.0, transform.c + col_min * dx,
            0.0, -dy, transform.f - row_min * dy,
        )
    else:
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

    n_rows, n_cols = len(yi), len(xi)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    # ── Build triangulation ────────────────────────────────────────────────
    triang = mtri.Triangulation(all_pts[:, 0], all_pts[:, 1], triangles)
    triang.set_mask(tri_mask)

    def _to_array(masked: Any) -> np.ndarray:
        """Convert a matplotlib masked array to a plain ndarray with NaN fill."""
        if hasattr(masked, "filled"):
            return masked.filled(np.nan)
        return np.asarray(masked, dtype=np.float64)

    # ── Render ────────────────────────────────────────────────────────────
    if render_mode == "horizontal":
        scalar = _render_horizontal(triang, tri_to_cell, cv, xi_grid, yi_grid)

    elif render_mode == "sloping":
        scalar = _to_array(
            mtri.LinearTriInterpolator(triang, all_vals)(xi_grid, yi_grid)
        )

    else:  # "hybrid"
        # Compute trifinder once; reuse for the horizontal render and for
        # mapping each pixel to its cell to determine the rendering mode.
        trifinder_fn = triang.get_trifinder()
        tri_idx_flat = trifinder_fn(xi_grid.ravel(), yi_grid.ravel())

        is_horiz_cell = _classify_horizontal_cells(
            face_cell_indexes, dry_mask, np.asarray(face_active, dtype=bool)
        )
        pixel_is_horiz = np.zeros(len(tri_idx_flat), dtype=bool)
        inside = tri_idx_flat >= 0
        pixel_is_horiz[inside] = is_horiz_cell[tri_to_cell[tri_idx_flat[inside]]]
        pixel_is_horiz = pixel_is_horiz.reshape(n_rows, n_cols)

        scalar_slop = _to_array(
            mtri.LinearTriInterpolator(triang, all_vals)(xi_grid, yi_grid)
        )
        scalar_horiz = _horizontal_from_trifind(
            tri_idx_flat, tri_to_cell, cv, (n_rows, n_cols)
        )
        scalar = np.where(pixel_is_horiz, scalar_horiz, scalar_slop)

    # ── Post-interpolation masking and band assembly ───────────────────────
    if min_value is not None:
        scalar[scalar < min_value] = np.nan
    if reference_raster is not None:
        with rasterio.open(reference_raster) as dem_src:
            dem_nodata = dem_src.nodata
            dem_data = dem_src.read(
                1, window=Window(col_min, row_min, n_cols, n_rows)
            ).astype(np.float64)
        threshold = min_above_ref if min_above_ref is not None else 0.0
        dry_pixel = (scalar - dem_data) < threshold
        if dem_nodata is not None:
            dry_pixel |= dem_data == dem_nodata
        scalar[dry_pixel] = np.nan
    band_arrays = [scalar]
    band_names = ["value"]

    # ── Write GeoTIFF ──────────────────────────────────────────────────────
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


def mesh_to_velocity_raster(
    cell_centers: np.ndarray,
    facepoint_coordinates: np.ndarray,
    face_facepoint_indexes: np.ndarray,
    face_cell_indexes: np.ndarray,
    cell_face_info: np.ndarray,
    cell_face_values: np.ndarray,
    cell_wse: np.ndarray,
    cell_velocity: np.ndarray,
    output_path: str | Path | None = None,
    *,
    cell_size: float | None = None,
    reference_transform: Any | None = None,
    reference_raster: str | Path | None = None,
    crs: Any | None = None,
    nodata: float = -9999.0,
    vel_min: float | None = None,
    depth_min: float | None = None,
    snap_to_reference_extent: bool = True,
    render_mode: str = "sloping",
    face_active: np.ndarray | None = None,
) -> Path | rasterio.io.DatasetReader:
    """Render a HEC-RAS velocity raster with WSE-based wet extent.

    HEC-RAS determines the wetted raster extent from the rendered
    (interpolated) water-surface elevation, then assigns velocity within
    those wet pixels.  This function replicates that two-step process:

    1. **Render WSE** in-memory using :func:`mesh_to_raster` with the
       requested *render_mode* — this determines which raster pixels are
       wet and, for sloping/hybrid modes, the pixel-level WSE.
    2. **Assign velocity horizontally** — each wet pixel is mapped to
       its parent mesh cell via a trifinder, then receives that cell's
       pre-computed WLS velocity vector ``[Vx, Vy]``.  Velocity is *not*
       spatially interpolated across cell boundaries, which would be
       physically incorrect because velocity can be discontinuous at faces.

    Parameters
    ----------
    cell_centers : ndarray, shape ``(n_cells, 2)``
        Cell-centre x, y coordinates (``FlowArea.cell_centers``).
    facepoint_coordinates : ndarray, shape ``(n_facepoints, 2)``
        Polygon-vertex x, y coordinates.
    face_facepoint_indexes : ndarray, shape ``(n_faces, 2)``
        Start/end facepoint index for each face.
    face_cell_indexes : ndarray, shape ``(n_faces, 2)``
        Left/right cell indices; ``-1`` = boundary.
    cell_face_info : ndarray, shape ``(n_cells, 2)``
        ``[start_idx, count]`` into *cell_face_values* for each cell.
    cell_face_values : ndarray, shape ``(total_entries, 2)``
        ``[face_idx, orientation]`` per cell-face association.
    cell_wse : ndarray, shape ``(n_cells,)``
        Cell-centre water-surface elevations.  ``NaN`` marks dry cells;
        these are excluded from both the WSE render and velocity output.
    cell_velocity : ndarray, shape ``(n_cells, 2)``
        Pre-computed WLS velocity vectors ``[Vx, Vy]`` at each cell centre.
    output_path : str, Path, or None
        Destination GeoTIFF.  ``None`` returns an open in-memory
        ``rasterio.DatasetReader``; the caller must close it.
    cell_size : float, optional
        Output pixel size.  Ignored when *reference_transform* or
        *reference_raster* is supplied.
    reference_transform : rasterio.transform.Affine, optional
        Affine transform for pixel-perfect grid alignment.
        Mutually exclusive with *reference_raster*.
    reference_raster : str or Path, optional
        Existing GeoTIFF whose transform and CRS are inherited.
        Mutually exclusive with *reference_transform*.
    crs : str, int, or rasterio.crs.CRS, optional
        Output CRS (overrides *reference_raster* CRS when given).
    nodata : float
        Fill value for pixels outside the wet mesh.
    vel_min : float, optional
        Speed threshold (m/s).  Cells whose WLS speed is below this are
        excluded from the WSE rendering (treated as dry) and from the
        final velocity output.
    depth_min : float, optional
        Minimum depth threshold (m) passed to the internal WSE render.
        Pixels where WSE minus DEM is less than this value are treated as
        dry and excluded from the velocity output.  Only relevant when
        *reference_raster* is given.
    snap_to_reference_extent : bool
        When *reference_raster* is given, extend the output to its full
        extent (default ``True``).
    render_mode : str
        Water-surface rendering mode — ``"sloping"`` (default),
        ``"horizontal"``, or ``"hybrid"``.  Controls which pixels are
        considered wet and therefore receive a velocity value.
    face_active : ndarray, shape ``(n_faces,)``, bool, optional
        Per-face hydraulic activity flag required by ``render_mode="hybrid"``.

    Returns
    -------
    Path
        Written GeoTIFF path (when *output_path* is given).
    rasterio.io.DatasetReader
        Open in-memory 4-band dataset ``[Vx, Vy, Speed, Direction_deg_from_N]``
        (when *output_path* is ``None``).  Caller must close it.

    Raises
    ------
    ImportError
        If ``rasterio`` or ``matplotlib`` are not installed.
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "mesh_to_velocity_raster requires rasterio. "
            "Install it with: pip install raspy[geo]"
        ) from exc
    try:
        import matplotlib.tri as mtri
    except ImportError as exc:
        raise ImportError(
            "mesh_to_velocity_raster requires matplotlib. "
            "Install it with: pip install raspy[geo]"
        ) from exc

    cell_velocity = np.asarray(cell_velocity, dtype=np.float64)  # (n_cells, 2)
    cell_wse_arr = np.asarray(cell_wse, dtype=np.float64)

    # Pre-filter: mask cells whose speed is below vel_min so that the WSE
    # rendering treats them as dry and excludes them from the wet extent.
    cell_wse_for_render = cell_wse_arr.copy()
    if vel_min is not None:
        with np.errstate(invalid="ignore"):
            speed_pre = np.linalg.norm(cell_velocity, axis=1)
        cell_wse_for_render[speed_pre < vel_min] = np.nan

    # ── Step 1: Render WSE in-memory to determine wet extent and grid. ──────
    wse_ds = mesh_to_raster(
        cell_centers=cell_centers,
        facepoint_coordinates=facepoint_coordinates,
        face_facepoint_indexes=face_facepoint_indexes,
        face_cell_indexes=face_cell_indexes,
        cell_face_info=cell_face_info,
        cell_face_values=cell_face_values,
        cell_values=cell_wse_for_render,
        output_path=None,
        cell_size=cell_size,
        reference_transform=reference_transform,
        reference_raster=reference_raster,
        crs=crs,
        nodata=nodata,
        min_value=None,  # dry cells already NaN-masked above
        min_above_ref=depth_min,
        snap_to_reference_extent=snap_to_reference_extent,
        render_mode=render_mode,
        face_active=face_active,
    )

    out_transform = wse_ds.transform
    n_rows = wse_ds.height
    n_cols = wse_ds.width
    out_crs = wse_ds.crs
    wse_nodata_val = wse_ds.nodata
    wse_pixel = wse_ds.read(1).astype(np.float64)
    wse_ds.close()

    # Pixels where WSE raster is valid → wet; velocity is nodata everywhere else.
    if wse_nodata_val is not None:
        wet_mask = wse_pixel != wse_nodata_val
    else:
        wet_mask = np.isfinite(wse_pixel)

    # ── Step 2: Build mesh triangulation for cell-ownership lookup. ──────────
    cell_centers_arr = np.asarray(cell_centers, dtype=np.float64)
    facepoint_coords_arr = np.asarray(facepoint_coordinates, dtype=np.float64)
    face_fp_idx = np.asarray(face_facepoint_indexes, dtype=np.int64)
    cell_face_values_arr = np.asarray(cell_face_values, dtype=np.int64)

    n_cells = len(cell_centers_arr)
    cell_face_info_arr = np.asarray(cell_face_info, dtype=np.int64)[:n_cells]

    dry_mask = np.isnan(cell_wse_for_render)

    all_pts = np.vstack([cell_centers_arr, facepoint_coords_arr])
    counts = cell_face_info_arr[:, 1]
    starts = cell_face_info_arr[:, 0]
    total = int(counts.sum())

    # Reproduce the vectorised entry-index array from mesh_to_raster.
    idx_step = np.ones(total, dtype=np.int64)
    if n_cells > 1:
        boundary_pos = np.cumsum(counts[:-1])
        idx_step[boundary_pos] = starts[1:] - starts[:-1] - counts[:-1] + 1
    idx_step[0] = starts[0]
    entry_indices = np.cumsum(idx_step)

    face_idx_arr = cell_face_values_arr[entry_indices, 0]
    cell_for_entry = np.repeat(np.arange(n_cells, dtype=np.int64), counts)

    fp0 = face_fp_idx[face_idx_arr, 0]
    fp1 = face_fp_idx[face_idx_arr, 1]
    valid_tris = fp0 != fp1

    triangles = np.column_stack([
        cell_for_entry[valid_tris],
        n_cells + fp0[valid_tris],
        n_cells + fp1[valid_tris],
    ])
    tri_to_cell = cell_for_entry[valid_tris]
    tri_mask_arr = dry_mask[tri_to_cell]

    triang = mtri.Triangulation(all_pts[:, 0], all_pts[:, 1], triangles)
    triang.set_mask(tri_mask_arr)

    # ── Step 3: Map each pixel to its parent cell; assign velocity. ──────────
    dx = abs(out_transform.a)
    dy = abs(out_transform.e)
    xi = out_transform.c + (np.arange(n_cols) + 0.5) * dx
    yi = out_transform.f - (np.arange(n_rows) + 0.5) * dy
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    trifinder_fn = triang.get_trifinder()
    tri_idx_flat = trifinder_fn(xi_grid.ravel(), yi_grid.ravel())

    cell_vel = cell_velocity.copy()
    cell_vel[dry_mask] = np.nan  # propagate dry-cell exclusion

    vel_flat = np.full((n_rows * n_cols, 2), np.nan)
    inside = tri_idx_flat >= 0
    vel_flat[inside] = cell_vel[tri_to_cell[tri_idx_flat[inside]]]

    vx = vel_flat[:, 0].reshape(n_rows, n_cols)
    vy = vel_flat[:, 1].reshape(n_rows, n_cols)

    # The WSE raster is the authoritative wet-extent mask.
    vx[~wet_mask] = np.nan
    vy[~wet_mask] = np.nan

    # ── Step 4: Speed, direction, min_value post-mask. ───────────────────────
    with np.errstate(invalid="ignore"):
        speed = np.sqrt(vx ** 2 + vy ** 2)
    direction = (90.0 - np.degrees(np.arctan2(vy, vx))) % 360.0
    direction = np.where(np.isnan(vx), np.nan, direction)

    if vel_min is not None:
        low = speed < vel_min
        vx[low] = vy[low] = speed[low] = direction[low] = np.nan

    band_arrays = [vx, vy, speed, direction]
    band_names = ["Vx", "Vy", "Speed", "Direction_deg_from_N"]

    # ── Step 5: Write output raster. ─────────────────────────────────────────
    profile: dict = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": n_cols,
        "height": n_rows,
        "count": 4,
        "nodata": nodata,
        "transform": out_transform,
        "compress": "lzw",
    }
    if out_crs is not None:
        profile["crs"] = out_crs

    def _write_vel_bands(dst: Any) -> None:
        for band_idx, (arr, bname) in enumerate(zip(band_arrays, band_names), start=1):
            data = arr.astype(np.float32)
            data[np.isnan(data)] = nodata
            dst.write(data, band_idx)
            dst.update_tags(band_idx, name=bname)

    if output_path is None:
        memfile = rasterio.MemoryFile()
        with memfile.open(**profile) as dst:
            _write_vel_bands(dst)
        return memfile.open()

    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        _write_vel_bands(dst)
    return out_path


def _velocity_raster_to_speed(
    vel_ds: Any,
    output_path: str | Path | None,
    nodata: float,
) -> Path | rasterio.io.DatasetReader:
    """Extract the Speed band from a 4-band velocity raster.

    Parameters
    ----------
    vel_ds : rasterio.io.DatasetReader
        Open 4-band velocity dataset produced by :func:`mesh_to_velocity_raster`.
        **Closed by this function.**
    output_path : str, Path, or None
        Destination GeoTIFF.  ``None`` returns an in-memory
        ``rasterio.DatasetReader``; caller must close it.
    nodata : float
        Nodata value for the output raster.

    Returns
    -------
    Path or rasterio.io.DatasetReader
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "_velocity_raster_to_speed requires rasterio. "
            "Install it with: pip install raspy[geo]"
        ) from exc

    speed_arr = vel_ds.read(3)  # band 3 = Speed (1-indexed)
    profile = dict(vel_ds.profile)
    profile["count"] = 1
    vel_ds.close()

    def _write(dst: Any) -> None:
        dst.write(speed_arr, 1)
        dst.update_tags(1, name="Speed")

    if output_path is None:
        memfile = rasterio.MemoryFile()
        with memfile.open(**profile) as dst:
            _write(dst)
        return memfile.open()

    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        _write(dst)
    return out_path


def _depth_from_wse_and_dem(
    wse_ds: rasterio.io.DatasetReader,
    reference_raster: str | Path,
    nodata: float,
    output_path: str | Path | None = None,
    min_value: float | None = None,
) -> Path | rasterio.io.DatasetReader:
    """Subtract a DEM from an interpolated WSE raster to produce a depth raster.

    The WSE dataset must be pixel-aligned with *reference_raster* (same pixel
    grid, possibly a sub-extent — as guaranteed when *reference_raster* was
    passed to :func:`points_to_raster`).  The DEM window that exactly
    overlaps *wse_ds* is read using a pixel-aligned ``rasterio.Window``; no
    resampling is performed.

    Negative depths (interpolated WSE below terrain) are clamped to 0.
    Pixels where either the WSE or the DEM is nodata, or where the depth is
    below *min_value*, are set to *nodata* in the output.

    Parameters
    ----------
    wse_ds:
        Open rasterio dataset containing the interpolated WSE (band 1).
    reference_raster:
        Path to the terrain DEM GeoTIFF used as the reference for *wse_ds*.
    nodata:
        Nodata value to use in the output depth raster.
    output_path:
        Destination file path.  ``None`` returns an in-memory
        ``DatasetReader``; the caller must close it.
    min_value:
        Depths below this threshold are set to *nodata*.  When ``None``
        (default) only negative depths (clamped to 0) and input nodata
        pixels are masked.

    Returns
    -------
    Path or rasterio.io.DatasetReader
    """
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as exc:
        raise ImportError(
            "Depth-from-DEM requires rasterio. "
            "Install it with: pip install raspy[geo]"
        ) from exc

    wse_transform = wse_ds.transform
    wse_nodata = wse_ds.nodata
    wse_data = wse_ds.read(1).astype(np.float64)

    with rasterio.open(reference_raster) as dem_src:
        dem_transform = dem_src.transform
        dem_nodata = dem_src.nodata
        dx = abs(dem_transform.a)
        dy = abs(dem_transform.e)
        col_off = round((wse_transform.c - dem_transform.c) / dx)
        row_off = round((dem_transform.f - wse_transform.f) / dy)
        window = Window(col_off, row_off, wse_ds.width, wse_ds.height)
        dem_data = dem_src.read(1, window=window).astype(np.float64)

    depth = wse_data - dem_data
    np.clip(depth, 0.0, None, out=depth)

    # Propagate nodata from either input; also mask shallow depths.
    out_nodata_mask = np.zeros(depth.shape, dtype=bool)
    if wse_nodata is not None:
        out_nodata_mask |= wse_data == wse_nodata
    if dem_nodata is not None:
        out_nodata_mask |= dem_data == dem_nodata
    if min_value is not None:
        out_nodata_mask |= depth < min_value
    depth[out_nodata_mask] = nodata

    profile = wse_ds.profile.copy()
    profile["nodata"] = nodata

    data = depth.astype(np.float32)

    if output_path is None:
        memfile = rasterio.MemoryFile()
        with memfile.open(**profile) as dst:
            dst.write(data, 1)
            dst.update_tags(1, name="depth")
        return memfile.open()

    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.update_tags(1, name="depth")
    return out_path
