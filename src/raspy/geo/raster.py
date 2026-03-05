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
    face_facepoint_indexes: np.ndarray,
    face_cell_indexes: np.ndarray,
    n_cells: int,
    n_facepoints: int,
) -> np.ndarray:
    """Assign interpolated values at mesh facepoints.

    For each facepoint the value is the mean of the face-midpoint values from
    all adjacent faces.  Each face midpoint is the NaN-aware mean of its two
    adjacent cell values (or the single cell value for boundary faces).
    Averaging via face midpoints avoids double-counting a cell that shares two
    faces at the same vertex.

    Parameters
    ----------
    cell_vals : ndarray, shape ``(n_cells,)``
        Cell-centre scalar values; ``NaN`` marks dry / excluded cells.
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
    left_cells = face_cell_indexes[:, 0]
    right_cells = face_cell_indexes[:, 1]

    left_vals = np.full(n_faces, np.nan)
    right_vals = np.full(n_faces, np.nan)

    valid_left = (left_cells >= 0) & (left_cells < n_cells)
    valid_right = (right_cells >= 0) & (right_cells < n_cells)
    left_vals[valid_left] = cell_vals[left_cells[valid_left]]
    right_vals[valid_right] = cell_vals[right_cells[valid_right]]

    # Face midpoint value: NaN-aware mean of adjacent cells.
    with np.errstate(all="ignore"):
        face_vals = np.nanmean(np.column_stack([left_vals, right_vals]), axis=1)

    # Accumulate face midpoint values at each of the two bounding facepoints.
    fp_sum = np.zeros(n_facepoints)
    fp_count = np.zeros(n_facepoints, dtype=np.int64)
    valid_faces = ~np.isnan(face_vals)
    fp0 = face_facepoint_indexes[:, 0]
    fp1 = face_facepoint_indexes[:, 1]
    np.add.at(fp_sum, fp0[valid_faces], face_vals[valid_faces])
    np.add.at(fp_count, fp0[valid_faces], 1)
    np.add.at(fp_sum, fp1[valid_faces], face_vals[valid_faces])
    np.add.at(fp_count, fp1[valid_faces], 1)

    with np.errstate(all="ignore"):
        return np.where(fp_count > 0, fp_sum / fp_count, np.nan)


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
    snap_to_reference_extent: bool = True,
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
    cell_values : ndarray, shape ``(n_cells,)`` or ``(n_cells, 2)``
        Scalar field or 2-component vector ``[Vx, Vy]`` at cell centres.
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
        Scalar / speed threshold: cells below this are treated as dry
        and excluded before interpolation.
    snap_to_reference_extent : bool
        When *reference_raster* is given, extend the output to its full
        extent (default ``True``).

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
    Band layout matches :func:`points_to_raster`:

    +---------------------+---------+--------------------------------------------+
    | *cell_values* shape | bands   | band names                                 |
    +=====================+=========+============================================+
    | ``(n,)`` scalar     | 1       | ``"value"``                                |
    +---------------------+---------+--------------------------------------------+
    | ``(n, 2)`` vector   | 4       | Vx, Vy, Speed, Direction_deg_from_N        |
    +---------------------+---------+--------------------------------------------+
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

    cell_centers = np.asarray(cell_centers, dtype=np.float64)
    facepoint_coordinates = np.asarray(facepoint_coordinates, dtype=np.float64)
    face_facepoint_indexes = np.asarray(face_facepoint_indexes, dtype=np.int64)
    face_cell_indexes = np.asarray(face_cell_indexes, dtype=np.int64)
    cell_face_values_arr = np.asarray(cell_face_values, dtype=np.int64)
    cell_values = np.asarray(cell_values, dtype=np.float64)

    n_cells = len(cell_centers)
    # Slice to real cells only — HDF may append ghost/padding rows beyond n_cells.
    cell_face_info = np.asarray(cell_face_info, dtype=np.int64)[:n_cells]
    n_facepoints = len(facepoint_coordinates)
    is_vector = cell_values.ndim == 2 and cell_values.shape[1] == 2

    # ── Dry-cell masking and facepoint value computation ───────────────────
    if is_vector:
        with np.errstate(invalid="ignore"):
            magnitude = np.sqrt(cell_values[:, 0] ** 2 + cell_values[:, 1] ** 2)
        dry_mask = (
            magnitude < min_value
            if min_value is not None
            else np.zeros(n_cells, dtype=bool)
        )
        cv = cell_values.copy()
        cv[dry_mask] = np.nan  # broadcasts over both components
        fp_vx = _compute_facepoint_values(
            cv[:, 0], face_facepoint_indexes, face_cell_indexes, n_cells, n_facepoints
        )
        fp_vy = _compute_facepoint_values(
            cv[:, 1], face_facepoint_indexes, face_cell_indexes, n_cells, n_facepoints
        )
        all_vx = np.concatenate([cv[:, 0], fp_vx])
        all_vy = np.concatenate([cv[:, 1], fp_vy])
        vertex_nan = np.isnan(all_vx) | np.isnan(all_vy)
    else:
        dry_mask = (
            cell_values < min_value
            if min_value is not None
            else np.zeros(n_cells, dtype=bool)
        )
        cv = cell_values.copy()
        cv[dry_mask] = np.nan
        fp_vals = _compute_facepoint_values(
            cv, face_facepoint_indexes, face_cell_indexes, n_cells, n_facepoints
        )
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

    # Mask triangles whose cell-centre vertex is dry or has any NaN vertex.
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

    # ── Build triangulation and interpolate ────────────────────────────────
    triang = mtri.Triangulation(all_pts[:, 0], all_pts[:, 1], triangles)
    triang.set_mask(tri_mask)

    def _to_array(masked: Any) -> np.ndarray:
        """Convert a matplotlib masked array to a plain ndarray with NaN fill."""
        if hasattr(masked, "filled"):
            return masked.filled(np.nan)
        return np.asarray(masked, dtype=np.float64)

    if is_vector:
        vx = _to_array(mtri.LinearTriInterpolator(triang, all_vx)(xi_grid, yi_grid))
        vy = _to_array(mtri.LinearTriInterpolator(triang, all_vy)(xi_grid, yi_grid))
        with np.errstate(invalid="ignore"):
            speed = np.sqrt(vx ** 2 + vy ** 2)
        direction = (90.0 - np.degrees(np.arctan2(vy, vx))) % 360.0
        direction = np.where(np.isnan(vx), np.nan, direction)
        if min_value is not None:
            _low = speed < min_value
            vx[_low] = vy[_low] = speed[_low] = direction[_low] = np.nan
        band_arrays = [vx, vy, speed, direction]
        band_names = ["Vx", "Vy", "Speed", "Direction_deg_from_N"]
    else:
        scalar = _to_array(
            mtri.LinearTriInterpolator(triang, all_vals)(xi_grid, yi_grid)
        )
        if min_value is not None:
            scalar[scalar < min_value] = np.nan
        if reference_raster is not None:
            with rasterio.open(reference_raster) as dem_src:
                dem_nodata = dem_src.nodata
                dem_data = dem_src.read(
                    1, window=Window(col_min, row_min, n_cols, n_rows)
                ).astype(np.float64)
            below_ground = scalar <= dem_data
            if dem_nodata is not None:
                below_ground |= dem_data == dem_nodata
            scalar[below_ground] = np.nan
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
