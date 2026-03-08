"""Raster utilities for HEC-RAS mesh rendering and export workflows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from raspy.utils import timed

if TYPE_CHECKING:
    import rasterio.io


# ---------------------------------------------------------------------------
# KDTree rendering helpers  called from mesh_to_wse_raster and mesh_to_velocity_raster
# ---------------------------------------------------------------------------


def _tight_pixel_bounds(
    transform: Any,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_cols: int | None,
    max_rows: int | None,
) -> tuple[int, int, int, int]:
    """Pixel column/row bounds for the geographic extent [x_min..x_max, y_min..y_max].

    Converts geographic coordinates to pixel indices aligned to *transform*.
    When *max_cols* and *max_rows* are given the result is clamped to
    ``[0, max_cols] x [0, max_rows]`` (pass ``None`` to skip clamping).

    Returns
    -------
    col_min, col_max, row_min, row_max : int
    """
    dx = abs(transform.a)
    dy = abs(transform.e)
    col_min = int(np.floor((x_min - transform.c) / dx))
    col_max = int(np.ceil((x_max - transform.c) / dx))
    row_min = int(np.floor((transform.f - y_max) / dy))
    row_max = int(np.ceil((transform.f - y_min) / dy))
    if max_cols is not None:
        col_min = max(col_min, 0)
        col_max = min(col_max, max_cols)
        row_min = max(row_min, 0)
        row_max = min(row_max, max_rows)  # type: ignore[arg-type]
    return col_min, col_max, row_min, row_max


@timed(logging.DEBUG)
def _build_wet_kdtree(
    cell_centers: np.ndarray,
    dry_mask: np.ndarray,
    facepoint_coordinates: np.ndarray,
) -> tuple[Any, np.ndarray, float]:
    """Build a ``cKDTree`` on wet cell centres and estimate the mesh boundary radius.

    Parameters
    ----------
    cell_centers : ndarray, shape ``(n_cells, 2)``
    dry_mask : ndarray, shape ``(n_cells,)``, bool
        True where a cell is dry (excluded from interpolation).
    facepoint_coordinates : ndarray, shape ``(n_facepoints, 2)``
        Used to estimate the maximum cell "radius" for boundary masking.

    Returns
    -------
    tree : cKDTree or None
        Built on wet cell-centre coordinates.  ``None`` when no wet cells exist.
    wet_indices : ndarray of int, shape ``(n_wet,)``
        Indices into the full cell arrays for each point in *tree*.
    max_radius : float
        Maximum distance from any facepoint to its nearest wet cell centre.
        Pixels farther than this from every wet centre are outside the mesh.
    """
    from scipy.spatial import cKDTree

    wet_indices = np.where(~dry_mask)[0]
    wet_centers = cell_centers[wet_indices]
    if len(wet_centers) == 0:
        return None, wet_indices, 0.0

    tree = cKDTree(wet_centers)
    if len(facepoint_coordinates) == 0:
        max_radius = 0.0
    else:
        fp_dists, _ = tree.query(facepoint_coordinates)
        max_radius = float(fp_dists.max())

    return tree, wet_indices, max_radius


@timed(logging.DEBUG)
def _kdtree_nearest(
    tree: Any,
    wet_indices: np.ndarray,
    cell_values: np.ndarray,
    max_radius: float,
    xi_grid: np.ndarray,
    yi_grid: np.ndarray,
) -> np.ndarray:
    """Assign each output pixel the value of its nearest wet cell centre.

    Pixels whose nearest wet centre is farther than *max_radius* receive NaN
    (outside the wet mesh extent).  Works for scalar ``(n_cells,)`` and
    vector ``(n_cells, k)`` *cell_values*.
    """
    extra = cell_values.shape[1:]  # () for scalar, (2,) for velocity vector
    if tree is None:
        return np.full(xi_grid.shape + extra, np.nan)

    pts = np.column_stack([xi_grid.ravel(), yi_grid.ravel()])
    # Example: wet_indices = [3, 7, 10], and query gives idx=1, dist=2.4
    # for a pixel.  The nearest wet centre is 2.4 units away, and idx=1 maps
    # back to original cell id wet_indices[1] == 7.
    dist, idx = tree.query(pts)
    out = np.full((xi_grid.size,) + extra, np.nan)
    in_mesh = dist <= max_radius
    # Example mapping:
    # in_mesh     = [True, False, True]
    # idx         = [1, 0, 2]      # nearest wet-centre index per pixel
    # wet_indices = [3, 7, 10]     # wet-centre index -> original cell id
    # idx[in_mesh] -> [1, 2] -> wet_indices[...] -> [7, 10]
    # out[in_mesh] gets values from original cells 7 and 10.
    # Shapes match because both sides use the same boolean mask in_mesh.
    # out[in_mesh] selects sum(in_mesh) pixels, and idx[in_mesh] selects the
    # same count of nearest-centre indices, which map to the same count of
    # source cell_values entries.
    out[in_mesh] = cell_values[wet_indices[idx[in_mesh]]]
    return out.reshape(xi_grid.shape + extra)


@timed(logging.DEBUG)
def _kdtree_idw(
    tree: Any,
    wet_indices: np.ndarray,
    cell_values: np.ndarray,
    max_radius: float,
    xi_grid: np.ndarray,
    yi_grid: np.ndarray,
    k: int = 4,
) -> np.ndarray:
    """Inverse-distance weighting (IDW) from nearby wet cell centres.

    For each raster pixel, this function queries the KDTree for the ``k``
    nearest wet cell centres, then computes a weighted average of the
    corresponding scalar ``cell_values`` using inverse-square distance
    weights (``1 / d^2``).

    Behaviour details:
    - Outside-mesh masking: if the nearest wet centre is farther than
      ``max_radius``, the pixel is marked ``NaN``.
    - ``k`` handling: uses ``min(k, n_wet)`` so it remains valid when few
      wet cells exist.
    - Exact-hit handling: if a pixel is exactly on a wet centre
      (distance = 0), that centre gets full weight and others get zero,
      avoiding numerical instability.

    Parameters
    ----------
    tree : cKDTree or None
        KDTree built on wet cell-centre coordinates.
    wet_indices : ndarray of int, shape ``(n_wet,)``
        Maps KDTree-local indices back to original cell indices.
    cell_values : ndarray, shape ``(n_cells,)``
        Scalar value per cell (for example, WSE). Vector values are not
        supported in this helper.
    max_radius : float
        Maximum allowed nearest-centre distance for a pixel to be treated
        as inside the wet mesh.
    xi_grid, yi_grid : ndarray, shape ``(n_rows, n_cols)``
        Pixel-centre coordinate grids.
    k : int, default 4
        Number of nearest wet centres to include in the IDW average.

    Returns
    -------
    ndarray, shape ``(n_rows, n_cols)``
        Interpolated scalar raster with ``NaN`` outside the wet extent.
    """
    if tree is None:
        return np.full(xi_grid.shape, np.nan)

    n_wet = len(wet_indices)
    k_use = min(k, n_wet)
    pts = np.column_stack([xi_grid.ravel(), yi_grid.ravel()])
    dist, idx = tree.query(pts, k=k_use)

    if k_use == 1:
        dist = dist[:, np.newaxis]
        idx = idx[:, np.newaxis]

    in_mesh = dist[:, 0] <= max_radius
    out = np.full(xi_grid.size, np.nan)
    if in_mesh.any():
        d = dist[in_mesh]
        i = idx[in_mesh]
        # Exact hit: pixel coincides with a cell centre  use that value directly.
        exact = d[:, 0] == 0.0
        weights = 1.0 / np.maximum(d, 1e-14) ** 2
        weights[exact, 0] = 1.0
        weights[exact, 1:] = 0.0
        vals = cell_values[wet_indices[i]]
        out[in_mesh] = (weights * vals).sum(axis=1) / weights.sum(axis=1)
    return out.reshape(xi_grid.shape)


@timed(logging.DEBUG)
def _rasterize_cell_values(
    cell_polygons: list[np.ndarray],
    cell_values: np.ndarray,
    dry_mask: np.ndarray,
    xi_grid: np.ndarray,
    yi_grid: np.ndarray,
) -> np.ndarray:
    """Assign each output pixel the value of the cell polygon it falls inside.

    Uses rasterio's C-accelerated scan-line rasterization to build a
    ``uint32`` cell-index map, then looks up *cell_values*.  This gives
    exact polygon-boundary accuracy for horizontal rendering, eliminating the
    Voronoi mis-assignment that occurs with KDTree nearest-centre when mesh
    cells are irregular (a pixel may be geometrically inside cell A but closer
    in Euclidean distance to the centre of an adjacent larger cell B).

    Parameters
    ----------
    cell_polygons : list of ndarray, shape ``(n_vertices, 2)``
        Exact boundary vertices per cell in polygon order.  Curved faces
        should already include interior perimeter points for accurate boundaries.
    cell_values : ndarray, shape ``(n_cells,)``
        Scalar value per cell; ``NaN`` for dry cells.
    dry_mask : ndarray, shape ``(n_cells,)``, bool
        ``True`` where a cell is dry (excluded from output).
    xi_grid, yi_grid : ndarray, shape ``(n_rows, n_cols)``
        Pixel-centre coordinate grids.

    Returns
    -------
    ndarray, shape ``(n_rows, n_cols)``
        Per-pixel cell values; ``NaN`` where no wet cell polygon covers the pixel.
    """
    from rasterio.features import rasterize as _rasterize
    from rasterio.transform import Affine as _Affine

    xi = xi_grid[0, :]
    yi = yi_grid[:, 0]
    dx = float(xi[1] - xi[0]) if len(xi) > 1 else 1.0
    dy = float(yi[0] - yi[1]) if len(yi) > 1 else 1.0  # positive (top > bottom)
    burn_transform = _Affine(dx, 0.0, float(xi[0]) - dx / 2.0,
                             0.0, -dy, float(yi[0]) + dy / 2.0)

    n_rows, n_cols = xi_grid.shape
    wet_idxs = np.where(~dry_mask)[0]
    shapes = []
    for ci in wet_idxs:
        verts = cell_polygons[ci]
        if len(verts) < 3:
            continue
        ring = verts.tolist()
        ring.append(ring[0])  # close the ring
        # Burn cell_index + 1 so that fill value 0 means "no cell".
        shapes.append(({"type": "Polygon", "coordinates": [ring]}, int(ci) + 1))

    if not shapes:
        return np.full((n_rows, n_cols), np.nan)

    # uint32 supports up to ~4 billion cells; sufficient for any HEC-RAS model.
    idx_raster = _rasterize(
        shapes,
        out_shape=(n_rows, n_cols),
        transform=burn_transform,
        fill=0,
        dtype=np.uint32,
    )

    out = np.full((n_rows, n_cols), np.nan)
    valid = idx_raster > 0
    # - 1 to retrive zero-based cell index
    out[valid] = cell_values[idx_raster[valid].astype(np.int64) - 1]
    return out


@timed(logging.DEBUG)
def _griddata_facepoint_sloping(
    facepoint_coordinates: np.ndarray,
    facepoint_values: np.ndarray,
    cell_polygons: list[np.ndarray] | None,
    dry_mask: np.ndarray,
    tree: Any,
    max_radius: float,
    xi_grid: np.ndarray,
    yi_grid: np.ndarray,
    method: str = "linear",
    facecenter_coordinates: np.ndarray | None = None,
    facecenter_values: np.ndarray | None = None,
) -> np.ndarray:
    """Interpolate facepoint WSE to raster pixels using ``scipy.interpolate.griddata``.

    Parameters
    ----------
    facepoint_coordinates : ndarray, shape ``(n_fp, 2)``
        Polygon-vertex coordinates.
    facepoint_values : ndarray, shape ``(n_fp,)``
        Pre-computed WSE at each facepoint (e.g. from
        ``FlowArea.wse_at_facepoints``).  ``NaN`` where all adjacent cells
        are dry.
    facecenter_coordinates : ndarray, shape ``(n_faces, 2)``, optional
        Centroid coordinate of each face (``FlowArea.face_centroids``).
        When provided together with *facecenter_values*, these points are
        appended to the facepoint scatter set before interpolation, giving
        a denser point cloud that better captures the mesh topology.
    facecenter_values : ndarray, shape ``(n_faces,)``, optional
        WSE at each face centroid (e.g. from
        ``FlowArea.wse_at_facecentroids``).  ``NaN`` where all adjacent
        cells are dry.
    cell_polygons : list of ndarray, shape ``(n_vertices, 2)``, or None
        Exact cell boundary vertices per cell in polygon order
        (``FlowArea.cell_polygons``).  Cells with curved faces already have
        the interior perimeter points included, giving accurate boundaries.
        Used to rasterize wet cell polygons so that pixels outside every wet
        cell are set to NaN.
    dry_mask : ndarray, shape ``(n_cells,)``, bool
        ``True`` where a cell is dry.
    tree : cKDTree or None
        KDTree built on wet cell centres (used to clip outside-mesh pixels
        when ``method='cubic'``).
    max_radius : float
        Maximum distance from a pixel to its nearest wet cell centre before
        it is considered outside the wet mesh.
    xi_grid, yi_grid : ndarray, shape ``(n_rows, n_cols)``
        Pixel-centre coordinate grids.
    method : str
        Passed directly to ``scipy.interpolate.griddata``: ``'nearest'``,
        ``'linear'``, or ``'cubic'``.

    Returns
    -------
    ndarray, shape ``(n_rows, n_cols)``
        Interpolated scalar raster with ``NaN`` outside the wet extent and
        inside dry cells.
    """
    from scipy.interpolate import griddata

    valid = ~np.isnan(facepoint_values)
    pts_valid = facepoint_coordinates[valid]
    vals_valid = facepoint_values[valid]

    if facecenter_coordinates is not None and facecenter_values is not None:
        fc_valid = ~np.isnan(facecenter_values)
        pts_valid = np.vstack([pts_valid, facecenter_coordinates[fc_valid]])
        vals_valid = np.concatenate([vals_valid, facecenter_values[fc_valid]])

    pts_query = np.column_stack([xi_grid.ravel(), yi_grid.ravel()])

    if len(pts_valid) < 3:
        return np.full(xi_grid.shape, np.nan)

    scalar_flat = griddata(pts_valid, vals_valid, pts_query, method=method).astype(
        np.float64
    )

    # Wet-cell mask: rasterize the exact wet cell polygons onto the output grid,
    # then NaN any pixel that does not fall inside at least one wet cell.
    # This is more robust than masking dry cells one-by-one because it handles
    # gaps between cells and avoids the polygon-ordering issues of the
    # ConvexHull approach.  Skipped when cell_facepoint_indexes is not provided.
    if cell_polygons is not None:
        from rasterio.features import rasterize as _rasterize
        from rasterio.transform import Affine as _Affine

        # raster transform
        xi = xi_grid[0, :]
        yi = yi_grid[:, 0]
        dx = float(xi[1] - xi[0]) if len(xi) > 1 else 1.0
        dy = float(yi[0] - yi[1]) if len(yi) > 1 else 1.0  # positive (top > bottom)
        # Affine maps pixel (col, row) → world (x, y): origin is top-left corner
        burn_transform = _Affine(dx, 0.0, float(xi[0]) - dx / 2.0,
                                 0.0, -dy, float(yi[0]) + dy / 2.0)

        wet_cell_idxs = np.where(~dry_mask)[0]
        shapes = []
        for ci in wet_cell_idxs:
            verts = cell_polygons[ci]
            if len(verts) < 3:
                continue
            ring = verts.tolist()
            ring.append(ring[0])  # close the ring
            shapes.append(({"type": "Polygon", "coordinates": [ring]}, 1))

        n_rows, n_cols = xi_grid.shape
        if shapes:
            wet_raster = _rasterize(
                shapes,
                out_shape=(n_rows, n_cols),
                transform=burn_transform,
                fill=0,
                dtype=np.uint8,
            )
            scalar_flat[wet_raster.ravel() == 0] = np.nan

    # For cubic, griddata may extrapolate beyond the convex hull of the wet
    # facepoints.  Apply the same boundary clamp used by other render paths.
    if method == "cubic" and tree is not None:
        dist, _ = tree.query(pts_query)
        scalar_flat[dist > max_radius] = np.nan

    return scalar_flat.reshape(xi_grid.shape)


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


@timed(logging.INFO)
def mesh_to_wse_raster(
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
    fix_triangulation: bool = True,
    extent_bbox: tuple[float, float, float, float] | None = None,
    facepoint_values: np.ndarray | None = None,
    scatter_interp_method: str = "linear",
    cell_polygons: list[np.ndarray] | None = None,
    sloping_method: Literal["corners", "corners_faces"] = "corners",
    facecenter_coordinates: np.ndarray | None = None,
    facecenter_values: np.ndarray | None = None,
) -> Path | rasterio.io.DatasetReader:
    """Interpolate HEC-RAS mesh results to a raster using mesh-conforming triangulation.

    This function builds a triangulation that exactly conforms to the
    cell polygons  mirroring how HEC-RAS RASMapper creates its result maps.

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

        ``"sloping"`` *(default)*  interpolates WSE between cell centres
        using distance-weighted face values and linear triangular
        interpolation, producing a smooth, continuous inundation surface.

        ``"horizontal"``  assigns each output pixel the flat cell-centre
        WSE of the cell it falls inside.  Produces a stepped "patchwork"
        appearance at cell boundaries; useful for checking raw model output.

        ``"hybrid"``  uses sloping rendering where adjacent wet cells are
        hydraulically connected (active face between them) and horizontal
        rendering where they are separated by a dry face.  Requires
        *face_active*.
    face_active : ndarray, shape ``(n_faces,)``, bool, optional
        Per-face hydraulic activity flag: ``True`` = face is wet/active.
        Required when *render_mode* is ``"hybrid"``; ignored otherwise.
        Typically derived from face velocity: ``abs(face_vel) > threshold``.
    fix_triangulation : bool
        When ``True`` (default), deduplicate coincident mesh vertices and
        remove zero-area triangles before building the triangulation.  These
        defects cause ``matplotlib``'s ``TrapezoidMapTriFinder`` to raise
        ``RuntimeError: Triangulation is invalid``.  Disable to skip the
        extra work on meshes known to be clean (saves a few milliseconds on
        very large models).
    extent_bbox : tuple[float, float, float, float], optional
        ``(x_min, y_min, x_max, y_max)`` geographic bounding box that
        overrides the automatic extent derived from the cell-centre and
        facepoint clouds.  When provided alongside a reference raster the
        bbox is snapped outward to the nearest reference pixel boundaries
        via :func:`_tight_pixel_bounds`, preserving pixel-grid alignment.
        Intended for use with ``clip_to_perimeter`` in the plan layer so
        the raster extent is driven by the mesh perimeter polygon rather
        than the full facepoint cloud.
    facepoint_values : ndarray, shape ``(n_facepoints,)``, optional
        Pre-computed WSE at every facepoint, typically obtained by calling
        ``FlowArea.wse_at_facepoints(cell_wse)``.  ``NaN`` where all
        adjacent cells are dry.  When supplied and
        ``render_mode='sloping'``, the sloping path uses
        :func:`_griddata_facepoint_sloping` instead of IDW: only facepoint
        coordinates and values are used as scatter points — cell centres
        are *not* included.  Ignored for ``'horizontal'`` and ``'hybrid'``
        modes.
    scatter_interp_method : str, default ``"linear"``
        Interpolation method forwarded to ``scipy.interpolate.griddata``
        when *facepoint_values* is provided.  One of ``'nearest'``,
        ``'linear'``, or ``'cubic'``.  Ignored when *facepoint_values* is
        ``None``.
    cell_polygons : list of ndarray, shape ``(n_vertices, 2)``, optional
        Exact cell boundary vertices per cell (``FlowArea.cell_polygons``).
        Cells with curved faces include interior perimeter points for accurate
        boundaries.  When provided, dry-cell masking in the sloping path
        rasterizes the true cell polygons instead of using a nearest-centre
        Voronoi approximation.
    sloping_method : {"corners", "corners_faces"}, default ``"corners"``
        Controls which scatter points are used when ``render_mode='sloping'``:

        ``"corners"`` *(default)*  uses only the polygon-corner facepoints
        (values from *facepoint_values*).

        ``"corners_faces"``  augments the corner facepoints with face-centroid
        points (values from *facecenter_values*, coordinates from
        *facecenter_coordinates*), giving a denser scatter set that better
        captures the mesh topology inside large cells.
    facecenter_coordinates : ndarray, shape ``(n_faces, 2)``, optional
        Centroid coordinate of each face (``FlowArea.face_centroids``).
        Required when *sloping_method* is ``"corners_faces"``.
    facecenter_values : ndarray, shape ``(n_faces,)``, optional
        WSE at each face centroid (``FlowArea.wse_at_facecentroids``).
        Required when *sloping_method* is ``"corners_faces"``.

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
    #  Deferred imports 
    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import Affine, from_origin
        from rasterio.windows import Window
    except ImportError as exc:
        raise ImportError(
            "mesh_to_wse_raster requires rasterio. "
            "Install it with: pip install raspy[geo]"
        ) from exc

    try:
        from scipy.spatial import cKDTree as _cKDTree  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "mesh_to_wse_raster requires scipy. "
            "Install it with: pip install raspy[geo]"
        ) from exc

    #  Validate 
    if reference_raster is not None and reference_transform is not None:
        raise ValueError(
            "Specify either reference_raster or reference_transform, not both."
        )
    if render_mode not in ("sloping", "horizontal", "hybrid"):
        raise ValueError(
            f"render_mode must be 'sloping', 'horizontal', or 'hybrid';"
            f" got {render_mode!r}."
        )
    if sloping_method not in ("corners", "corners_faces"):
        raise ValueError(
            f"sloping_method must be 'corners' or 'corners_faces';"
            f" got {sloping_method!r}."
        )
    if sloping_method == "corners_faces" and (
        facecenter_coordinates is None or facecenter_values is None
    ):
        raise ValueError(
            "facecenter_coordinates and facecenter_values are required "
            "when sloping_method='corners_faces'."
        )
    if render_mode == "sloping" and facepoint_values is None:
        raise ValueError(
            "facepoint_values is required when render_mode='sloping'. "
            "Compute it with FlowArea.wse_at_facepoints(cell_wse)."
        )
    if scatter_interp_method not in ("nearest", "linear", "cubic"):
        raise ValueError(
            f"scatter_interp_method must be 'nearest', 'linear', or 'cubic';"
            f" got {scatter_interp_method!r}."
        )
    cell_values = np.asarray(cell_values, dtype=np.float64)
    if cell_values.ndim != 1:
        raise ValueError(
            "mesh_to_wse_raster only accepts scalar cell_values (shape (n_cells,)). "
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
    # Slice to real cells only  HDF may append ghost/padding rows beyond n_cells.
    cell_face_info = np.asarray(cell_face_info, dtype=np.int64)[:n_cells]
    n_facepoints = len(facepoint_coordinates)

    #  Dry-cell masking
    # NaN cell values always indicate dry cells (set by caller before passing in).
    # min_value provides an additional scalar threshold.
    dry_mask = np.isnan(cell_values)
    if min_value is not None:
        with np.errstate(invalid="ignore"):
            dry_mask = dry_mask | (cell_values < min_value)
    cv = cell_values.copy()
    cv[dry_mask] = np.nan

    #  Build unified point set for extent calculation 
    all_pts = np.vstack([cell_centers, facepoint_coordinates])

    #  Resolve transform and CRS 
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

    #  Build output pixel grid 
    # When extent_bbox is supplied (e.g. from the perimeter polygon) use it
    # directly; otherwise derive the extent from the full point cloud so the
    # grid covers the mesh boundary rather than just the cell-centre cloud.
    if extent_bbox is not None:
        x_min, y_min, x_max, y_max = extent_bbox
    else:
        x_min, y_min = all_pts.min(axis=0)
        x_max, y_max = all_pts.max(axis=0)

    if transform is not None:
        dx = abs(transform.a)
        dy = abs(transform.e)

        # Tight pixel bounds from the point-cloud extent, aligned to the
        # reference grid.  KDTree queries run only on this sub-grid, skipping
        # pixels outside the mesh bounding box that are guaranteed to be NaN.
        # ref_width/ref_height may be None (reference_transform given without a
        # reference_raster), in which case _tight_pixel_bounds skips clamping.
        col_min, col_max, row_min, row_max = _tight_pixel_bounds(
            transform, x_min, x_max, y_min, y_max, ref_width, ref_height
        )
        xi = transform.c + (np.arange(col_min, col_max) + 0.5) * dx
        yi = transform.f - (np.arange(row_min, row_max) + 0.5) * dy

        if snap_to_reference_extent and ref_width is not None:
            # Output covers the full reference raster extent so rasters from
            # different point clouds are pixel-aligned.  The tight KDTree result
            # is pasted into a full-size NaN array after rendering.
            embed_tight = True
            n_rows, n_cols = ref_height, ref_width  # type: ignore[assignment]
            out_transform = Affine(dx, 0.0, transform.c, 0.0, -dy, transform.f)
        else:
            embed_tight = False
            n_rows, n_cols = len(yi), len(xi)
            out_transform = Affine(
                dx, 0.0, transform.c + col_min * dx,
                0.0, -dy, transform.f - row_min * dy,
            )
    else:
        embed_tight = False
        if cell_size is None or cell_size <= 0:
            raise ValueError(
                "cell_size must be a positive number when transform and "
                "reference_raster are both None."
            )
        west = np.floor(x_min / cell_size) * cell_size
        north = np.ceil(y_max / cell_size) * cell_size
        xi = np.arange(west + cell_size / 2, x_max + cell_size, cell_size)
        yi = np.arange(north - cell_size / 2, y_min - cell_size, -cell_size)
        n_rows, n_cols = len(yi), len(xi)
        out_transform = from_origin(west, north, cell_size, cell_size)

    xi_grid, yi_grid = np.meshgrid(xi, yi)

    #  Build KDTree on wet cell centres 
    tree, wet_indices, max_radius = _build_wet_kdtree(
        cell_centers, dry_mask, facepoint_coordinates
    )

    #  Render
    if render_mode == "horizontal":
        if cell_polygons is not None:
            scalar = _rasterize_cell_values(
                cell_polygons, cv, dry_mask, xi_grid, yi_grid
            )
        else:
            scalar = _kdtree_nearest(
                tree, wet_indices, cv, max_radius, xi_grid, yi_grid
            )

    elif render_mode == "sloping":
        scalar = _griddata_facepoint_sloping(
            facepoint_coordinates,
            np.asarray(facepoint_values, dtype=np.float64),
            cell_polygons,
            dry_mask,
            tree,
            max_radius,
            xi_grid,
            yi_grid,
            scatter_interp_method,
            facecenter_coordinates=(
                np.asarray(facecenter_coordinates, dtype=np.float64)
                if sloping_method == "corners_faces"
                and facecenter_coordinates is not None
                else None
            ),
            facecenter_values=(
                np.asarray(facecenter_values, dtype=np.float64)
                if sloping_method == "corners_faces"
                and facecenter_values is not None
                else None
            ),
        )

    else:  # "hybrid"
        is_horiz_cell = _classify_horizontal_cells(
            face_cell_indexes, dry_mask, np.asarray(face_active, dtype=bool)
        )
        if tree is not None:
            pts_flat = np.column_stack([xi_grid.ravel(), yi_grid.ravel()])
            dist_flat, idx_flat = tree.query(pts_flat)
            in_mesh = dist_flat <= max_radius
            pixel_is_horiz = np.zeros(xi_grid.size, dtype=bool)
            pixel_is_horiz[in_mesh] = is_horiz_cell[wet_indices[idx_flat[in_mesh]]]
        else:
            pixel_is_horiz = np.zeros(xi_grid.size, dtype=bool)
        pixel_is_horiz = pixel_is_horiz.reshape(xi_grid.shape)
        if cell_polygons is not None:
            scalar_horiz = _rasterize_cell_values(
                cell_polygons, cv, dry_mask, xi_grid, yi_grid
            )
        else:
            scalar_horiz = _kdtree_nearest(
                tree, wet_indices, cv, max_radius, xi_grid, yi_grid
            )
        scalar_slop = _kdtree_idw(
            tree, wet_indices, cv, max_radius, xi_grid, yi_grid
        )
        scalar = np.where(pixel_is_horiz, scalar_horiz, scalar_slop)

    #  Embed tight result into full output array 
    # When snap_to_reference_extent=True the KDTree ran on a tight sub-grid;
    # paste that result into a full-size NaN array now.
    if embed_tight:
        full = np.full((n_rows, n_cols), np.nan)
        full[row_min:row_max, col_min:col_max] = scalar
        scalar = full

    #  Post-interpolation masking and band assembly 
    if min_value is not None:
        scalar[scalar < min_value] = np.nan
    if reference_raster is not None:
        with rasterio.open(reference_raster) as dem_src:
            dem_nodata = dem_src.nodata
            if embed_tight:
                # Output covers the full reference raster; read the entire DEM.
                dem_data = dem_src.read(1).astype(np.float64)
            else:
                # Output is a tight sub-grid; read only the matching DEM window.
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

    #  Write GeoTIFF 
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


@timed(logging.INFO)
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
    fix_triangulation: bool = True,
    extent_bbox: tuple[float, float, float, float] | None = None,
    facepoint_values: np.ndarray | None = None,
    scatter_interp_method: str = "linear",
    cell_facepoint_indexes: np.ndarray | None = None,
) -> Path | rasterio.io.DatasetReader:
    """Render a HEC-RAS velocity raster with WSE-based wet extent.

    HEC-RAS determines the wetted raster extent from the rendered
    (interpolated) water-surface elevation, then assigns velocity within
    those wet pixels.  This function replicates that two-step process:

    1. **Render WSE** in-memory using :func:`mesh_to_wse_raster` with the
       requested *render_mode*  this determines which raster pixels are
       wet and, for sloping/hybrid modes, the pixel-level WSE.
    2. **Assign velocity horizontally**  each wet pixel is mapped to
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
        Water-surface rendering mode  ``"sloping"`` (default),
        ``"horizontal"``, or ``"hybrid"``.  Controls which pixels are
        considered wet and therefore receive a velocity value.
    face_active : ndarray, shape ``(n_faces,)``, bool, optional
        Per-face hydraulic activity flag required by ``render_mode="hybrid"``.
    extent_bbox : tuple[float, float, float, float], optional
        Override bounding box ``(x_min, y_min, x_max, y_max)`` passed
        through to the internal :func:`mesh_to_wse_raster` call.  See that
        function for full description.
    facepoint_values : ndarray, shape ``(n_facepoints,)``, optional
        Pre-computed WSE at every facepoint passed through to the internal
        :func:`mesh_to_wse_raster` call.  See that function for full description.
    scatter_interp_method : str, default ``"linear"``
        Griddata interpolation method passed through to
        :func:`mesh_to_wse_raster` when *facepoint_values* is provided.

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
    cell_velocity = np.asarray(cell_velocity, dtype=np.float64)  # (n_cells, 2)
    cell_wse_arr = np.asarray(cell_wse, dtype=np.float64)

    # Pre-filter: mask cells whose speed is below vel_min so that the WSE
    # rendering treats them as dry and excludes them from the wet extent.
    cell_wse_for_render = cell_wse_arr.copy()
    if vel_min is not None:
        with np.errstate(invalid="ignore"):
            speed_pre = np.linalg.norm(cell_velocity, axis=1)
        cell_wse_for_render[speed_pre < vel_min] = np.nan

    #  Step 1: Render WSE in-memory to determine wet extent and grid. 
    wse_ds = mesh_to_wse_raster(
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
        fix_triangulation=fix_triangulation,
        extent_bbox=extent_bbox,
        facepoint_values=facepoint_values,
        scatter_interp_method=scatter_interp_method,
        cell_facepoint_indexes=cell_facepoint_indexes,
    )

    out_transform = wse_ds.transform
    n_rows = wse_ds.height
    n_cols = wse_ds.width
    out_crs = wse_ds.crs
    wse_nodata_val = wse_ds.nodata
    wse_pixel = wse_ds.read(1).astype(np.float64)
    wse_ds.close()

    # Pixels where WSE raster is valid  wet; velocity is nodata everywhere else.
    if wse_nodata_val is not None:
        wet_mask = wse_pixel != wse_nodata_val
    else:
        wet_mask = np.isfinite(wse_pixel)

    #  Step 2: Build KDTree for cell-ownership lookup; assign velocity. 
    cell_centers_arr = np.asarray(cell_centers, dtype=np.float64)
    facepoint_coords_arr = np.asarray(facepoint_coordinates, dtype=np.float64)
    dry_mask = np.isnan(cell_wse_for_render)

    cell_vel = cell_velocity.copy()
    cell_vel[dry_mask] = np.nan  # propagate dry-cell exclusion

    #  Step 3: Map each pixel to its parent cell; assign velocity. 
    dx = abs(out_transform.a)
    dy = abs(out_transform.e)

    # Build and query the KDTree on the tight mesh bounding box only; embed
    # the result into the full output grid to avoid querying distant NaN pixels.
    mesh_pts = np.vstack([cell_centers_arr, facepoint_coords_arr])
    x_min, y_min = mesh_pts.min(axis=0)
    x_max, y_max = mesh_pts.max(axis=0)
    col_min, col_max, row_min, row_max = _tight_pixel_bounds(
        out_transform, x_min, x_max, y_min, y_max, n_cols, n_rows
    )
    xi_grid, yi_grid = np.meshgrid(
        out_transform.c + (np.arange(col_min, col_max) + 0.5) * dx,
        out_transform.f - (np.arange(row_min, row_max) + 0.5) * dy,
    )

    tree, wet_indices, max_radius = _build_wet_kdtree(
        cell_centers_arr, dry_mask, facepoint_coords_arr
    )
    vel_tight = _kdtree_nearest(
        tree, wet_indices, cell_vel, max_radius, xi_grid, yi_grid
    )
    vel_full = np.full((n_rows, n_cols, 2), np.nan)
    vel_full[row_min:row_max, col_min:col_max] = vel_tight
    vx = vel_full[:, :, 0]
    vy = vel_full[:, :, 1]

    # The WSE raster is the authoritative wet-extent mask.
    vx[~wet_mask] = np.nan
    vy[~wet_mask] = np.nan

    #  Step 4: Speed, direction, min_value post-mask. 
    with np.errstate(invalid="ignore"):
        speed = np.sqrt(vx ** 2 + vy ** 2)
    direction = (90.0 - np.degrees(np.arctan2(vy, vx))) % 360.0
    direction = np.where(np.isnan(vx), np.nan, direction)

    if vel_min is not None:
        low = speed < vel_min
        vx[low] = vy[low] = speed[low] = direction[low] = np.nan

    band_arrays = [vx, vy, speed, direction]
    band_names = ["Vx", "Vy", "Speed", "Direction_deg_from_N"]

    #  Step 5: Write output raster. 
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


@timed(logging.DEBUG)
def _compute_face_velocity_2d(
    face_normals: np.ndarray,
    face_vel: np.ndarray,
    face_cell_indexes: np.ndarray,
    cell_velocity: np.ndarray,
    dry_mask: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    """Reconstruct full 2D face velocity using the HEC-RAS double-C stencil.

    HEC-RAS stores only the face-normal velocity ``vn``.  The tangential
    component is recovered by projecting each adjacent cell's WLS velocity
    vector onto the face tangential direction, then arithmetically averaging
    the two sides (left and right).  This mirrors the double-C stencil
    described in the HEC-RAS 2D Technical Reference Manual:

    .. code-block:: text

        vt_L = cell_velocity[L] * t_hat
        vt_R = cell_velocity[R] * t_hat
        vt   = (vt_L + vt_R) / 2          (both cells hydraulically connected)
        V_face = vn * n_hat + vt * t_hat

    The normal component ``vn`` is kept exactly as stored in the HDF; only
    the tangential component is estimated from the WLS reconstruction.

    Parameters
    ----------
    face_normals : ndarray, shape ``(n_faces, 3)``
        ``[nx, ny, face_length]``; ``(nx, ny)`` must be a unit normal vector.
    face_vel : ndarray, shape ``(n_faces,)``
        Signed face-normal velocities for this timestep.
    face_cell_indexes : ndarray, shape ``(n_faces, 2)``
        Left and right cell indices; ``-1`` = boundary.
    cell_velocity : ndarray, shape ``(n_cells, 2)``
        WLS velocity vectors ``[Vx, Vy]`` at each cell centre.
    dry_mask : ndarray, shape ``(n_cells,)``, bool
        ``True`` where a cell is dry; its WLS velocity is excluded.
    n_cells : int
        Number of real computational cells.

    Returns
    -------
    ndarray, shape ``(n_faces, 2)``
        Full ``[Vx, Vy]`` velocity at each face midpoint.
        Faces where *both* adjacent cells are dry receive ``[0, 0]``.
    """
    n_faces = len(face_normals)
    nx_ny = face_normals[:, :2]                         # (n_faces, 2)
    t_hat = np.column_stack([-nx_ny[:, 1], nx_ny[:, 0]])  # (n_faces, 2)  90 CCW

    left  = face_cell_indexes[:, 0]
    right = face_cell_indexes[:, 1]

    # Clamp indices for safe array lookup (out-of-range cases handled by masks).
    left_safe  = np.where((left  >= 0) & (left  < n_cells), left,  0)
    right_safe = np.where((right >= 0) & (right < n_cells), right, 0)

    valid_left  = (left  >= 0) & (left  < n_cells) & ~dry_mask[left_safe]
    valid_right = (right >= 0) & (right < n_cells) & ~dry_mask[right_safe]

    # Tangential projection: dot(V_cell, t_hat) for each side.
    vt_L = np.sum(cell_velocity[left_safe]  * t_hat, axis=1)  # (n_faces,)
    vt_R = np.sum(cell_velocity[right_safe] * t_hat, axis=1)

    # Arithmetic average following the manual; fall back to single side at
    # boundary or dry-neighbour faces.
    vt = np.zeros(n_faces, dtype=np.float64)
    both       = valid_left & valid_right
    left_only  = valid_left & ~valid_right
    right_only = ~valid_left & valid_right

    vt[both]       = 0.5 * (vt_L[both] + vt_R[both])
    vt[left_only]  = vt_L[left_only]
    vt[right_only] = vt_R[right_only]
    # Faces with no valid neighbour: vt stays 0  normal component still correct.

    # Compose: preserve measured vn on the normal axis; add estimated vt.
    return face_vel[:, np.newaxis] * nx_ny + vt[:, np.newaxis] * t_hat


def _cell_center_barycentric_weight(
    px: np.ndarray,
    py: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
) -> np.ndarray:
    """Barycentric weight of vertex A (cell center) for points P in triangle ABC.

    All arrays must be 1-D with the same length (one entry per pixel).

    The weight is 1 at the cell center, 0 on the opposite face edge (BC),
    and varies linearly in between.  Numerically degenerate triangles
    (area  0) return weight 1 so they fall back to the cell-centre WLS
    value.
    """
    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    # Guard against degenerate triangles (fp0 == fp1 already filtered, but
    # floating-point coincident points can still produce near-zero area).
    safe_denom = np.where(np.abs(denom) > 1e-12, denom, 1.0)
    w0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / safe_denom
    w0 = np.where(np.abs(denom) > 1e-12, w0, 1.0)
    return np.clip(w0, 0.0, 1.0)


def _barycentric_weights(
    px: np.ndarray,
    py: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full barycentric weights ``(w0, w1, w2)`` for points P in triangles ABC.

    All arrays must be 1-D with the same length (one entry per pixel).

    ``w0`` is the weight of vertex A, ``w1`` of B, ``w2`` of C.
    The weights sum to 1; all are in ``[0, 1]`` for points strictly inside
    the triangle.  Degenerate triangles (area  0) return ``(1, 0, 0)`` so
    the result falls back to the A-vertex value.
    """
    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    degenerate = np.abs(denom) <= 1e-12
    safe_denom = np.where(degenerate, 1.0, denom)

    w0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / safe_denom
    w1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / safe_denom

    w0 = np.where(degenerate, 1.0, w0)
    w1 = np.where(degenerate, 0.0, w1)
    w2 = 1.0 - w0 - w1
    return w0, w1, w2


@timed(logging.DEBUG)
def _compute_facepoint_velocities(
    face_facepoint_indexes: np.ndarray,
    face_vel_2d: np.ndarray,
    n_facepoints: int,
    face_cell_indexes: np.ndarray,
    dry_mask: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    """Average face-midpoint 2D velocities onto each facepoint (polygon vertex).

    Each facepoint is an endpoint of one or more faces.  Its velocity is the
    unweighted average of the full 2D velocities of all *wet* adjacent faces.
    A face is considered wet when at least one of its two neighbouring cells
    is not dry.  Facepoints that border only dry or boundary-only faces
    receive ``[0, 0]``.

    Parameters
    ----------
    face_facepoint_indexes : ndarray, shape ``(n_faces, 2)``
        Start and end facepoint index for each face.
    face_vel_2d : ndarray, shape ``(n_faces, 2)``
        Full 2D ``[Vx, Vy]`` velocity at each face midpoint.
    n_facepoints : int
        Total number of facepoints.
    face_cell_indexes : ndarray, shape ``(n_faces, 2)``
        Left/right cell indices; ``-1`` = boundary.
    dry_mask : ndarray, shape ``(n_cells,)``, bool
        ``True`` for dry cells.
    n_cells : int
        Number of real computational cells.

    Returns
    -------
    ndarray, shape ``(n_facepoints, 2)``
        ``[Vx, Vy]`` velocity estimate at each facepoint.
    """
    left = face_cell_indexes[:, 0]
    right = face_cell_indexes[:, 1]
    left_safe  = np.where((left  >= 0) & (left  < n_cells), left,  0)
    right_safe = np.where((right >= 0) & (right < n_cells), right, 0)
    valid_left  = (left  >= 0) & (left  < n_cells) & ~dry_mask[left_safe]
    valid_right = (right >= 0) & (right < n_cells) & ~dry_mask[right_safe]
    wet_face = valid_left | valid_right   # at least one wet neighbour

    wet_fi = np.where(wet_face)[0]
    fp0 = face_facepoint_indexes[wet_fi, 0]
    fp1 = face_facepoint_indexes[wet_fi, 1]
    fv  = face_vel_2d[wet_fi]              # (n_wet, 2)

    fp_vel_sum = np.zeros((n_facepoints, 2), dtype=np.float64)
    fp_count   = np.zeros(n_facepoints,      dtype=np.float64)
    np.add.at(fp_vel_sum, fp0, fv)
    np.add.at(fp_vel_sum, fp1, fv)
    np.add.at(fp_count,   fp0, 1.0)
    np.add.at(fp_count,   fp1, 1.0)

    safe_count = np.maximum(fp_count, 1.0)[:, np.newaxis]
    return fp_vel_sum / safe_count




@timed(logging.DEBUG)
def _interp_face_idw(
    px: np.ndarray,
    py: np.ndarray,
    cell_idx_hit: np.ndarray,
    n_cells: int,
    cell_face_info: np.ndarray,
    cell_face_values: np.ndarray,
    face_centroids: np.ndarray,
    face_vel_2d: np.ndarray,
    dry_mask: np.ndarray,
    power: float = 2.0,
    min_dist: float = 1e-3,
) -> np.ndarray:
    """Interpolate pixel velocities using IDW from face midpoints.

    For each pixel P inside cell C the velocity is the inverse-distance-
    weighted average of the full 2D face velocities located at the midpoints
    of all faces bounding C:

    .. code-block:: text

        w_i  = 1 / max(d(P, m_i), min_dist) ^ power
        V(P) = sum(w_i * V_face_i) / sum(w_i)

    This eliminates the triangulation-edge discontinuities of the
    ``"triangle_blend"`` method while preserving the exact face-normal
    velocity at each face midpoint.

    Parameters
    ----------
    px, py : ndarray, shape ``(m,)``
        Pixel x, y coordinates (one entry per wet pixel).
    cell_idx_hit : ndarray, shape ``(m,)``
        Index of the cell that contains each pixel.
    n_cells : int
        Number of real computational cells.
    cell_face_info : ndarray, shape ``(>= n_cells, 2)``
        ``[start_index, count]`` into *cell_face_values* for each cell.
    cell_face_values : ndarray, shape ``(total, 2)``
        ``[face_index, orientation]`` per cell-face association.
    face_centroids : ndarray, shape ``(n_faces, 2)``
        Centroid coordinate of each face.
    face_vel_2d : ndarray, shape ``(n_faces, 2)``
        Full 2D ``[Vx, Vy]`` velocity at each face midpoint.
    dry_mask : ndarray, shape ``(n_cells,)``, bool
        ``True`` for dry cells; their pixels receive ``[0, 0]``.
    power : float
        IDW exponent (default 2).
    min_dist : float
        Distance floor to avoid division by zero (default 1e-3 m).

    Returns
    -------
    ndarray, shape ``(m, 2)``
        Interpolated ``[Vx, Vy]`` at each pixel.
    """
    m = len(px)
    result = np.zeros((m, 2), dtype=np.float64)

    cell_face_info_arr = np.asarray(cell_face_info, dtype=np.int64)[:n_cells]
    cell_face_values_arr = np.asarray(cell_face_values, dtype=np.int64)

    # Group pixels by cell for vectorised per-cell IDW.
    sort_order = np.argsort(cell_idx_hit, kind="stable")
    sorted_cells = cell_idx_hit[sort_order]
    unique_cells, seg_start = np.unique(sorted_cells, return_index=True)
    seg_end = np.empty_like(seg_start)
    seg_end[:-1] = seg_start[1:]
    seg_end[-1] = m

    for i, c in enumerate(unique_cells):
        if dry_mask[c]:
            continue

        pix_slice = sort_order[seg_start[i] : seg_end[i]]
        ppx = px[pix_slice]
        ppy = py[pix_slice]

        start = int(cell_face_info_arr[c, 0])
        count = int(cell_face_info_arr[c, 1])
        fi = cell_face_values_arr[start : start + count, 0].astype(int)

        mp = face_centroids[fi]   # (N, 2)
        fv = face_vel_2d[fi]      # (N, 2)

        # Distance from each pixel to each face midpoint: (n_pix, N)
        ddx = ppx[:, np.newaxis] - mp[np.newaxis, :, 0]
        ddy = ppy[:, np.newaxis] - mp[np.newaxis, :, 1]
        dist = np.hypot(ddx, ddy)

        w = 1.0 / np.maximum(dist, min_dist) ** power   # (n_pix, N)
        result[pix_slice] = (w @ fv) / w.sum(axis=1, keepdims=True)

    return result


@timed(logging.DEBUG)
def _interp_face_gradient(
    px: np.ndarray,
    py: np.ndarray,
    cell_idx_hit: np.ndarray,
    n_cells: int,
    cell_centers: np.ndarray,
    cell_face_info: np.ndarray,
    cell_face_values: np.ndarray,
    face_centroids: np.ndarray,
    face_vel_2d: np.ndarray,
    dry_mask: np.ndarray,
    cell_velocity: np.ndarray,
) -> np.ndarray:
    """Interpolate pixel velocities using a per-cell linear gradient fit.

    For each cell C a local linear velocity field is fitted to the face
    midpoint velocities via ordinary least squares:

    .. code-block:: text

        V(x, y) = V0 + Gx * (x - x0) + Gy * (y - y0)

    where ``(x0, y0)`` is the cell centre and the 2x3 coefficient matrix
    ``[V0, Gx, Gy]`` is found by solving the (N x 3) system

    .. code-block:: text

        A @ coeff = fv,   A = [[1, xi-x0, yi-y0], ...]

    using ``numpy.linalg.lstsq``.  Cells with fewer than 3 faces (degenerate)
    fall back to the constant cell-centre WLS velocity.

    Parameters
    ----------
    px, py : ndarray, shape ``(m,)``
        Pixel x, y coordinates.
    cell_idx_hit : ndarray, shape ``(m,)``
        Cell index for each pixel.
    n_cells : int
        Number of real computational cells.
    cell_centers : ndarray, shape ``(n_cells, 2)``
        X, Y coordinates of each cell centre.
    cell_face_info : ndarray, shape ``(>= n_cells, 2)``
        ``[start_index, count]`` into *cell_face_values* per cell.
    cell_face_values : ndarray, shape ``(total, 2)``
        ``[face_index, orientation]`` per cell-face association.
    face_centroids : ndarray, shape ``(n_faces, 2)``
        Centroid coordinate of each face.
    face_vel_2d : ndarray, shape ``(n_faces, 2)``
        Full 2D ``[Vx, Vy]`` velocity at each face midpoint.
    dry_mask : ndarray, shape ``(n_cells,)``, bool
        ``True`` for dry cells; their pixels receive ``[0, 0]``.
    cell_velocity : ndarray, shape ``(n_cells, 2)``
        WLS cell-centre velocities used as fallback when N < 3.

    Returns
    -------
    ndarray, shape ``(m, 2)``
        Interpolated ``[Vx, Vy]`` at each pixel.
    """
    m = len(px)
    result = np.zeros((m, 2), dtype=np.float64)

    cell_face_info_arr = np.asarray(cell_face_info, dtype=np.int64)[:n_cells]
    cell_face_values_arr = np.asarray(cell_face_values, dtype=np.int64)
    cell_centers_arr = np.asarray(cell_centers, dtype=np.float64)

    sort_order = np.argsort(cell_idx_hit, kind="stable")
    sorted_cells = cell_idx_hit[sort_order]
    unique_cells, seg_start = np.unique(sorted_cells, return_index=True)
    seg_end = np.empty_like(seg_start)
    seg_end[:-1] = seg_start[1:]
    seg_end[-1] = m

    for i, c in enumerate(unique_cells):
        if dry_mask[c]:
            continue

        pix_slice = sort_order[seg_start[i] : seg_end[i]]
        ppx = px[pix_slice]
        ppy = py[pix_slice]

        start = int(cell_face_info_arr[c, 0])
        count = int(cell_face_info_arr[c, 1])
        fi = cell_face_values_arr[start : start + count, 0].astype(int)

        mp = face_centroids[fi]   # (N, 2)
        fv = face_vel_2d[fi]      # (N, 2)
        N = len(fi)

        if N < 3:
            # Degenerate cell: fall back to constant cell-centre velocity.
            result[pix_slice] = cell_velocity[c]
            continue

        x0, y0 = cell_centers_arr[c]
        xi = mp[:, 0] - x0   # (N,)
        yi = mp[:, 1] - y0   # (N,)
        A = np.column_stack([np.ones(N), xi, yi])   # (N, 3)

        # Solve A @ coeff = fv    coeff shape (3, 2)
        coeff, _, _, _ = np.linalg.lstsq(A, fv, rcond=None)

        dxp = ppx - x0   # (n_pix,)
        dyp = ppy - y0   # (n_pix,)
        # V(P) = coeff[0] + dxp * coeff[1] + dyp * coeff[2]
        result[pix_slice] = (
            coeff[0]
            + dxp[:, np.newaxis] * coeff[1]
            + dyp[:, np.newaxis] * coeff[2]
        )

    return result


@timed(logging.INFO)
def mesh_to_velocity_raster_interp(
    cell_centers: np.ndarray,
    facepoint_coordinates: np.ndarray,
    face_facepoint_indexes: np.ndarray,
    face_cell_indexes: np.ndarray,
    cell_face_info: np.ndarray,
    cell_face_values: np.ndarray,
    face_normals: np.ndarray,
    face_vel: np.ndarray,
    face_centroids: np.ndarray,
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
    method: str = "triangle_blend",
    scatter_scatter_interp_method: str = "linear",
    fix_triangulation: bool = True,
    extent_bbox: tuple[float, float, float, float] | None = None,
) -> Path | rasterio.io.DatasetReader:
    """Render a spatially varying velocity raster constrained within each mesh cell.

    Extends :func:`mesh_to_velocity_raster` by replacing the uniform
    cell-centre assignment with a **per-pixel barycentric blend** between
    the cell-centre WLS velocity and the face-normal velocity on the
    bounding face of each sub-triangle.

    Within every triangle ``(cell_center, fp0, fp1)`` the velocity at a
    pixel P is:

    .. code-block:: text

        V(P) = w0 * V_cell_WLS  +  (1 - w0) * V_face

    where ``w0`` is the barycentric weight of the cell-center vertex,
    equal to 1 at the centre and 0 on the face edge, and ``V_face`` is
    the full 2D face velocity reconstructed via the **double-C stencil**:

    .. code-block:: text

        vt   = (V_cell[L] * t_hat + V_cell[R] * t_hat) / 2
        V_face = vn * n_hat  +  vt * t_hat

    ``vn`` is preserved exactly from the HDF; only the tangential component
    ``vt`` is estimated from adjacent-cell WLS velocities.  Interpolation is
    strictly confined to each mesh cell; no values bleed across face
    boundaries.

    Parameters
    ----------
    cell_centers : ndarray, shape ``(n_cells, 2)``
        Cell-centre x, y coordinates.
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
    face_normals : ndarray, shape ``(n_faces, 3)``
        ``[nx, ny, face_length]`` unit-normal vector and face length for
        each face.  ``nx, ny`` must be a unit vector.
    face_vel : ndarray, shape ``(n_faces,)``
        Signed face-normal velocity for this timestep (positive = flow in
        the direction of the normal).
    cell_wse : ndarray, shape ``(n_cells,)``
        Cell-centre water-surface elevations.  ``NaN`` marks dry cells.
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
        Output CRS.
    nodata : float
        Fill value for dry pixels.
    vel_min : float, optional
        Speed threshold (m/s).  Cells whose WLS speed is below this are
        treated as dry.
    depth_min : float, optional
        Minimum depth threshold passed to the internal WSE render.
    snap_to_reference_extent : bool
        Extend output to the full reference raster extent (default ``True``).
    render_mode : str
        WSE rendering mode  ``"sloping"`` (default), ``"horizontal"``,
        or ``"hybrid"``.
    face_active : ndarray, shape ``(n_faces,)``, bool, optional
        Per-face hydraulic activity flag required by ``render_mode="hybrid"``.
    method : str
        Intra-cell velocity interpolation method:

        ``"triangle_blend"`` *(default)*
            Fan-triangulate each cell into ``(cell_center, fp0, fp1)``
            sub-triangles and blend the WLS cell-centre velocity with the
            face's full 2D velocity (double-C stencil) using the barycentric
            weight of the cell-centre vertex.  Fast but shows a discontinuity
            at sub-triangle edges inside each cell.

        ``"face_idw"``
            Inverse-distance-weighted average of all face-midpoint 2D
            velocities within the owning cell.  Eliminates triangulation
            artefacts; velocity varies smoothly inside each cell.  Face
            normal velocity is exactly reproduced at each face midpoint.

        ``"face_gradient"``
            Fit a local linear velocity gradient inside each cell by solving
            a least-squares system built from all face-midpoint velocities.
            Gives a smooth linear field ``V(x,y) = V0 + Gx*(x-x0) +
            Gy*(y-y0)`` centred at the cell centre.  Cells with fewer than 3
            faces fall back to the constant WLS cell-centre velocity.

        ``"facepoint_blend"``
            Assign a velocity to every facepoint (polygon vertex) by
            averaging the full 2D face velocities of all adjacent wet faces,
            then perform full 3-vertex barycentric interpolation within each
            fan-triangle ``(cell_center, fp0, fp1)``.  Because facepoints are
            shared between adjacent cells the field is C0-continuous across
            all cell-face boundaries, eliminating the hard breaks seen in
            ``"face_idw"``.  Produces smooth flow-arrow fields suitable for
            vector rendering.

        ``"scatter_interp"``
            Build a global point cloud from wet cell centres and wet face
            midpoints, then use ``scipy.interpolate.LinearNDInterpolator``
            over the entire mesh in a single call.  Because face midpoints
            are shared between adjacent cells the result is C0-continuous
            across all cell boundaries.  Pixels outside the convex hull of
            the point cloud are filled with NaN and subsequently masked by
            the WSE wet-extent raster.  Does not require fan-triangulation.

        ``"scatter_interp2"``
            Same as ``"scatter_interp"`` but the point cloud consists of
            **wet face midpoints only**  cell-centre WLS velocities are
            excluded.  This removes discontinuities that can originate at
            cell centres when the WLS velocity differs from the surrounding
            face-midpoint field.
    scatter_scatter_interp_method : str
        ``scipy.interpolate.griddata`` *method* argument used by
        ``"scatter_interp"`` and ``"scatter_interp2"``.  Accepted values:
        ``"nearest"``, ``"linear"`` *(default)*, ``"cubic"``.  Ignored for
        all other interpolation methods.
    extent_bbox : tuple[float, float, float, float], optional
        Override bounding box ``(x_min, y_min, x_max, y_max)`` passed
        through to the internal :func:`mesh_to_wse_raster` call.  See that
        function for full description.

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
    ValueError
        If *method* is not one of the recognised strings.

    Notes
    -----
    All three methods use the double-C stencil to reconstruct the full 2D
    face velocity (``vn * n_hat + vt * t_hat``), keeping ``vn`` exact as stored in
    the HDF and estimating ``vt`` from adjacent-cell WLS projections.
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "mesh_to_velocity_raster_interp requires rasterio. "
            "Install it with: pip install raspy[geo]"
        ) from exc

    _valid_methods = {
        "triangle_blend", "face_idw", "face_gradient",
        "facepoint_blend", "scatter_interp", "scatter_interp2",
    }
    if method not in _valid_methods:
        raise ValueError(
            f"method must be one of {sorted(_valid_methods)}; got {method!r}"
        )
    _valid_scatter = {"nearest", "linear", "cubic"}
    if scatter_scatter_interp_method not in _valid_scatter:
        raise ValueError(
            f"scatter_scatter_interp_method must be one of {sorted(_valid_scatter)}; "
            f"got {scatter_scatter_interp_method!r}"
        )

    cell_velocity = np.asarray(cell_velocity, dtype=np.float64)
    cell_wse_arr = np.asarray(cell_wse, dtype=np.float64)
    face_vel_arr = np.asarray(face_vel, dtype=np.float64)
    face_normals_arr = np.asarray(face_normals, dtype=np.float64)

    # Pre-filter: mask cells whose WLS speed is below vel_min.
    cell_wse_for_render = cell_wse_arr.copy()
    if vel_min is not None:
        with np.errstate(invalid="ignore"):
            speed_pre = np.linalg.norm(cell_velocity, axis=1)
        cell_wse_for_render[speed_pre < vel_min] = np.nan

    #  Step 1: Render WSE in-memory to determine wet extent and grid. 
    wse_ds = mesh_to_wse_raster(
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
        min_value=None,
        min_above_ref=depth_min,
        snap_to_reference_extent=snap_to_reference_extent,
        render_mode=render_mode,
        face_active=face_active,
        extent_bbox=extent_bbox,
    )

    out_transform = wse_ds.transform
    n_rows = wse_ds.height
    n_cols = wse_ds.width
    out_crs = wse_ds.crs
    wse_nodata_val = wse_ds.nodata
    wse_pixel = wse_ds.read(1).astype(np.float64)
    wse_ds.close()

    if wse_nodata_val is not None:
        wet_mask = wse_pixel != wse_nodata_val
    else:
        wet_mask = np.isfinite(wse_pixel)

    #  Step 2: Array conversions (shared by all methods). 
    cell_centers_arr = np.asarray(cell_centers, dtype=np.float64)
    facepoint_coords_arr = np.asarray(facepoint_coordinates, dtype=np.float64)
    face_fp_idx = np.asarray(face_facepoint_indexes, dtype=np.int64)
    cell_face_values_arr = np.asarray(cell_face_values, dtype=np.int64)

    n_cells = len(cell_centers_arr)
    cell_face_info_arr = np.asarray(cell_face_info, dtype=np.int64)[:n_cells]

    dry_mask = np.isnan(cell_wse_for_render)

    #  Step 2a: Fan-triangulation (cell-local methods only). 
    if method not in {"scatter_interp", "scatter_interp2"}:
        all_pts = np.vstack([cell_centers_arr, facepoint_coords_arr])
        counts = cell_face_info_arr[:, 1]
        starts = cell_face_info_arr[:, 0]
        total = int(counts.sum())

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
        tri_to_face = face_idx_arr[valid_tris]

        if fix_triangulation:
            # Remove degenerate and zero-area triangles  both can invalidate
            # TrapezoidMapTriFinder.  Note: full point deduplication is not
            # applied here because facepoint_blend indexes into v_fp_all via
            # `vertex_idx - n_cells`, which relies on the original index layout.
            _t0, _t1, _t2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
            _nondegen = (_t0 != _t1) & (_t1 != _t2) & (_t0 != _t2)
            _p0, _p1, _p2 = all_pts[_t0], all_pts[_t1], all_pts[_t2]
            _cross_z = (
                (_p1[:, 0] - _p0[:, 0]) * (_p2[:, 1] - _p0[:, 1])
                - (_p2[:, 0] - _p0[:, 0]) * (_p1[:, 1] - _p0[:, 1])
            )
            _keep = _nondegen & (_cross_z != 0.0)
            triangles  = triangles[_keep]
            tri_to_cell = tri_to_cell[_keep]
            tri_to_face = tri_to_face[_keep]

        tri_mask_arr = dry_mask[tri_to_cell]

        try:
            import matplotlib.tri as mtri
        except ImportError as exc:
            raise ImportError(
                "mesh_to_velocity_raster_interp requires matplotlib. "
                "Install it with: pip install raspy[geo]"
            ) from exc
        triang = mtri.Triangulation(all_pts[:, 0], all_pts[:, 1], triangles)
        triang.set_mask(tri_mask_arr)

    #  Step 2b: Full 2D face velocity  double-C stencil (shared). 
    face_ci_arr = np.asarray(face_cell_indexes, dtype=np.int64)
    face_vel_2d = _compute_face_velocity_2d(
        face_normals=face_normals_arr,
        face_vel=face_vel_arr,
        face_cell_indexes=face_ci_arr,
        cell_velocity=cell_velocity,
        dry_mask=dry_mask,
        n_cells=n_cells,
    )

    #  Step 3: Tight raster grid (mesh bounding box only). 
    # KDTree and trifinder queries run only on this sub-grid; the result is
    # embedded into the full output array at the end.
    dx = abs(out_transform.a)
    dy = abs(out_transform.e)
    mesh_pts = np.vstack([cell_centers_arr, facepoint_coords_arr])
    x_min, y_min = mesh_pts.min(axis=0)
    x_max, y_max = mesh_pts.max(axis=0)
    col_min, col_max, row_min, row_max = _tight_pixel_bounds(
        out_transform, x_min, x_max, y_min, y_max, n_cols, n_rows
    )
    n_tight_rows = row_max - row_min
    n_tight_cols = col_max - col_min
    wet_mask_tight = wet_mask[row_min:row_max, col_min:col_max]
    xi_grid, yi_grid = np.meshgrid(
        out_transform.c + (np.arange(col_min, col_max) + 0.5) * dx,
        out_transform.f - (np.arange(row_min, row_max) + 0.5) * dy,
    )

    #  Step 4: Interpolate velocities. 
    if method in {"scatter_interp", "scatter_interp2"}:
        # Global scattered interpolation.  scipy LinearNDInterpolator builds
        # one Delaunay triangulation over the whole mesh and interpolates
        # continuously, eliminating all cell-boundary artefacts.
        try:
            from scipy.interpolate import griddata
        except ImportError as exc:
            raise ImportError(
                "scatter_interp requires scipy. "
                "Install it with: pip install raspy[geo]"
            ) from exc

        # Wet-face mask: at least one wet neighbour.
        left  = face_ci_arr[:, 0]
        right = face_ci_arr[:, 1]
        left_safe  = np.where((left  >= 0) & (left  < n_cells), left,  0)
        right_safe = np.where((right >= 0) & (right < n_cells), right, 0)
        wet_face = (
            ((left  >= 0) & (left  < n_cells) & ~dry_mask[left_safe])
            | ((right >= 0) & (right < n_cells) & ~dry_mask[right_safe])
        )

        if method == "scatter_interp":
            # Include wet cell centres in addition to face centroids.
            pts = np.vstack([cell_centers_arr[~dry_mask], face_centroids[wet_face]])
            vel = np.vstack([cell_velocity[~dry_mask],    face_vel_2d[wet_face]])
        else:
            # Face centroids only — avoids discontinuities at cell centres.
            pts = face_centroids[wet_face]
            vel = face_vel_2d[wet_face]

        xi_flat = np.column_stack([xi_grid.ravel(), yi_grid.ravel()])
        # griddata returns (n_pixels, 2); NaN outside convex hull.
        vel_flat = griddata(pts, vel, xi_flat, method=scatter_scatter_interp_method,
                            fill_value=np.nan, rescale=True)
        vx = vel_flat[:, 0].reshape(n_tight_rows, n_tight_cols)
        vy = vel_flat[:, 1].reshape(n_tight_rows, n_tight_cols)

    else:
        #  Cell-local methods: map each pixel to its fan-triangle. 
        # If TrapezoidMapTriFinder fails (e.g. overlapping fan-triangles from
        # non-convex cells), fall back to KDTree flat cell-centre assignment.
        try:
            trifinder_fn = triang.get_trifinder()
            _trifinder_ok = True
        except RuntimeError:
            _trifinder_ok = False

        if not _trifinder_ok:
            cell_vel_nd = cell_velocity.copy()
            cell_vel_nd[dry_mask] = np.nan
            _tree, _wet_idx, _max_r = _build_wet_kdtree(
                cell_centers_arr, dry_mask, facepoint_coords_arr
            )
            vel_flat = _kdtree_nearest(
                _tree, _wet_idx, cell_vel_nd, _max_r, xi_grid, yi_grid
            ).reshape(-1, 2)
        else:
            #  trifinder succeeded  run cell-local interpolation 
            tri_idx_flat = trifinder_fn(xi_grid.ravel(), yi_grid.ravel())

            inside = (tri_idx_flat >= 0) & wet_mask_tight.ravel()
            inside_idx = np.where(inside)[0]
            tri_hit = tri_idx_flat[inside_idx]

            px = xi_grid.ravel()[inside_idx]
            py = yi_grid.ravel()[inside_idx]

            cell_idx_hit = tri_to_cell[tri_hit]

            if method == "triangle_blend":
                v_cell_idx = triangles[tri_hit, 0]
                v_fp0_idx  = triangles[tri_hit, 1]
                v_fp1_idx  = triangles[tri_hit, 2]

                ax, ay = all_pts[v_cell_idx, 0], all_pts[v_cell_idx, 1]
                bx, by = all_pts[v_fp0_idx,  0], all_pts[v_fp0_idx,  1]
                cx, cy = all_pts[v_fp1_idx,  0], all_pts[v_fp1_idx,  1]

                w0 = _cell_center_barycentric_weight(px, py, ax, ay, bx, by, cx, cy)

                v_cell = cell_velocity[cell_idx_hit]
                v_cell = np.where(dry_mask[cell_idx_hit, np.newaxis], np.nan, v_cell)

                face_idx_hit = tri_to_face[tri_hit]
                v_face = face_vel_2d[face_idx_hit]

                v_pixel = (
                    w0[:, np.newaxis] * v_cell
                    + (1.0 - w0[:, np.newaxis]) * v_face
                )

            elif method == "facepoint_blend":
                v_fp_all = _compute_facepoint_velocities(
                    face_fp_idx, face_vel_2d, len(facepoint_coords_arr),
                    face_ci_arr, dry_mask, n_cells,
                )

                v_cell_idx = triangles[tri_hit, 0]
                v_fp0_idx  = triangles[tri_hit, 1]
                v_fp1_idx  = triangles[tri_hit, 2]

                ax, ay = all_pts[v_cell_idx, 0], all_pts[v_cell_idx, 1]
                bx, by = all_pts[v_fp0_idx,  0], all_pts[v_fp0_idx,  1]
                cx, cy = all_pts[v_fp1_idx,  0], all_pts[v_fp1_idx,  1]

                w0, w1, w2 = _barycentric_weights(px, py, ax, ay, bx, by, cx, cy)

                v_cell = cell_velocity[cell_idx_hit]
                v_cell = np.where(dry_mask[cell_idx_hit, np.newaxis], np.nan, v_cell)

                v_fp0 = v_fp_all[v_fp0_idx - n_cells]
                v_fp1 = v_fp_all[v_fp1_idx - n_cells]

                v_pixel = (
                    w0[:, np.newaxis] * v_cell
                    + w1[:, np.newaxis] * v_fp0
                    + w2[:, np.newaxis] * v_fp1
                )

            else:
                if method == "face_idw":
                    v_pixel = _interp_face_idw(
                        px, py, cell_idx_hit, n_cells,
                        cell_face_info_arr, cell_face_values_arr,
                        face_centroids, face_vel_2d, dry_mask,
                    )
                else:  # face_gradient
                    v_pixel = _interp_face_gradient(
                        px, py, cell_idx_hit, n_cells,
                        cell_centers_arr, cell_face_info_arr, cell_face_values_arr,
                        face_centroids, face_vel_2d, dry_mask, cell_velocity,
                    )

            vel_flat = np.full((n_tight_rows * n_tight_cols, 2), np.nan)
            vel_flat[inside_idx] = v_pixel

        vx = vel_flat[:, 0].reshape(n_tight_rows, n_tight_cols)
        vy = vel_flat[:, 1].reshape(n_tight_rows, n_tight_cols)

    # Embed tight result into full output arrays.
    vx_full = np.full((n_rows, n_cols), np.nan)
    vy_full = np.full((n_rows, n_cols), np.nan)
    vx_full[row_min:row_max, col_min:col_max] = vx
    vy_full[row_min:row_max, col_min:col_max] = vy
    vx, vy = vx_full, vy_full

    # WSE raster is the authoritative wet-extent mask (applied to both paths).
    vx[~wet_mask] = np.nan
    vy[~wet_mask] = np.nan

    #  Step 4: Speed, direction, vel_min post-mask. 
    with np.errstate(invalid="ignore"):
        speed = np.sqrt(vx ** 2 + vy ** 2)
    direction = (90.0 - np.degrees(np.arctan2(vy, vx))) % 360.0
    direction = np.where(np.isnan(vx), np.nan, direction)

    if vel_min is not None:
        low = speed < vel_min
        vx[low] = vy[low] = speed[low] = direction[low] = np.nan

    band_arrays = [vx, vy, speed, direction]
    band_names = ["Vx", "Vy", "Speed", "Direction_deg_from_N"]

    #  Step 5: Write output raster. 
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


@timed(logging.DEBUG)
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


def _write_dataset(
    src: rasterio.io.DatasetReader,
    output_path: str | Path,
) -> Path:
    """Write an open rasterio dataset to a GeoTIFF file.

    The source dataset is **not** closed; the caller remains responsible for
    closing it (useful when *src* is shared with other operations).

    Parameters
    ----------
    src:
        Open rasterio dataset to copy.
    output_path:
        Destination GeoTIFF path.

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    import rasterio

    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **src.profile) as dst:
        dst.write(src.read())
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
    grid, possibly a sub-extent  as guaranteed when *reference_raster* was
    used to create the WSE raster).  The DEM window that exactly
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


def _mask_outside_polygon(
    ds: rasterio.io.DatasetReader,
    polygon_xy: np.ndarray,
    nodata: float,
    output_path: str | Path | None,
) -> Path | rasterio.io.DatasetReader:
    """Set all pixels outside *polygon_xy* to *nodata* in every band.

    The input dataset *ds* is not closed; the caller remains responsible for
    closing it.

    Parameters
    ----------
    ds:
        Open rasterio dataset to mask.  Must be in the same coordinate system
        as *polygon_xy*.
    polygon_xy:
        Mesh boundary polygon as an ``(n_pts, 2)`` float array of
        ``[x, y]`` pairs in model coordinates.  The ring is automatically
        closed if the first and last points differ.
    nodata:
        Fill value written to pixels outside the polygon.
    output_path:
        Destination GeoTIFF path.  ``None`` returns an open in-memory
        ``rasterio.DatasetReader``; the caller must close it.

    Returns
    -------
    Path or rasterio.io.DatasetReader
    """
    import rasterio
    from rasterio.features import geometry_mask

    coords = polygon_xy.tolist()
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    geom = {"type": "Polygon", "coordinates": [coords]}

    outside = geometry_mask(
        [geom],
        transform=ds.transform,
        out_shape=(ds.height, ds.width),
        invert=False,  # False -> True where OUTSIDE the polygon
    )

    data = ds.read()  # shape: (n_bands, height, width)
    data[:, outside] = nodata

    profile = ds.profile.copy()
    profile["nodata"] = nodata

    if output_path is None:
        memfile = rasterio.MemoryFile()
        with memfile.open(**profile) as dst:
            dst.write(data)
        return memfile.open()

    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)
    return out_path
