"""Raster utilities for HEC-RAS mesh rendering and export workflows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from raspy.utils import log_call, timed

logger = logging.getLogger("raspy.geo")

if TYPE_CHECKING:
    import rasterio.io


@log_call(logging.INFO)
@timed(logging.INFO)
def rasmap_raster(
    variable: Literal["wse", "water_surface", "depth", "velocity", "velocity_vector"],
    cell_wse: np.ndarray,
    cell_min_elevation: np.ndarray,
    face_min_elevation: np.ndarray,
    face_cell_indexes: np.ndarray,
    cell_face_info: np.ndarray,
    cell_face_values: np.ndarray,
    face_facepoint_indexes: np.ndarray,
    fp_coords: np.ndarray,
    face_normals: np.ndarray,
    fp_face_info: np.ndarray,
    fp_face_values: np.ndarray,
    cell_polygons: list[np.ndarray],
    face_normal_velocity: np.ndarray | None = None,
    output_path: str | Path | None = None,
    *,
    cell_centers: np.ndarray | None = None,
    cell_surface_area: np.ndarray | None = None,
    reference_raster: str | Path | None = None,
    cell_size: float | None = None,
    crs: Any | None = None,
    nodata: float = -9999.0,
    render_mode: Literal["horizontal", "sloping", "hybrid"] = "sloping",
    use_depth_weights: bool = False,
    shallow_to_flat: bool = False,
    depth_threshold: float = 0.001,
    tight_extent: bool = True,
    perimeter: np.ndarray | None = None,
    use_numba: bool | None = None,
) -> Path | rasterio.io.DatasetReader:
    """Rasterize HEC-RAS 2D mesh results using the RASMapper-exact algorithm.

    Implements the pixel-perfect pipeline reverse-engineered from
    ``archive/DLLs/RasMapperLib/`` (decompiled C# source, HEC-RAS 6.6).
    Produces output matching
    RASMapper's ``"Horizontal"``, ``"Sloping Cell Corners"``, and
    ``"Sloping Cell Corners + Face Centers"`` render modes.

    See ``docs/export_raster2_plan.md`` for a full description of the
    algorithm steps.

    Pipeline summary
    ----------------
    **water_surface / depth — horizontal** (``render_mode="horizontal"``)

    1. ``build_cell_id_raster`` — scan-line rasterize wet cell polygons
       (Numba, mirrors ``RasterizePolygon.ComputeCells``).
    2. *(pixel loop)* — write ``cell_wse`` directly; every pixel in a cell
       gets the same flat value.

    **water_surface / depth — sloping** (``render_mode="sloping"``)

    Matches RasMapperLib ``SetSlopingRenderingMode()`` (SharedData.cs:1778):
    ``CellStencilMethod.JustFacepoints`` + ``ShallowBehavior.None``.
    GUI label: *"Sloping (Cell Corners)"*.  The "Shallow Water reduces to
    Horizontal" checkbox is a sub-option of WithFaces only and does not
    apply to this mode.

    A. ``compute_face_wss`` — hydraulic connectivity (``face_connected``),
       per-face WSE values (``face_value_a``, ``face_value_b``), and full
       connection classification (``face_hconn``, one of the ``HC_*``
       constants).
    B. ``compute_facepoint_wse`` — planar regression fitting a plane through
       the face midpoint WSE samples; the intercept ``c`` at each facepoint
       is the corner WSE.
    4a. ``build_cell_id_raster`` — scan-line rasterize wet cell polygons.
    4b. ``sample_terrain_at_facepoints`` — terrain elevation at facepoints
        for depth rebalancing (only when a DEM is supplied).
    4c. ``rasterize_rasmap`` — per-pixel barycentric interpolation of the
        facepoint WSE values within each cell's triangles (``with_faces=False``,
        ``shallow_to_flat=False``).

    **water_surface / depth — hybrid** (``render_mode="hybrid"``)

    Matches RasMapperLib ``WithFaces`` path.  Same steps as ``sloping`` but
    calls ``rasterize_rasmap`` with ``with_faces=True``; ``use_depth_weights``
    and ``shallow_to_flat`` are user-configurable.

    **velocity / velocity_vector — sloping / hybrid**

    A. ``compute_face_wss`` — hydraulic connectivity + ``face_hconn`` (same as WSE pipeline).
    2. ``reconstruct_face_velocities`` — C-stencil least-squares
       reconstruction of full ``(Vx, Vy)`` at each face from the stored
       face-normal velocity scalar.
    3. ``compute_facepoint_velocities`` — inverse-face-length weighted
       averaging of face velocity vectors to facepoints.
    3.5. ``replace_face_velocities_sloped`` — replace each face velocity
         with the average of its two endpoint facepoint velocities.
    4a. ``build_cell_id_raster`` — scan-line rasterize wet cell polygons.
    4c. ``rasterize_rasmap`` — per-pixel barycentric interpolation of
        facepoint velocity vectors; speed magnitude ``sqrt(Vx²+Vy²)``
        computed per pixel.  ``"velocity"`` returns the speed band only;
        ``"velocity_vector"`` returns all four bands.

    **velocity / velocity_vector — horizontal** (``render_mode="horizontal"``)

    RASMapper uses its stencil pipeline (``Render2D_8Stencil``) for velocity
    even in horizontal mode whenever cells are large relative to the pixel
    size (threshold = ``pixel_size² × 5``, ``MeshFV2D.cs`` line 1431).  For
    typical HEC-RAS 2D meshes all cells exceed this threshold, so this
    implementation routes horizontal-velocity requests through the same
    sloping stencil pipeline (Steps A, 2, 3, 3.5, 4) as ``"sloping"``
    mode.  ``shallow_to_flat`` is always ``False`` for this route.  WSE and
    depth still use the flat per-cell paint path.

    Parameters
    ----------
    variable:
        ``"wse"`` / ``"water_surface"`` — water-surface elevation (aliases).
        ``"depth"``          — water depth (WSE minus terrain); requires
                               *reference_raster* (DEM).
        ``"velocity"``       — 1-band speed raster ``sqrt(Vx²+Vy²)``; requires
                               *face_normal_velocity*.
        ``"velocity_vector"``— 4-band velocity raster ``[Vx, Vy, speed,
                               direction_deg]`` (bands 1–4); requires
                               *face_normal_velocity*.  ``direction_deg`` is
                               degrees clockwise from north.
    cell_wse:
        ``(n_cells,)`` water-surface elevation per cell.
    cell_min_elevation:
        ``(n_cells + n_ghost,)`` minimum bed elevation per cell including
        ghost (virtual boundary) cell rows.  Must be the full unsliced array
        (:attr:`~raspy.hdf.FlowArea.cell_min_elevation` returns only real
        cells — pass the raw HDF dataset or append ghost rows).  Ghost rows
        have ``NaN`` in HEC-RAS output.  The extra rows are required because
        :func:`~raspy.geo._rasmap.compute_face_wss` indexes into this array
        using ``face_cell_indexes`` which contains ghost cell indices on the
        cellB side of perimeter faces.
    face_min_elevation:
        ``(n_faces,)`` minimum bed elevation at each face
        (:attr:`~raspy.hdf.FlowArea.face_min_elevation`).
    face_cell_indexes:
        ``(n_faces, 2)`` — ``[cellA, cellB]``
        (:attr:`~raspy.hdf.FlowArea.face_cell_indexes`).
    cell_face_info:
        ``(n_cells, 2)`` ``[start, count]``
        (first element of :attr:`~raspy.hdf.FlowArea.cell_face_info` tuple).
    cell_face_values:
        ``(total, 2)`` ``[face_idx, orientation]``
        (second element of :attr:`~raspy.hdf.FlowArea.cell_face_info` tuple).
    face_facepoint_indexes:
        ``(n_faces, 2)`` (:attr:`~raspy.hdf.FlowArea.face_facepoint_indexes`).
    fp_coords:
        ``(n_fp, 2)`` (:attr:`~raspy.hdf.FlowArea.facepoint_coordinates`).
    face_normals:
        ``(n_faces, 3)`` ``[nx, ny, length]``
        (:attr:`~raspy.hdf.FlowArea.face_normals`).  Columns 0–1 are used as
        unit normal vectors; column 2 as face lengths.
    fp_face_info:
        ``(n_fp, 2)`` angle-sorted CSR start/count from
        :attr:`~raspy.hdf.FlowArea.facepoint_face_orientation`.
    fp_face_values:
        ``(total, 2)`` angle-sorted ``[face_idx, orientation]`` from
        :attr:`~raspy.hdf.FlowArea.facepoint_face_orientation`.
    cell_polygons:
        Per-cell polygon vertex arrays from
        :attr:`~raspy.hdf.FlowArea.cell_polygons`.
    face_normal_velocity:
        ``(n_faces,)`` signed face-normal velocity scalars.  Required for
        ``variable="velocity"``.
    output_path:
        Destination ``.tif`` file path.  ``None`` returns an open in-memory
        ``rasterio.DatasetReader``; the caller must close it.
    cell_centers:
        ``(n_cells, 2)`` cell-centre XY coordinates
        (:attr:`~raspy.hdf.FlowArea.cell_centers`).  When provided, face
        application points for the PlanarRegressionZ in Step B are computed
        as the intersection of the cell-centre-to-cell-centre line with each
        face chord (the RASMapper-exact ``GetFaceMidSide`` algorithm).  When
        ``None`` the chord midpoint of the two endpoint facepoints is used
        instead, which is adequate for orthogonal meshes.
    cell_surface_area:
        ``(n_cells,)`` plan-view cell areas in model area units
        (:attr:`~raspy.hdf.FlowArea.cell_surface_area`).  Used to split cells
        into "flat" (area ≤ ``pixel_size² × 5``) and "sloping" groups,
        matching RASMapper's ``SplitCellsOnThreshold`` /
        ``PixelRenderingCutoff = 5`` logic.  When ``None`` all cells are
        treated as sloping (current behaviour, correct for typical meshes but
        incorrect for fine-resolution refinement areas such as channels).
        Only relevant for velocity variables.
    reference_raster:
        Existing GeoTIFF whose transform and CRS are inherited.  Also used
        as the terrain DEM for depth computation and per-pixel wet/dry
        masking.  Mutually exclusive with *cell_size*.
    cell_size:
        Output pixel size in model coordinate units.  Used when no
        *reference_raster* is provided; the grid origin is derived from
        *perimeter* (if given) or from the facepoint bounding box.
    crs:
        Output CRS.  Inherited from *reference_raster* when ``None``.
    nodata:
        Fill value for dry / out-of-domain pixels.
    render_mode:
        ``"sloping"`` (default) — RASMapper "Sloping (Cell Corners)" pipeline;
        N-point barycentric interpolation using corner facepoints only
        (``with_faces=False``, ``shallow_to_flat=False``).  Matches
        ``SetSlopingRenderingMode()`` (SharedData.cs:1778) and
        ``store_map(render_mode="sloping")``.
        ``"hybrid"`` — RASMapper "Sloping Cell Corners + Face Centers"
        (``with_faces=True``); more accurate near cell edges.
        ``use_depth_weights`` and ``shallow_to_flat`` are honoured as
        supplied.  Matches ``store_map(render_mode="hybrid")``.
        ``"horizontal"`` — flat per-cell value painted over each owned pixel;
        facepoint interpolation is skipped entirely.
        Matches ``store_map(render_mode="horizontal")``.
    use_depth_weights:
        When ``True``, face weights in the ``hybrid`` stencil are
        proportional to water depth at each face.  **Ignored** (forced
        ``False``) for ``"sloping"`` and ``"horizontal"`` modes.
        Requires *reference_raster* when ``True``.
    shallow_to_flat:
        When ``True``, cells with no hydraulically-connected faces are
        rendered flat (horizontal).  **Ignored** for ``"horizontal"`` and
        ``"sloping"`` modes (forced ``False`` — "Shallow Water reduces to
        Horizontal" is a WithFaces-only sub-option in the RASMapper GUI).
        Only user-configurable for ``"hybrid"`` mode.
    depth_threshold:
        Minimum depth for a pixel to be considered wet (default ``0.001``).
        Matches ``RASResults.MinWSPlotTolerance``.
    tight_extent:
        When ``True`` (default), pixels outside *perimeter* are set to
        *nodata*.  Has no effect when *perimeter* is ``None``.
    perimeter:
        ``(n_pts, 2)`` boundary polygon used for extent and clipping.
        Pass :attr:`~raspy.hdf.FlowArea.perimeter` here.  When ``None`` the
        bounding box of *fp_coords* is used for the grid extent and no
        polygon clipping is applied.
    use_numba:
        ``True`` — require Numba JIT (raises ``ImportError`` if absent).
        ``False`` — force pure-Python fallback.
        ``None`` (default) — use Numba if available, otherwise fall back
        silently. *(Numba path not yet implemented; reserved for future.)*

    Returns
    -------
    Path
        Written GeoTIFF path when *output_path* is given.
    rasterio.io.DatasetReader
        Open in-memory dataset when *output_path* is ``None``.

    Raises
    ------
    ImportError
        If ``rasterio`` or ``shapely`` are not installed.
    ValueError
        If ``variable="depth"`` and *reference_raster* is ``None``.
        If ``variable="velocity"`` and *face_normal_velocity* is ``None``.
        If ``use_depth_weights=True`` and *reference_raster* is ``None``.
        If both *reference_raster* and *cell_size* are ``None``.
    """
    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_origin
    except ImportError as exc:
        raise ImportError(
            "rasmap_raster requires rasterio.  "
            "Install it with: pip install raspy[geo]"
        ) from exc

    from raspy.geo import _rasmap

    # -- 0. Guards ----------------------------------------------------------
    if variable == "wse":
        variable = "water_surface"

    # Translate public velocity names to internal rasterize_rasmap names.
    # "velocity"        → "speed"    (1-band magnitude output)
    # "velocity_vector" → "velocity" (4-band Vx/Vy/speed/direction output)
    if variable == "velocity":
        variable = "speed"
    elif variable == "velocity_vector":
        variable = "velocity"

    if variable == "depth" and reference_raster is None:
        raise ValueError(
            "reference_raster is required when variable='depth'. "
            "Provide a path to a terrain DEM GeoTIFF."
        )
    if variable in ("speed", "velocity") and face_normal_velocity is None:
        raise ValueError(
            "face_normal_velocity is required when variable='velocity' "
            "or variable='velocity_vector'."
        )
    if reference_raster is None and cell_size is None:
        raise ValueError(
            "Provide either reference_raster or cell_size to define the output grid."
        )
    if reference_raster is not None and cell_size is not None:
        raise ValueError(
            "Specify either reference_raster or cell_size, not both."
        )

    # Map render_mode to internal pipeline flags per RasMapperLib SharedData defaults.
    # shallow_to_flat and use_depth_weights are only user-controllable for hybrid;
    # for other modes the values are dictated by the RasMapperLib implementation.
    if render_mode == "sloping":
        # CellStencilMethod.JustFacepoints + ShallowBehavior.None
        # SetSlopingRenderingMode() in SharedData.cs:1778.
        # "Shallow Water reduces to Horizontal" is a sub-option of WithFaces only
        # (GUI: "Sloping Cell Corners + Face Centers") — not available for this mode.
        shallow_to_flat = False
        use_depth_weights = False
    elif render_mode == "horizontal":
        shallow_to_flat = False
        use_depth_weights = False
    # render_mode == "hybrid": keep user-provided shallow_to_flat and use_depth_weights

    if use_depth_weights and reference_raster is None:
        raise ValueError(
            "use_depth_weights=True requires reference_raster (terrain DEM) "
            "to sample facepoint elevations.  Provide a terrain GeoTIFF."
        )

    # -- 1. Resolve output grid and optionally load terrain -----------------
    terrain_grid: np.ndarray | None = None
    out_transform: Any
    out_width: int
    out_height: int
    out_crs: Any

    if reference_raster is not None:
        import rasterio.windows as _rwin
        with rasterio.open(reference_raster) as src:
            out_crs = crs if crs is not None else src.crs
            nd = src.nodata

            if tight_extent and perimeter is not None and len(perimeter) >= 3:
                # Derive a tight window snapped to the reference pixel grid.
                # window_transform(win) stays pixel-aligned with the full
                # reference raster, so terrain values are consistent.
                px = abs(src.transform.a)   # pixel width  (model units)
                py = abs(src.transform.e)   # pixel height (model units)
                x_min = float(perimeter[:, 0].min()) - px
                x_max = float(perimeter[:, 0].max()) + px
                y_min = float(perimeter[:, 1].min()) - py
                y_max = float(perimeter[:, 1].max()) + py
                _win_f = _rwin.from_bounds(
                    x_min, y_min, x_max, y_max, src.transform
                )
                # Floor offset, ceil size so we never clip the perimeter
                _col_off = max(0, int(np.floor(_win_f.col_off)))
                _row_off = max(0, int(np.floor(_win_f.row_off)))
                _win_w   = min(src.width  - _col_off,
                               int(np.ceil(_win_f.col_off + _win_f.width))  - _col_off)
                _win_h   = min(src.height - _row_off,
                               int(np.ceil(_win_f.row_off + _win_f.height)) - _row_off)
                win = _rwin.Window(_col_off, _row_off, _win_w, _win_h)
                out_transform = src.window_transform(win)
                out_width  = _win_w
                out_height = _win_h
                raw = src.read(1, window=win).astype(np.float32)
            else:
                out_transform = src.transform
                out_width     = src.width
                out_height    = src.height
                raw = src.read(1).astype(np.float32)

        if nd is not None:
            raw[raw == nd] = np.nan
        terrain_grid = raw.astype(np.float64)
    else:
        # Derive grid from perimeter bbox or facepoint bbox
        if perimeter is not None and len(perimeter) >= 3:
            x_min = float(perimeter[:, 0].min())
            x_max = float(perimeter[:, 0].max())
            y_min = float(perimeter[:, 1].min())
            y_max = float(perimeter[:, 1].max())
        else:
            x_min = float(fp_coords[:, 0].min())
            x_max = float(fp_coords[:, 0].max())
            y_min = float(fp_coords[:, 1].min())
            y_max = float(fp_coords[:, 1].max())
        west  = float(np.floor(x_min / cell_size) * cell_size)  # type: ignore[operator]
        north = float(np.ceil(y_max  / cell_size) * cell_size)  # type: ignore[operator]
        out_transform = from_origin(west, north, cell_size, cell_size)
        east  = float(np.ceil(x_max / cell_size) * cell_size)  # type: ignore[operator]
        south = float(np.floor(y_min / cell_size) * cell_size)  # type: ignore[operator]
        out_width  = int(round((east - west) / cell_size))
        out_height = int(round((north - south) / cell_size))
        out_crs    = CRS.from_user_input(crs) if crs is not None else None

    if out_crs is not None and not isinstance(out_crs, CRS):
        out_crs = CRS.from_user_input(out_crs)

    n_cells = len(cell_wse)
    n_faces = len(face_cell_indexes)

    logger.info(
        "rasmap_raster: variable=%r, render_mode=%r, n_cells=%d, n_faces=%d, "
        "grid=%dx%d, pixel_size=%.2f",
        variable, render_mode, n_cells, n_faces,
        out_width, out_height, abs(out_transform.a),
    )

    logger.info("raster_map: output_path=%r", str(Path(output_path).resolve()) if output_path is not None else None)

    # -- 2. Wet-cell mask (common to flat and sloping) ----------------------
    # Number of faces per cell (needed for virtual-cell detection in Step A)
    _cell_face_count_arr = cell_face_info[:, 1].astype(np.int32)

    wet_mask = (cell_wse - cell_min_elevation[:n_cells]) > depth_threshold

    # -- 3a. Flat mode — paint cell values directly -------------------------
    # RASMapper uses the stencil pipeline for velocity even in "horizontal"
    # mode whenever cells are large relative to the pixel size (threshold =
    # pixel_size² × 5).  For typical HEC-RAS meshes all cells exceed this
    # threshold, so horizontal velocity is rendered with the sloping stencil.
    # We mirror this by directing velocity to the sloping pipeline below.
    # WSE and depth keep their flat (per-cell) rendering.
    _flat_velocity = render_mode == "horizontal" and variable in ("speed", "velocity")
    if render_mode == "horizontal" and not _flat_velocity:
        # This path runs for WSE and depth in horizontal model
        cell_id_grid = _rasmap.build_cell_id_raster(
            cell_polygons, wet_mask, out_transform, out_height, out_width
        )
        valid_rows, valid_cols = np.where(cell_id_grid > 0)

        out_arr: np.ndarray = np.full((out_height, out_width), nodata, dtype=np.float32)
        # ci_arr: 0-based cell index for every valid pixel (vectorised lookup).
        ci_arr = cell_id_grid[valid_rows, valid_cols] - 1

        if variable == "water_surface":
            wse_vals = cell_wse[ci_arr].astype(np.float32)
            if terrain_grid is not None:
                t_vals = terrain_grid[valid_rows, valid_cols]
                ok = (
                    ~np.isnan(t_vals)
                    & (t_vals != nodata)
                    & (wse_vals >= t_vals + depth_threshold)
                )
                out_arr[valid_rows[ok], valid_cols[ok]] = wse_vals[ok]
            else:
                out_arr[valid_rows, valid_cols] = wse_vals

        elif variable == "depth":
            # terrain_grid is guaranteed non-None for depth (guarded at entry).
            t_vals = terrain_grid[valid_rows, valid_cols]  # type: ignore[index]
            ok = ~np.isnan(t_vals) & (t_vals != nodata)
            dep = cell_wse[ci_arr[ok]].astype(np.float64) - t_vals[ok]
            wet = dep > 0
            r_wet = valid_rows[ok][wet]
            c_wet = valid_cols[ok][wet]
            out_arr[r_wet, c_wet] = dep[wet].astype(np.float32)

    # -- 3b. Sloping/hybrid mode — full RASMapper stencil pipeline ----------
    # Also used for horizontal-mode velocity (see comment above).
    if render_mode != "horizontal" or _flat_velocity:
        # Step A: hydraulic connectivity
        face_value_a, face_value_b, face_hconn = _rasmap.compute_face_wss(
            cell_wse, cell_min_elevation, face_min_elevation,
            face_cell_indexes, _cell_face_count_arr,
        )
        face_connected = (face_hconn >= _rasmap.HC_BACKFILL) & (face_hconn <= _rasmap.HC_DOWNHILL_SHALLOW)

        # Step B: facepoint WSE (for WSE/depth sloping render, and for
        # velocity wet-pixel masking — velocity wet extent must match the
        # sloped WSE wet extent, not the coarser horizontal cell-WSE check).
        fp_wse: np.ndarray | None = None
        if render_mode != "horizontal":
            # Precompute face midsides (RASMapper-exact application points for
            # PlanarRegressionZ) when cell centres are available.  Pass an
            # empty (0, 2) array when not available — compute_facepoint_wse
            # uses shape[0] > 0 as the "available" sentinel.
            if cell_centers is not None:
                face_midsides = _rasmap._compute_face_midsides(
                    fp_coords, face_facepoint_indexes,
                    face_cell_indexes, np.asarray(cell_centers, dtype=np.float64),
                )
            else:
                face_midsides = np.empty((0, 2), dtype=np.float64)
            fp_wse = _rasmap.compute_facepoint_wse(
                fp_coords, fp_face_info, fp_face_values,
                face_facepoint_indexes, face_value_a, face_value_b,
                face_connected,
                face_midsides=face_midsides,
            )

        # Steps 2 / 3 / 3.5: velocity reconstruction (velocity only)
        fp_vel_data: np.ndarray | None = None
        face_fp_local_idx: np.ndarray | None = None
        replaced_face_vel = None
        face_vel_A = None
        face_vel_B = None
        if variable in ("speed", "velocity") and face_normal_velocity is not None:
            face_normals_2d = face_normals[:, :2]
            face_vel_A, face_vel_B = _rasmap.reconstruct_face_velocities(
                face_normal_velocity.astype(np.float64),
                face_normals_2d, face_connected,
                face_cell_indexes, cell_face_info[:n_cells], cell_face_values,
            )
            fp_vel_data, face_fp_local_idx = _rasmap.compute_facepoint_velocities(
                face_vel_A, face_vel_B, face_connected,
                face_normals[:, 2],  # face lengths
                face_facepoint_indexes, face_cell_indexes,
                cell_wse, fp_face_info, fp_face_values,
                face_value_a, face_value_b,
            )
            replaced_face_vel = _rasmap.replace_face_velocities_sloped(
                fp_vel_data, fp_face_info, face_fp_local_idx, face_facepoint_indexes,
            )

        # Step 4a: cell-ID raster
        cell_id_grid = _rasmap.build_cell_id_raster(
            cell_polygons, wet_mask, out_transform, out_height, out_width
        )

        # Sample terrain at facepoints for depth-weighted rebalancing
        fp_elev: np.ndarray | None = None
        if terrain_grid is not None and fp_wse is not None:
            fp_elev = _rasmap.sample_terrain_at_facepoints(
                fp_coords, terrain_grid, out_transform
            )

        # -- Flat-cell velocity (SplitCellsOnThreshold / PixelRenderingCutoff) --
        # Cells with area ≤ pixel_size² × 5 receive a uniform whole-cell
        # least-squares velocity instead of the stencil, matching RASMapper's
        # ComputeFromFacePerpValues(FlatMeshMap) path (Renderer.cs).
        # NaN sentinel means "use stencil" for that cell.
        flat_cell_vx: np.ndarray | None = None
        flat_cell_vy: np.ndarray | None = None
        if variable in ("speed", "velocity") and cell_surface_area is not None:
            _px = abs(out_transform.a)
            _threshold = _px * _px * 5.0
            _flat_mask = np.asarray(cell_surface_area[:n_cells]) <= _threshold
            if _flat_mask.any():
                flat_cell_vx = np.full(n_cells, np.nan, dtype=np.float64)
                flat_cell_vy = np.full(n_cells, np.nan, dtype=np.float64)
                _fvx, _fvy = _rasmap.compute_cell_flat_velocities(
                    cell_face_info[:n_cells],
                    cell_face_values,
                    np.asarray(face_normal_velocity, dtype=np.float64),
                    face_normals[:, :2].astype(np.float64),
                    wet_mask & _flat_mask,
                )
                flat_cell_vx[_flat_mask] = _fvx[_flat_mask]
                flat_cell_vy[_flat_mask] = _fvy[_flat_mask]

        # Step 4b–4e: pixel loop
        # rasterize_rasmap uses "speed" for 1-band magnitude, "velocity" for
        # 4-band (Vx, Vy, speed, direction).  The public variable="velocity"
        # returns the 4-band array so callers can access vector components.
        out_arr = _rasmap.rasterize_rasmap(
            variable=variable,
            cell_id_grid=cell_id_grid,
            transform=out_transform,
            terrain_grid=terrain_grid,
            cell_wse=cell_wse,
            cell_face_info=cell_face_info[:n_cells],
            cell_face_values=cell_face_values,
            face_facepoint_indexes=face_facepoint_indexes,
            face_cell_indexes=face_cell_indexes,
            face_min_elev=face_min_elevation,
            fp_coords=fp_coords,
            fp_wse=fp_wse,
            face_value_a=face_value_a,
            face_value_b=face_value_b,
            fp_vel_data=fp_vel_data,
            fp_face_info=fp_face_info,
            face_fp_local_idx=face_fp_local_idx,
            replaced_face_vel=replaced_face_vel,
            face_vel_A=face_vel_A,
            face_vel_B=face_vel_B,
            fp_elev=fp_elev,
            face_hconn=face_hconn,
            nodata=nodata,
            depth_threshold=depth_threshold,
            with_faces=(render_mode == "hybrid"),
            use_depth_weights=use_depth_weights,
            shallow_to_flat=shallow_to_flat,
            flat_cell_vx=flat_cell_vx,
            flat_cell_vy=flat_cell_vy,
        )

    # -- 4. Write output ----------------------------------------------------
    # velocity_vector (internal: "velocity") → 4-band; all others → 1-band.
    n_bands = 4 if variable == "velocity" else 1
    profile: dict[str, Any] = dict(
        driver="GTiff",
        dtype="float32",
        count=n_bands,
        width=out_width,
        height=out_height,
        transform=out_transform,
        nodata=nodata,
    )
    if out_crs is not None:
        profile["crs"] = out_crs

    # Apply perimeter mask if requested
    if tight_extent and perimeter is not None:
        if n_bands == 1:
            out_arr = _mask_outside_polygon_array(
                out_arr, perimeter, out_transform, nodata
            )
        else:
            for b in range(n_bands):
                out_arr[b] = _mask_outside_polygon_array(
                    out_arr[b], perimeter, out_transform, nodata
                )

    out_f32 = out_arr.astype(np.float32)
    # rasterio.write(arr) requires shape (bands, H, W); 1-band arrays need
    # an explicit band index or a leading dimension.
    if n_bands == 1:
        out_f32 = out_f32[np.newaxis, :, :]  # (1, H, W)

    if output_path is None:
        memfile = rasterio.MemoryFile()
        with memfile.open(**profile) as dst:
            dst.write(out_f32)
        return memfile.open()

    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_f32)
    return out_path


def _mask_outside_polygon_array(
    arr: np.ndarray,
    polygon_xy: np.ndarray,
    transform: Any,
    nodata: float,
) -> np.ndarray:
    """Set pixels outside *polygon_xy* to *nodata* in a 2D float array.

    Thin wrapper around rasterio.features.geometry_mask so the caller
    doesn't need to manage an in-memory dataset just for masking.
    """
    import rasterio.features
    from shapely.geometry import Polygon as _Polygon

    out = arr.copy()
    poly = _Polygon(polygon_xy)
    mask = rasterio.features.geometry_mask(
        [poly],
        out_shape=arr.shape,
        transform=transform,
        invert=False,  # True = inside polygon; False = mask outside
        all_touched=False,
    )
    out[mask] = nodata
    return out

