"""RASMapper-exact 2D mesh rasterization pipeline.

Implements the pixel-perfect algorithm reverse-engineered from the decompiled
C#/.NET source of ``RasMapperLib.dll`` (HEC-RAS 6.6).  All functions are
pure-numpy / pure-Python with an optional Numba JIT path for the hot
pixel-interpolation loop.

Reference
---------
``archive/DLLs/RasMapperLib/`` — decompiled C# source of ``RasMapperLib.dll``
(HEC-RAS 6.6).  Key files: ``MeshFV2D.cs``, ``Renderer.cs``,
``RASGeometryMapPoints.cs``, ``FaceVelocityCoef.cs``.  Validated pixel-perfect
against RASMapper VRT exports (median |diff| = 0.000000 across all test plans).

Pipeline stages (see ``export_raster2_plan.md`` for full description)
----------------------------------------------------------------------
Step A  — Hydraulic connectivity + per-face WSE values
Step B  — Facepoint WSE via PlanarRegressionZ
Step 2  — C-stencil tangential velocity reconstruction
Step 3  — Inverse-face-length weighted facepoint velocity averaging
Step 3.5— Sloped face velocity replacement
Step 4  — Polygon barycentric weights + donate + pixel interpolation

Public API
----------
compute_face_wss
compute_facepoint_wse
reconstruct_face_velocities
compute_facepoint_velocities
replace_face_velocities_sloped
build_cell_id_raster
sample_terrain_at_facepoints
rasterize_rasmap
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import rasterio

# ---------------------------------------------------------------------------
# Optional Numba JIT
# ---------------------------------------------------------------------------

try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False


# ---------------------------------------------------------------------------
# Step A — Hydraulic connectivity
# ---------------------------------------------------------------------------

_MIN_WS_PLOT_TOLERANCE = 0.001  # matches C# RASResults.MinWSPlotTolerance
_NODATA = -9999.0


def _avg_wse_with_crit_check(
    wse_a: float, wse_b: float, max_wse: float, min_z_face: float
) -> tuple[float, bool]:
    """Average WSE capped at the critical-flow WSE over the face sill.

    For a rectangular cross-section, specific energy at critical flow is
    E_c = (3/2) * y_c, so critical depth y_c = (2/3) * E_c.  Taking the
    upstream head H = max_wse - min_z_face as the specific energy (negligible
    approach-velocity head), the WSE at the crest under critical flow is::

        WSE_crit = min_z_face + (2/3) * H

    If the simple average of the two cell WSEs falls below this threshold,
    the face is at or below critical depth — flow is accelerating to critical
    velocity at the crest (weir / levee overtopping condition) — and the
    average is capped at WSE_crit.  Returns ``(face_wse, was_crit_cap_used)``.

    Note: the (2/3) rule is exact for rectangular sections; HEC-RAS applies it
    as a simplifying assumption at the face level.

    C# equivalent: ``_avg_water_surface_with_crit_check``.
    """
    avg = (wse_a + wse_b) * 0.5
    crit = (max_wse - min_z_face) * (2.0 / 3.0) + min_z_face  # WSE at critical depth
    return (avg, False) if avg > crit else (crit, True)


def compute_face_wss(
    cell_wse: np.ndarray,
    cell_min_elev: np.ndarray,
    face_min_elev: np.ndarray,
    face_cell_indexes: np.ndarray,
    cell_face_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step A — hydraulic connectivity and per-face WSE values.

    Replicates ``ComputeFaceWSS`` from ``archive/DLLs/RasMapperLib/MeshFV2D.cs``
    (``RASResults.cs`` internal logic).

    For each face the function determines whether water is actively flowing
    across it (hydraulic connectivity) and assigns a WSE value to each side
    (cellA side -> ``face_value_a``, cellB side -> ``face_value_b``).  These
    per-side values are later used by :func:`compute_facepoint_wse` to fit a
    sloped water surface across the mesh.

    Decision logic (applied in order)
    ----------------------------------
    1. **No real neighbour** (``cellA < 0`` or ``cellB < 0``): face sits on
       the outer mesh boundary with only one real cell.  Skip entirely —
       outputs remain at ``_NODATA`` / ``False``.

    2. **Virtual cell or dry cell**: cells with ``face_count == 1`` are
       virtual boundary cells inserted by HEC-RAS as computational
       placeholders.  cellA and cellB are the left and right cells of the
       face respectively (relative to the face normal orientation).  By
       HEC-RAS mesh construction convention, the virtual cell is always
       placed on the right (cellB) side of perimeter faces, so
       ``face_is_perimeter`` is determined solely by ``cell_b_virtual``.  If
       either cell is dry (``wse <= min_elev + _MIN_WS_PLOT_TOLERANCE``) or
       the face is a perimeter face, connectivity is ``False``.  The side
       belonging to a dry or virtual cell gets ``_NODATA``; a wet real cell
       keeps its WSE.

    3. **Both cells below the face invert** (``wse_a <= min_elev_face`` and
       ``wse_b <= min_elev_face``): water on both sides is ponded below the
       face sill — no flow over the face.  Both sides take the cell WSE and
       connectivity is ``False``.

    4. **Overtopping / weir-crest condition** (``was_crit_cap_used`` and
       depth ratio ``depth_higher / depth_above_face > 2``): the face is
       acting as a crested structure (levee, weir, or high sill) where water
       overtops with critical flow.  ``was_crit_cap_used`` indicates the
       average WSE fell below the critical-depth threshold (2/3 of head above
       the sill), and the depth ratio confirms the upstream cell is a deep
       pool with only a small head driving flow over the crest.  Both a levee
       and a weir satisfy these same two criteria — they are the same
       hydraulic condition.  Connectivity is ``False``; the upstream
       (higher) side takes ``higher_wse`` and the downstream side keeps its
       own cell WSE.

    5. **Backfill condition** (``(wse_b - wse_a) * (min_elev_b - min_elev_a) <= 0``):
       WSE gradient and bed-elevation gradient point in opposite directions —
       the lower cell sits on higher ground, which can occur when flow backs
       up into a depression.  Both sides take ``face_ws`` and connectivity is
       ``True``.

    6. **Deep flow** (``depth_higher >= 2 * delta_min_elev``): the water depth
       in the higher cell is at least twice the bed-elevation difference
       between cells, so the slope effect is negligible.  Both sides take
       ``face_ws`` and connectivity is ``True``.

    7. **Transitional flow** (everything else): the bed-elevation difference
       is significant relative to depth.  A quadratic interpolation blends
       between a depth-based estimate and ``face_ws``, giving a smoothly
       varying WSE that accounts for the sloping bed.  Both sides take this
       interpolated value and connectivity is ``True``.

    Parameters
    ----------
    cell_wse:
        Water-surface elevation per cell, shape ``(n_cells,)``.
    cell_min_elev:
        Minimum bed elevation per cell, shape ``(n_cells,)``.
    face_min_elev:
        Minimum bed elevation at each face, shape ``(n_faces,)``.
    face_cell_indexes:
        ``(n_faces, 2)`` — ``[cellA, cellB]``.  ``-1`` = boundary / no cell.
    cell_face_count:
        Number of faces per cell, shape ``(n_cells,)``.  Cells with count == 1
        are treated as virtual boundary cells (also called ghost cells in
        general CFD literature).

    Returns
    -------
    face_connected : bool ndarray, shape ``(n_faces,)``
        True for hydraulically connected faces (water actively flowing).
    face_value_a : float64 ndarray, shape ``(n_faces,)``
        WSE on the cellA side of each face; ``-9999`` where nodata.
    face_value_b : float64 ndarray, shape ``(n_faces,)``
        WSE on the cellB side of each face; ``-9999`` where nodata.
    """
    n_faces = len(face_cell_indexes)
    face_connected = np.zeros(n_faces, dtype=bool)
    face_value_a = np.full(n_faces, _NODATA, dtype=np.float64)
    face_value_b = np.full(n_faces, _NODATA, dtype=np.float64)

    for f in range(n_faces):
        cellA = int(face_cell_indexes[f, 0])
        cellB = int(face_cell_indexes[f, 1])

        # Logic 1: no real neighbour on either side
        if cellA < 0 or cellB < 0:
            continue

        wse_a = float(cell_wse[cellA])
        wse_b = float(cell_wse[cellB])
        min_elev_a = float(cell_min_elev[cellA])
        min_elev_b = float(cell_min_elev[cellB])
        min_elev_face = float(face_min_elev[f])

        cell_a_virtual = int(cell_face_count[cellA]) == 1
        cell_b_virtual = int(cell_face_count[cellB]) == 1
        face_is_perimeter = cell_b_virtual

        flag_a_dry = (
            True if np.isnan(min_elev_a)
            else wse_a <= min_elev_a + _MIN_WS_PLOT_TOLERANCE
        )
        flag_b_dry = (
            True if np.isnan(min_elev_b)
            else wse_b <= min_elev_b + _MIN_WS_PLOT_TOLERANCE
        )

        # Logic 2: virtual cell or dry cell
        if flag_a_dry or flag_b_dry or face_is_perimeter:
            face_value_a[f] = _NODATA if (flag_a_dry or cell_a_virtual) else wse_a
            face_value_b[f] = _NODATA if (flag_b_dry or cell_b_virtual) else wse_b
            face_connected[f] = False
            continue

        # Logic 3: both cells below face invert
        if wse_a <= min_elev_face and wse_b <= min_elev_face:
            face_value_a[f] = wse_a
            face_value_b[f] = wse_b
            face_connected[f] = False
            continue

        # Strict `>` mirrors C# MeshFV2D.cs — when wse_a == wse_b, cellB is
        # arbitrarily labeled "higher".  The tie-breaking choice is harmless:
        # the equal-WSE case always satisfies flag_backfill (0 * x <= 0), so
        # Logic 5 fires and both face values are set to the same WSE regardless
        # of which cell was chosen.  Conceptually this is the wrong branch
        # (there is no gradient, so "backfill" is a misnomer), but the output
        # is identical to what Logic 6/7 would produce.  RasMapper makes the
        # same "accidental" shortcut in its C# implementation.
        if wse_a > wse_b:
            higher_cell = cellA
            higher_wse = wse_a
            lower_wse = wse_b
        else:
            higher_cell = cellB
            higher_wse = wse_b
            lower_wse = wse_a

        # Bed elevation of the higher-WSE cell — used as the reference datum
        # for all depth calculations below.
        higher_min_elev = float(cell_min_elev[higher_cell])

        # Signed bed-elevation difference: positive when B's bed is higher
        # than A's bed, negative when A's bed is higher.
        delta_min_elev_signed = min_elev_b - min_elev_a

        # Backfill flag: True when the WSE gradient and the bed gradient point
        # in opposite directions.  Example: A has the higher WSE but B sits on
        # higher ground.  This means water is backing up into a depression
        # (the low-WSE cell is actually elevated), so both sides of the face
        # share the same WSE.  The product (wse_b - wse_a) * (min_elev_b -
        # min_elev_a) is negative when the gradients oppose; <= 0 also catches
        # the equal-WSE edge case (product == 0).
        flag_backfill = (wse_b - wse_a) * delta_min_elev_signed <= 0.0

        # Absolute bed-elevation difference between the two cells — used as a
        # length scale to decide between deep-flow and transitional-flow.
        delta_min_elev = abs(delta_min_elev_signed)

        # Water depth at the higher-WSE cell above its own bed minimum.
        depth_higher = higher_wse - higher_min_elev

        # Effective lower reference level for interpolation.  The lower cell's
        # WSE may sit below the face sill (the sill is above the lower water
        # surface), meaning water on the lower side has not yet reached the
        # face.  In that case the sill elevation is used as the lower boundary
        # instead of lower_wse, so the reference never drops below the face
        # invert.  When lower_wse >= min_elev_face, water on both sides
        # touches the face and lower_wse is used directly.
        eff_lower = max(min_elev_face, lower_wse)

        # Distance from the higher cell's bed to the effective lower reference.
        # Used in the transitional quadratic interpolation.  Can be negative
        # when the face sill sits below the higher cell's bed minimum.
        depth_ref_lower = eff_lower - higher_min_elev

        # Head driving flow over the face sill from the higher side.
        # Positive whenever the higher WSE is above the face invert.
        depth_above_face = higher_wse - min_elev_face

        # Check whether the average WSE falls below the critical-flow
        # threshold (2/3 of the head above the face sill).  was_crit_cap_used
        # == True means the face is at or below critical depth — a weir-like
        # condition.  The average itself is not used here (hence `_`).
        _, was_crit_cap_used = _avg_wse_with_crit_check(
            wse_a, wse_b, higher_wse, min_elev_face
        )

        # The face WSE is always anchored to the higher cell's WSE.
        face_ws = higher_wse

        # Levee / weir-crest flag: two conditions must both be met:
        #   1. was_crit_cap_used — flow is at critical depth over the face
        #      (i.e. the face is a prominent crest, not just a slight rise).
        #   2. depth_higher / depth_above_face > 2 — the upstream pool is
        #      deep relative to the head above the crest.  A ratio > 2 means
        #      most of the water in the higher cell sits *below* the face sill;
        #      the face is a significant barrier, not just a gentle slope.
        if depth_above_face > 0:
            flag_levee = was_crit_cap_used and (depth_higher / depth_above_face > 2.0)
        else:
            flag_levee = False

        if flag_levee or flag_backfill:
            if flag_levee:
                # Logic 4: overtopping / weir-crest
                # Each side keeps its own cell WSE (the higher side's WSE ==
                # face_ws by definition, so the if/else in the C# source is
                # redundant).  Connectivity is False — the two pools are
                # hydraulically separated at the crest.
                face_value_a[f] = wse_a
                face_value_b[f] = wse_b
                face_connected[f] = False
            else:
                # Logic 5: backfill
                # WSE and bed gradients oppose — water backs into a depression.
                # Both sides share face_ws and the face is connected.
                face_value_a[f] = face_ws
                face_value_b[f] = face_ws
                face_connected[f] = True
        elif depth_higher >= 2.0 * delta_min_elev:
            # Logic 6: deep flow
            # The water depth in the higher cell is at least twice the
            # bed-elevation step between the two cells.  The step is
            # effectively submerged — like a small bump at the bottom of a
            # deep pool — and has negligible influence on the water surface.
            # The two cells behave as a single connected pool with a nearly
            # horizontal WSE, so both sides of the face are assigned face_ws.
            #
            # The threshold 2*delta_min_elev is not arbitrary: it is exactly
            # the point where Logic 7's quadratic blend (step 7b) converges
            # to face_ws (blend weight of face_ws -> 1 as depth_higher ->
            # 2*delta_min_elev).  This ensures a smooth, continuous transition
            # between Logic 6 and Logic 7 with no jump at the boundary.
            face_value_a[f] = face_ws
            face_value_b[f] = face_ws
            face_connected[f] = True
        else:
            # Logic 7: transitional flow
            # The bed-elevation difference is significant relative to depth.
            # A quadratic formula (derived from assuming a linearly sloping
            # water surface) interpolates the WSE at the face.
            #
            # Step 7a — quadratic interpolation:
            #   num9 = eff_lower + (depth_higher**2 - depth_ref_lower**2)
            #                      / (2 * delta_min_elev)
            # This balances depths on both sides of the sloping bed.
            if delta_min_elev > 1e-12:
                num9 = eff_lower + (depth_higher ** 2 - depth_ref_lower ** 2) / (
                    2.0 * delta_min_elev
                )
            else:
                # Flat bed — no interpolation needed; use higher WSE directly.
                num9 = face_ws

            # Step 7b — linear blend toward face_ws as depth increases:
            # When depth_higher is between delta_min_elev and 2*delta_min_elev,
            # blend num9 with face_ws so the result transitions smoothly into
            # Logic 6 (deep flow) at the upper boundary.
            #   weight of num9    = (2*dz - d) / dz  ->  1 at d=dz, 0 at d=2*dz
            #   weight of face_ws = (d - dz)  / dz  ->  0 at d=dz, 1 at d=2*dz
            if depth_higher > delta_min_elev and delta_min_elev > 1e-12:
                num9 = (
                    (2.0 * delta_min_elev - depth_higher) * num9
                    + (depth_higher - delta_min_elev) * face_ws
                ) / delta_min_elev
            face_connected[f] = True
            face_value_a[f] = num9
            face_value_b[f] = num9

    return face_connected, face_value_a, face_value_b


# ---------------------------------------------------------------------------
# Step B — Facepoint WSE via PlanarRegressionZ
# ---------------------------------------------------------------------------


def _compute_face_midsides(
    fp_coords: np.ndarray,
    face_facepoint_indexes: np.ndarray,
    face_cell_indexes: np.ndarray,
    cell_centers: np.ndarray,
) -> np.ndarray:
    """Precompute face application points (midsides) for PlanarRegressionZ.

    Replicates the non-cached ``GetFaceMidSide`` path from
    ``archive/DLLs/RasMapperLib/MeshFV2D.cs``
    (``USE_FACE_MIDSIDE_CACHING = false``).

    For **internal faces** (both cells real, i.e. cell index >= 0): finds the
    intersection of the cell-centre-to-cell-centre infinite line with the face
    chord.  The intersection is used if it lies within [5 %, 95 %] of the face
    chord length; otherwise the chord midpoint is used as a fallback.

    For **boundary faces** (either cell index < 0): always uses the chord
    midpoint of the two endpoint facepoints.

    Only straight faces (defined by two endpoint facepoints) are handled
    exactly.  Curved faces with intermediate perimeter points are treated as
    straight — the chord midpoint is used as the fallback in those cases.

    Parameters
    ----------
    fp_coords:
        ``(n_fp, 2)`` facepoint XY coordinates.
    face_facepoint_indexes:
        ``(n_faces, 2)`` — ``[fpA, fpB]`` endpoint facepoint indexes.
    face_cell_indexes:
        ``(n_faces, 2)`` — ``[cellA, cellB]``; ``-1`` for boundary faces.
    cell_centers:
        ``(n_cells, 2)`` cell-centre XY coordinates (real cells only).

    Returns
    -------
    midsides : float64 ndarray, shape ``(n_faces, 2)``
        Application point (x, y) for each face.
    """
    n_faces = len(face_facepoint_indexes)
    n_cells = len(cell_centers)
    midsides = np.empty((n_faces, 2), dtype=np.float64)

    for fi in range(n_faces):
        fpA = int(face_facepoint_indexes[fi, 0])
        fpB = int(face_facepoint_indexes[fi, 1])
        pAx = float(fp_coords[fpA, 0])
        pAy = float(fp_coords[fpA, 1])
        pBx = float(fp_coords[fpB, 0])
        pBy = float(fp_coords[fpB, 1])
        mid_x = (pAx + pBx) * 0.5
        mid_y = (pAy + pBy) * 0.5

        cA = int(face_cell_indexes[fi, 0])
        cB = int(face_cell_indexes[fi, 1])
        # Guard against boundary faces: -1 means no neighbour; indices >= n_cells
        # are ghost (virtual boundary) cells inserted by HEC-RAS.  Both cases
        # fall back to chord midpoint since no real cell centre is available.
        if cA < 0 or cB < 0 or cA >= n_cells or cB >= n_cells:
            midsides[fi, 0] = mid_x
            midsides[fi, 1] = mid_y
            continue

        # Direction vectors
        ccAx = float(cell_centers[cA, 0])
        ccAy = float(cell_centers[cA, 1])
        ccBx = float(cell_centers[cB, 0])
        ccBy = float(cell_centers[cB, 1])
        d1x = ccBx - ccAx  # cell-centre line direction
        d1y = ccBy - ccAy
        d2x = pBx - pAx  # face chord direction
        d2y = pBy - pAy
        rx = pAx - ccAx  # offset: fpA from cellA
        ry = pAy - ccAy

        # d1 × d2 (2-D cross product); zero means parallel
        denom = d1x * d2y - d1y * d2x
        if abs(denom) < 1e-12:
            midsides[fi, 0] = mid_x
            midsides[fi, 1] = mid_y
            continue

        # s = parameter along face chord where the cell-centre LINE intersects.
        # Derived from: ccA + t*d1 = fpA + s*d2  →  s = (r × d1) / (d1 × d2)
        # where (a × b) = ax*by - ay*bx, so r × d1 = rx*d1y - ry*d1x
        s = (rx * d1y - ry * d1x) / denom

        if 0.05 < s < 0.95:
            midsides[fi, 0] = pAx + s * d2x
            midsides[fi, 1] = pAy + s * d2y
        else:
            midsides[fi, 0] = mid_x
            midsides[fi, 1] = mid_y

    return midsides


# ---------------------------------------------------------------------------
# PlanarRegressionZ helpers
# ---------------------------------------------------------------------------


def _planar_z_intercept(
    base_x: float,
    base_y: float,
    app_xs: list[float],
    app_ys: list[float],
    zs: list[float],
) -> float:
    """Fit a plane Z = a*dx + b*dy + c and return c at (base_x, base_y).

    Matches ``PlanarRegressionZ`` in ``archive/DLLs/RasMapperLib/MeshFV2D.cs``.  Working in local
    coordinates ``dx = x - base_x``, ``dy = y - base_y`` so that evaluating
    the plane at the origin (the facepoint) gives Z = c directly.

    Degenerate cases:

    * n = 0 → return ``_NODATA``
    * n = 1 → return ``zs[0]``
    * n = 2 → return average
    * det = 0 (collinear) → return average
    """
    n = len(zs)
    if n == 0:
        return _NODATA
    if n == 1:
        return zs[0]
    if n == 2:
        return (zs[0] + zs[1]) * 0.5

    sumX2 = 0.0
    sumX = 0.0
    sumY2 = 0.0
    sumY = 0.0
    sumZ = 0.0
    sumXY = 0.0
    sumYZ = 0.0
    sumXZ = 0.0
    for i in range(n):
        dx = app_xs[i] - base_x
        dy = app_ys[i] - base_y
        z = zs[i]
        sumX2 += dx * dx
        sumX += dx
        sumY2 += dy * dy
        sumY += dy
        sumZ += z
        sumXY += dx * dy
        sumYZ += dy * z
        sumXZ += dx * z

    det = (
        sumX2 * (sumY2 * n - sumY * sumY)
        - sumXY * (sumXY * n - sumY * sumX)
        + sumX * (sumXY * sumY - sumY2 * sumX)
    )
    if det == 0.0:
        return sumZ / n
    return (
        sumX2 * (sumY2 * sumZ - sumYZ * sumY)
        - sumXY * (sumXY * sumZ - sumYZ * sumX)
        + sumXZ * (sumXY * sumY - sumY2 * sumX)
    ) / det


def _face_app_point(
    fi: int,
    fp_coords: np.ndarray,
    face_facepoint_indexes: np.ndarray,
    face_midsides: np.ndarray | None,
) -> tuple[float, float]:
    """Return the application point (x, y) for face *fi*.

    Uses the precomputed midside when available; falls back to chord midpoint.
    """
    if face_midsides is not None:
        return float(face_midsides[fi, 0]), float(face_midsides[fi, 1])
    fpA = int(face_facepoint_indexes[fi, 0])
    fpB = int(face_facepoint_indexes[fi, 1])
    return (
        (float(fp_coords[fpA, 0]) + float(fp_coords[fpB, 0])) * 0.5,
        (float(fp_coords[fpA, 1]) + float(fp_coords[fpB, 1])) * 0.5,
    )


def compute_facepoint_wse(
    fp_coords: np.ndarray,
    fp_face_info: np.ndarray,
    fp_face_values: np.ndarray,
    face_facepoint_indexes: np.ndarray,
    face_value_a: np.ndarray,
    face_value_b: np.ndarray,
    face_connected: np.ndarray,
    face_midsides: np.ndarray | None = None,
) -> np.ndarray:
    """Step B — arc-based per-facepoint WSE via PlanarRegressionZ.

    Replicates ``ComputeFacePointWSs`` from
    ``archive/DLLs/RasMapperLib/MeshFV2D.cs`` exactly, including the arc
    decomposition at wet/dry boundaries.

    **Arc-based algorithm** (``MeshFV2D.cs:9483``)

    Faces adjacent to each facepoint are angle-sorted CCW (from
    ``fp_face_values``).  Hydraulically-connected faces form *arcs*; arcs
    are separated by disconnected (shallow/dry) faces.

    For each arc a separate ``PlanarRegressionZ`` is fitted:

    * **Arc faces** (connected): contribute their *own-side* WSE —
      ``face_value_a`` when the facepoint is ``fpA`` (orientation ``-1``),
      ``face_value_b`` when it is ``fpB`` (``+1``).
    * **Terminal face** (the first disconnected face CCW past the arc):
      contributes its *opposite-side* WSE as a boundary anchor.
    * The regression intercept at the facepoint coordinate is assigned to
      **all faces in the arc**.

    Different arcs around the same facepoint therefore produce different
    WSE values for the faces in each arc; these per-arc values are stored
    separately so the pixel loop can retrieve the correct one for each cell.

    Parameters
    ----------
    fp_coords:
        ``(n_fp, 2)`` — facepoint XY coordinates.
    fp_face_info:
        ``(n_fp, 2)`` int32 — ``[start, count]`` into ``fp_face_values``.
        From :attr:`~raspy.hdf.FlowArea.facepoint_face_orientation_info`.
        **Faces must be in angle-sorted CCW order** (as stored in the HDF
        ``FacePoints Face and Orientation`` datasets).
    fp_face_values:
        ``(total, 2)`` int32 — ``[face_idx, orientation]``.
        ``orientation = -1`` → this facepoint is ``fpA``; ``+1`` → ``fpB``.
    face_facepoint_indexes:
        ``(n_faces, 2)`` — ``[fpA, fpB]`` for each face.
    face_value_a:
        ``(n_faces,)`` — WSE on the cellA side from :func:`compute_face_wss`.
    face_value_b:
        ``(n_faces,)`` — WSE on the cellB side from :func:`compute_face_wss`.
    face_connected:
        ``(n_faces,)`` bool — hydraulic connectivity from
        :func:`compute_face_wss`.  Disconnected faces (``False``) separate arcs.
    face_midsides:
        ``(n_faces, 2)`` — precomputed face application points from
        :func:`_compute_face_midsides`.  When ``None`` the chord midpoint is
        used (adequate for orthogonal meshes).

    Returns
    -------
    fp_wse_at_face : float64 ndarray, shape ``(n_faces, 2)``
        Per-face arc WSE:

        * ``[fi, 0]`` — regression result at face ``fi`` from fpA's arc.
        * ``[fi, 1]`` — regression result at face ``fi`` from fpB's arc.

        ``-9999`` where the arc has no valid sample points.
    """
    n_faces = len(face_facepoint_indexes)
    fp_wse_at_face = np.full((n_faces, 2), _NODATA, dtype=np.float64)

    for fp_idx in range(len(fp_coords)):
        base_x = float(fp_coords[fp_idx, 0])
        base_y = float(fp_coords[fp_idx, 1])
        fp_start = int(fp_face_info[fp_idx, 0])
        fp_count = int(fp_face_info[fp_idx, 1])

        # Early exit: skip facepoints where every adjacent face is completely dry
        # (both sides -9999).  Matches C# flag=true early-return.
        any_wet = False
        for j in range(fp_count):
            fi = int(fp_face_values[fp_start + j, 0])
            if float(face_value_a[fi]) != _NODATA or float(face_value_b[fi]) != _NODATA:
                any_wet = True
                break
        if not any_wet:
            continue

        # processed[j] tracks which local face indices have been assigned
        # by a completed arc.  Prevents double-processing.
        processed = [False] * fp_count

        # Outer loop: find the next unprocessed local face index j and
        # process its arc.  Mirrors the C# for(j=0..num) with continue.
        for j in range(fp_count):
            if processed[j]:
                continue

            # --- Find num4: first disconnected face going CCW from j ---
            # (or j itself if all faces are connected → full ring arc)
            num4 = j
            while True:
                num4 = (num4 + 1) % fp_count
                fi_num4 = int(fp_face_values[fp_start + num4, 0])
                if not face_connected[fi_num4] or num4 == j:
                    break

            # --- Find num5: first disconnected face going CW from j ---
            # If num5 == num4 (all connected), the whole ring is one arc.
            num5 = j
            if num5 != num4:
                while face_connected[int(fp_face_values[fp_start + num5, 0])]:
                    num5 = (num5 - 1 + fp_count) % fp_count

            # --- Collect arc faces (num5 → num4, CCW exclusive) ---
            app_xs: list[float] = []
            app_ys: list[float] = []
            zs: list[float] = []
            raw: list[float] = []  # own-side raw values (one per arc face)

            num6 = num5
            while True:
                fi_cur = int(fp_face_values[fp_start + num6, 0])
                ori_cur = int(fp_face_values[fp_start + num6, 1])
                # Own-side: fpA (ori=-1) → face_value_a; fpB (ori=+1) → face_value_b
                if ori_cur == -1:
                    wse = float(face_value_a[fi_cur])
                else:
                    wse = float(face_value_b[fi_cur])
                if wse != _NODATA:
                    ax, ay = _face_app_point(
                        fi_cur, fp_coords, face_facepoint_indexes, face_midsides
                    )
                    app_xs.append(ax)
                    app_ys.append(ay)
                    zs.append(wse)
                raw.append(wse)  # may be _NODATA; overridden by arc result below
                num6 = (num6 + 1) % fp_count
                if num6 == num4:
                    break

            # --- Add terminal face (num4, if disconnected) using OPPOSITE side ---
            # Provides a slope anchor at the wet/dry boundary.
            fi_term = int(fp_face_values[fp_start + num4, 0])
            if not face_connected[fi_term]:
                ori_term = int(fp_face_values[fp_start + num4, 1])
                # Opposite: fpA (ori=-1) → face_value_b; fpB (ori=+1) → face_value_a
                if ori_term == -1:
                    wse_term = float(face_value_b[fi_term])
                else:
                    wse_term = float(face_value_a[fi_term])
                if wse_term != _NODATA:
                    ax, ay = _face_app_point(
                        fi_term, fp_coords, face_facepoint_indexes, face_midsides
                    )
                    app_xs.append(ax)
                    app_ys.append(ay)
                    zs.append(wse_term)

            # --- Solve regression and store result for arc faces ---
            arc_result = _planar_z_intercept(base_x, base_y, app_xs, app_ys, zs)

            num6 = num5
            arc_idx = 0
            while True:
                fi_cur = int(fp_face_values[fp_start + num6, 0])
                ori_cur = int(fp_face_values[fp_start + num6, 1])
                # Store arc result (or raw value if regression had no points)
                value = arc_result if arc_result != _NODATA else raw[arc_idx]
                side = 0 if ori_cur == -1 else 1  # -1=fpA→col0; +1=fpB→col1
                if value != _NODATA:
                    fp_wse_at_face[fi_cur, side] = value
                processed[num6] = True
                arc_idx += 1
                num6 = (num6 + 1) % fp_count
                if num6 == num4:
                    break

            # All-connected ring: one arc covers everything → done.
            if num5 == num4:
                break

    return fp_wse_at_face


# ---------------------------------------------------------------------------
# Step 2 — C-stencil tangential velocity reconstruction
# ---------------------------------------------------------------------------


class _FaceVelocityCoef:
    """Symmetric 2x2 normal-equation matrix for the C-stencil WLS solve.

    Mirrors ``FaceVelocityCoef`` in
    ``archive/DLLs/RasMapperLib/MeshFV2D.cs``.
    """
    __slots__ = ("A11", "A22", "A12", "_ct")

    def __init__(self) -> None:
        self.A11 = 0.0
        self.A22 = 0.0
        self.A12 = 0.0
        self._ct = 0

    def add_face_normal(self, nx: float, ny: float) -> None:
        self.A11 += nx * nx
        self.A22 += ny * ny
        self.A12 += nx * ny
        self._ct += 1

    def complete(self) -> None:
        det = self.A11 * self.A22 - self.A12 * self.A12
        if det == 0.0:
            inv_ct = 1.0 / self._ct if self._ct > 0 else 1.0
            self.A11 = inv_ct
            self.A12 = 0.0
            self.A22 = inv_ct
        else:
            inv_det = 1.0 / det
            self.A11 *= inv_det
            self.A22 *= inv_det
            self.A12 *= inv_det

    def solve(self, B1: float, B2: float) -> tuple[float, float]:
        return (self.A22 * B1 - self.A12 * B2, -self.A12 * B1 + self.A11 * B2)


def _cw_ccw_neighbors(
    target_face: int,
    cell_idx: int,
    cell_face_info: np.ndarray,
    cell_face_values: np.ndarray,
) -> tuple[int, int]:
    """Return (CW neighbor face, CCW neighbor face) of *target_face* within *cell_idx*.

    Returns -1 for a neighbor that doesn't exist.
    """
    start = int(cell_face_info[cell_idx, 0])
    count = int(cell_face_info[cell_idx, 1])
    target_pos = -1
    for k in range(count):
        if int(cell_face_values[start + k, 0]) == target_face:
            target_pos = k
            break
    if target_pos < 0:
        return -1, -1
    cw_pos  = (target_pos + 1) % count
    ccw_pos = (target_pos - 1 + count) % count
    return int(cell_face_values[start + cw_pos, 0]), int(cell_face_values[start + ccw_pos, 0])


def _solve_face_vector(
    coef: _FaceVelocityCoef,
    fn_x: float, fn_y: float,
    face_vel: float,
    cw_normal: np.ndarray, cw_vel: float, cw_connected: bool,
    ccw_normal: np.ndarray, ccw_vel: float, ccw_connected: bool,
) -> tuple[float, float]:
    """Reconstruct 2D velocity vector at a face from a 3-face C-stencil.

    Mirrors the C-stencil solve in
    ``archive/DLLs/RasMapperLib/MeshFV2D.cs``.
    """
    tan_x = fn_y
    tan_y = -fn_x
    tangential = 0.0
    if cw_connected or ccw_connected:
        B1 = cw_normal[0] * cw_vel + ccw_normal[0] * ccw_vel + fn_x * face_vel
        B2 = cw_normal[1] * cw_vel + ccw_normal[1] * ccw_vel + fn_y * face_vel
        sx, sy = coef.solve(B1, B2)
        tangential = sx * tan_x + sy * tan_y
    Vx = face_vel * fn_x + tangential * tan_x
    Vy = face_vel * fn_y + tangential * tan_y
    return Vx, Vy


def reconstruct_face_velocities(
    face_normal_vel: np.ndarray,
    face_normals_2d: np.ndarray,
    face_connected: np.ndarray,
    face_cell_indexes: np.ndarray,
    cell_face_info: np.ndarray,
    cell_face_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Step 2 — C-stencil least-squares tangential velocity reconstruction.

    HEC-RAS stores only face-normal velocity scalars ``vn`` (the component
    perpendicular to each face).  This function reconstructs the full 2D
    ``(Vx, Vy)`` velocity vector at every face by estimating the missing
    tangential component via a 3-face C-stencil.

    Replicates ``ReconstructFaceVelocitiesLeastSquares`` from
    ``archive/DLLs/RasMapperLib/MeshFV2D.cs``.

    Algorithm
    ---------
    For every face *f* with unit normal ``n_hat = (fn_x, fn_y)`` and stored
    face-normal velocity ``vn``, the reconstruction is performed **twice** —
    once from each adjacent cell's perspective (cellA and cellB):

    1. **C-stencil selection** — within the current cell, find the two faces
       immediately adjacent to *f* in the cell's face-ordering ring:

       - *cw*  : the face one step clockwise from *f* (index ``(pos+1) % count``)
       - *ccw* : the face one step counter-clockwise (index ``(pos-1) % count``)

       These three faces (cw, ccw, and *f* itself) form the "C-stencil".

    2. **Normal-equation matrix** — accumulate the 2x2 symmetric WLS matrix
       using the unit normals of the three C-stencil faces::

           A = S  [nx*nx   nx*ny]     (sum over cw, ccw, and f)
                  [nx*ny   ny*ny]

       The inverse ``A_inv`` is computed analytically (Cramer's rule).  If
       ``det(A) == 0`` (degenerate geometry), the identity scaled by 1/count
       is substituted so the solve degrades gracefully.

    3. **RHS vector** — using only the *connected* (wet) CW and CCW neighbors,
       assemble::

           B = n_hat_cw * vn_cw  +  n_hat_ccw * vn_ccw  +  n_hat_f * vn_f

    4. **Least-squares solve** — recover the full velocity vector ``(sx, sy)``
       as ``A_inv * B``.  Project it onto the tangential direction
       ``t_hat = (-fn_y, fn_x)`` to obtain the scalar tangential component::

           tangential = sx * t_hat_x + sy * t_hat_y

    5. **Compose face velocity**::

           Vx = vn * fn_x + tangential * t_hat_x
           Vy = vn * fn_y + tangential * t_hat_y

       The normal component is kept exactly as stored in the HDF; only the
       tangential component is estimated.

    6. **Averaging for connected faces** — when a face is marked connected
       (``face_connected[f] == True``) *and* both cellA and cellB exist, the
       two independent reconstructions are averaged::

           face_vel_A[f] = face_vel_B[f] = (velA + velB) / 2

    Parameters
    ----------
    face_normal_vel : ndarray, shape ``(n_faces,)``
        Signed face-normal velocity scalars read directly from the HDF result.
    face_normals_2d : ndarray, shape ``(n_faces, 2)``
        Unit normal vectors ``[nx, ny]`` for each face.
        Pass ``face_normals[:, :2]`` from :attr:`~raspy.hdf.FlowArea.face_normals`.
    face_connected : ndarray, shape ``(n_faces,)``, bool
        ``True`` for faces that are hydraulically connected (wet) for this
        timestep.  Obtained from :func:`compute_face_wss`.
    face_cell_indexes : ndarray, shape ``(n_faces, 2)``
        ``[cellA, cellB]`` — indices of the left and right cell for each face.
        ``-1`` indicates no neighbour (boundary face).
    cell_face_info : ndarray, shape ``(n_cells, 2)``
        ``[start, count]`` — index into *cell_face_values* for each cell's
        face list.  From :attr:`~raspy.hdf.FlowArea.cell_face_info`.
    cell_face_values : ndarray, shape ``(total, 2)``
        ``[face_idx, orientation]`` — the ordered face ring for each cell,
        used to find CW/CCW neighbors.
        From :attr:`~raspy.hdf.FlowArea.cell_face_values`.

    Returns
    -------
    face_vel_A : float64 ndarray, shape ``(n_faces, 2)``
        Full ``[Vx, Vy]`` velocity at each face reconstructed from cellA's
        C-stencil (Item1 in RASMapper terminology).
    face_vel_B : float64 ndarray, shape ``(n_faces, 2)``
        Full ``[Vx, Vy]`` velocity at each face reconstructed from cellB's
        C-stencil (Item2 in RASMapper terminology).
        For connected faces both arrays hold the averaged value.
        Boundary faces (no cellA or cellB) fall back to ``vn * n_hat``.
    """
    n_faces = len(face_cell_indexes)
    face_vel_A = np.zeros((n_faces, 2), dtype=np.float64)
    face_vel_B = np.zeros((n_faces, 2), dtype=np.float64)
    _zeros2 = np.zeros(2, dtype=np.float64)

    for fidx in range(n_faces):
        cellA = int(face_cell_indexes[fidx, 0])
        cellB = int(face_cell_indexes[fidx, 1])
        fv = float(face_normal_vel[fidx])
        fn = face_normals_2d[fidx]  # view, no copy
        fn_x = float(fn[0])
        fn_y = float(fn[1])

        # ---- cellA stencil (Item1) ----------------------------------------
        if cellA >= 0:
            cw_a, ccw_a = _cw_ccw_neighbors(fidx, cellA, cell_face_info, cell_face_values)
            s = _FaceVelocityCoef()
            if cw_a >= 0:
                s.add_face_normal(float(face_normals_2d[cw_a, 0]), float(face_normals_2d[cw_a, 1]))
            if ccw_a >= 0:
                s.add_face_normal(float(face_normals_2d[ccw_a, 0]), float(face_normals_2d[ccw_a, 1]))
            s.add_face_normal(fn_x, fn_y)
            s.complete()
            vx, vy = _solve_face_vector(
                s, fn_x, fn_y, fv,
                face_normals_2d[cw_a]  if cw_a  >= 0 else _zeros2,
                float(face_normal_vel[cw_a])  if cw_a  >= 0 else 0.0,
                bool(face_connected[cw_a])    if cw_a  >= 0 else False,
                face_normals_2d[ccw_a] if ccw_a >= 0 else _zeros2,
                float(face_normal_vel[ccw_a]) if ccw_a >= 0 else 0.0,
                bool(face_connected[ccw_a])   if ccw_a >= 0 else False,
            )
            face_vel_A[fidx, 0] = vx
            face_vel_A[fidx, 1] = vy
        else:
            # Boundary: degenerate stencil
            face_vel_A[fidx, 0] = fv * fn_x
            face_vel_A[fidx, 1] = fv * fn_y

        # ---- cellB stencil (Item2) ----------------------------------------
        if cellB >= 0:
            cw_b, ccw_b = _cw_ccw_neighbors(fidx, cellB, cell_face_info, cell_face_values)
            s = _FaceVelocityCoef()
            if cw_b >= 0:
                s.add_face_normal(float(face_normals_2d[cw_b, 0]), float(face_normals_2d[cw_b, 1]))
            if ccw_b >= 0:
                s.add_face_normal(float(face_normals_2d[ccw_b, 0]), float(face_normals_2d[ccw_b, 1]))
            s.add_face_normal(fn_x, fn_y)
            s.complete()
            vx, vy = _solve_face_vector(
                s, fn_x, fn_y, fv,
                face_normals_2d[cw_b]  if cw_b  >= 0 else _zeros2,
                float(face_normal_vel[cw_b])  if cw_b  >= 0 else 0.0,
                bool(face_connected[cw_b])    if cw_b  >= 0 else False,
                face_normals_2d[ccw_b] if ccw_b >= 0 else _zeros2,
                float(face_normal_vel[ccw_b]) if ccw_b >= 0 else 0.0,
                bool(face_connected[ccw_b])   if ccw_b >= 0 else False,
            )
            face_vel_B[fidx, 0] = vx
            face_vel_B[fidx, 1] = vy
        else:
            face_vel_B[fidx, 0] = fv * fn_x
            face_vel_B[fidx, 1] = fv * fn_y

        # Connected faces: average Item1 and Item2 (C# MeshFV2D.cs)
        if bool(face_connected[fidx]) and cellA >= 0 and cellB >= 0:
            avg_x = (face_vel_A[fidx, 0] + face_vel_B[fidx, 0]) / 2.0
            avg_y = (face_vel_A[fidx, 1] + face_vel_B[fidx, 1]) / 2.0
            face_vel_A[fidx, 0] = avg_x;  face_vel_A[fidx, 1] = avg_y
            face_vel_B[fidx, 0] = avg_x;  face_vel_B[fidx, 1] = avg_y

    return face_vel_A, face_vel_B


# ---------------------------------------------------------------------------
# Step 3 — Inverse-face-length weighted facepoint velocity averaging
# ---------------------------------------------------------------------------


def _connected_arc(
    local_start: int,
    fp_faces: list[tuple[int, int]],
    face_connected: np.ndarray,
) -> tuple[int, int]:
    """Find the hydraulically-connected arc starting at *local_start*.

    Returns ``(arc_start, arc_end)`` as inclusive/exclusive local indices into
    the angle-sorted ``fp_faces`` list for this facepoint.
    """
    n = len(fp_faces)
    if n == 0:
        return local_start, local_start

    # Walk forward to find arc end (first disconnected face after arc)
    end_idx = (local_start + 1) % n
    while end_idx != local_start:
        if not face_connected[fp_faces[end_idx][0]]:
            break
        end_idx = (end_idx + 1) % n

    # Walk backward from start to find actual arc start
    start_idx = local_start
    if start_idx != end_idx:
        while face_connected[fp_faces[start_idx][0]]:
            start_idx = (start_idx - 1 + n) % n
            if face_connected[fp_faces[start_idx][0]] is False:
                break

    return start_idx, end_idx


def compute_facepoint_velocities(
    face_vel_A: np.ndarray,
    face_vel_B: np.ndarray,
    face_connected: np.ndarray,
    face_lengths: np.ndarray,
    face_facepoint_indexes: np.ndarray,
    face_cell_indexes: np.ndarray,
    cell_wse: np.ndarray,
    fp_face_info: np.ndarray,
    fp_face_values: np.ndarray,
    face_value_a: np.ndarray,
    face_value_b: np.ndarray,
) -> tuple[list[np.ndarray], dict[tuple[int, int], int]]:
    """Step 3 — inverse-face-length weighted facepoint velocity averaging.

    Replicates ``ComputeVertexVelocities`` from
    ``archive/DLLs/RasMapperLib/MeshFV2D.cs``.

    **Goal**

    Produce a velocity vector at each facepoint for use in barycentric
    interpolation (Step 4).  Because a facepoint sits at the junction of
    multiple faces, its velocity depends on which *connected arc* of wet
    faces surrounds it — giving a different result depending on which
    adjacent face's arc context is being evaluated.  The function therefore
    returns **one vector per adjacent face**, not a single vector per
    facepoint.

    **Algorithm (per facepoint ``fp``, per adjacent face ``j``)**

    1. *Connected arc* — starting from face ``j`` in the angle-sorted face
       ring of ``fp``, walk forward through consecutive hydraulically
       connected faces (``face_connected[fi] == True``) to find the arc
       end, then walk backward to find the true arc start
       (:func:`_connected_arc`).  The arc is the maximal contiguous run of
       wet faces that includes face ``j``.

    2. *Weighted sum* — accumulate an inverse-face-length weighted velocity
       over all faces in the arc::

           sum_vx += vel[0] / face_length[fi]
           sum_vy += vel[1] / face_length[fi]
           total_w += 1 / face_length[fi]

       Shorter faces get higher weight (they represent sharper local
       geometry and should dominate the velocity estimate at the corner).

    3. *Velocity selection* — which of Item1/Item2 (``face_vel_A`` /
       ``face_vel_B``) to use for each face in the arc:

       - **Arc-start face**: use Item1 (``face_vel_A``) if ``fp`` is fpA of
         that face, otherwise Item2 (``face_vel_B``).  This picks the
         reconstruction that was done from the cell whose boundary this
         facepoint lies on.
       - **Interior connected faces**: always use Item1 (``face_vel_A``).
       - **Arc-end boundary face** (the first disconnected face past the
         arc): always use the *opposite* selection from the start face —
         Item2 if the start used Item1, and vice versa.  This boundary face
         is included in the weighted sum to anchor the average at the wet/dry
         interface.

    4. *Normalise*::

           vel_j = (sum_vx / total_w, sum_vy / total_w)

       All accumulators are **float32** to exactly match
       ``MeshFV2D.cs:8677`` (``float num5=0f, num6=0f, num7=0f``).

    5. *Dry facepoints* — if all adjacent face WSE values are ``_NODATA``
       the facepoint is considered fully dry and all velocity vectors are
       set to ``(0, 0)``.

    Parameters
    ----------
    face_vel_A, face_vel_B:
        ``(n_faces, 2)`` — from :func:`reconstruct_face_velocities`.
    face_connected:
        ``(n_faces,)`` bool — from :func:`compute_face_wss`.
    face_lengths:
        ``(n_faces,)`` — face plan-view lengths.
        Use ``face_normals[:, 2]`` from :attr:`~raspy.hdf.FlowArea.face_normals`.
    face_facepoint_indexes:
        ``(n_faces, 2)`` — ``[fpA, fpB]``.
    face_cell_indexes:
        ``(n_faces, 2)`` — ``[cellA, cellB]``.
    cell_wse:
        ``(n_cells,)`` — water-surface elevation per cell.
    fp_face_info, fp_face_values:
        Angle-sorted facepoint-to-face CSR arrays from
        :attr:`~raspy.hdf.FlowArea.facepoint_face_orientation`.
        The angular order is required here so that arc traversal visits
        faces in consistent counter-clockwise order.
    face_value_a, face_value_b:
        ``(n_faces,)`` — from :func:`compute_face_wss`.  Used only for
        the dry-facepoint check.

    Returns
    -------
    fp_velocities : list of ndarray, length ``n_fp``
        ``fp_velocities[fp]`` is shape ``(n_adj_faces, 2)`` — one
        ``[Vx, Vy]`` vector per adjacent face of facepoint ``fp``,
        in the same order as the angle-sorted face ring.
    fp_face_local_map : dict ``(fp_idx, face_idx) -> local_j``
        Maps a ``(facepoint, face)`` pair to the row index ``local_j``
        in ``fp_velocities[fp]``.  Used by :func:`replace_face_velocities_sloped`
        and Step 4 to look up the arc-context velocity for a given face.
    """
    n_fp = len(fp_face_info)
    face_inv_lengths = 1.0 / np.maximum(face_lengths, 1e-12)

    fp_velocities: list[np.ndarray] = []
    fp_face_local_map: dict[tuple[int, int], int] = {}

    for fp in range(n_fp):
        fp_start = int(fp_face_info[fp, 0])
        fp_count = int(fp_face_info[fp, 1])

        # Build local face list: list of (face_idx, orientation)
        fp_faces: list[tuple[int, int]] = [
            (int(fp_face_values[fp_start + j, 0]), int(fp_face_values[fp_start + j, 1]))
            for j in range(fp_count)
        ]

        # Register local index map for Step 3.5 and Step 4 lookups
        for j, (fi, _) in enumerate(fp_faces):
            fp_face_local_map[(fp, fi)] = j

        if fp_count == 0:
            fp_velocities.append(np.zeros((0, 2), dtype=np.float64))
            continue

        # Dry check: skip if all adjacent cells have no wet WSE
        all_dry = True
        for fi, _ in fp_faces:
            va = float(face_value_a[fi])
            vb = float(face_value_b[fi])
            if va != _NODATA or vb != _NODATA:
                all_dry = False
                break

        if all_dry:
            fp_velocities.append(np.zeros((fp_count, 2), dtype=np.float64))
            continue

        vels = np.zeros((fp_count, 2), dtype=np.float64)

        for j in range(fp_count):
            arc_start, arc_end = _connected_arc(j, fp_faces, face_connected)

            # float32 accumulators — critical to match C# MeshFV2D.cs:8677
            sum_vx = np.float32(0.0)
            sum_vy = np.float32(0.0)
            total_w = np.float32(0.0)

            current = arc_start
            while True:
                fi = fp_faces[current][0]
                inv_len = np.float32(face_inv_lengths[fi])

                if current == arc_start:
                    # Start face: Item1 if fpA == fp, else Item2
                    fpA = int(face_facepoint_indexes[fi, 0])
                    vel = face_vel_A[fi] if fpA == fp else face_vel_B[fi]
                else:
                    # Interior connected faces: always Item1
                    vel = face_vel_A[fi]

                sum_vx += np.float32(vel[0]) * inv_len
                sum_vy += np.float32(vel[1]) * inv_len
                total_w += inv_len

                current = (current + 1) % fp_count
                if current == arc_end:
                    break

            # Terminal disconnected face (arc end boundary)
            if arc_end != arc_start or not face_connected[fp_faces[arc_end][0]]:
                end_fi = fp_faces[arc_end][0]
                if not face_connected[end_fi]:
                    inv_len = np.float32(face_inv_lengths[end_fi])
                    fpA = int(face_facepoint_indexes[end_fi, 0])
                    # Opposite selection from start face
                    vel = face_vel_B[end_fi] if fpA == fp else face_vel_A[end_fi]
                    sum_vx += np.float32(vel[0]) * inv_len
                    sum_vy += np.float32(vel[1]) * inv_len
                    total_w += inv_len

            if float(total_w) > 1e-12:
                # Explicit float32 division — matches C# MeshFV2D.cs result type
                vels[j, 0] = float(np.float32(sum_vx / total_w))
                vels[j, 1] = float(np.float32(sum_vy / total_w))

        fp_velocities.append(vels)

    return fp_velocities, fp_face_local_map


# ---------------------------------------------------------------------------
# Step 3.5 — Sloped face velocity replacement
# ---------------------------------------------------------------------------


def replace_face_velocities_sloped(
    fp_velocities: list[np.ndarray],
    fp_face_local_map: dict[tuple[int, int], int],
    face_facepoint_indexes: np.ndarray,
) -> np.ndarray:
    """Step 3.5 — replace face velocity with average of endpoint facepoint velocities.

    Replicates ``ReplaceFaceVelocitiesSloped`` from
    ``archive/DLLs/RasMapperLib/MeshFV2D.cs``.

    **Context**

    Step 3 (:func:`compute_facepoint_velocities`) produces, for every
    facepoint ``fp``, an array ``fp_velocities[fp]`` of one velocity vector
    per adjacent face.  These are *arc-context* velocities: each entry
    reflects the weighted average of the connected-face arc that includes
    that particular adjacent face, so the velocity at a facepoint can vary
    depending on which face's arc is being considered.

    **What this step does**

    For every face ``f`` with endpoint facepoints ``fpA`` and ``fpB``:

    1. Look up the arc-context velocity that Step 3 assigned to ``fpA``
       while processing face ``f``:
       ``vel_A = fp_velocities[fpA][fp_face_local_map[(fpA, f)]]``
    2. Do the same for ``fpB``:
       ``vel_B = fp_velocities[fpB][fp_face_local_map[(fpB, f)]]``
    3. Replace the face velocity with the component-wise average:
       ``replaced[f] = (vel_A + vel_B) / 2``

    If either facepoint has no entry for face ``f`` in the map (e.g. a
    boundary facepoint not covered by the arc), the missing velocity
    defaults to ``(0, 0)``.

    **Why**

    The sloped-cell render mode needs a velocity that is spatially smooth
    across each cell polygon, not just at face midpoints.  By averaging the
    two endpoint facepoint velocities, this step produces a face-centre
    value that is consistent with the corner values used for barycentric
    interpolation in Step 4.  In Step 4 this ``replaced_face_vel`` is used
    for face-midpoint sample contributions; corner (facepoint) contributions
    use ``fp_velocities`` directly.

    Parameters
    ----------
    fp_velocities:
        ``list[ndarray]``, length ``n_fp`` — from
        :func:`compute_facepoint_velocities`.  ``fp_velocities[fp]`` has
        shape ``(n_adj_faces, 2)``, one row per adjacent face of that
        facepoint.
    fp_face_local_map:
        ``dict[(fp_idx, face_idx) -> local_j]`` — from
        :func:`compute_facepoint_velocities`.  Maps a ``(facepoint, face)``
        pair to the row index in ``fp_velocities[fp]``.
    face_facepoint_indexes:
        ``(n_faces, 2)`` — ``[fpA, fpB]`` endpoint facepoint indices for
        each face.

    Returns
    -------
    replaced_face_vel : float64 ndarray, shape ``(n_faces, 2)``
        ``[Vx, Vy]`` sloped replacement velocity for each face.
    """
    n_faces = len(face_facepoint_indexes)
    replaced = np.zeros((n_faces, 2), dtype=np.float64)
    for f in range(n_faces):
        fpA = int(face_facepoint_indexes[f, 0])
        fpB = int(face_facepoint_indexes[f, 1])
        key_A = (fpA, f)
        key_B = (fpB, f)
        vel_A = fp_velocities[fpA][fp_face_local_map[key_A]] if key_A in fp_face_local_map else np.zeros(2)
        vel_B = fp_velocities[fpB][fp_face_local_map[key_B]] if key_B in fp_face_local_map else np.zeros(2)
        replaced[f, 0] = (vel_A[0] + vel_B[0]) / 2.0
        replaced[f, 1] = (vel_A[1] + vel_B[1]) / 2.0
    return replaced


# ---------------------------------------------------------------------------
# Step 4 helpers — barycentric weights, donate, WSE + velocity interpolation
# ---------------------------------------------------------------------------


def _barycentric_weights(px: float, py: float, verts_x: np.ndarray, verts_y: np.ndarray) -> np.ndarray:
    """Generalised polygon barycentric coordinates, cast to float32.

    Matches ``archive/DLLs/RasMapperLib/RASGeometryMapPoints.cs:2956``
    ``fpWeights[l] = (float)(array[l] / num3)``.
    """
    N = len(verts_x)
    if N < 3:
        return np.ones(N, dtype=np.float32) / max(N, 1)

    xp = np.empty(N, dtype=np.float64)
    for i in range(N):
        ax = verts_x[i];  ay = verts_y[i]
        bx = verts_x[(i + 1) % N];  by = verts_y[(i + 1) % N]
        xp[i] = (ax - px) * (by - py) - (ay - py) * (bx - px)
    xp[xp == 0.0] = 1e-5

    weights = np.empty(N, dtype=np.float64)
    for j in range(N):
        product = 1.0
        prev_j = (j - 1 + N) % N
        for k in range(N):
            if k != j and k != prev_j:
                product *= xp[k]
        p1x = verts_x[prev_j];  p1y = verts_y[prev_j]
        p2x = verts_x[j];       p2y = verts_y[j]
        p3x = verts_x[(j + 1) % N]; p3y = verts_y[(j + 1) % N]
        cp = (p2x - p1x) * (p3y - p1y) - (p2y - p1y) * (p3x - p1x)
        weights[j] = product * cp

    total = weights.sum()
    if abs(total) > 1e-20:
        weights /= total
    weights[weights < 0.0] = 0.0
    total = weights.sum()
    if total > 1e-20:
        weights /= total

    return weights.astype(np.float32)


def _donate(fp_weights: np.ndarray) -> np.ndarray:
    """Redistribute facepoint weights to edge midpoints.

    Returns ``velocity_weights`` of length ``2*N``.  First N entries are
    updated facepoint weights; last N entries are face-midpoint weights.

    Matches ``redistribute_weights_to_edge_midpoints`` /
    ``archive/DLLs/RasMapperLib/RASGeometryMapPoints.cs`` donate logic.
    """
    N = len(fp_weights)
    w = np.zeros(2 * N, dtype=np.float64)
    w[:N] = fp_weights.copy()

    cw_give  = np.zeros(N, dtype=np.float64)
    ccw_give = np.zeros(N, dtype=np.float64)
    for i in range(N):
        prev_w = float(fp_weights[(i - 1 + N) % N])
        next_w = float(fp_weights[(i + 1) % N])
        denom  = prev_w + next_w
        if denom > 1e-20:
            ratio = float(fp_weights[i]) / denom
            cw_give[i]  = ratio * prev_w
            ccw_give[i] = ratio * next_w

    for j in range(N):
        nxt = (j + 1) % N
        can_donate = min(ccw_give[j], cw_give[nxt])
        w[j]     -= can_donate
        w[nxt]   -= can_donate
        w[N + j]  = can_donate * 2.0

    return w


def _downward_adjust_fp_wse(cell_wse: float, fp_local_wse: np.ndarray) -> np.ndarray:
    """Drag facepoint WSE values toward cell-average WSE.

    C# ``Renderer.cs:3267 DownwardAdjustFPValues``.
    ``correction = (cell_wse - mean(fp_wse)) / N`` added to each fp.
    """
    n = len(fp_local_wse)
    if n == 0:
        return fp_local_wse.copy()
    avg = fp_local_wse.sum() / n
    return fp_local_wse + (cell_wse - avg) / n


def _depth_weights_for_cell(
    cell_idx: int,
    cell_face_info: np.ndarray,
    cell_face_values: np.ndarray,
    face_facepoint_indexes: np.ndarray,
    face_cell_indexes: np.ndarray,
    fp_wse_at_face: np.ndarray,
    fp_elev: np.ndarray,
    face_value_a: np.ndarray,
    face_value_b: np.ndarray,
    face_min_elev: np.ndarray,
) -> np.ndarray:
    """Compute per-sample depth weights for the 2N-point stencil of one cell.

    Used by ``PaintCell_8Stencil_RebalanceWeights`` to bias the barycentric
    blend toward deeper (wetter) sample points.  Shallow samples near the
    wet/dry fringe receive less weight, so they cannot pull the interpolated
    WSE below the surrounding terrain and create spurious dry pixels.

    The stencil has ``2 * count`` samples, where ``count`` is the number of
    faces (= polygon corners) of the cell:

    * Indices ``0 .. count-1`` — one **corner facepoint** per face, selected
      by orientation (``fpA`` for cellA, ``fpB`` for cellB).  The depth
      weight is ``WSE_facepoint − elevation_facepoint``, clamped to 0.01.
    * Indices ``count .. 2*count-1`` — one **face midpoint** per face.  The
      depth weight is ``WSE_face − minimum_elevation_face``, clamped to 0.01.

    All weights are initialised to the minimum value 0.01 so that every
    sample contributes at least a token weight even when completely dry
    (matches the C# minimum depth floor).

    The returned array is passed directly to
    :func:`_pixel_wse_sloped` as ``depth_weights``, where it element-wise
    multiplies the donated barycentric weights before normalisation.

    C# reference: ``WaterSurfaceRenderer.cs:2842
    ComputeDepthWeightedValuesPerCell``.

    Parameters
    ----------
    cell_idx:
        0-based index of the cell being processed.
    cell_face_info:
        ``(n_cells, 2)`` CSR offsets — ``[start, count]`` into
        ``cell_face_values`` for each cell.
    cell_face_values:
        ``(total, 2)`` CSR data — ``[face_index, orientation]`` for each
        cell–face entry.
    face_facepoint_indexes:
        ``(n_faces, 2)`` — ``[fpA, fpB]`` endpoint facepoints of each face.
    face_cell_indexes:
        ``(n_faces, 2)`` — ``[cellA, cellB]`` for each face; used to select
        the correct side (``face_value_a`` vs ``face_value_b``).
    fp_wse_at_face:
        ``(n_faces, 2)`` — arc-based WSE from :func:`compute_facepoint_wse`.
        Column 0 is the fpA arc value; column 1 is the fpB arc value.
    fp_elev:
        ``(n_fp,)`` — terrain elevation at each facepoint.
    face_value_a, face_value_b:
        ``(n_faces,)`` — face-side WSE for cellA and cellB respectively.
    face_min_elev:
        ``(n_faces,)`` — minimum bed elevation along each face.

    Returns
    -------
    dw : float64 ndarray, shape ``(2 * count,)``
        Depth weights, minimum 0.01.  First ``count`` entries are facepoint
        depths; last ``count`` entries are face-midpoint depths.
    """
    start = int(cell_face_info[cell_idx, 0])
    count = int(cell_face_info[cell_idx, 1])
    dw = np.full(2 * count, 0.01, dtype=np.float64)

    for k in range(count):
        fi  = int(cell_face_values[start + k, 0])
        ori = int(cell_face_values[start + k, 1])
        # Facepoint for this face edge (orientation-based selection)
        fp  = int(face_facepoint_indexes[fi, 0]) if ori > 0 else int(face_facepoint_indexes[fi, 1])

        # Facepoint depth weight: look up from the arc-based table.
        # ori > 0 → this cell is cellA → facepoint is fpA → column 0.
        # ori < 0 → this cell is cellB → facepoint is fpB → column 1.
        side = 0 if ori > 0 else 1
        fp_w = float(fp_wse_at_face[fi, side])
        fp_e = float(fp_elev[fp]) if fp < len(fp_elev) else _NODATA
        if fp_w != _NODATA and fp_e != _NODATA and fp_w > fp_e:
            dw[k] = max(0.01, fp_w - fp_e)

        # Face midpoint depth weight
        cellA = int(face_cell_indexes[fi, 0])
        face_ws = float(face_value_a[fi]) if cell_idx == cellA else float(face_value_b[fi])
        f_min_e = float(face_min_elev[fi])
        if f_min_e != _NODATA and face_ws != _NODATA and face_ws > f_min_e:
            dw[count + k] = max(0.01, face_ws - f_min_e)

    return dw


def _all_shallow(
    cell_idx: int,
    cell_face_info: np.ndarray,
    cell_face_values: np.ndarray,
    face_connected: np.ndarray,
) -> bool:
    """Return True when ALL faces of this cell are disconnected (shallow).

    C# ``Renderer.cs:3079-3097``.
    """
    start = int(cell_face_info[cell_idx, 0])
    count = int(cell_face_info[cell_idx, 1])
    for k in range(count):
        fi = int(cell_face_values[start + k, 0])
        if face_connected[fi]:
            return False
    return True


def _pixel_wse_sloped(
    vel_weights: np.ndarray,
    fp_local_wse: np.ndarray,
    face_local_wse: np.ndarray,
    depth_weights: np.ndarray | None,
) -> float:
    """Interpolate WSE at a pixel using donated barycentric weights.

    C# ``Renderer.cs:3139 PaintCell_8Stencil`` or
    ``Renderer.cs:3166 PaintCell_8Stencil_RebalanceWeights``.
    """
    N = len(fp_local_wse)
    if N == 0:
        return _NODATA
    base_val = float(face_local_wse[0])

    if depth_weights is not None:
        eff_w = vel_weights * depth_weights
        w_sum = float(eff_w.sum())
        if w_sum < 1e-20:
            return _NODATA
        result = 0.0
        for i in range(N):
            result += (float(fp_local_wse[i]) - base_val) * eff_w[i]
        for j in range(N):
            result += (float(face_local_wse[j]) - base_val) * eff_w[N + j]
        return result / w_sum + base_val
    else:
        result = 0.0
        for i in range(N):
            result += (float(fp_local_wse[i]) - base_val) * vel_weights[i]
        for j in range(N):
            result += (float(face_local_wse[j]) - base_val) * vel_weights[N + j]
        return result + base_val


# ---------------------------------------------------------------------------
# Step 4 — Build cell-ID raster
# ---------------------------------------------------------------------------


def build_cell_id_raster(
    cell_polygons: list[np.ndarray],
    wet_mask: np.ndarray,
    transform: "rasterio.transform.Affine",
    height: int,
    width: int,
) -> np.ndarray:
    """Rasterize cell ownership: pixel value = cell_idx + 1, 0 = outside.

    Only wet cells (``wet_mask[c] == True``) are rasterized.

    Mirrors ``RasterizePolygon.ComputeCells``
    (``archive/DLLs/RasMapperLib/RasterizePolygon.cs``) called from
    ``MeshFV2D.cs``.

    **Differences from RasMapperLib**

    * *Algorithm:* RasMapperLib uses a custom scan-line fill that walks polygon
      edges and collects X-intersections at each row's cell-center Y, then fills
      between sorted intersection pairs.  This implementation delegates to
      ``rasterio.features.rasterize`` (GDAL center-point test,
      ``all_touched=False``).  Both apply the same owning rule — a pixel is
      owned if its center falls inside the polygon — so results agree for the
      vast majority of pixels.
    * *Degenerate vertices:* RasMapperLib has explicit
      ``IsVertexVerticallyBetweenNeighbors`` logic to handle the case where a
      polygon vertex lies exactly on a row's center-Y line, preventing
      double-counting at those edges.  GDAL's behavior at such positions may
      differ slightly.
    * *Invalid polygons:* this implementation calls ``Polygon.is_valid`` via
      Shapely and skips degenerate geometries; RasMapperLib has no such check.

    Parameters
    ----------
    cell_polygons:
        From :attr:`~raspy.hdf.FlowArea.cell_polygons`.  Each element is
        ``(n_vertices, 2)`` XY coordinates.  Empty arrays are skipped.
    wet_mask:
        ``(n_cells,)`` bool — which cells have water.
    transform:
        Rasterio affine transform for the output grid.
    height, width:
        Output raster dimensions.

    Returns
    -------
    cell_id_grid : int32 ndarray, shape ``(height, width)``
        ``cell_id_grid[r, c] = cell_idx + 1`` for the owning cell (1-based),
        or 0 where no cell covers that pixel.
    """
    import rasterio.features
    from shapely.geometry import Polygon as _Polygon
    from shapely.ops import polygonize as _polygonize

    shapes: list[tuple] = []
    for ci in range(len(cell_polygons)):
        if not wet_mask[ci]:
            continue
        pts = cell_polygons[ci]
        if len(pts) < 3:
            continue
        poly = _Polygon(pts)
        if not poly.is_valid or poly.is_empty:
            continue
        shapes.append((poly, ci + 1))  # 1-based so 0 = no cell

    cell_id = rasterio.features.rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.int32,
        all_touched=False,
    )
    return cell_id


# ---------------------------------------------------------------------------
# Terrain sampling at facepoints
# ---------------------------------------------------------------------------


def sample_terrain_at_facepoints(
    fp_coords: np.ndarray,
    terrain_grid: np.ndarray,
    transform: "rasterio.transform.Affine",
) -> np.ndarray:
    """Bilinear-sample terrain elevation at each facepoint coordinate.

    Replicates the per-facepoint terrain lookup used by RasMapperLib when
    computing depth-weighted stencil weights (``UseDepthWeightedFaces``).
    In HEC-RAS this data is pre-computed and cached in ``PostProcessing.hdf``
    as ``FacePoint Elevation``; here we derive it on-the-fly from the terrain
    raster so no separate file is required.

    Parameters
    ----------
    fp_coords:
        ``(n_fp, 2)`` facepoint XY coordinates in the raster CRS.
    terrain_grid:
        ``(H, W)`` terrain elevation array (float64, NaN where nodata).
    transform:
        Rasterio affine transform for *terrain_grid*.

    Returns
    -------
    fp_elev : ndarray, shape ``(n_fp,)``
        Terrain elevation at each facepoint.  Facepoints outside the raster
        extent or at NaN terrain pixels receive ``NaN``.
    """
    H, W = terrain_grid.shape
    n_fp = len(fp_coords)
    fp_elev = np.full(n_fp, np.nan, dtype=np.float64)

    # Rasterio affine: x = c + a*col + e*row, y = f + b*col + d*row
    # For north-up rasters: a > 0, d < 0, b=e=0
    a, c = transform.a, transform.c
    d, f = transform.e, transform.f  # note: rasterio uses .e for the y-scale

    xs = fp_coords[:, 0]
    ys = fp_coords[:, 1]

    # Fractional pixel coordinates (0-based, pixel-centre at 0.5)
    cols_f = (xs - c) / a - 0.5
    rows_f = (ys - f) / d - 0.5

    for i in range(n_fp):
        c0 = int(np.floor(cols_f[i]))
        r0 = int(np.floor(rows_f[i]))
        c1 = c0 + 1
        r1 = r0 + 1
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            continue
        dc = cols_f[i] - c0
        dr = rows_f[i] - r0
        v00 = terrain_grid[r0, c0]
        v01 = terrain_grid[r0, c1]
        v10 = terrain_grid[r1, c0]
        v11 = terrain_grid[r1, c1]
        if np.isnan(v00) or np.isnan(v01) or np.isnan(v10) or np.isnan(v11):
            continue
        fp_elev[i] = (
            v00 * (1 - dr) * (1 - dc)
            + v01 * (1 - dr) * dc
            + v10 * dr * (1 - dc)
            + v11 * dr * dc
        )

    return fp_elev


# ---------------------------------------------------------------------------
# Step 4 — Main pixel rasterization loop
# ---------------------------------------------------------------------------


def rasterize_rasmap(
    variable: str,
    cell_id_grid: np.ndarray,
    transform: "rasterio.transform.Affine",
    terrain_grid: np.ndarray | None,
    cell_wse: np.ndarray,
    cell_face_info: np.ndarray,
    cell_face_values: np.ndarray,
    face_facepoint_indexes: np.ndarray,
    face_cell_indexes: np.ndarray,
    face_min_elev: np.ndarray,
    fp_coords: np.ndarray,
    fp_wse: np.ndarray | None,
    face_value_a: np.ndarray,
    face_value_b: np.ndarray,
    fp_velocities: list[np.ndarray] | None,
    fp_face_local_map: dict[tuple[int, int], int] | None,
    replaced_face_vel: np.ndarray | None,
    face_vel_A: np.ndarray | None,
    face_vel_B: np.ndarray | None,
    fp_elev: np.ndarray | None,
    face_connected: np.ndarray,
    nodata: float,
    depth_threshold: float = _MIN_WS_PLOT_TOLERANCE,
    with_faces: bool = True,
    use_depth_weights: bool = False,
    shallow_to_flat: bool = False,
) -> np.ndarray:
    """Step 4 — pixel-level barycentric interpolation for all wet cells.

    Replicates ``PaintCell_8Stencil`` / ``PaintCell_8Stencil_RebalanceWeights``
    from ``archive/DLLs/RasMapperLib/Renderer.cs``.

    **Algorithm**

    *Setup*

    Collect all pixels with a valid owning cell from ``cell_id_grid`` and
    convert their row/col positions to model XY coordinates.  Group pixels
    by owning cell so each cell's geometry is built only once.

    *Per-cell geometry (done once per cell)*

    For each wet cell, walk its ordered face ring (``cell_face_info`` /
    ``cell_face_values``) to collect the cell's polygon vertices.  Each
    face contributes one corner facepoint, selected by orientation:
    ``fpA`` if ``orientation > 0``, else ``fpB``.  This gives an ordered
    polygon of ``N`` facepoints (``N`` = number of faces = number of
    corners).

    *Per-cell value arrays (done once per cell)*

    For sloping WSE / depth, two value arrays of length ``N`` are built:

    - ``fp_local_wse[i]`` — facepoint WSE from :func:`compute_facepoint_wse`,
      one per polygon corner.
    - ``face_local_wse[i]`` — face-side WSE from :func:`compute_face_wss`,
      selecting ``face_value_a`` or ``face_value_b`` based on whether this
      cell is cellA or cellB of that face.

    ``fp_local_wse`` is then adjusted toward the cell-average WSE
    (``DownwardAdjustFPValues``, ``Renderer.cs:3267``) so corner values do
    not drift above the cell mean.

    If terrain elevations at facepoints are available (``fp_elev``), depth
    weights are computed per facepoint and face midpoint — deeper locations
    get higher weight in the final interpolation
    (``ComputeDepthWeightedValuesPerCell``, ``WaterSurfaceRenderer.cs:2842``).

    For speed / velocity, facepoint velocity arrays (``nb_fp_vx``,
    ``nb_fp_vy``) and face-midpoint velocity arrays (``nb_face_vx``,
    ``nb_face_vy``) of length ``N`` are built from ``replaced_face_vel``
    and ``face_vel_A`` / ``face_vel_B``.

    *Per-pixel interpolation (inner loop)*

    For every pixel ``(px, py)`` owned by the cell:

    1. **Barycentric weights** — compute generalised polygon barycentric
       weights ``fw[0..N-1]`` for the pixel position relative to the ``N``
       polygon corners (:func:`_barycentric_weights`,
       ``RASGeometryMapPoints.cs:2956``).

    2. **Donate redistribution** — redistribute weight from each corner
       toward the midpoints of its two adjacent edges (:func:`_donate`).
       This produces a ``2*N`` weight vector ``vel_w``:

       - ``vel_w[0..N-1]`` — updated corner (facepoint) weights.
       - ``vel_w[N..2N-1]`` — face-midpoint weights (one per face edge).

       Using both corner and midpoint values gives an 8-sample stencil
       per cell (for ``N=4``: 4 corners + 4 midpoints = 8 samples).

    3. **Wet/dry check** — for sloping mode, the pixel WSE is first
       estimated via :func:`_pixel_wse_sloped` and compared against the
       terrain elevation.  If ``pixel_wse < terrain + depth_threshold``
       the pixel is marked dry and skipped.  Without a terrain grid the
       cell-centre WSE is used for the check instead.

    4. **Value interpolation** — depends on ``variable``:

       - ``"water_surface"`` — :func:`_pixel_wse_sloped` with
         depth-weighted rebalancing if ``fp_elev`` is available, otherwise
         plain donated barycentric blend of ``fp_local_wse`` and
         ``face_local_wse``.
       - ``"depth"`` — same WSE interpolation then subtract terrain
         elevation; clamped to zero from below.
       - ``"speed"`` — donated barycentric blend of facepoint and
         face-midpoint velocity vectors::

             Vx = sum(vel_w[i] * nb_fp_vx[i]) + sum(vel_w[N+j] * nb_face_vx[j])
             Vy = sum(vel_w[i] * nb_fp_vy[i]) + sum(vel_w[N+j] * nb_face_vy[j])

         Magnitude = ``sqrt(Vx**2 + Vy**2)``.  This is the internal name
         for what the public API calls ``"velocity"``; the translation
         ``"velocity"`` → ``"speed"`` is performed by :func:`rasmap_raster`
         before this function is called.

    Parameters
    ----------
    variable:
        ``"water_surface"``, ``"depth"``, or ``"speed"`` (velocity magnitude).
        ``"speed"`` is the internal name for what the public API exposes as
        ``"velocity"``; callers should never pass ``"velocity"`` directly.
    cell_id_grid:
        ``(H, W)`` int32 — from :func:`build_cell_id_raster`.
    transform:
        Rasterio affine transform (pixel center coords).
    terrain_grid:
        ``(H, W)`` float — terrain elevation; required for ``"depth"``;
        optional for ``"speed"`` / ``"water_surface"`` (``None`` skips
        per-pixel depth masking).
    cell_wse:
        ``(n_cells,)`` — water-surface elevation per cell.
    cell_face_info, cell_face_values:
        From :attr:`~raspy.hdf.FlowArea.cell_face_info`.
    face_facepoint_indexes:
        ``(n_faces, 2)``.
    face_cell_indexes:
        ``(n_faces, 2)``.
    face_min_elev:
        ``(n_faces,)``.
    fp_coords:
        ``(n_fp, 2)``.
    fp_wse:
        ``(n_faces, 2)`` from :func:`compute_facepoint_wse`; required for
        sloping.  Column 0 = fpA arc value; column 1 = fpB arc value.
    face_value_a, face_value_b:
        ``(n_faces,)`` from :func:`compute_face_wss`.
    fp_velocities, fp_face_local_map, replaced_face_vel:
        From Steps 3 / 3.5; required for ``"speed"`` (velocity magnitude).
    face_vel_A, face_vel_B:
        ``(n_faces, 2)`` from Step 2; required for ``"speed"``.
    fp_elev:
        ``(n_fp,)`` terrain elevation sampled at facepoints; enables
        depth-weighted rebalancing (``PaintCell_8Stencil_RebalanceWeights``).
        Pass ``None`` to skip rebalancing.
    face_connected:
        ``(n_faces,)`` bool — from :func:`compute_face_wss`.  Used by the
        all-shallow check (``shallow_to_flat``) to detect cells where every
        bounding face is hydraulically disconnected.  Must be the real
        connectivity array, not the ``face_value_a != _NODATA`` proxy (which
        misclassifies Logic 3/4 faces as connected).
    nodata:
        Fill value for dry / out-of-domain pixels.
    depth_threshold:
        Minimum depth for a pixel to be considered wet (default 0.001).
    with_faces:
        When ``True`` (default), the sloped interpolation uses a 2N-point
        stencil: N polygon corner facepoints **plus** N face-midpoint
        values, blended via donated barycentric weights.  Matches
        ``CellStencilMethod.WithFaces`` in ``WaterSurfaceRenderer.cs``.
        When ``False``, only the N corner facepoints are used (plain
        barycentric blend, no donation to face midpoints).  Matches
        ``CellStencilMethod.JustFacepoints``.  Has no effect when
        ``fp_wse`` is ``None`` (horizontal mode).
    use_depth_weights:
        When ``True``, each sample in the 2N-point stencil is weighted by
        its water depth before the barycentric blend, so deeper (wetter)
        locations dominate over shallow margins.  Matches
        ``UseDepthWeightedFaces`` / ``PaintCell_8Stencil_RebalanceWeights``
        in ``WaterSurfaceRenderer.cs``.  Requires ``fp_elev`` to be
        provided; if ``fp_elev`` is ``None`` depth weighting is silently
        skipped even when this flag is ``True``.  Default ``False``
        (matches RasMapper default).
    shallow_to_flat:
        When ``True``, cells whose every bounding face is hydraulically
        disconnected (all-shallow) are rendered with the flat cell-average
        WSE and no barycentric interpolation.  Matches
        ``ShallowBehavior.ReduceToHorizontal`` in ``Renderer.cs``.
        Default ``False`` (matches RasMapper default).

    Returns
    -------
    output : float32 ndarray, shape ``(H, W)``
        Interpolated values; ``nodata`` where dry or outside mesh.
        For ``"velocity"``: 4-band array ``(4, H, W)`` — ``[Vx, Vy, speed, direction_deg]``.
    """
    import rasterio.transform as _rt

    H, W = cell_id_grid.shape
    is_velocity_4band = variable == "velocity"

    # Allocate output: velocity uses 4 bands (Vx, Vy, speed, direction),
    # all other variables use a single band.  Initialise to nodata.
    if is_velocity_4band:
        output = np.full((4, H, W), nodata, dtype=np.float32)
    else:
        output = np.full((H, W), nodata, dtype=np.float32)

    # Extract the subset of pixels that belong to a mesh cell (cell_id > 0).
    # Pixels outside the mesh stay nodata and are never visited.
    valid_mask = cell_id_grid > 0
    valid_rows, valid_cols = np.where(valid_mask)
    if len(valid_rows) == 0:
        return output

    # Convert raster row/col to model XY coordinates (pixel-centre convention).
    xs, ys = _rt.xy(transform, valid_rows, valid_cols)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    cell_ids = cell_id_grid[valid_rows, valid_cols]  # 1-based

    # Group pixels by owning cell for batch processing
    pixel_groups: dict[int, list[int]] = defaultdict(list)
    for i, cid in enumerate(cell_ids):
        pixel_groups[int(cid)].append(i)

    for raster_id, pix_indices in pixel_groups.items():
        cell_idx = raster_id - 1  # back to 0-based

        # ---- Build ordered facepoint polygon for this cell ---------------
        # Read the CSR slice: face_indices are in CCW order around the cell;
        # face_orients encode whether this cell is cellA (+1) or cellB (-1)
        # for each face (see cell_face_info docstring for the full invariant).
        start = int(cell_face_info[cell_idx, 0])
        count = int(cell_face_info[cell_idx, 1])
        if count < 3:
            continue  # degenerate cell — skip

        face_indices = [int(cell_face_values[start + k, 0]) for k in range(count)]
        face_orients = [int(cell_face_values[start + k, 1]) for k in range(count)]

        # Pick the CCW-entry facepoint for each face.
        # RasMapper invariant (Face.cs / GetFPPrev): for cellA (ori=+1) the
        # face runs fpA→fpB in CCW order, so fpA is the entry point; for
        # cellB (ori=-1) it runs fpB→fpA, so fpB is the entry point.
        # Equivalent to Face.GetFPPrev(cellIdx) in RasMapperLib.
        verts_fp = [
            int(face_facepoint_indexes[fi, 0]) if ori > 0  # cellA → fpA
            else int(face_facepoint_indexes[fi, 1])          # cellB → fpB
            for fi, ori in zip(face_indices, face_orients)
        ]
        verts_x = np.array([float(fp_coords[fp, 0]) for fp in verts_fp], dtype=np.float64)
        verts_y = np.array([float(fp_coords[fp, 1]) for fp in verts_fp], dtype=np.float64)
        N = count

        # ---- Per-cell WSE arrays (for sloping WSE / depth) ---------------
        # Sloping interpolation needs one WSE value per polygon corner
        # (facepoint) and one per face midpoint.  Both arrays have length N
        # and are indexed in the same CCW order as verts_fp / face_indices.
        if fp_wse is not None:
            # Corner WSEs: arc-based facepoint WSE from Step B.
            # fp_wse shape is (n_faces, 2): col 0 = fpA arc, col 1 = fpB arc.
            # ori > 0 → this cell is cellA → corner facepoint is fpA → col 0.
            # ori < 0 → this cell is cellB → corner facepoint is fpB → col 1.
            fp_local_wse = np.array(
                [float(fp_wse[fi, 0 if ori > 0 else 1])
                 for fi, ori in zip(face_indices, face_orients)],
                dtype=np.float64,
            )
            if with_faces:
                # Face-midpoint WSEs: select the side of the face that belongs
                # to this cell.  face_value_a corresponds to cellA (column 0
                # of face_cell_indexes); face_value_b to cellB (column 1).
                face_local_wse = np.array([
                    float(face_value_a[fi]) if cell_idx == int(face_cell_indexes[fi, 0])
                    else float(face_value_b[fi])
                    for fi in face_indices
                ], dtype=np.float64)
                # Valid when at least one corner OR face-midpoint has a real WSE.
                has_valid_wse = (fp_local_wse != _NODATA).any() or (face_local_wse != _NODATA).any()
            else:
                # Corners-only mode (CellStencilMethod.JustFacepoints):
                # face midpoint values are not used; validity is based on
                # facepoint WSEs alone.
                face_local_wse = None
                has_valid_wse = (fp_local_wse != _NODATA).any()
            # Use sloped mode whenever at least one facepoint WSE is valid.
            # Per-pixel terrain masking is applied in the pixel branches
            # when terrain_grid is available; without terrain all cell
            # pixels pass the wet/dry check (cell_id_raster already
            # filtered dry cells via build_cell_id_raster).
            use_sloped = has_valid_wse
        else:
            fp_local_wse = None
            face_local_wse = None
            use_sloped = False

        # ShallowBehavior.ReduceToHorizontal (C# Renderer.cs:3099 / 2226):
        # A cell is "all-shallow" when every one of its bounding faces (the
        # shared edges between this cell and each of its neighbours) is
        # hydraulically disconnected (face WSE == _NODATA), meaning no
        # active hydraulic connection exists across any edge of the cell.
        # When shallow_to_flat=True, such cells are
        # rendered with the flat cell-average WSE: skipping
        # DownwardAdjustFPValues and the entire sloped paint path.  Nulling
        # fp_local_wse_adj and face_local_wse here makes the pixel loop fall
        # through to the flat branch (B/C) automatically, so no extra logic
        # is needed inside the loop.
        # When shallow_to_flat=False, the all-shallow condition is ignored and
        # sloped interpolation continues as normal (C# non-ReduceToHorizontal).
        if use_sloped and shallow_to_flat and _all_shallow(cell_idx, cell_face_info, cell_face_values, face_connected):
            use_sloped = False
            fp_local_wse_adj = None
            face_local_wse = None
        elif use_sloped:
            # FacepointAdjustmentMode = None in all C# render configurations
            # (SharedData.cs:1768, 1781, 1807, 1929) — DownwardAdjustFPValues
            # is never called.
            # Substitute NODATA facepoints with the cell's own WSE: boundary
            # facepoints (on mesh perimeter, adjacent only to boundary faces)
            # have no arc information so their best estimate is the cell WSE.
            _cws = float(cell_wse[cell_idx])
            fp_local_wse_adj = np.where(fp_local_wse == _NODATA, _cws, fp_local_wse)
            if face_local_wse is not None:
                face_local_wse = np.where(face_local_wse == _NODATA, _cws, face_local_wse)
        else:
            fp_local_wse_adj = fp_local_wse.copy() if fp_local_wse is not None else None

        # Depth weights (C# UseDepthWeightedFaces, WaterSurfaceRenderer.cs:2842)
        # Only computed when use_depth_weights=True (opt-in, default off).
        # Requires fp_elev; silently skipped if unavailable (cell_dw=None
        # causes _pixel_wse_sloped to use plain donated weights instead).
        cell_dw: np.ndarray | None = None
        if use_sloped and with_faces and use_depth_weights and fp_elev is not None:
            cell_dw = _depth_weights_for_cell(
                cell_idx, cell_face_info, cell_face_values,
                face_facepoint_indexes, face_cell_indexes,
                fp_wse, fp_elev, face_value_a, face_value_b, face_min_elev,
            )  # fp_wse is (n_faces, 2) from compute_facepoint_wse

        # ---- Per-cell velocity arrays -------------------------------------
        # Build two N-length velocity arrays for the 8-sample stencil:
        #   nb_fp_v*   — velocity at each polygon corner (facepoint)
        #   nb_face_v* — velocity at each face midpoint
        # These are parallel to verts_fp / face_indices (same CCW order).
        nb_fp_vx: np.ndarray | None = None
        nb_fp_vy: np.ndarray | None = None
        nb_face_vx: np.ndarray | None = None
        nb_face_vy: np.ndarray | None = None
        if variable in ("speed", "velocity") and fp_velocities is not None:
            # Corner (facepoint) velocities: use the arc-context facepoint
            # velocity from Step 3 — fp_velocities[fp_i][lj] is the weighted
            # average of C-stencil face velocities in the hydraulically
            # connected arc that includes face fi, viewed from facepoint fp_i.
            # This matches C# GetLocalFacepointValues → fpVelocityRing[fPPrev]
            # .Velocity[...].  Do NOT use replaced_face_vel (face-averaged
            # facepoint velocity) — that was wrong for the pixel stencil.
            nb_fp_vx = np.zeros(N, dtype=np.float64)
            nb_fp_vy = np.zeros(N, dtype=np.float64)
            for i in range(N):
                fp_i = verts_fp[i]
                fi   = face_indices[i]
                key  = (fp_i, fi)
                if key in fp_face_local_map:
                    lj = fp_face_local_map[key]
                    nb_fp_vx[i] = float(fp_velocities[fp_i][lj, 0])
                    nb_fp_vy[i] = float(fp_velocities[fp_i][lj, 1])
            # Face-midpoint velocities: select the cell-side value from the
            # two face velocity arrays (face_vel_A for cellA, face_vel_B for
            # cellB), matching the same cellA/cellB logic as the WSE arrays.
            nb_face_vx = np.zeros(N, dtype=np.float64)
            nb_face_vy = np.zeros(N, dtype=np.float64)
            for j in range(N):
                fi  = face_indices[j]
                ori = face_orients[j]
                src = face_vel_A[fi] if ori > 0 else face_vel_B[fi]
                nb_face_vx[j] = float(src[0])
                nb_face_vy[j] = float(src[1])

        # ---- Pixel loop for this cell -------------------------------------
        # Gather all pixels belonging to this cell into contiguous arrays
        # to avoid repeated index arithmetic inside the inner loop.
        pix_arr = np.array(pix_indices, dtype=np.int64)
        pix_rows = valid_rows[pix_arr]
        pix_cols = valid_cols[pix_arr]
        pix_xs   = xs[pix_arr]
        pix_ys   = ys[pix_arr]

        for pi in range(len(pix_arr)):
            px = float(pix_xs[pi])
            py = float(pix_ys[pi])
            r  = int(pix_rows[pi])
            c  = int(pix_cols[pi])

            # ---- Wet/dry check + barycentric weights ---------------------
            # Four branches depending on sloped mode and available data:
            #
            # (A1) Sloped corners + faces (with_faces=True, default):
            #      Interpolate per-pixel WSE from the full 2N-point stencil
            #      (N corner facepoints + N face midpoints, donated weights).
            #      Most accurate wet/dry boundary.
            if use_sloped and fp_local_wse_adj is not None and face_local_wse is not None:
                fw = _barycentric_weights(px, py, verts_x, verts_y)
                vel_w = _donate(fw)
                pixel_wse = _pixel_wse_sloped(vel_w, fp_local_wse_adj, face_local_wse, cell_dw)
                if pixel_wse == _NODATA:
                    continue
                if terrain_grid is not None:
                    t_elev = float(terrain_grid[r, c])
                    if np.isnan(t_elev) or t_elev == _NODATA:
                        continue
                    if pixel_wse < t_elev + depth_threshold:
                        continue
                else:
                    t_elev = 0.0
            # (A2) Sloped corners only (with_faces=False):
            #      Interpolate per-pixel WSE using plain barycentric weights
            #      over the N corner facepoints only — no donation to face
            #      midpoints (CellStencilMethod.JustFacepoints in C#).
            elif use_sloped and fp_local_wse_adj is not None:
                fw = _barycentric_weights(px, py, verts_x, verts_y)
                vel_w = _donate(fw)  # still donate for velocity interpolation
                pixel_wse = float(np.dot(fw, fp_local_wse_adj))
                if pixel_wse == _NODATA:
                    continue
                if terrain_grid is not None:
                    t_elev = float(terrain_grid[r, c])
                    if np.isnan(t_elev) or t_elev == _NODATA:
                        continue
                    if pixel_wse < t_elev + depth_threshold:
                        continue
                else:
                    t_elev = 0.0
            # (B) Terrain available but no sloped WSE data: use the uniform
            #     cell-average WSE for the depth test.  Applies when fp_wse
            #     is None (horizontal mode) or cell reverted to flat.
            elif terrain_grid is not None:
                t_elev = float(terrain_grid[r, c])
                if np.isnan(t_elev) or t_elev == _NODATA:
                    continue
                if float(cell_wse[cell_idx]) < t_elev + depth_threshold:
                    continue
                fw = _barycentric_weights(px, py, verts_x, verts_y)
                vel_w = _donate(fw)
                pixel_wse = float(cell_wse[cell_idx])
                t_elev = t_elev
            # (C) No terrain at all: every pixel in the cell is treated as
            #     wet; depth output will be meaningless so callers should
            #     not request "depth" without a terrain grid.
            else:
                fw = _barycentric_weights(px, py, verts_x, verts_y)
                vel_w = _donate(fw)
                pixel_wse = float(cell_wse[cell_idx])
                t_elev = 0.0

            # ---- Value interpolation -------------------------------------
            if variable == "water_surface":
                if fp_local_wse_adj is not None and face_local_wse is not None:
                    # Sloped corners + faces: donated blend, optional depth weights.
                    val = _pixel_wse_sloped(vel_w, fp_local_wse_adj, face_local_wse, cell_dw)
                    output[r, c] = np.float32(val) if val != _NODATA else nodata
                elif fp_local_wse_adj is not None:
                    # Sloped corners only: plain barycentric blend of N corners.
                    output[r, c] = np.float32(np.dot(fw, fp_local_wse_adj))
                else:
                    # Horizontal: uniform cell-average WSE for every pixel.
                    output[r, c] = np.float32(cell_wse[cell_idx])

            elif variable == "depth":
                # Interpolate WSE then subtract terrain; clamp to zero so no
                # negative depths are written.  Pixel is nodata when dep ≤ 0
                # (already passed the depth_threshold check above, but rounding
                # can produce a tiny negative after the subtract).
                if fp_local_wse_adj is not None and face_local_wse is not None:
                    pix_wse = _pixel_wse_sloped(vel_w, fp_local_wse_adj, face_local_wse, cell_dw)
                elif fp_local_wse_adj is not None:
                    pix_wse = float(np.dot(fw, fp_local_wse_adj))
                else:
                    pix_wse = float(cell_wse[cell_idx])
                dep = pix_wse - t_elev
                output[r, c] = np.float32(max(0.0, dep)) if dep > 0 else nodata

            elif variable in ("speed", "velocity"):
                # Donated barycentric blend over the 2N-sample stencil:
                #   vel_w[0..N-1]   × corner (facepoint) velocities
                #   vel_w[N..2N-1]  × face-midpoint velocities
                # vel_w always uses donated weights regardless of with_faces,
                # since velocity always uses the full stencil in RasMapper.
                if nb_fp_vx is None:
                    continue
                Vx = 0.0;  Vy = 0.0
                for i in range(N):
                    Vx += vel_w[i]     * nb_fp_vx[i]
                    Vy += vel_w[i]     * nb_fp_vy[i]
                for j in range(N):
                    Vx += vel_w[N + j] * nb_face_vx[j]
                    Vy += vel_w[N + j] * nb_face_vy[j]
                spd = float(np.sqrt(Vx * Vx + Vy * Vy))
                if variable == "speed":
                    output[r, c] = np.float32(spd)
                else:  # 4-band velocity: Vx, Vy, speed, compass direction
                    # arctan2(Vx, Vy) gives bearing from North (East=90°, etc.)
                    direction = float(np.degrees(np.arctan2(Vx, Vy))) % 360.0
                    output[0, r, c] = np.float32(Vx)
                    output[1, r, c] = np.float32(Vy)
                    output[2, r, c] = np.float32(spd)
                    output[3, r, c] = np.float32(direction)

    return output
