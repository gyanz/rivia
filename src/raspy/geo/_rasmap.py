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
Step 4  — Polygon scan-line rasterization + barycentric pixel interpolation

Public API
----------
HC_NONE, HC_BACKFILL, HC_DOWNHILL_DEEP, HC_DOWNHILL_INTERMEDIATE, HC_DOWNHILL_SHALLOW, HC_LEVEE
compute_face_wss
compute_facepoint_wse
reconstruct_face_velocities
compute_facepoint_velocities
replace_face_velocities_sloped
build_cell_id_raster        (default: Numba scan-line; GDAL fallback via use_scanline=False)
sample_terrain_at_facepoints
_rasterize_rasmap
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import rasterio

# ---------------------------------------------------------------------------
# Optional Numba JIT
# ---------------------------------------------------------------------------

from numba import njit, prange
import numba as _numba

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step A — Hydraulic connectivity
# ---------------------------------------------------------------------------

_MIN_WS_PLOT_TOLERANCE = 0.001  # matches C# RASResults.MinWSPlotTolerance
_NODATA = -9999.0

# HydraulicConnection enum values — mirror C# RasMapperLib.Mesh.HydraulicConnection
# (archive/DLLs/RasMapperLib/RasMapperLib.Mesh/HydraulicConnection.cs)
HC_NONE                  = 0  # disconnected: dry cell, boundary, or both below sill
HC_BACKFILL              = 1  # connected: WSE gradient opposes bed-elevation gradient
HC_DOWNHILL_DEEP         = 2  # connected: higher-cell depth >= 2 × bed-elevation diff
HC_DOWNHILL_INTERMEDIATE = 3  # connected: depth between bed_diff and 2×bed_diff
HC_DOWNHILL_SHALLOW      = 4  # connected: higher-cell depth <= bed-elevation diff
HC_LEVEE                 = 5  # disconnected: critical-flow overtopping (levee/weir crest)


@njit(cache=True)
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


@njit(cache=True, parallel=True)
def compute_face_wss(
    cell_wse: np.ndarray,
    cell_min_elev: np.ndarray,
    face_min_elev: np.ndarray,
    face_cell_indexes: np.ndarray,
    cell_face_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step A — hydraulic connectivity and per-face WSE values.

    Replicates ``ComputeFaceWaterSurfaces`` → ``ComputeFaceWSsNew`` from
    ``archive/DLLs/RasMapperLib/RasMapperLib/MeshFV2D.cs``.

    For each face the function determines the hydraulic-connection type
    (``face_hconn``, one of the ``HC_*`` constants), whether water is actively
    flowing across the face (``face_connected``), and the WSE to assign to
    each side (cellA → ``face_value_a``, cellB → ``face_value_b``).  The
    per-side values are used by :func:`compute_facepoint_wse` to fit a sloped
    water surface; ``face_hconn`` is used by :func:`_rasterize_rasmap` to
    determine the all-shallow rendering fallback.

    Decision logic (applied in order)
    ----------------------------------
    1. **No real neighbour** (``cellA < 0`` or ``cellB < 0``): face sits on
       the outer mesh boundary with only one real cell.  Skip entirely —
       outputs remain at ``_NODATA`` / ``False`` / ``HC_NONE``.

    2. **Virtual cell or dry cell** → ``HC_NONE``: cells with
       ``face_count == 1`` are virtual boundary cells inserted by HEC-RAS as
       computational placeholders.  By HEC-RAS convention the virtual cell is
       always cellB, so ``face_is_perimeter = cell_b_virtual``.  If either
       cell is dry (``wse <= min_elev + _MIN_WS_PLOT_TOLERANCE``) or the face
       is a perimeter face, ``face_connected = False``.  The dry/virtual side
       gets ``_NODATA``; a wet real cell keeps its own WSE.

    3. **Both cells below the face invert** → ``HC_NONE``: both WSEs are
       at or below ``face_min_elev`` — water is ponded on both sides without
       reaching the sill.  Both sides take the cell WSE; ``face_connected =
       False``.

    4. **Overtopping / weir-crest** → ``HC_LEVEE``: ``was_crit_cap_used``
       (average WSE < 2/3 of head above sill) AND
       ``depth_higher / depth_above_face > 2`` (upstream pool is deep relative
       to the driving head).  The face acts as a crested structure.
       ``face_connected = False``; each side keeps its own cell WSE.

    5. **Backfill** → ``HC_BACKFILL``, ``face_connected = True``:
       ``(wse_b - wse_a) * (bed_b - bed_a) <= 0`` — WSE and bed gradients
       oppose each other (water backs up into a depression).  Both sides take
       ``face_ws`` (the higher cell WSE).

    6. **Deep flow** → ``HC_DOWNHILL_DEEP``, ``face_connected = True``:
       ``depth_higher >= 2 * delta_bed`` — the higher cell is deep enough
       that the bed step is negligible (submerged bump).  Both sides take
       ``face_ws``.

    7. **Transitional flow** — quadratic interpolation; ``face_connected =
       True``:

       * ``depth_higher > delta_bed`` → ``HC_DOWNHILL_INTERMEDIATE``: blends
         the quadratic estimate with ``face_ws`` for a smooth transition into
         the deep-flow regime.
       * ``depth_higher <= delta_bed`` → ``HC_DOWNHILL_SHALLOW``: pure
         quadratic estimate; the higher cell is too shallow for blending.
         Does **not** clear the all-shallow rendering flag.

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
    face_value_a : float64 ndarray, shape ``(n_faces,)``
        WSE on the cellA side of each face; ``-9999`` where nodata.
    face_value_b : float64 ndarray, shape ``(n_faces,)``
        WSE on the cellB side of each face; ``-9999`` where nodata.
    face_hconn : uint8 ndarray, shape ``(n_faces,)``
        Hydraulic-connection classification per face (``HC_*`` constants).
        Mirrors C# ``FaceValues.HydraulicConnection``.  Connectedness is a
        derived property: ``face_connected = (face_hconn >= HC_BACKFILL) &
        (face_hconn <= HC_DOWNHILL_SHALLOW)``, matching C#
        ``FaceValues.IsHydraulicallyConnected``.
    """
    n_faces = len(face_cell_indexes)
    face_value_a = np.full(n_faces, _NODATA, dtype=np.float64)
    face_value_b = np.full(n_faces, _NODATA, dtype=np.float64)
    face_hconn = np.zeros(n_faces, dtype=np.uint8)  # HC_NONE by default

    for f in prange(n_faces):
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

        # Logic 2 → HC_NONE: virtual cell or dry cell
        if flag_a_dry or flag_b_dry or face_is_perimeter:
            face_value_a[f] = _NODATA if (flag_a_dry or cell_a_virtual) else wse_a
            face_value_b[f] = _NODATA if (flag_b_dry or cell_b_virtual) else wse_b
            # face_hconn[f] stays HC_NONE (default)
            continue

        # Logic 3 → HC_NONE: both cells below face invert
        if wse_a <= min_elev_face and wse_b <= min_elev_face:
            face_value_a[f] = wse_a
            face_value_b[f] = wse_b
            # face_hconn[f] stays HC_NONE (default)
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
                # Logic 4 → HC_LEVEE: overtopping / weir-crest.
                # Each side keeps its own cell WSE.  The two pools are
                # hydraulically separated at the crest — face_connected = False.
                face_value_a[f] = wse_a
                face_value_b[f] = wse_b
                face_hconn[f] = HC_LEVEE
            else:
                # Logic 5 → HC_BACKFILL: WSE and bed gradients oppose each
                # other — water backs into a depression.  Both sides share face_ws.
                face_value_a[f] = face_ws
                face_value_b[f] = face_ws
                face_hconn[f] = HC_BACKFILL
        elif depth_higher >= 2.0 * delta_min_elev:
            # Logic 6 → HC_DOWNHILL_DEEP: higher cell is deep enough that the
            # bed step is negligible (submerged bump).  Both sides share face_ws.
            # The 2× threshold is where Logic 7's blend weight for face_ws
            # reaches 1, ensuring a smooth, jump-free transition.
            face_value_a[f] = face_ws
            face_value_b[f] = face_ws
            face_hconn[f] = HC_DOWNHILL_DEEP
        else:
            # Logic 7: transitional flow — bed-elevation difference is
            # significant relative to depth.  Quadratic formula interpolates
            # the face WSE assuming a linearly sloping water surface.
            #
            # Step 7a — quadratic interpolation:
            #   num9 = eff_lower + (depth_higher² - depth_ref_lower²)
            #                      / (2 × delta_min_elev)
            if delta_min_elev > 1e-12:
                num9 = eff_lower + (depth_higher ** 2 - depth_ref_lower ** 2) / (
                    2.0 * delta_min_elev
                )
            else:
                # Flat bed — quadratic collapses to face_ws.
                num9 = face_ws

            # Step 7b — linear blend toward face_ws as depth approaches 2×dz.
            # Weights: num9 → (2dz-d)/dz, face_ws → (d-dz)/dz.
            # depth_higher > delta_min_elev → HC_DOWNHILL_INTERMEDIATE (blended).
            # depth_higher <= delta_min_elev → HC_DOWNHILL_SHALLOW (pure quadratic).
            # HC_DOWNHILL_SHALLOW does NOT clear the all-shallow rendering flag
            # (C# Renderer.cs:3094 — treated identically to a disconnected face
            # for the purpose of ShallowBehavior.ReduceToHorizontal).
            if depth_higher > delta_min_elev and delta_min_elev > 1e-12:
                num9 = (
                    (2.0 * delta_min_elev - depth_higher) * num9
                    + (depth_higher - delta_min_elev) * face_ws
                ) / delta_min_elev
                face_hconn[f] = HC_DOWNHILL_INTERMEDIATE
            else:
                face_hconn[f] = HC_DOWNHILL_SHALLOW
            face_value_a[f] = num9
            face_value_b[f] = num9

    return face_value_a, face_value_b, face_hconn


# ---------------------------------------------------------------------------
# Step B — Facepoint WSE via PlanarRegressionZ
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
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

    for fi in prange(n_faces):
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


@njit(cache=True)
def _planar_z_intercept(
    base_x: float,
    base_y: float,
    app_xs: np.ndarray,
    app_ys: np.ndarray,
    zs: np.ndarray,
    n: int,
) -> float:
    """Fit a plane Z = a*dx + b*dy + c and return c at (base_x, base_y).

    Matches ``PlanarRegressionZ`` in ``archive/DLLs/RasMapperLib/MeshFV2D.cs``.
    Working in local coordinates ``dx = x - base_x``, ``dy = y - base_y`` so
    that evaluating the plane at the origin (the facepoint) gives Z = c directly.

    Takes pre-allocated numpy arrays and an explicit count ``n`` (the first
    ``n`` elements are used) so the function can be compiled with Numba
    ``@njit``.

    Degenerate cases:

    * n = 0 → return ``_NODATA``
    * n = 1 → return ``zs[0]``
    * n = 2 → return average
    * det = 0 (collinear) → return average
    """
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

    fn = float(n)
    det = (
        sumX2 * (sumY2 * fn - sumY * sumY)
        - sumXY * (sumXY * fn - sumY * sumX)
        + sumX * (sumXY * sumY - sumY2 * sumX)
    )
    if det == 0.0:
        return sumZ / fn
    return (
        sumX2 * (sumY2 * sumZ - sumYZ * sumY)
        - sumXY * (sumXY * sumZ - sumYZ * sumX)
        + sumXZ * (sumXY * sumY - sumY2 * sumX)
    ) / det


@njit(cache=True)
def _face_app_point(
    fi: int,
    fp_coords: np.ndarray,
    face_facepoint_indexes: np.ndarray,
    face_midsides: np.ndarray,
) -> tuple[float, float]:
    """Return the application point (x, y) for face *fi*.

    Uses the precomputed midside when available (``face_midsides.shape[0] > 0``);
    falls back to chord midpoint of the two endpoint facepoints.
    """
    if face_midsides.shape[0] > 0:
        return face_midsides[fi, 0], face_midsides[fi, 1]
    fpA = face_facepoint_indexes[fi, 0]
    fpB = face_facepoint_indexes[fi, 1]
    return (
        (fp_coords[fpA, 0] + fp_coords[fpB, 0]) * 0.5,
        (fp_coords[fpA, 1] + fp_coords[fpB, 1]) * 0.5,
    )


@njit(cache=True, parallel=True)
def _compute_facepoint_wse_nb(
    fp_coords: np.ndarray,
    fp_face_info: np.ndarray,
    fp_face_values: np.ndarray,
    face_facepoint_indexes: np.ndarray,
    face_value_a: np.ndarray,
    face_value_b: np.ndarray,
    face_connected: np.ndarray,
    face_midsides: np.ndarray,
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
    n_fp = len(fp_coords)
    fp_wse_at_face = np.full((n_faces, 2), _NODATA, dtype=np.float64)

    # Maximum arc buffer size: fp_count arc faces + 1 terminal face.
    # Computed once (serial reduction) so each parallel iteration can
    # allocate thread-local buffers of the right size without dynamic growth.
    max_fp_count = 0
    for k in range(n_fp):
        c = fp_face_info[k, 1]
        if c > max_fp_count:
            max_fp_count = c

    for fp_idx in prange(n_fp):
        # Thread-local buffers — allocated inside prange so each thread owns
        # its own copy; size is bounded by max_fp_count (a few dozen entries
        # for any real HEC-RAS mesh) so allocation overhead is negligible.
        buf_xs = np.empty(max_fp_count + 1, dtype=np.float64)
        buf_ys = np.empty(max_fp_count + 1, dtype=np.float64)
        buf_zs = np.empty(max_fp_count + 1, dtype=np.float64)
        buf_raw = np.empty(max_fp_count, dtype=np.float64)
        processed = np.zeros(max_fp_count, dtype=np.bool_)

        base_x = fp_coords[fp_idx, 0]
        base_y = fp_coords[fp_idx, 1]
        fp_start = fp_face_info[fp_idx, 0]
        fp_count = fp_face_info[fp_idx, 1]

        # Early exit: skip facepoints where every adjacent face is completely dry
        # (both sides -9999).  Matches C# flag=true early-return.
        any_wet = False
        for j in range(fp_count):
            fi = fp_face_values[fp_start + j, 0]
            if face_value_a[fi] != _NODATA or face_value_b[fi] != _NODATA:
                any_wet = True
                break
        if not any_wet:
            continue

        # processed[j] tracks which local face indices have been assigned
        # by a completed arc.  Prevents double-processing.
        for j in range(fp_count):
            processed[j] = False

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
                fi_num4 = fp_face_values[fp_start + num4, 0]
                if not face_connected[fi_num4] or num4 == j:
                    break

            # --- Find num5: first disconnected face going CW from j ---
            # If num5 == num4 (all connected), the whole ring is one arc.
            num5 = j
            if num5 != num4:
                while face_connected[fp_face_values[fp_start + num5, 0]]:
                    num5 = (num5 - 1 + fp_count) % fp_count

            # --- Collect arc faces (num5 → num4, CCW exclusive) ---
            # Use pre-allocated buffers instead of Python lists.
            n_buf = 0   # count of regression sample points
            n_raw = 0   # count of raw (own-side) values for arc faces

            num6 = num5
            while True:
                fi_cur = fp_face_values[fp_start + num6, 0]
                ori_cur = fp_face_values[fp_start + num6, 1]
                # Own-side: fpA (ori=-1) → face_value_a; fpB (ori=+1) → face_value_b
                if ori_cur == -1:
                    wse = face_value_a[fi_cur]
                else:
                    wse = face_value_b[fi_cur]
                if wse != _NODATA:
                    ax, ay = _face_app_point(
                        fi_cur, fp_coords, face_facepoint_indexes, face_midsides
                    )
                    buf_xs[n_buf] = ax
                    buf_ys[n_buf] = ay
                    buf_zs[n_buf] = wse
                    n_buf += 1
                buf_raw[n_raw] = wse  # may be _NODATA; overridden by arc result below
                n_raw += 1
                num6 = (num6 + 1) % fp_count
                if num6 == num4:
                    break

            # --- Add terminal face (num4, if disconnected) using OPPOSITE side ---
            # Provides a slope anchor at the wet/dry boundary.
            fi_term = fp_face_values[fp_start + num4, 0]
            if not face_connected[fi_term]:
                ori_term = fp_face_values[fp_start + num4, 1]
                # Opposite: fpA (ori=-1) → face_value_b; fpB (ori=+1) → face_value_a
                if ori_term == -1:
                    wse_term = face_value_b[fi_term]
                else:
                    wse_term = face_value_a[fi_term]
                if wse_term != _NODATA:
                    ax, ay = _face_app_point(
                        fi_term, fp_coords, face_facepoint_indexes, face_midsides
                    )
                    buf_xs[n_buf] = ax
                    buf_ys[n_buf] = ay
                    buf_zs[n_buf] = wse_term
                    n_buf += 1

            # --- Solve regression and store result for arc faces ---
            arc_result = _planar_z_intercept(base_x, base_y, buf_xs, buf_ys, buf_zs, n_buf)

            num6 = num5
            arc_idx = 0
            while True:
                fi_cur = fp_face_values[fp_start + num6, 0]
                ori_cur = fp_face_values[fp_start + num6, 1]
                # Store arc result (or raw value if regression had no points)
                value = arc_result if arc_result != _NODATA else buf_raw[arc_idx]
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

    Public wrapper around the ``@njit`` kernel :func:`_compute_facepoint_wse_nb`.
    ``face_midsides`` may be omitted or ``None``; an empty ``(0, 2)`` sentinel
    is passed to the kernel in that case so that the midpoint branch is skipped.
    """
    ms = face_midsides if face_midsides is not None else np.empty((0, 2), dtype=np.float64)
    return _compute_facepoint_wse_nb(
        fp_coords, fp_face_info, fp_face_values,
        face_facepoint_indexes, face_value_a, face_value_b,
        face_connected, ms,
    )


# ---------------------------------------------------------------------------
# Step 2 — C-stencil tangential velocity reconstruction
# ---------------------------------------------------------------------------


@njit(cache=True)
def _cw_ccw_neighbors(
    target_face: int,
    cell_idx: int,
    cell_face_info: np.ndarray,
    cell_face_values: np.ndarray,
) -> tuple[int, int]:
    """Return (CW neighbor face, CCW neighbor face) of *target_face* within *cell_idx*.

    Returns -1 for a neighbor that doesn't exist.
    Mirrors ``FaceVelocityCoef`` helper in
    ``archive/DLLs/RasMapperLib/MeshFV2D.cs``.
    """
    start = cell_face_info[cell_idx, 0]
    count = cell_face_info[cell_idx, 1]
    target_pos = -1
    for k in range(count):
        if cell_face_values[start + k, 0] == target_face:
            target_pos = k
            break
    if target_pos < 0:
        return -1, -1
    cw_pos  = (target_pos + 1) % count
    ccw_pos = (target_pos - 1 + count) % count
    return cell_face_values[start + cw_pos, 0], cell_face_values[start + ccw_pos, 0]


@njit(cache=True, parallel=True)
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

    for fidx in prange(n_faces):
        cellA = face_cell_indexes[fidx, 0]
        cellB = face_cell_indexes[fidx, 1]
        fv = face_normal_vel[fidx]
        fn_x = face_normals_2d[fidx, 0]
        fn_y = face_normals_2d[fidx, 1]
        tan_x = fn_y
        tan_y = -fn_x

        # ---- cellA stencil (Item1) ----------------------------------------
        # Inline _FaceVelocityCoef: accumulate 2×2 symmetric WLS matrix,
        # invert via Cramer's rule, reconstruct the tangential component.
        if cellA >= 0:
            cw_a, ccw_a = _cw_ccw_neighbors(fidx, cellA, cell_face_info, cell_face_values)
            A11 = fn_x * fn_x
            A22 = fn_y * fn_y
            A12 = fn_x * fn_y
            ct = 1
            if cw_a >= 0:
                nx = face_normals_2d[cw_a, 0]
                ny = face_normals_2d[cw_a, 1]
                A11 += nx * nx
                A22 += ny * ny
                A12 += nx * ny
                ct += 1
            if ccw_a >= 0:
                nx = face_normals_2d[ccw_a, 0]
                ny = face_normals_2d[ccw_a, 1]
                A11 += nx * nx
                A22 += ny * ny
                A12 += nx * ny
                ct += 1
            det = A11 * A22 - A12 * A12
            if det == 0.0:
                inv_ct = 1.0 / ct
                A11 = inv_ct
                A22 = inv_ct
                A12 = 0.0
            else:
                inv_det = 1.0 / det
                A11 *= inv_det
                A22 *= inv_det
                A12 *= inv_det

            tangential = 0.0
            cw_conn  = face_connected[cw_a]  if cw_a  >= 0 else False
            ccw_conn = face_connected[ccw_a] if ccw_a >= 0 else False
            if cw_conn or ccw_conn:
                cw_vn  = face_normal_vel[cw_a]       if cw_a  >= 0 else 0.0
                cw_nx  = face_normals_2d[cw_a,  0]   if cw_a  >= 0 else 0.0
                cw_ny  = face_normals_2d[cw_a,  1]   if cw_a  >= 0 else 0.0
                ccw_vn = face_normal_vel[ccw_a]      if ccw_a >= 0 else 0.0
                ccw_nx = face_normals_2d[ccw_a, 0]   if ccw_a >= 0 else 0.0
                ccw_ny = face_normals_2d[ccw_a, 1]   if ccw_a >= 0 else 0.0
                B1 = cw_nx * cw_vn + ccw_nx * ccw_vn + fn_x * fv
                B2 = cw_ny * cw_vn + ccw_ny * ccw_vn + fn_y * fv
                sx = A22 * B1 - A12 * B2
                sy = -A12 * B1 + A11 * B2
                tangential = sx * tan_x + sy * tan_y
            face_vel_A[fidx, 0] = fv * fn_x + tangential * tan_x
            face_vel_A[fidx, 1] = fv * fn_y + tangential * tan_y
        else:
            face_vel_A[fidx, 0] = fv * fn_x
            face_vel_A[fidx, 1] = fv * fn_y

        # ---- cellB stencil (Item2) ----------------------------------------
        if cellB >= 0:
            cw_b, ccw_b = _cw_ccw_neighbors(fidx, cellB, cell_face_info, cell_face_values)
            A11 = fn_x * fn_x
            A22 = fn_y * fn_y
            A12 = fn_x * fn_y
            ct = 1
            if cw_b >= 0:
                nx = face_normals_2d[cw_b, 0]
                ny = face_normals_2d[cw_b, 1]
                A11 += nx * nx
                A22 += ny * ny
                A12 += nx * ny
                ct += 1
            if ccw_b >= 0:
                nx = face_normals_2d[ccw_b, 0]
                ny = face_normals_2d[ccw_b, 1]
                A11 += nx * nx
                A22 += ny * ny
                A12 += nx * ny
                ct += 1
            det = A11 * A22 - A12 * A12
            if det == 0.0:
                inv_ct = 1.0 / ct
                A11 = inv_ct
                A22 = inv_ct
                A12 = 0.0
            else:
                inv_det = 1.0 / det
                A11 *= inv_det
                A22 *= inv_det
                A12 *= inv_det

            tangential = 0.0
            cw_conn  = face_connected[cw_b]  if cw_b  >= 0 else False
            ccw_conn = face_connected[ccw_b] if ccw_b >= 0 else False
            if cw_conn or ccw_conn:
                cw_vn  = face_normal_vel[cw_b]       if cw_b  >= 0 else 0.0
                cw_nx  = face_normals_2d[cw_b,  0]   if cw_b  >= 0 else 0.0
                cw_ny  = face_normals_2d[cw_b,  1]   if cw_b  >= 0 else 0.0
                ccw_vn = face_normal_vel[ccw_b]      if ccw_b >= 0 else 0.0
                ccw_nx = face_normals_2d[ccw_b, 0]   if ccw_b >= 0 else 0.0
                ccw_ny = face_normals_2d[ccw_b, 1]   if ccw_b >= 0 else 0.0
                B1 = cw_nx * cw_vn + ccw_nx * ccw_vn + fn_x * fv
                B2 = cw_ny * cw_vn + ccw_ny * ccw_vn + fn_y * fv
                sx = A22 * B1 - A12 * B2
                sy = -A12 * B1 + A11 * B2
                tangential = sx * tan_x + sy * tan_y
            face_vel_B[fidx, 0] = fv * fn_x + tangential * tan_x
            face_vel_B[fidx, 1] = fv * fn_y + tangential * tan_y
        else:
            face_vel_B[fidx, 0] = fv * fn_x
            face_vel_B[fidx, 1] = fv * fn_y

        # Connected faces: average Item1 and Item2 (C# MeshFV2D.cs)
        if face_connected[fidx] and cellA >= 0 and cellB >= 0:
            avg_x = (face_vel_A[fidx, 0] + face_vel_B[fidx, 0]) / 2.0
            avg_y = (face_vel_A[fidx, 1] + face_vel_B[fidx, 1]) / 2.0
            face_vel_A[fidx, 0] = avg_x
            face_vel_A[fidx, 1] = avg_y
            face_vel_B[fidx, 0] = avg_x
            face_vel_B[fidx, 1] = avg_y

    return face_vel_A, face_vel_B


# ---------------------------------------------------------------------------
# Step 3 — Inverse-face-length weighted facepoint velocity averaging
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _build_face_fp_local_idx(
    face_facepoint_indexes: np.ndarray,
    fp_face_info: np.ndarray,
    fp_face_values: np.ndarray,
) -> np.ndarray:
    """Precompute local ring index of each face within its two endpoint facepoints.

    Returns ``face_fp_local_idx`` shape ``(n_faces, 2)`` where:

    * ``[fi, 0]`` — position *j* of face *fi* in fpA's angle-sorted ring
      (``fp_face_values[fp_face_info[fpA, 0] + j, 0] == fi``).
    * ``[fi, 1]`` — same for fpB.

    ``-1`` if a mapping entry is not found (should not occur for valid meshes).

    This replaces the ``fp_face_local_map`` dict so that Step 3 / 3.5 / 4
    lookups are Numba-compatible array accesses.
    """
    n_faces = len(face_facepoint_indexes)
    n_fp = len(fp_face_info)
    face_fp_local_idx = np.full((n_faces, 2), np.int32(-1), dtype=np.int32)
    # Each fp writes to face_fp_local_idx[fi, 0] (fpA) or [fi, 1] (fpB).
    # A face's fpA and fpB are distinct facepoints, so different threads
    # write to different columns — no write conflict.
    for fp in prange(n_fp):
        start = fp_face_info[fp, 0]
        count = fp_face_info[fp, 1]
        for j in range(count):
            fi = fp_face_values[start + j, 0]
            if face_facepoint_indexes[fi, 0] == fp:
                face_fp_local_idx[fi, 0] = np.int32(j)
            else:
                face_fp_local_idx[fi, 1] = np.int32(j)
    return face_fp_local_idx


@njit(cache=True, parallel=True)
def _compute_facepoint_velocities_nb(
    fp_vel_data: np.ndarray,
    face_vel_A: np.ndarray,
    face_vel_B: np.ndarray,
    face_connected: np.ndarray,
    face_inv_lengths: np.ndarray,
    face_facepoint_indexes: np.ndarray,
    fp_face_info: np.ndarray,
    fp_face_values: np.ndarray,
    face_value_a: np.ndarray,
    face_value_b: np.ndarray,
) -> None:
    """Inner @njit kernel for :func:`compute_facepoint_velocities`.

    Writes directly into ``fp_vel_data`` (shape ``(total_fp_face_entries, 2)``,
    pre-zeroed).  Mirrors ``ComputeVertexVelocities`` in
    ``archive/DLLs/RasMapperLib/MeshFV2D.cs``.
    """
    n_fp = len(fp_face_info)

    # Each fp owns a non-overlapping slice of fp_vel_data via the CSR layout,
    # so prange writes are conflict-free across threads.
    for fp in prange(n_fp):
        fp_start = fp_face_info[fp, 0]
        fp_count = fp_face_info[fp, 1]
        if fp_count == 0:
            continue

        # Dry check: skip if all adjacent faces have no wet WSE
        all_dry = True
        for j in range(fp_count):
            fi = fp_face_values[fp_start + j, 0]
            if face_value_a[fi] != _NODATA or face_value_b[fi] != _NODATA:
                all_dry = False
                break
        if all_dry:
            continue  # fp_vel_data already zero-initialised

        for j in range(fp_count):
            n = fp_count

            # --- Find connected arc starting from local index j ---
            # Walk forward to find arc end (first disconnected face CCW from j+1)
            end_idx = (j + 1) % n
            while end_idx != j:
                fi_e = fp_face_values[fp_start + end_idx, 0]
                if not face_connected[fi_e]:
                    break
                end_idx = (end_idx + 1) % n

            # Walk backward from j to find arc start
            start_idx = j
            if start_idx != end_idx:
                fi_s = fp_face_values[fp_start + start_idx, 0]
                while face_connected[fi_s]:
                    start_idx = (start_idx - 1 + n) % n
                    fi_s = fp_face_values[fp_start + start_idx, 0]

            arc_start = start_idx
            arc_end = end_idx

            # --- Weighted sum over arc faces (float32 matches C# MeshFV2D.cs:8677) ---
            sum_vx = np.float32(0.0)
            sum_vy = np.float32(0.0)
            total_w = np.float32(0.0)

            current = arc_start
            while True:
                fi = fp_face_values[fp_start + current, 0]
                inv_len = np.float32(face_inv_lengths[fi])

                if current == arc_start:
                    # Start face: Item1 if fpA == fp, else Item2
                    if face_facepoint_indexes[fi, 0] == fp:
                        vx = face_vel_A[fi, 0]
                        vy = face_vel_A[fi, 1]
                    else:
                        vx = face_vel_B[fi, 0]
                        vy = face_vel_B[fi, 1]
                else:
                    # Interior connected faces: always Item1
                    vx = face_vel_A[fi, 0]
                    vy = face_vel_A[fi, 1]

                sum_vx += np.float32(vx) * inv_len
                sum_vy += np.float32(vy) * inv_len
                total_w += inv_len

                current = (current + 1) % n
                if current == arc_end:
                    break

            # --- Terminal disconnected face (boundary anchor) ---
            fi_end = fp_face_values[fp_start + arc_end, 0]
            if arc_end != arc_start or not face_connected[fi_end]:
                if not face_connected[fi_end]:
                    inv_len = np.float32(face_inv_lengths[fi_end])
                    # Opposite selection from start face
                    if face_facepoint_indexes[fi_end, 0] == fp:
                        vx = face_vel_B[fi_end, 0]
                        vy = face_vel_B[fi_end, 1]
                    else:
                        vx = face_vel_A[fi_end, 0]
                        vy = face_vel_A[fi_end, 1]
                    sum_vx += np.float32(vx) * inv_len
                    sum_vy += np.float32(vy) * inv_len
                    total_w += inv_len

            gj = fp_start + j
            if total_w > np.float32(1e-12):
                # Explicit float32 division — matches C# MeshFV2D.cs result type
                fp_vel_data[gj, 0] = float(np.float32(sum_vx / total_w))
                fp_vel_data[gj, 1] = float(np.float32(sum_vy / total_w))


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
) -> tuple[np.ndarray, np.ndarray]:
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
    fp_vel_data : float64 ndarray, shape ``(total_fp_face_entries, 2)``
        Flat CSR velocity store.  ``fp_vel_data[fp_face_info[fp, 0] + j]``
        gives the ``[Vx, Vy]`` arc-context velocity for facepoint ``fp``,
        local face index ``j`` (same order as the angle-sorted face ring).
    face_fp_local_idx : int32 ndarray, shape ``(n_faces, 2)``
        For each face *fi*:

        * ``[fi, 0]`` — local ring index of *fi* in fpA's face ring.
        * ``[fi, 1]`` — local ring index of *fi* in fpB's face ring.

        ``-1`` if not found.  Replaces the ``fp_face_local_map`` dict so
        Step 3.5 and Step 4 lookups are Numba-compatible.
    """
    face_inv_lengths = 1.0 / np.maximum(face_lengths, 1e-12)

    total = int(fp_face_info[:, 1].sum())
    fp_vel_data = np.zeros((total, 2), dtype=np.float64)

    # np.asarray avoids an allocation when the arrays are already int64.
    _ffi64 = np.asarray(face_facepoint_indexes, dtype=np.int64)
    _fpi64 = np.asarray(fp_face_info,           dtype=np.int64)
    _fpv64 = np.asarray(fp_face_values,          dtype=np.int64)

    face_fp_local_idx = _build_face_fp_local_idx(_ffi64, _fpi64, _fpv64)

    _compute_facepoint_velocities_nb(
        fp_vel_data,
        face_vel_A, face_vel_B, face_connected,
        face_inv_lengths,
        _ffi64, _fpi64, _fpv64,
        face_value_a, face_value_b,
    )

    return fp_vel_data, face_fp_local_idx


# ---------------------------------------------------------------------------
# Step 3.5 — Sloped face velocity replacement
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def replace_face_velocities_sloped(
    fp_vel_data: np.ndarray,
    fp_face_info: np.ndarray,
    face_fp_local_idx: np.ndarray,
    face_facepoint_indexes: np.ndarray,
) -> np.ndarray:
    """Step 3.5 — replace face velocity with average of endpoint facepoint velocities.

    Replicates ``ReplaceFaceVelocitiesSloped`` from
    ``archive/DLLs/RasMapperLib/MeshFV2D.cs``.

    For every face ``f`` with endpoint facepoints ``fpA`` and ``fpB``:

    1. Look up the arc-context velocity for ``fpA`` at face ``f`` via the
       CSR table: ``fp_vel_data[fp_face_info[fpA, 0] + face_fp_local_idx[f, 0]]``
    2. Same for ``fpB``.
    3. Replace the face velocity with ``(vel_A + vel_B) / 2``.

    Missing entries (``face_fp_local_idx == -1``) default to ``(0, 0)``.

    Parameters
    ----------
    fp_vel_data:
        ``(total_fp_face_entries, 2)`` CSR velocity data from
        :func:`compute_facepoint_velocities`.
    fp_face_info:
        ``(n_fp, 2)`` — ``[start, count]`` into ``fp_vel_data``.
    face_fp_local_idx:
        ``(n_faces, 2)`` int32 — local ring indices from
        :func:`compute_facepoint_velocities`.  ``-1`` = not found.
    face_facepoint_indexes:
        ``(n_faces, 2)`` — ``[fpA, fpB]``.

    Returns
    -------
    replaced_face_vel : float64 ndarray, shape ``(n_faces, 2)``
        ``[Vx, Vy]`` sloped replacement velocity for each face.
    """
    n_faces = len(face_facepoint_indexes)
    replaced = np.zeros((n_faces, 2), dtype=np.float64)
    for f in prange(n_faces):
        fpA = face_facepoint_indexes[f, 0]
        fpB = face_facepoint_indexes[f, 1]
        jA = face_fp_local_idx[f, 0]
        jB = face_fp_local_idx[f, 1]
        vx_A = fp_vel_data[fp_face_info[fpA, 0] + jA, 0] if jA >= 0 else 0.0
        vy_A = fp_vel_data[fp_face_info[fpA, 0] + jA, 1] if jA >= 0 else 0.0
        vx_B = fp_vel_data[fp_face_info[fpB, 0] + jB, 0] if jB >= 0 else 0.0
        vy_B = fp_vel_data[fp_face_info[fpB, 0] + jB, 1] if jB >= 0 else 0.0
        replaced[f, 0] = (vx_A + vx_B) / 2.0
        replaced[f, 1] = (vy_A + vy_B) / 2.0
    return replaced


# ---------------------------------------------------------------------------
# Step 4 helpers — barycentric weights, donate, WSE + velocity interpolation
# ---------------------------------------------------------------------------


@njit(cache=True)
def _barycentric_weights(px: float, py: float, verts_x: np.ndarray, verts_y: np.ndarray) -> np.ndarray:
    """Generalised polygon barycentric coordinates, cast to float32.

    Matches ``archive/DLLs/RasMapperLib/RASGeometryMapPoints.cs:2956``
    ``fpWeights[l] = (float)(array[l] / num3)``.
    """
    N = len(verts_x)
    if N < 3:
        out = np.empty(N, dtype=np.float32)
        v = 1.0 / float(max(N, 1))
        for i in range(N):
            out[i] = np.float32(v)
        return out

    xp = np.empty(N, dtype=np.float64)
    for i in range(N):
        ax = verts_x[i];  ay = verts_y[i]
        bx = verts_x[(i + 1) % N];  by = verts_y[(i + 1) % N]
        val = (ax - px) * (by - py) - (ay - py) * (bx - px)
        xp[i] = val if val != 0.0 else 1e-5

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

    total = 0.0
    for i in range(N):
        total += weights[i]
    if abs(total) > 1e-20:
        for i in range(N):
            weights[i] /= total
    for i in range(N):
        if weights[i] < 0.0:
            weights[i] = 0.0
    total = 0.0
    for i in range(N):
        total += weights[i]
    if total > 1e-20:
        for i in range(N):
            weights[i] /= total

    out = np.empty(N, dtype=np.float32)
    for i in range(N):
        out[i] = np.float32(weights[i])
    return out


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
def _all_shallow(
    cell_idx: int,
    cell_face_info: np.ndarray,
    cell_face_values: np.ndarray,
    face_hconn: np.ndarray,
) -> bool:
    """Return True when ALL faces of this cell are ``HC_NONE``, ``HC_LEVEE``, or
    ``HC_DOWNHILL_SHALLOW`` — i.e. no face has an *active* hydraulic connection.

    Mirrors C# ``Renderer.cs:3079-3097`` line 3094::

        if (faceValues.IsHydraulicallyConnected &
                (faceValues.HydraulicConnection != HydraulicConnection.DownhillShallow))
            flag = false;

    Only ``HC_BACKFILL`` (1), ``HC_DOWNHILL_DEEP`` (2), and
    ``HC_DOWNHILL_INTERMEDIATE`` (3) clear the all-shallow flag.
    ``HC_DOWNHILL_SHALLOW`` (4) is connected but shallow — it does *not* clear
    the flag.  ``HC_NONE`` (0) and ``HC_LEVEE`` (5) are disconnected — they
    also do not clear the flag.
    """
    start = int(cell_face_info[cell_idx, 0])
    count = int(cell_face_info[cell_idx, 1])
    for k in range(count):
        fi = int(cell_face_values[start + k, 0])
        hc = int(face_hconn[fi])
        # Clears flag only for Backfill=1, DownhillDeep=2, DownhillIntermediate=3.
        if HC_BACKFILL <= hc <= HC_DOWNHILL_INTERMEDIATE:
            return False
    return True


@njit(cache=True)
def _pixel_wse_sloped(
    vel_weights: np.ndarray,
    fp_local_wse: np.ndarray,
    face_local_wse: np.ndarray,
    depth_weights: np.ndarray,
) -> float:
    """Interpolate WSE at a pixel using donated barycentric weights.

    C# ``Renderer.cs:3139 PaintCell_8Stencil`` or
    ``Renderer.cs:3166 PaintCell_8Stencil_RebalanceWeights``.

    Pass ``depth_weights`` as a length-0 array to skip depth-weighted
    rebalancing (plain donated-barycentric blend).
    """
    N = len(fp_local_wse)
    if N == 0:
        return _NODATA
    base_val = float(face_local_wse[0])

    if depth_weights.shape[0] > 0:
        w_sum = 0.0
        for i in range(2 * N):
            w_sum += vel_weights[i] * depth_weights[i]
        if w_sum < 1e-20:
            return _NODATA
        result = 0.0
        for i in range(N):
            result += (float(fp_local_wse[i]) - base_val) * vel_weights[i] * depth_weights[i]
        for j in range(N):
            result += (float(face_local_wse[j]) - base_val) * vel_weights[N + j] * depth_weights[N + j]
        return result / w_sum + base_val
    else:
        result = 0.0
        for i in range(N):
            result += (float(fp_local_wse[i]) - base_val) * vel_weights[i]
        for j in range(N):
            result += (float(face_local_wse[j]) - base_val) * vel_weights[N + j]
        return result + base_val


@njit(cache=True)
def _compute_cell_pixel_weights(
    pxs: np.ndarray,
    pys: np.ndarray,
    verts_x: np.ndarray,
    verts_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute barycentric and donated weights for all pixels of one cell.

    Runs :func:`_barycentric_weights` and :func:`_donate` for every pixel
    in a single Numba dispatch, eliminating per-pixel Python→Numba overhead.

    Returns
    -------
    fw_batch : float32 ``(n_pixels, N)``
        Barycentric weights for each pixel over the N polygon corners.
    vel_w_batch : float64 ``(n_pixels, 2*N)``
        Donated weights for the full 2N-point stencil (N corners + N faces).
    """
    n = len(pxs)
    N = len(verts_x)
    fw_batch    = np.empty((n, N),      dtype=np.float32)
    vel_w_batch = np.empty((n, 2 * N),  dtype=np.float64)
    for i in range(n):
        fw              = _barycentric_weights(pxs[i], pys[i], verts_x, verts_y)
        fw_batch[i]     = fw
        vel_w_batch[i]  = _donate(fw)
    return fw_batch, vel_w_batch


# ---------------------------------------------------------------------------
# Step 4 — Build cell-ID raster
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _scanline_rasterize_nb(
    poly_xy: np.ndarray,       # (total_verts, 2) float64 — all polygons concatenated
    poly_offsets: np.ndarray,  # (n_polys + 1,) int32 — CSR: poly i → [offsets[i]:offsets[i+1]]
    wet_mask: np.ndarray,      # (n_polys,) bool
    tf_c: float,               # transform.c — x coordinate of left edge of col 0
    tf_f: float,               # transform.f — y coordinate of top  edge of row 0
    tf_a: float,               # transform.a — pixel width  (> 0)
    tf_e: float,               # transform.e — pixel height (< 0)
    H: int,
    W: int,
    out: np.ndarray,           # (H, W) int32, zeroed before call, mutated in-place
) -> None:
    """Scan-line polygon rasterization matching ``RasterizePolygon.ComputeCells``.

    For each wet cell polygon:

    1. Walk every non-horizontal edge; for each row whose center-Y is **strictly
       above the lower endpoint and at-or-below the upper endpoint** (half-open
       interval ``y_lo < y_c <= y_hi``), record the X-intersection.  This
       convention handles degenerate vertices (polygon tip exactly on a center-Y)
       correctly without the ``IsVertexVerticallyBetweenNeighbors`` special-case:

       * ``∧`` tip (both neighbors below): two intersections at ``x_v`` → empty
         fill range → no pixels painted (bounce). ✓
       * ``∨`` tip (both neighbors above): zero intersections → row skipped. ✓
       * True crossing (neighbors on opposite sides): exactly one intersection. ✓

    2. Sort the intersections for the row; fill between each consecutive pair
       using the same center-point rule as RASMapper: include column *c* if
       ``x_left ≤ center_x(c) ≤ x_right``.

    Results match ``rasterio.features.rasterize(all_touched=False)`` for all
    well-behaved interior pixels; the two algorithms may differ by ≤ 1 pixel at
    polygon edges where a vertex Y lands exactly on a row's center Y.

    **Parallelism** (``parallel=True``, ``prange`` over polygons)

    The outer loop is parallelised because HEC-RAS 2D meshes are conformal
    finite-volume grids: cells share faces exactly with no geometric overlap,
    so no two polygons ever fill the same pixel under the center-point rule.
    Empirically verified with ``diag_scanline_overlap.py`` on the Tulloch mesh
    (1 452 cells) at pixel sizes 5 m, 2 m, 1 m, and 0.5 m — zero conflicts
    found across all 702 K pixels at the finest resolution.
    """
    py = -tf_e   # positive pixel height
    px =  tf_a   # positive pixel width
    x_raster_max = tf_c + W * px

    # HEC-RAS 2D mesh cells tile the domain without overlap, so no two
    # polygons ever fill the same pixel (verified — see docstring).
    # prange writes to out[r,c] are conflict-free across threads.
    n_polys = len(wet_mask)
    for pi in prange(n_polys):
        if not wet_mask[pi]:
            continue

        # Per-polygon intersection buffer — thread-local (inside prange).
        # Max intersections per row = n_verts; 64 covers all HEC-RAS cells.
        xs_buf = np.empty(64, np.float64)

        v_start = int(poly_offsets[pi])
        v_end   = int(poly_offsets[pi + 1])
        n_verts = v_end - v_start

        # Drop a repeated closing vertex (first == last).
        while n_verts > 3:
            if (poly_xy[v_start + n_verts - 1, 0] == poly_xy[v_start, 0] and
                    poly_xy[v_start + n_verts - 1, 1] == poly_xy[v_start, 1]):
                n_verts -= 1
            else:
                break

        if n_verts < 3:
            continue

        cell_val = np.int32(pi + 1)  # 1-based

        # --- Polygon bounding box -------------------------------------------
        y_min = poly_xy[v_start, 1]
        y_max = poly_xy[v_start, 1]
        for vi in range(v_start + 1, v_start + n_verts):
            y = poly_xy[vi, 1]
            if y < y_min:
                y_min = y
            if y > y_max:
                y_max = y

        # Row range.  Pixel (r, c) center: y_c = tf_f + (r + 0.5) * tf_e
        # Since tf_e < 0, y decreases as r increases.
        # Row containing y: r = floor((tf_f - y) / py)
        r_top = int(np.floor((tf_f - y_max) / py))
        r_bot = int(np.floor((tf_f - y_min) / py))
        if r_top < 0:
            r_top = 0
        if r_bot >= H:
            r_bot = H - 1
        if r_top > r_bot:
            continue

        # --- Scan rows -------------------------------------------------------
        for r in range(r_top, r_bot + 1):
            yc = tf_f + (r + 0.5) * tf_e   # pixel center Y (= tf_f - (r+0.5)*py)

            n_xs = 0
            for k in range(n_verts):
                k_next = k + 1
                if k_next == n_verts:
                    k_next = 0

                y0 = poly_xy[v_start + k,      1]
                y1 = poly_xy[v_start + k_next, 1]
                x0 = poly_xy[v_start + k,      0]
                x1 = poly_xy[v_start + k_next, 0]

                dy = y1 - y0
                if dy == 0.0:
                    continue  # horizontal edge — skip

                # Half-open interval: include if y_lo < yc <= y_hi
                if dy > 0.0:
                    if not (y0 < yc <= y1):
                        continue
                else:
                    if not (y1 < yc <= y0):
                        continue

                xi = x0 + (yc - y0) * (x1 - x0) / dy
                if xi < x_raster_max and n_xs < 64:
                    xs_buf[n_xs] = xi
                    n_xs += 1

            if n_xs < 2:
                continue

            # Insertion sort (n_xs is small — ≤ n_verts)
            for i in range(1, n_xs):
                key = xs_buf[i]
                j = i - 1
                while j >= 0 and xs_buf[j] > key:
                    xs_buf[j + 1] = xs_buf[j]
                    j -= 1
                xs_buf[j + 1] = key

            # Fill between consecutive pairs of intersections
            i = 0
            while i + 1 < n_xs:
                x_left  = xs_buf[i]
                x_right = xs_buf[i + 1]
                i += 2

                if x_right <= tf_c:
                    continue  # pair lies entirely left of raster

                # Left column: first c where center_x(c) >= x_left
                # Mirrors RasMapper: col = floor((x-tf_c)/px); if x > center_x(col): col++
                c_left = int(np.floor((x_left - tf_c) / px))
                if c_left < 0:
                    c_left = 0
                elif c_left < W:
                    if x_left > tf_c + (c_left + 0.5) * px:
                        c_left += 1

                # Right column: last c where center_x(c) <= x_right
                c_right = int(np.floor((x_right - tf_c) / px))
                if c_right >= W:
                    c_right = W - 1
                elif c_right >= 0:
                    if x_right < tf_c + (c_right + 0.5) * px:
                        c_right -= 1

                if c_left < 0:
                    c_left = 0
                if c_right >= W:
                    c_right = W - 1
                if c_left > c_right:
                    continue

                for c in range(c_left, c_right + 1):
                    out[r, c] = cell_val


def build_cell_id_raster(
    cell_polygons: list[np.ndarray],
    wet_mask: np.ndarray,
    transform: "rasterio.transform.Affine",
    height: int,
    width: int,
    use_scanline: bool = True,
) -> np.ndarray:
    """Rasterize cell ownership: pixel value = cell_idx + 1, 0 = outside.

    Only wet cells (``wet_mask[c] == True``) are rasterized.

    Mirrors ``RasterizePolygon.ComputeCells``
    (``archive/DLLs/RasMapperLib/RasterizePolygon.cs``) called from
    ``MeshFV2D.cs``.

    **Scan-line path** (``use_scanline=True``, default)

    Delegates to :func:`_scanline_rasterize_nb` — a Numba JIT scan-line fill
    that closely follows RasMapperLib:

    * *Edge rule:* half-open interval ``y_lo < y_c ≤ y_hi`` (vs RASMapper's
      ``IsVertexVerticallyBetweenNeighbors``).  Both handle degenerate vertices
      (polygon tip exactly on a center-Y row) correctly; they may disagree on
      ≤ 1 pixel at those positions.
    * *Fill rule:* ``x_left ≤ center_x(c) ≤ x_right`` — identical to
      RASMapper.
    * ~22× faster than the GDAL path on a typical mesh (no Shapely
      allocation, single compiled loop).

    **GDAL path** (``use_scanline=False``)

    Delegates to ``rasterio.features.rasterize`` with ``all_touched=False``
    (GDAL center-point test).  Requires ``shapely``.  Both paths produce
    pixel-perfect agreement for all well-behaved interior pixels; edge pixels
    at polygon boundaries may differ by ≤ 1 pixel.

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
    use_scanline:
        ``True`` (default) — Numba scan-line path.
        ``False`` — GDAL / ``rasterio.features.rasterize`` path.

    Returns
    -------
    cell_id_grid : int32 ndarray, shape ``(height, width)``
        ``cell_id_grid[r, c] = cell_idx + 1`` for the owning cell (1-based),
        or 0 where no cell covers that pixel.
    """
    if use_scanline:
        # --- Numba scan-line path -------------------------------------------
        # Build CSR representation: flat (total_verts, 2) array + int32 offsets.
        n_polys = len(cell_polygons)
        offsets = np.zeros(n_polys + 1, dtype=np.int32)
        for ci in range(n_polys):
            offsets[ci + 1] = offsets[ci] + len(cell_polygons[ci])
        total_verts = int(offsets[n_polys])
        poly_xy = np.empty((total_verts, 2), dtype=np.float64)
        for ci in range(n_polys):
            s = int(offsets[ci])
            e = int(offsets[ci + 1])
            if e > s:
                poly_xy[s:e] = cell_polygons[ci]

        out = np.zeros((height, width), dtype=np.int32)
        _scanline_rasterize_nb(
            poly_xy, offsets,
            wet_mask.astype(np.bool_),
            float(transform.c), float(transform.f),
            float(transform.a), float(transform.e),
            height, width,
            out,
        )
        return out

    import rasterio.features
    from shapely.geometry import Polygon as _Polygon

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

    return rasterio.features.rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.int32,
        all_touched=False,
    )


# ---------------------------------------------------------------------------
# Terrain sampling at facepoints
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _sample_terrain_nb(
    fp_coords: np.ndarray,
    terrain_grid: np.ndarray,
    a: float, c: float,
    d: float, f: float,
) -> np.ndarray:
    """Numba kernel: bilinear terrain sampling at facepoint coordinates.

    ``a, c`` are the x-scale and x-origin from the rasterio affine transform;
    ``d, f`` are the y-scale (negative for north-up) and y-origin.
    """
    H, W = terrain_grid.shape
    n_fp = len(fp_coords)
    fp_elev = np.full(n_fp, np.nan)

    for i in prange(n_fp):
        col_f = (fp_coords[i, 0] - c) / a - 0.5
        row_f = (fp_coords[i, 1] - f) / d - 0.5
        c0 = int(np.floor(col_f))
        r0 = int(np.floor(row_f))
        c1 = c0 + 1
        r1 = r0 + 1
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            continue
        dc = col_f - c0
        dr = row_f - r0
        v00 = terrain_grid[r0, c0]
        v01 = terrain_grid[r0, c1]
        v10 = terrain_grid[r1, c0]
        v11 = terrain_grid[r1, c1]
        if np.isnan(v00) or np.isnan(v01) or np.isnan(v10) or np.isnan(v11):
            continue
        fp_elev[i] = (
            v00 * (1.0 - dr) * (1.0 - dc)
            + v01 * (1.0 - dr) * dc
            + v10 * dr * (1.0 - dc)
            + v11 * dr * dc
        )

    return fp_elev


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
    # Extract scalar affine coefficients — rasterio Affine objects cannot be
    # passed into Numba @njit functions.
    # Rasterio affine: x = c + a*col, y = f + d*row  (north-up: a>0, d<0)
    a_coef = transform.a
    c_coef = transform.c
    d_coef = transform.e  # rasterio uses .e for the y-scale
    f_coef = transform.f
    grid = np.asarray(terrain_grid, dtype=np.float64)
    coords = np.asarray(fp_coords, dtype=np.float64)
    return _sample_terrain_nb(coords, grid, a_coef, c_coef, d_coef, f_coef)


# ---------------------------------------------------------------------------
# Flat-cell velocity helper
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def compute_cell_flat_velocities(
    cell_face_info: np.ndarray,
    cell_face_values: np.ndarray,
    face_normal_vel: np.ndarray,
    face_normals_2d: np.ndarray,
    flat_wet_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Whole-cell least-squares velocity for small (flat) cells.

    Mirrors ``MeshFV2D.ComputeCellValue_FacePerpLeastSquares`` from
    ``archive/DLLs/RasMapperLib/MeshFV2D.cs``, called by
    ``ComputeFromFacePerpValues(FlatMeshMap, ...)`` in ``Renderer.cs``.

    RASMapper classifies each cell as *flat* when its plan-view area is at or
    below ``pixel_size² × PixelRenderingCutoff`` (cutoff = 5, defined in
    ``MeshFV2D.cs``).  Flat cells receive a single uniform velocity vector
    painted over all their pixels, rather than the C-stencil + facepoint
    interpolation used for larger (sloping) cells.

    The vector is the least-squares solution to ``V · nᵢ ≈ vᵢ`` over all
    faces of the cell, where ``nᵢ`` is the unit face normal and ``vᵢ`` is
    the face-perpendicular velocity:

    .. code-block:: text

        M  = Σᵢ nᵢ nᵢᵀ        (2×2 symmetric normal matrix)
        B  = Σᵢ nᵢ vᵢ
        V  = M⁻¹ B

    Parameters
    ----------
    cell_face_info : ndarray, shape ``(n_cells, 2)``
        ``[start, count]`` CSR offsets into *cell_face_values*.
    cell_face_values : ndarray, shape ``(n_csr, 2)``
        ``[face_index, orientation]`` pairs in CCW order per cell.
    face_normal_vel : ndarray, shape ``(n_faces,)``
        Signed face-perpendicular velocity (positive = A→B direction).
    face_normals_2d : ndarray, shape ``(n_faces, 2)``
        Unit normal vectors ``[nx, ny]`` for each face (A→B direction).
    flat_wet_mask : ndarray, shape ``(n_cells,)``, bool
        ``True`` for cells that are both flat AND wet; all others are skipped.

    Returns
    -------
    vx, vy : ndarray, shape ``(n_cells,)``
        Velocity components.  Cells not in *flat_wet_mask* are left at 0.0.
    """
    n_cells = flat_wet_mask.shape[0]
    vx = np.zeros(n_cells, dtype=np.float64)
    vy = np.zeros(n_cells, dtype=np.float64)

    for ci in prange(n_cells):
        if not flat_wet_mask[ci]:
            continue
        start = int(cell_face_info[ci, 0])
        count = int(cell_face_info[ci, 1])
        if count < 3:
            continue  # degenerate cell

        # Accumulate normal matrix M = Σ nᵢnᵢᵀ and rhs B = Σ nᵢvᵢ.
        A11 = 0.0; A22 = 0.0; A12 = 0.0
        B1  = 0.0; B2  = 0.0
        for k in range(count):
            fi  = int(cell_face_values[start + k, 0])
            nx  = float(face_normals_2d[fi, 0])
            ny  = float(face_normals_2d[fi, 1])
            v   = float(face_normal_vel[fi])
            A11 += nx * nx
            A22 += ny * ny
            A12 += nx * ny
            B1  += nx * v
            B2  += ny * v

        # Invert M analytically and solve V = M⁻¹B.
        # FaceVelocityCoef.Complete() / Compute() in FaceVelocityCoef.cs.
        det = A11 * A22 - A12 * A12
        if det == 0.0:
            # Degenerate matrix: fallback to uniform average (1/count weight).
            k_inv = 1.0 / count
            vx[ci] = k_inv * B1
            vy[ci] = k_inv * B2
        else:
            inv_det = 1.0 / det
            vx[ci] = (A22 * B1 - A12 * B2) * inv_det
            vy[ci] = (-A12 * B1 + A11 * B2) * inv_det

    return vx, vy


# ---------------------------------------------------------------------------
# Step 4 — Variable flags for Numba kernel dispatch
# ---------------------------------------------------------------------------

_VAR_WSE      = 0  # "water_surface"
_VAR_DEPTH    = 1  # "depth"
_VAR_SPEED    = 2  # "speed" (1-band magnitude)
_VAR_VELOCITY = 3  # "velocity" (4-band: Vx, Vy, speed, direction)

_VARIABLE_FLAGS: dict[str, int] = {
    "water_surface": _VAR_WSE,
    "depth":         _VAR_DEPTH,
    "speed":         _VAR_SPEED,
    "velocity":      _VAR_VELOCITY,
}


# ---------------------------------------------------------------------------
# Step 4 — Numba parallel cell-rasterization kernel
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _rasterize_cells_nb(
    # Sorted pixel arrays (indexed in cell-sorted order)
    pix_rows: np.ndarray,           # (n_valid,) int64
    pix_cols: np.ndarray,           # (n_valid,) int64
    pix_xs: np.ndarray,             # (n_valid,) float64
    pix_ys: np.ndarray,             # (n_valid,) float64
    # Cell groups (CSR over sorted pixels)
    group_cell_idxs: np.ndarray,    # (n_groups,) int64 — 0-based cell index
    group_starts: np.ndarray,       # (n_groups,) int64 — start in sorted arrays
    group_ends: np.ndarray,         # (n_groups,) int64 — exclusive end
    # Mesh geometry
    cell_face_info: np.ndarray,     # (n_cells, 2) int64
    cell_face_values: np.ndarray,   # (total_cf, 2) int64
    face_facepoint_indexes: np.ndarray,  # (n_faces, 2) int64
    face_cell_indexes: np.ndarray,  # (n_faces, 2) int64
    face_min_elev: np.ndarray,      # (n_faces,) float64
    fp_coords: np.ndarray,          # (n_fp, 2) float64
    # WSE
    cell_wse: np.ndarray,           # (n_cells,) float64
    face_value_a: np.ndarray,       # (n_faces,) float64
    face_value_b: np.ndarray,       # (n_faces,) float64
    fp_wse: np.ndarray,             # (n_faces, 2) float64  or  (0, 2) sentinel
    has_fp_wse: bool,
    # Terrain
    terrain_grid: np.ndarray,       # (H_t, W_t) float64  or  (1, 1) sentinel
    has_terrain: bool,
    # Velocity data
    fp_vel_data: np.ndarray,        # (total, 2) float64  or  (0, 2) sentinel
    fp_face_info_arr: np.ndarray,   # (n_fp, 2) int64  or  (0, 2) sentinel
    face_fp_local_idx: np.ndarray,  # (n_faces, 2) int32  or  (0, 2) sentinel
    face_vel_A: np.ndarray,         # (n_faces, 2) float64  or  (0, 2) sentinel
    face_vel_B: np.ndarray,         # (n_faces, 2) float64  or  (0, 2) sentinel
    has_vel_data: bool,
    # Flat-cell velocity
    flat_cell_vx: np.ndarray,       # (n_cells,) float64  or  (0,) sentinel
    flat_cell_vy: np.ndarray,       # (n_cells,) float64  or  (0,) sentinel
    has_flat_vel: bool,
    # Facepoint terrain elevation (depth-weight rebalancing)
    fp_elev: np.ndarray,            # (n_fp,) float64  or  (0,) sentinel
    has_fp_elev: bool,
    face_hconn: np.ndarray,         # (n_faces,) uint8
    # Render options
    variable_flag: int,             # _VAR_* constant
    nodata: float,
    depth_threshold: float,
    with_faces: bool,
    use_depth_weights: bool,
    shallow_to_flat: bool,
    # Output: always (n_bands, H, W) — n_bands=1 for scalar, 4 for velocity
    output: np.ndarray,
) -> None:
    """Numba parallel kernel for Step 4 pixel rasterization.

    Iterates over cell groups with ``prange``; each group owns a disjoint
    slice of ``pix_rows/cols/xs/ys`` (built from the sort-based CSR in the
    Python wrapper).  Because every pixel belongs to exactly one cell the
    writes to ``output`` are conflict-free across threads.

    Replicates the per-cell and per-pixel logic of ``_rasterize_rasmap``
    inside a single ``@njit(parallel=True)`` function so the Python
    interpreter overhead of the outer cell loop is eliminated entirely.

    String inputs (``variable``) are replaced by integer flags
    (``_VAR_*`` constants); ``None`` inputs are replaced by sentinel empty
    arrays accompanied by companion ``bool`` flags.
    """
    n_groups = len(group_cell_idxs)

    for gi in prange(n_groups):
        cell_idx = int(group_cell_idxs[gi])
        g_start  = int(group_starts[gi])
        g_end    = int(group_ends[gi])

        # ---- Build ordered facepoint polygon --------------------------------
        cf_start = int(cell_face_info[cell_idx, 0])
        count    = int(cell_face_info[cell_idx, 1])
        if count < 3:
            continue

        N = count
        face_indices = np.empty(N, dtype=np.int64)
        face_orients = np.empty(N, dtype=np.int64)
        verts_fp     = np.empty(N, dtype=np.int64)
        verts_x      = np.empty(N, dtype=np.float64)
        verts_y      = np.empty(N, dtype=np.float64)

        for k in range(N):
            fi  = int(cell_face_values[cf_start + k, 0])
            ori = int(cell_face_values[cf_start + k, 1])
            face_indices[k] = fi
            face_orients[k] = ori
            fp_col = 0 if ori > 0 else 1
            fp_i   = int(face_facepoint_indexes[fi, fp_col])
            verts_fp[k] = fp_i
            verts_x[k]  = fp_coords[fp_i, 0]
            verts_y[k]  = fp_coords[fp_i, 1]

        # ---- Per-cell WSE arrays --------------------------------------------
        fp_local_wse     = np.full(N, _NODATA, dtype=np.float64)
        fp_local_wse_adj = np.full(N, _NODATA, dtype=np.float64)
        face_local_wse   = np.full(N, _NODATA, dtype=np.float64)
        use_sloped  = False
        has_face_wse = False

        if has_fp_wse:
            for k in range(N):
                fi  = face_indices[k]
                col = 0 if face_orients[k] > 0 else 1
                fp_local_wse[k] = fp_wse[fi, col]

            has_valid = False
            for k in range(N):
                if fp_local_wse[k] != _NODATA:
                    has_valid = True
                    break

            if with_faces:
                for k in range(N):
                    fi = face_indices[k]
                    is_cellA = cell_idx == int(face_cell_indexes[fi, 0])
                    face_local_wse[k] = face_value_a[fi] if is_cellA else face_value_b[fi]
                has_face_wse = True
                if not has_valid:
                    for k in range(N):
                        if face_local_wse[k] != _NODATA:
                            has_valid = True
                            break

            use_sloped = has_valid

        # ---- Shallow-to-flat ------------------------------------------------
        if use_sloped and shallow_to_flat:
            if _all_shallow(cell_idx, cell_face_info, cell_face_values, face_hconn):
                use_sloped   = False
                has_face_wse = False

        # ---- Substitute NODATA facepoints with cell WSE ---------------------
        if use_sloped:
            cws = float(cell_wse[cell_idx])
            for k in range(N):
                fp_local_wse_adj[k] = cws if fp_local_wse[k] == _NODATA else fp_local_wse[k]
            if has_face_wse:
                for k in range(N):
                    if face_local_wse[k] == _NODATA:
                        face_local_wse[k] = cws

        # ---- Depth weights --------------------------------------------------
        cell_dw = np.empty(0, dtype=np.float64)
        if use_sloped and with_faces and use_depth_weights and has_fp_elev:
            cell_dw = _depth_weights_for_cell(
                cell_idx, cell_face_info, cell_face_values,
                face_facepoint_indexes, face_cell_indexes,
                fp_wse, fp_elev, face_value_a, face_value_b, face_min_elev,
            )

        # ---- Per-cell velocity arrays ---------------------------------------
        nb_fp_vx   = np.empty(0, dtype=np.float64)
        nb_fp_vy   = np.empty(0, dtype=np.float64)
        nb_face_vx = np.empty(0, dtype=np.float64)
        nb_face_vy = np.empty(0, dtype=np.float64)
        has_vel_arrays = False

        if (variable_flag == _VAR_SPEED or variable_flag == _VAR_VELOCITY) and has_vel_data:
            nb_fp_vx = np.zeros(N, dtype=np.float64)
            nb_fp_vy = np.zeros(N, dtype=np.float64)
            for i in range(N):
                fp_i = verts_fp[i]
                fi   = face_indices[i]
                fpA  = int(face_facepoint_indexes[fi, 0])
                j_local = int(face_fp_local_idx[fi, 0]) if fp_i == fpA else int(face_fp_local_idx[fi, 1])
                if j_local >= 0:
                    offset = int(fp_face_info_arr[fp_i, 0]) + j_local
                    nb_fp_vx[i] = fp_vel_data[offset, 0]
                    nb_fp_vy[i] = fp_vel_data[offset, 1]

            nb_face_vx = np.zeros(N, dtype=np.float64)
            nb_face_vy = np.zeros(N, dtype=np.float64)
            for j in range(N):
                fi  = face_indices[j]
                ori = face_orients[j]
                if ori > 0:
                    nb_face_vx[j] = face_vel_A[fi, 0]
                    nb_face_vy[j] = face_vel_A[fi, 1]
                else:
                    nb_face_vx[j] = face_vel_B[fi, 0]
                    nb_face_vy[j] = face_vel_B[fi, 1]
            has_vel_arrays = True

        # ---- Batch pixel weights --------------------------------------------
        fw_batch, vel_w_batch = _compute_cell_pixel_weights(
            pix_xs[g_start:g_end], pix_ys[g_start:g_end], verts_x, verts_y,
        )

        # ---- Per-pixel loop -------------------------------------------------
        for pi in range(g_end - g_start):
            r = int(pix_rows[g_start + pi])
            c = int(pix_cols[g_start + pi])
            fw    = fw_batch[pi]
            vel_w = vel_w_batch[pi]

            # Terrain elevation for this pixel
            t_elev = 0.0
            if has_terrain:
                t_elev = terrain_grid[r, c]
                if np.isnan(t_elev) or t_elev == _NODATA:
                    continue

            # WSE / wet-dry check
            pixel_wse = _NODATA
            if use_sloped and has_face_wse:
                pixel_wse = _pixel_wse_sloped(vel_w, fp_local_wse_adj, face_local_wse, cell_dw)
                if pixel_wse == _NODATA:
                    continue
                if has_terrain and pixel_wse < t_elev + depth_threshold:
                    continue
            elif use_sloped:
                # JustFacepoints: plain barycentric blend of N corners
                pixel_wse = 0.0
                for k in range(N):
                    pixel_wse += float(fw[k]) * fp_local_wse_adj[k]
                if pixel_wse == _NODATA:
                    continue
                if has_terrain and pixel_wse < t_elev + depth_threshold:
                    continue
            elif has_terrain:
                cws = float(cell_wse[cell_idx])
                if cws < t_elev + depth_threshold:
                    continue
                pixel_wse = cws
            else:
                pixel_wse = float(cell_wse[cell_idx])

            # Value interpolation
            if variable_flag == _VAR_WSE:
                if has_face_wse:
                    val = _pixel_wse_sloped(vel_w, fp_local_wse_adj, face_local_wse, cell_dw)
                    output[0, r, c] = np.float32(val) if val != _NODATA else np.float32(nodata)
                else:
                    output[0, r, c] = np.float32(pixel_wse)

            elif variable_flag == _VAR_DEPTH:
                if has_face_wse:
                    pix_wse = _pixel_wse_sloped(vel_w, fp_local_wse_adj, face_local_wse, cell_dw)
                else:
                    pix_wse = pixel_wse
                dep = pix_wse - t_elev
                output[0, r, c] = np.float32(max(0.0, dep)) if dep > 0.0 else np.float32(nodata)

            elif variable_flag == _VAR_SPEED or variable_flag == _VAR_VELOCITY:
                if has_flat_vel and not np.isnan(flat_cell_vx[cell_idx]):
                    Vx = float(flat_cell_vx[cell_idx])
                    Vy = float(flat_cell_vy[cell_idx])
                elif has_vel_arrays:
                    Vx = 0.0
                    Vy = 0.0
                    for k in range(N):
                        Vx += vel_w[k] * nb_fp_vx[k]
                        Vy += vel_w[k] * nb_fp_vy[k]
                    for k in range(N):
                        Vx += vel_w[N + k] * nb_face_vx[k]
                        Vy += vel_w[N + k] * nb_face_vy[k]
                else:
                    continue
                spd = np.sqrt(Vx * Vx + Vy * Vy)
                if variable_flag == _VAR_SPEED:
                    output[0, r, c] = np.float32(spd)
                else:
                    direction = np.degrees(np.arctan2(Vx, Vy)) % 360.0
                    output[0, r, c] = np.float32(Vx)
                    output[1, r, c] = np.float32(Vy)
                    output[2, r, c] = np.float32(spd)
                    output[3, r, c] = np.float32(direction)


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
    fp_vel_data: np.ndarray | None,
    fp_face_info: np.ndarray | None,
    face_fp_local_idx: np.ndarray | None,
    replaced_face_vel: np.ndarray | None,
    face_vel_A: np.ndarray | None,
    face_vel_B: np.ndarray | None,
    fp_elev: np.ndarray | None,
    face_hconn: np.ndarray,
    nodata: float,
    depth_threshold: float = _MIN_WS_PLOT_TOLERANCE,
    with_faces: bool = True,
    use_depth_weights: bool = False,
    shallow_to_flat: bool = False,
    flat_cell_vx: np.ndarray | None = None,
    flat_cell_vy: np.ndarray | None = None,
) -> np.ndarray:
    """Step 4 — pixel-level barycentric interpolation for all wet cells.

    Public wrapper that delegates to :func:`_rasterize_cells_nb`, a
    ``@njit(parallel=True)`` Numba kernel that processes cells in parallel
    via ``prange``.  Pixel ownership is conflict-free because each pixel
    in ``cell_id_grid`` belongs to exactly one cell.

    For the full algorithm description and parameter documentation see
    :func:`_rasterize_rasmap` (the serial reference implementation).

    **Performance** (1 414-cell mesh, median cell area ~48 sq ft,
    4 logical cores, Numba ``prange`` parallel kernel vs serial loop):

    .. list-table::
       :header-rows: 1
       :widths: 20 20 12 12 10

       * - variable
         - render_mode
         - parallel
         - serial
         - speedup
       * - wse
         - sloping
         - 599 ms
         - 10 878 ms
         - 18x
       * - wse
         - hybrid (s2f + dw)
         - 738 ms
         - 8 375 ms
         - 11x
       * - depth
         - hybrid
         - 733 ms
         - 9 109 ms
         - 12x
       * - velocity
         - horizontal
         - 684 ms
         - 8 188 ms
         - 12x
       * - velocity
         - hybrid
         - 399 ms
         - 7 892 ms
         - 20x

    Thread count is controlled by ``NUMBA_NUM_THREADS`` or
    :func:`numba.set_num_threads` at runtime.
    """
    import rasterio.transform as _rt

    _log.info(
        "rasterize_rasmap: %d/%d cores available for parallel kernels "
        "(NUMBA_NUM_THREADS=%s)",
        _numba.get_num_threads(),
        os.cpu_count() or 1,
        os.environ.get("NUMBA_NUM_THREADS", "unset"),
    )

    variable_flag    = _VARIABLE_FLAGS[variable]
    is_velocity_4band = variable == "velocity"
    n_bands = 4 if is_velocity_4band else 1

    H, W = cell_id_grid.shape
    output_nb = np.full((n_bands, H, W), nodata, dtype=np.float32)

    valid_mask = cell_id_grid > 0
    valid_rows, valid_cols = np.where(valid_mask)
    if len(valid_rows) == 0:
        return output_nb[0] if not is_velocity_4band else output_nb

    xs, ys = _rt.xy(transform, valid_rows, valid_cols)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    cell_ids = cell_id_grid[valid_rows, valid_cols]

    # Sort pixels by cell_id → CSR grouping (same logic as _rasterize_rasmap)
    sort_order    = np.argsort(cell_ids, kind="stable")
    sorted_cids   = cell_ids[sort_order]
    unique_cids, grp_starts = np.unique(sorted_cids, return_index=True)
    grp_ends          = np.empty(len(unique_cids), dtype=np.int64)
    grp_ends[:-1]     = grp_starts[1:]
    grp_ends[-1]      = len(sorted_cids)
    group_cell_idxs   = (unique_cids - 1).astype(np.int64)  # 0-based

    pix_rows = valid_rows[sort_order].astype(np.int64)
    pix_cols = valid_cols[sort_order].astype(np.int64)
    pix_xs   = xs[sort_order]
    pix_ys   = ys[sort_order]

    # Sentinel arrays for optional inputs (Numba requires consistent types)
    _e2f  = np.empty((0, 2), dtype=np.float64)   # (0,2) float64 sentinel
    _e2i  = np.empty((0, 2), dtype=np.int64)      # (0,2) int64 sentinel
    _e2i32= np.empty((0, 2), dtype=np.int32)      # (0,2) int32 sentinel
    _e1f  = np.empty(0,      dtype=np.float64)    # (0,)  float64 sentinel
    _terr = np.empty((1, 1), dtype=np.float64)    # terrain sentinel (non-zero shape)

    _rasterize_cells_nb(
        pix_rows, pix_cols, pix_xs, pix_ys,
        group_cell_idxs,
        grp_starts.astype(np.int64),
        grp_ends,
        # Mesh geometry
        np.asarray(cell_face_info,          dtype=np.int64),
        np.asarray(cell_face_values,        dtype=np.int64),
        np.asarray(face_facepoint_indexes,  dtype=np.int64),
        np.asarray(face_cell_indexes,       dtype=np.int64),
        np.asarray(face_min_elev,           dtype=np.float64),
        np.asarray(fp_coords,               dtype=np.float64),
        # WSE
        np.asarray(cell_wse,     dtype=np.float64),
        np.asarray(face_value_a, dtype=np.float64),
        np.asarray(face_value_b, dtype=np.float64),
        np.asarray(fp_wse,       dtype=np.float64) if fp_wse is not None else _e2f,
        fp_wse is not None,
        # Terrain
        np.asarray(terrain_grid, dtype=np.float64) if terrain_grid is not None else _terr,
        terrain_grid is not None,
        # Velocity
        np.asarray(fp_vel_data,       dtype=np.float64) if fp_vel_data       is not None else _e2f,
        np.asarray(fp_face_info,      dtype=np.int64)   if fp_face_info      is not None else _e2i,
        np.asarray(face_fp_local_idx, dtype=np.int32)   if face_fp_local_idx is not None else _e2i32,
        np.asarray(face_vel_A,        dtype=np.float64) if face_vel_A        is not None else _e2f,
        np.asarray(face_vel_B,        dtype=np.float64) if face_vel_B        is not None else _e2f,
        fp_vel_data is not None,
        # Flat-cell velocity
        np.asarray(flat_cell_vx, dtype=np.float64) if flat_cell_vx is not None else _e1f,
        np.asarray(flat_cell_vy, dtype=np.float64) if flat_cell_vy is not None else _e1f,
        flat_cell_vx is not None,
        # Facepoint elevation
        np.asarray(fp_elev, dtype=np.float64) if fp_elev is not None else _e1f,
        fp_elev is not None,
        np.asarray(face_hconn, dtype=np.uint8),
        # Options
        variable_flag,
        float(nodata),
        float(depth_threshold),
        with_faces,
        use_depth_weights,
        shallow_to_flat,
        # Output
        output_nb,
    )

    return output_nb if is_velocity_4band else output_nb[0]


def _rasterize_rasmap(
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
    fp_vel_data: np.ndarray | None,
    fp_face_info: np.ndarray | None,
    face_fp_local_idx: np.ndarray | None,
    replaced_face_vel: np.ndarray | None,
    face_vel_A: np.ndarray | None,
    face_vel_B: np.ndarray | None,
    fp_elev: np.ndarray | None,
    face_hconn: np.ndarray,
    nodata: float,
    depth_threshold: float = _MIN_WS_PLOT_TOLERANCE,
    with_faces: bool = True,
    use_depth_weights: bool = False,
    shallow_to_flat: bool = False,
    flat_cell_vx: np.ndarray | None = None,
    flat_cell_vy: np.ndarray | None = None,
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
    fp_vel_data, fp_face_info, face_fp_local_idx, replaced_face_vel:
        From Steps 3 / 3.5; required for ``"speed"`` (velocity magnitude).
        ``fp_vel_data`` shape ``(total_fp_face_entries, 2)`` — CSR flat array
        indexed via ``fp_face_info[fp, 0] + j``; ``face_fp_local_idx`` shape
        ``(n_faces, 2)`` int32 — local ring index of face ``fi`` in fpA (col 0)
        and fpB (col 1) rings.
    face_vel_A, face_vel_B:
        ``(n_faces, 2)`` from Step 2; required for ``"speed"``.
    fp_elev:
        ``(n_fp,)`` terrain elevation sampled at facepoints; enables
        depth-weighted rebalancing (``PaintCell_8Stencil_RebalanceWeights``).
        Pass ``None`` to skip rebalancing.
    face_hconn:
        ``(n_faces,)`` uint8 — hydraulic-connection classification from
        :func:`compute_face_wss` (``HC_*`` constants).  Used by the
        all-shallow check (``shallow_to_flat``): only ``HC_BACKFILL``,
        ``HC_DOWNHILL_DEEP``, and ``HC_DOWNHILL_INTERMEDIATE`` clear the
        all-shallow flag; ``HC_DOWNHILL_SHALLOW`` is connected but does not
        clear it (C# ``Renderer.cs:3094``).
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
        When ``True``, cells where every bounding face has ``HC_NONE``,
        ``HC_LEVEE``, or ``HC_DOWNHILL_SHALLOW`` (i.e. no face with an
        active hydraulic connection) are rendered with the flat cell-average
        WSE instead of barycentric interpolation.  Matches
        ``ShallowBehavior.ReduceToHorizontal`` in ``Renderer.cs``.
        Default ``False`` (matches RASMapper default).
    flat_cell_vx, flat_cell_vy : ndarray, shape ``(n_cells,)``, optional
        Pre-computed whole-cell least-squares velocity from
        :func:`compute_cell_flat_velocities`.  ``NaN`` entries mean "use the
        stencil for this cell" (i.e. the cell is not flat or was dry).
        When both are ``None`` (default) the stencil is used for all cells.
        Only consulted for ``variable in ("speed", "velocity")``.

    Returns
    -------
    output : float32 ndarray, shape ``(H, W)``
        Interpolated values; ``nodata`` where dry or outside mesh.
        For ``"velocity"``: 4-band array ``(4, H, W)`` — ``[Vx, Vy, speed, direction_deg]``.
    """
    import rasterio.transform as _rt

    _log.info(
        "_rasterize_rasmap: %d/%d cores available for parallel kernels "
        "(NUMBA_NUM_THREADS=%s)",
        _numba.get_num_threads(),
        os.cpu_count() or 1,
        os.environ.get("NUMBA_NUM_THREADS", "unset"),
    )

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

    # Group pixels by owning cell using a sort-based CSR representation.
    # Avoids an O(n_pixels) Python loop that dominates for large meshes.
    _sort_order  = np.argsort(cell_ids, kind="stable")
    _sorted_cids = cell_ids[_sort_order]
    _unique_cids, _group_starts = np.unique(_sorted_cids, return_index=True)
    _group_ends       = np.empty(len(_unique_cids), dtype=np.intp)
    _group_ends[:-1]  = _group_starts[1:]
    _group_ends[-1]   = len(_sorted_cids)

    for _gi, raster_id in enumerate(_unique_cids):
        cell_idx = int(raster_id) - 1  # back to 0-based
        pix_arr  = _sort_order[_group_starts[_gi]:_group_ends[_gi]]

        # ---- Build ordered facepoint polygon for this cell ---------------
        # Read the CSR slice: face_indices are in CCW order around the cell;
        # face_orients encode whether this cell is cellA (+1) or cellB (-1)
        # for each face (see cell_face_info docstring for the full invariant).
        start = int(cell_face_info[cell_idx, 0])
        count = int(cell_face_info[cell_idx, 1])
        if count < 3:
            continue  # degenerate cell — skip

        face_indices = cell_face_values[start:start + count, 0].astype(np.int64)
        face_orients = cell_face_values[start:start + count, 1].astype(np.int64)

        # Pick the CCW-entry facepoint for each face.
        # RasMapper invariant (Face.cs / GetFPPrev): for cellA (ori=+1) the
        # face runs fpA→fpB in CCW order, so fpA is the entry point; for
        # cellB (ori=-1) it runs fpB→fpA, so fpB is the entry point.
        # Equivalent to Face.GetFPPrev(cellIdx) in RasMapperLib.
        _fp_col  = np.where(face_orients > 0, 0, 1)
        verts_fp = face_facepoint_indexes[face_indices, _fp_col]
        verts_x  = fp_coords[verts_fp, 0].astype(np.float64)
        verts_y  = fp_coords[verts_fp, 1].astype(np.float64)
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
            fp_local_wse = fp_wse[face_indices, _fp_col].astype(np.float64)
            if with_faces:
                # Face-midpoint WSEs: select the side of the face that belongs
                # to this cell.  face_value_a corresponds to cellA (column 0
                # of face_cell_indexes); face_value_b to cellB (column 1).
                _is_cellA      = cell_idx == face_cell_indexes[face_indices, 0]
                face_local_wse = np.where(
                    _is_cellA, face_value_a[face_indices], face_value_b[face_indices]
                ).astype(np.float64)
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
        # A cell is "all-shallow" when every bounding face has HC_NONE,
        # HC_LEVEE, or HC_DOWNHILL_SHALLOW.  Only HC_BACKFILL, HC_DOWNHILL_DEEP,
        # and HC_DOWNHILL_INTERMEDIATE clear the all-shallow flag
        # (C# Renderer.cs:3094).  When shallow_to_flat=True, all-shallow cells
        # are rendered with the flat cell-average WSE: skipping
        # DownwardAdjustFPValues and the entire sloped paint path.  Nulling
        # fp_local_wse_adj and face_local_wse here makes the pixel loop fall
        # through to the flat branch (B/C) automatically, so no extra logic
        # is needed inside the loop.
        # When shallow_to_flat=False, the all-shallow condition is ignored and
        # sloped interpolation continues as normal (C# non-ReduceToHorizontal).
        if use_sloped and shallow_to_flat and _all_shallow(
            cell_idx, cell_face_info, cell_face_values, face_hconn,
        ):
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
        # Requires fp_elev; silently skipped if unavailable (cell_dw length-0
        # causes _pixel_wse_sloped to use plain donated weights instead).
        cell_dw: np.ndarray = np.empty(0, dtype=np.float64)
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
        if variable in ("speed", "velocity") and fp_vel_data is not None:
            # Corner (facepoint) velocities: use the arc-context facepoint
            # velocity from Step 3.  fp_vel_data is a CSR flat array indexed
            # via fp_face_info[fp, 0] + j_local, where j_local is the local
            # ring index of face fi in fp's ring (face_fp_local_idx[fi, 0/1]).
            # Matches C# GetLocalFacepointValues → fpVelocityRing[fPPrev]
            # .Velocity[...].  Do NOT use replaced_face_vel (face-averaged
            # facepoint velocity) — that was wrong for the pixel stencil.
            nb_fp_vx = np.zeros(N, dtype=np.float64)
            nb_fp_vy = np.zeros(N, dtype=np.float64)
            for i in range(N):
                fp_i = verts_fp[i]
                fi   = face_indices[i]
                fpA = int(face_facepoint_indexes[fi, 0])
                fpB = int(face_facepoint_indexes[fi, 1])
                if fp_i == fpA:
                    j_local = int(face_fp_local_idx[fi, 0])
                else:
                    j_local = int(face_fp_local_idx[fi, 1])
                if j_local >= 0:
                    offset = int(fp_face_info[fp_i, 0]) + j_local
                    nb_fp_vx[i] = float(fp_vel_data[offset, 0])
                    nb_fp_vy[i] = float(fp_vel_data[offset, 1])
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
        pix_rows = valid_rows[pix_arr]
        pix_cols = valid_cols[pix_arr]
        pix_xs   = xs[pix_arr]
        pix_ys   = ys[pix_arr]

        # Pre-batch terrain values (Opportunity C): one vectorised numpy index
        # per cell instead of per-pixel 2-D lookups from Python.
        _t_vals: np.ndarray | None = (
            terrain_grid[pix_rows, pix_cols] if terrain_grid is not None else None
        )

        # Pre-batch pixel weights (Opportunity A): run _barycentric_weights
        # and _donate for every pixel of this cell in a single Numba dispatch,
        # eliminating 2×n_pixels Python→Numba round-trips.
        fw_batch, vel_w_batch = _compute_cell_pixel_weights(
            pix_xs, pix_ys, verts_x, verts_y
        )

        for pi in range(len(pix_arr)):
            r = int(pix_rows[pi])
            c = int(pix_cols[pi])

            fw    = fw_batch[pi]
            vel_w = vel_w_batch[pi]

            # ---- Wet/dry check -------------------------------------------
            # Four branches depending on sloped mode and available data:
            #
            # (A1) Sloped corners + faces (with_faces=True, default):
            #      Interpolate per-pixel WSE from the full 2N-point stencil
            #      (N corner facepoints + N face midpoints, donated weights).
            if use_sloped and fp_local_wse_adj is not None and face_local_wse is not None:
                pixel_wse = _pixel_wse_sloped(vel_w, fp_local_wse_adj, face_local_wse, cell_dw)
                if pixel_wse == _NODATA:
                    continue
                if _t_vals is not None:
                    t_elev = float(_t_vals[pi])
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
                pixel_wse = float(np.dot(fw, fp_local_wse_adj))
                if pixel_wse == _NODATA:
                    continue
                if _t_vals is not None:
                    t_elev = float(_t_vals[pi])
                    if np.isnan(t_elev) or t_elev == _NODATA:
                        continue
                    if pixel_wse < t_elev + depth_threshold:
                        continue
                else:
                    t_elev = 0.0
            # (B) Terrain available but no sloped WSE data: use the uniform
            #     cell-average WSE for the depth test.  Applies when fp_wse
            #     is None (horizontal mode) or cell reverted to flat.
            elif _t_vals is not None:
                t_elev = float(_t_vals[pi])
                if np.isnan(t_elev) or t_elev == _NODATA:
                    continue
                if float(cell_wse[cell_idx]) < t_elev + depth_threshold:
                    continue
                pixel_wse = float(cell_wse[cell_idx])
            # (C) No terrain at all: every pixel in the cell is treated as
            #     wet; depth output will be meaningless so callers should
            #     not request "depth" without a terrain grid.
            else:
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
                # Flat-cell override: cells with area ≤ pixel² × 5 receive a
                # uniform whole-cell least-squares vector (ComputeFromFacePerpValues
                # FlatMeshMap path in C# Renderer.cs).  Non-NaN sentinel marks
                # flat cells; NaN means "use the stencil" for this cell.
                if (flat_cell_vx is not None
                        and not np.isnan(float(flat_cell_vx[cell_idx]))):
                    Vx = float(flat_cell_vx[cell_idx])
                    Vy = float(flat_cell_vy[cell_idx])  # type: ignore[index]
                else:
                    # Sloping cell: donated barycentric blend over the 2N-sample
                    # stencil (corner facepoints + face midpoints).
                    if nb_fp_vx is None:
                        continue
                    # Opportunity D: replace four Python for-loops with np.dot.
                    Vx = float(
                        np.dot(vel_w[:N], nb_fp_vx) + np.dot(vel_w[N:], nb_face_vx)
                    )
                    Vy = float(
                        np.dot(vel_w[:N], nb_fp_vy) + np.dot(vel_w[N:], nb_face_vy)
                    )
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
