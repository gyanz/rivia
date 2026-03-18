"""RASMapper-exact 2D mesh rasterization pipeline.

Implements the pixel-perfect algorithm reverse-engineered from RasMapperLib.dll
(C#/.NET) by CLB Engineering Corporation (2026).  All functions are pure-numpy /
pure-Python with an optional Numba JIT path for the hot pixel-interpolation loop.

Reference
---------
``archive/velocity_rasterizer_standalone/velocity_rasterizer_standalone/
velocity_rasterizer_combined.py`` — validated pixel-perfect against RASMapper
VRT exports (median |diff| = 0.000000 ft/s across all test plans).

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

    Replicates ``compute_face_wss_new`` from ``velocity_rasterizer_combined.py``
    which mirrors ``MeshFV2D.cs`` / ``RASResults.cs`` internal logic.

    For each face the function determines whether water is actively flowing
    across it (hydraulic connectivity) and assigns a WSE value to each side
    (cellA side → ``face_value_a``, cellB side → ``face_value_b``).  These
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
       either cell is dry (``wse ≤ min_elev + _MIN_WS_PLOT_TOLERANCE``) or
       the face is a perimeter face, connectivity is ``False``.  The side
       belonging to a dry or virtual cell gets ``_NODATA``; a wet real cell
       keeps its WSE.

    3. **Both cells below the face invert** (``wse_a ≤ min_elev_face`` and
       ``wse_b ≤ min_elev_face``): water on both sides is ponded below the
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

    5. **Backfill condition** (``(wse_b - wse_a) * (min_elev_b - min_elev_a) ≤ 0``):
       WSE gradient and bed-elevation gradient point in opposite directions —
       the lower cell sits on higher ground, which can occur when flow backs
       up into a depression.  Both sides take ``face_ws`` and connectivity is
       ``True``.

    6. **Deep flow** (``depth_higher ≥ 2 × delta_min_elev``): the water depth
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
            # to face_ws (blend weight of face_ws → 1 as depth_higher →
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
            #   num9 = eff_lower + (depth_higher² - depth_ref_lower²)
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
            #   weight of num9    = (2*Δz - d) / Δz  →  1 at d=Δz, 0 at d=2Δz
            #   weight of face_ws = (d - Δz)  / Δz  →  0 at d=Δz, 1 at d=2Δz
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


def compute_facepoint_wse(
    fp_coords: np.ndarray,
    fp_face_info: np.ndarray,
    fp_face_values: np.ndarray,
    face_facepoint_indexes: np.ndarray,
    face_value_a: np.ndarray,
    face_value_b: np.ndarray,
) -> np.ndarray:
    """Step B — WSE at each facepoint by PlanarRegressionZ fitting.

    Fits a weighted least-squares plane through the WSE values at adjacent
    face midpoints for each facepoint.  Replicates ``ComputeFacePointWSs``
    from ``RASMapper`` (``MeshFV2D.cs``).

    Parameters
    ----------
    fp_coords:
        ``(n_fp, 2)`` — facepoint XY coordinates.
    fp_face_info:
        ``(n_fp, 2)`` int32 — ``[start, count]`` into ``fp_face_values``.
        First element of :attr:`~raspy.hdf.FlowArea.facepoint_face_orientation`
        (HDF: ``FacePoints Face and Orientation Info``).
    fp_face_values:
        ``(total, 2)`` int32 — ``[face_idx, orientation]``.
        Second element of :attr:`~raspy.hdf.FlowArea.facepoint_face_orientation`
        (HDF: ``FacePoints Face and Orientation Values``).
    face_facepoint_indexes:
        ``(n_faces, 2)`` — ``[fpA, fpB]`` for each face.
    face_value_a:
        ``(n_faces,)`` — WSE on the cellA side from :func:`compute_face_wss`.
    face_value_b:
        ``(n_faces,)`` — WSE on the cellB side from :func:`compute_face_wss`.

    Returns
    -------
    fp_wse : float64 ndarray, shape ``(n_fp,)``
        WSE at each facepoint; ``-9999`` where all adjacent faces are dry.
    """
    n_fp = len(fp_coords)
    fp_wse = np.full(n_fp, _NODATA, dtype=np.float64)

    for fp_idx in range(n_fp):
        base_x = float(fp_coords[fp_idx, 0])
        base_y = float(fp_coords[fp_idx, 1])
        fp_start = int(fp_face_info[fp_idx, 0])
        fp_count = int(fp_face_info[fp_idx, 1])

        sumX2 = 0.0; sumX = 0.0
        sumY2 = 0.0; sumY = 0.0
        sumZ = 0.0;  sumXY = 0.0
        sumYZ = 0.0; sumXZ = 0.0
        n = 0

        for j in range(fp_count):
            fi = int(fp_face_values[fp_start + j, 0])
            fpA_idx = int(face_facepoint_indexes[fi, 0])
            fpB_idx = int(face_facepoint_indexes[fi, 1])
            # Application point = midpoint of the face's two endpoint facepoints
            app_x = (float(fp_coords[fpA_idx, 0]) + float(fp_coords[fpB_idx, 0])) * 0.5
            app_y = (float(fp_coords[fpA_idx, 1]) + float(fp_coords[fpB_idx, 1])) * 0.5
            dx = app_x - base_x
            dy = app_y - base_y

            for wse in (float(face_value_a[fi]), float(face_value_b[fi])):
                if wse != _NODATA:
                    sumX2 += dx * dx;  sumX  += dx
                    sumY2 += dy * dy;  sumY  += dy
                    sumZ  += wse;      sumXY += dx * dy
                    sumYZ += dy * wse; sumXZ += dx * wse
                    n += 1

        if n == 0:
            continue
        elif n == 1:
            fp_wse[fp_idx] = sumZ
        elif n == 2:
            fp_wse[fp_idx] = sumZ / 2.0
        else:
            det = (
                sumX2 * (sumY2 * n - sumY * sumY)
                - sumXY * (sumXY * n - sumY * sumX)
                + sumX * (sumXY * sumY - sumY2 * sumX)
            )
            if det == 0.0:
                fp_wse[fp_idx] = sumZ / n
            else:
                fp_wse[fp_idx] = (
                    sumX2 * (sumY2 * sumZ - sumYZ * sumY)
                    - sumXY * (sumXY * sumZ - sumYZ * sumX)
                    + sumXZ * (sumXY * sumY - sumY2 * sumX)
                ) / det

    return fp_wse


# ---------------------------------------------------------------------------
# Step 2 — C-stencil tangential velocity reconstruction
# ---------------------------------------------------------------------------


class _FaceVelocityCoef:
    """Symmetric 2×2 normal-equation matrix for the C-stencil WLS solve.

    Mirrors ``FaceVelocityCoef`` in ``MeshFV2D.cs``.
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

    Mirrors ``solve_face_vector_c`` in ``velocity_rasterizer_combined.py``
    (``MeshFV2D.cs``).
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

    Replicates ``reconstruct_face_velocities_least_squares`` from RASMapper's
    ``MeshFV2D.cs`` (also exposed in ``velocity_rasterizer_combined.py``).

    Algorithm
    ---------
    For every face *f* with unit normal ``n̂ = (fn_x, fn_y)`` and stored
    face-normal velocity ``vn``, the reconstruction is performed **twice** —
    once from each adjacent cell's perspective (cellA and cellB):

    1. **C-stencil selection** — within the current cell, find the two faces
       immediately adjacent to *f* in the cell's face-ordering ring:

       - *cw*  : the face one step clockwise from *f* (index ``(pos+1) % count``)
       - *ccw* : the face one step counter-clockwise (index ``(pos-1) % count``)

       These three faces (cw, ccw, and *f* itself) form the "C-stencil".

    2. **Normal-equation matrix** — accumulate the 2×2 symmetric WLS matrix
       using the unit normals of the three C-stencil faces::

           A = Σ  [nx²   nx·ny]     (sum over cw, ccw, and f)
                  [nx·ny  ny²]

       The inverse ``A⁻¹`` is computed analytically (Cramer's rule).  If
       ``det(A) == 0`` (degenerate geometry), the identity scaled by 1/count
       is substituted so the solve degrades gracefully.

    3. **RHS vector** — using only the *connected* (wet) CW and CCW neighbors,
       assemble::

           B = n̂_cw * vn_cw  +  n̂_ccw * vn_ccw  +  n̂_f * vn_f

    4. **Least-squares solve** — recover the full velocity vector ``(sx, sy)``
       as ``A⁻¹ · B``.  Project it onto the tangential direction
       ``t̂ = (-fn_y, fn_x)`` to obtain the scalar tangential component::

           tangential = sx * t̂_x + sy * t̂_y

    5. **Compose face velocity**::

           Vx = vn * fn_x + tangential * t̂_x
           Vy = vn * fn_y + tangential * t̂_y

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
        Boundary faces (no cellA or cellB) fall back to ``vn * n̂``.
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

    For each facepoint, computes one velocity vector per adjacent face
    (arc-context specific).  Uses **float32 accumulators** to match C#
    ``MeshFV2D.cs:8677`` (``float num5=0f, num6=0f, num7=0f``).

    Replicates ``compute_vertex_velocities`` from
    ``velocity_rasterizer_combined.py``.

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
        Angle-sorted facepoint→face CSR arrays from
        :attr:`~raspy.hdf.FlowArea.facepoint_face_orientation`.
    face_value_a, face_value_b:
        ``(n_faces,)`` — from :func:`compute_face_wss`.

    Returns
    -------
    fp_velocities : list of ndarray, length ``n_fp``
        ``fp_velocities[fp]`` is shape ``(n_adj_faces, 2)`` — one velocity
        vector per adjacent face of this facepoint (arc-context specific).
    fp_face_local_map : dict ``(fp_idx, face_idx) -> local_j``
        Maps ``(facepoint, face)`` to the local index in ``fp_velocities[fp]``.
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

    For each face, the "sloped" velocity is the mean of the arc-context
    velocities at its two endpoint facepoints.  Used for **facepoint** weight
    contributions in Step 4; face midpoint contributions still use the
    original Item1/Item2 from Step 2.

    Replicates ``replace_face_velocities_sloped`` from
    ``velocity_rasterizer_combined.py``.

    Parameters
    ----------
    fp_velocities:
        From :func:`compute_facepoint_velocities`.
    fp_face_local_map:
        From :func:`compute_facepoint_velocities`.
    face_facepoint_indexes:
        ``(n_faces, 2)`` — ``[fpA, fpB]``.

    Returns
    -------
    replaced_face_vel : float64 ndarray, shape ``(n_faces, 2)``
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

    Matches C# ``RASGeometryMapPoints.cs:2956``
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
    C# ``RASGeometryMapPoints.cs`` donate logic.
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
    fp_wse: np.ndarray,
    fp_elev: np.ndarray,
    face_value_a: np.ndarray,
    face_value_b: np.ndarray,
    face_min_elev: np.ndarray,
) -> np.ndarray:
    """Depth weights for PaintCell_8Stencil_RebalanceWeights.

    C# ``WaterSurfaceRenderer.cs:2842 ComputeDepthWeightedValuesPerCell``.
    Returns float64 array of length ``2*count`` (facepoints then face midpoints).
    Minimum depth = 0.01 (matches C#).
    """
    start = int(cell_face_info[cell_idx, 0])
    count = int(cell_face_info[cell_idx, 1])
    dw = np.full(2 * count, 0.01, dtype=np.float64)

    for k in range(count):
        fi  = int(cell_face_values[start + k, 0])
        ori = int(cell_face_values[start + k, 1])
        # Facepoint for this face edge (orientation-based selection)
        fp  = int(face_facepoint_indexes[fi, 0]) if ori > 0 else int(face_facepoint_indexes[fi, 1])

        # Facepoint depth weight
        fp_w = float(fp_wse[fp])  if fp < len(fp_wse)  else _NODATA
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
    nodata: float,
    depth_threshold: float = _MIN_WS_PLOT_TOLERANCE,
) -> np.ndarray:
    """Step 4 — pixel-level barycentric interpolation for all wet cells.

    Processes all pixels whose owning cell is recorded in ``cell_id_grid``,
    applies polygon barycentric weights + donate redistribution, and
    interpolates the requested variable.

    Parameters
    ----------
    variable:
        ``"water_surface"``, ``"depth"``, ``"speed"``, or ``"velocity"``.
    cell_id_grid:
        ``(H, W)`` int32 — from :func:`build_cell_id_raster`.
    transform:
        Rasterio affine transform (pixel center coords).
    terrain_grid:
        ``(H, W)`` float — terrain elevation; required for ``"depth"``,
        ``"speed"``, ``"velocity"``; ``None`` skips per-pixel depth masking.
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
        ``(n_fp,)`` from :func:`compute_facepoint_wse`; required for sloping.
    face_value_a, face_value_b:
        ``(n_faces,)`` from :func:`compute_face_wss`.
    fp_velocities, fp_face_local_map, replaced_face_vel:
        From Steps 3 / 3.5; required for ``"speed"`` / ``"velocity"``.
    face_vel_A, face_vel_B:
        ``(n_faces, 2)`` from Step 2; required for ``"speed"`` / ``"velocity"``.
    fp_elev:
        ``(n_fp,)`` terrain elevation sampled at facepoints; enables
        depth-weighted rebalancing (``PaintCell_8Stencil_RebalanceWeights``).
        Pass ``None`` to skip rebalancing.
    nodata:
        Fill value for dry / out-of-domain pixels.
    depth_threshold:
        Minimum depth for a pixel to be considered wet (default 0.001).

    Returns
    -------
    output : float32 ndarray, shape ``(H, W)``
        Interpolated values; ``nodata`` where dry or outside mesh.
        For ``"velocity"``: 4-band array ``(4, H, W)`` — ``[Vx, Vy, speed, direction_deg]``.
    """
    import rasterio.transform as _rt

    H, W = cell_id_grid.shape
    is_velocity_4band = variable == "velocity"

    if is_velocity_4band:
        output = np.full((4, H, W), nodata, dtype=np.float32)
    else:
        output = np.full((H, W), nodata, dtype=np.float32)

    valid_mask = cell_id_grid > 0
    valid_rows, valid_cols = np.where(valid_mask)
    if len(valid_rows) == 0:
        return output

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

        # ---- Build ordered facepoint list for this cell ------------------
        start = int(cell_face_info[cell_idx, 0])
        count = int(cell_face_info[cell_idx, 1])
        if count < 3:
            continue

        face_indices = [int(cell_face_values[start + k, 0]) for k in range(count)]
        face_orients = [int(cell_face_values[start + k, 1]) for k in range(count)]

        # One facepoint per face edge (orientation selects fpA or fpB)
        verts_fp = [
            int(face_facepoint_indexes[fi, 0]) if ori > 0
            else int(face_facepoint_indexes[fi, 1])
            for fi, ori in zip(face_indices, face_orients)
        ]
        verts_x = np.array([float(fp_coords[fp, 0]) for fp in verts_fp], dtype=np.float64)
        verts_y = np.array([float(fp_coords[fp, 1]) for fp in verts_fp], dtype=np.float64)
        N = count

        # ---- Per-cell WSE arrays (for sloping WSE / depth) ---------------
        if fp_wse is not None:
            fp_local_wse = np.array([float(fp_wse[fp]) for fp in verts_fp], dtype=np.float64)
            face_local_wse = np.array([
                float(face_value_a[fi]) if cell_idx == int(face_cell_indexes[fi, 0])
                else float(face_value_b[fi])
                for fi in face_indices
            ], dtype=np.float64)
            has_valid_wse = (fp_local_wse != _NODATA).any() or (face_local_wse != _NODATA).any()
            use_sloped = has_valid_wse and terrain_grid is not None
        else:
            fp_local_wse = None
            face_local_wse = None
            use_sloped = False

        # DownwardAdjustFPValues (C# Renderer.cs:3267)
        if use_sloped and not _all_shallow(cell_idx, cell_face_info, cell_face_values, face_value_a != _NODATA):
            fp_local_wse_adj = _downward_adjust_fp_wse(float(cell_wse[cell_idx]), fp_local_wse)
        else:
            fp_local_wse_adj = fp_local_wse.copy() if fp_local_wse is not None else None

        # Depth weights (C# UseDepthWeightedFaces)
        cell_dw: np.ndarray | None = None
        if use_sloped and fp_elev is not None:
            cell_dw = _depth_weights_for_cell(
                cell_idx, cell_face_info, cell_face_values,
                face_facepoint_indexes, face_cell_indexes,
                fp_wse, fp_elev, face_value_a, face_value_b, face_min_elev,
            )

        # ---- Per-cell velocity arrays -------------------------------------
        nb_fp_vx: np.ndarray | None = None
        nb_fp_vy: np.ndarray | None = None
        nb_face_vx: np.ndarray | None = None
        nb_face_vy: np.ndarray | None = None
        if variable in ("speed", "velocity") and fp_velocities is not None:
            nb_fp_vx = np.zeros(N, dtype=np.float64)
            nb_fp_vy = np.zeros(N, dtype=np.float64)
            for i in range(N):
                fp_i = verts_fp[i]
                fi   = face_indices[i]
                key  = (fp_i, fi)
                if key in fp_face_local_map:
                    lj = fp_face_local_map[key]
                    nb_fp_vx[i] = float(replaced_face_vel[fi, 0]) if replaced_face_vel is not None else float(fp_velocities[fp_i][lj, 0])
                    nb_fp_vy[i] = float(replaced_face_vel[fi, 1]) if replaced_face_vel is not None else float(fp_velocities[fp_i][lj, 1])
            nb_face_vx = np.zeros(N, dtype=np.float64)
            nb_face_vy = np.zeros(N, dtype=np.float64)
            for j in range(N):
                fi  = face_indices[j]
                ori = face_orients[j]
                src = face_vel_A[fi] if ori > 0 else face_vel_B[fi]
                nb_face_vx[j] = float(src[0])
                nb_face_vy[j] = float(src[1])

        # ---- Pixel loop for this cell -------------------------------------
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

            # ---- Wet/dry check -------------------------------------------
            if use_sloped and fp_local_wse_adj is not None and face_local_wse is not None:
                # Per-pixel WSE check via barycentric interpolation
                fw = _barycentric_weights(px, py, verts_x, verts_y)
                vel_w = _donate(fw)
                pixel_wse = _pixel_wse_sloped(vel_w, fp_local_wse_adj, face_local_wse, cell_dw)
                t_elev = float(terrain_grid[r, c]) if terrain_grid is not None else float("nan")
                if pixel_wse == _NODATA or np.isnan(t_elev) or t_elev == _NODATA:
                    continue
                if pixel_wse < t_elev + depth_threshold:
                    continue
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
            else:
                fw = _barycentric_weights(px, py, verts_x, verts_y)
                vel_w = _donate(fw)
                pixel_wse = float(cell_wse[cell_idx])
                t_elev = 0.0

            # ---- Interpolation -------------------------------------------
            if variable == "water_surface":
                if fp_local_wse_adj is not None and face_local_wse is not None:
                    val = _pixel_wse_sloped(vel_w, fp_local_wse_adj, face_local_wse, cell_dw)
                    output[r, c] = np.float32(val) if val != _NODATA else nodata
                else:
                    output[r, c] = np.float32(cell_wse[cell_idx])

            elif variable == "depth":
                if fp_local_wse_adj is not None and face_local_wse is not None:
                    pix_wse = _pixel_wse_sloped(vel_w, fp_local_wse_adj, face_local_wse, cell_dw)
                else:
                    pix_wse = float(cell_wse[cell_idx])
                dep = pix_wse - t_elev
                output[r, c] = np.float32(max(0.0, dep)) if dep > 0 else nodata

            elif variable in ("speed", "velocity"):
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
                else:  # 4-band velocity
                    direction = float(np.degrees(np.arctan2(Vx, Vy))) % 360.0
                    output[0, r, c] = np.float32(Vx)
                    output[1, r, c] = np.float32(Vy)
                    output[2, r, c] = np.float32(spd)
                    output[3, r, c] = np.float32(direction)

    return output
