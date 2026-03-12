"""WLS cell-centre velocity reconstruction from HEC-RAS face normal velocities.

This module contains pure-numpy functions that have no dependency on h5py,
rasterio, or any other optional library.  They are called by
``FlowAreaResults`` which provides the mesh geometry arrays.

Background
----------
HEC-RAS 2-D solves for *face normal velocities* (velocity component
perpendicular to each cell face) as primary unknowns.  Cell-centre velocity
vectors are a derived quantity reconstructed via the weighted least-squares
(WLS) method described in the HEC-RAS Technical Reference Manual
(Section: 2D Unsteady Flow - Numerical Methods - Cell Velocity).

The 2x2 WLS system (per cell, summing over its k faces):

    [a11  a12] [u]   [b1]
    [a12  a22] [v] = [b2]

    a11 = sum(w_k * nx_k**2)
    a22 = sum(w_k * ny_k**2)
    a12 = sum(w_k * nx_k * ny_k)
    b1  = sum(w_k * V_n,k * nx_k)
    b2  = sum(w_k * V_n,k * ny_k)

Solved via Cramer's rule:
    det = a11*a22 - a12**2
    u   = (a22*b1 - a12*b2) / det
    v   = (a11*b2 - a12*b1) / det

Weights *w_k* are either:
  - ``"area_weighted"``  : wetted face flow areas interpolated from the
      per-face hydraulic property tables at the estimated face WSE.  This
      matches HEC-RAS's internal reconstruction.
  - ``"length_weighted"``: face plan-view lengths (column 2 of
      ``Faces NormalUnitVector and Length``).  Simpler; no table lookup.

Sign convention note
--------------------
The stored face normal ``n_hat_stored`` points from the *left* cell to the
*right* cell (``Faces Cell Indexes`` col 0 -> col 1).  Orientation (+1/-1)
records whether ``n_hat_stored`` is *outward* (+1) or *inward* (-1) relative
to the current cell.  Orientation cancels in the WLS product
``V_n * n_hat``, so the stored values are used directly.

"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Single-cell helpers (used internally and exposed for testing)
# ---------------------------------------------------------------------------


def _wls_velocity(
    vn: np.ndarray,
    weights: np.ndarray,
    normals: np.ndarray,
) -> np.ndarray:
    """Solve the 2x2 WLS system for one cell.

    Parameters
    ----------
    vn : shape ``(n_faces,)``
        Signed face normal velocities.
    weights : shape ``(n_faces,)``
        Per-face weights (flow areas or lengths).
    normals : shape ``(n_faces, 2)``
        Unit normal vectors ``[nx, ny]``.

    Returns
    -------
    np.ndarray, shape ``(2,)``
        ``[u, v]`` cell-centre velocity components.
    """
    nx = normals[:, 0]
    ny = normals[:, 1]

    a11 = np.dot(weights, nx * nx)
    a22 = np.dot(weights, ny * ny)
    a12 = np.dot(weights, nx * ny)
    b1 = np.dot(weights * vn, nx)
    b2 = np.dot(weights * vn, ny)

    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-30:
        return np.zeros(2)

    u = (a22 * b1 - a12 * b2) / det
    v = (a11 * b2 - a12 * b1) / det
    return np.array([u, v])


def _interpolate_face_flow_area(
    face_idx: int,
    wse: float,
    ae_info: np.ndarray,
    ae_values: np.ndarray,
) -> float:
    """Interpolate flow area for one face from its hydraulic property table.

    Parameters
    ----------
    face_idx : int
        0-based face index.
    wse : float
        Water-surface elevation at the face.
    ae_info : ndarray, shape ``(n_faces, 2)``
        ``[start_index, count]`` for each face into *ae_values*.
    ae_values : ndarray, shape ``(total, 4)``
        Columns: ``[elevation, flow_area, wetted_perimeter, mannings_n]``.

    Returns
    -------
    float
        Wetted cross-sectional flow area at the given WSE.
    """
    start = int(ae_info[face_idx, 0])
    count = int(ae_info[face_idx, 1])
    table = ae_values[start : start + count]
    elevs = table[:, 0]
    areas = table[:, 1]

    if wse <= elevs[0]:
        return 0.0
    if wse >= elevs[-1]:
        return float(areas[-1])
    return float(np.interp(wse, elevs, areas))


def _estimate_face_wse_average(
    face_cell_indexes: np.ndarray,
    cell_wse: np.ndarray,
) -> np.ndarray:
    """Estimate face WSE as the simple average of the two adjacent cell WSEs.

    For boundary faces (one neighbour index is -1 or beyond the length of
    *cell_wse*), the single available cell's WSE is used.  Passing an
    extended *cell_wse* that includes ghost-cell rows (indices
    ``n_cells_real .. n_total-1``) makes boundary faces use the
    ghost-cell WSE instead of falling back to the inner cell only.

    Parameters
    ----------
    face_cell_indexes : ndarray, shape ``(n_faces, 2)``
        Left and right cell index per face; -1 = no neighbour.
    cell_wse : ndarray, shape ``(n_total,)``
        Water-surface elevation indexed by cell index.  May be real cells
        only (length ``n_cells``) or extended with ghost cells
        (length ``n_cells + n_ghost``).

    Returns
    -------
    ndarray, shape ``(n_faces,)``
    """
    n_total = len(cell_wse)
    n_faces = face_cell_indexes.shape[0]
    face_wse = np.zeros(n_faces)

    left = face_cell_indexes[:, 0].astype(int)
    right = face_cell_indexes[:, 1].astype(int)

    l_real = (left >= 0) & (left < n_total)
    r_real = (right >= 0) & (right < n_total)

    both = l_real & r_real
    face_wse[both] = 0.5 * (cell_wse[left[both]] + cell_wse[right[both]])

    left_only = l_real & ~r_real
    face_wse[left_only] = cell_wse[left[left_only]]

    right_only = ~l_real & r_real
    face_wse[right_only] = cell_wse[right[right_only]]

    return face_wse


def _estimate_face_wse_max(
    face_cell_indexes: np.ndarray,
    cell_wse: np.ndarray,
) -> np.ndarray:
    """Estimate face WSE as the maximum of the two adjacent cell WSEs.

    For boundary faces (one neighbour index is -1 or beyond the length of
    *cell_wse*), the single available cell's WSE is used.

    Parameters
    ----------
    face_cell_indexes : ndarray, shape ``(n_faces, 2)``
        Left and right cell index per face; -1 = no neighbour.
    cell_wse : ndarray, shape ``(n_total,)``
        Water-surface elevation indexed by cell index.  May include ghost cells.

    Returns
    -------
    ndarray, shape ``(n_faces,)``
    """
    n_total = len(cell_wse)
    n_faces = face_cell_indexes.shape[0]
    face_wse = np.zeros(n_faces)

    left = face_cell_indexes[:, 0].astype(int)
    right = face_cell_indexes[:, 1].astype(int)

    l_real = (left >= 0) & (left < n_total)
    r_real = (right >= 0) & (right < n_total)

    both = l_real & r_real
    face_wse[both] = np.maximum(cell_wse[left[both]], cell_wse[right[both]])

    left_only = l_real & ~r_real
    face_wse[left_only] = cell_wse[left[left_only]]

    right_only = ~l_real & r_real
    face_wse[right_only] = cell_wse[right[right_only]]

    return face_wse


def _estimate_face_wse_sloped(
    face_cell_indexes: np.ndarray,
    cell_wse: np.ndarray,
    cell_coords: np.ndarray,
    face_coords: np.ndarray,
) -> np.ndarray:
    """Estimate face WSE via distance-weighted linear interpolation.

    The face WSE is placed at the face's actual position along the line
    connecting the two adjacent cell centres:

        t        = d_left / (d_left + d_right)
        wse_face = (1 - t) * wse_left + t * wse_right

    where *d_left* (*d_right*) is the Euclidean distance from the face
    centroid to the left (right) cell centre.  When the two distances sum
    to nearly zero (degenerate geometry), the simple average is used as a
    fallback.

    For boundary faces (one neighbour index is -1 or beyond the length of
    *cell_wse*), the single available cell's WSE is used unchanged.
    Passing extended *cell_wse* and *cell_coords* that include ghost cells
    makes boundary faces use distance-weighted interpolation like interior
    faces.

    Parameters
    ----------
    face_cell_indexes : ndarray, shape ``(n_faces, 2)``
        Left and right cell index per face; -1 = no neighbour.
    cell_wse : ndarray, shape ``(n_total,)``
        Water-surface elevation indexed by cell index.  May include ghost
        cells (length ``n_cells + n_ghost``).
    cell_coords : ndarray, shape ``(n_total, 2)``
        X, Y coordinates indexed by cell index.  Must have the same length
        as *cell_wse* so ghost-cell coordinates are accessible.
    face_coords : ndarray, shape ``(n_faces, 2)``
        X, Y coordinates of each face centroid.

    Returns
    -------
    ndarray, shape ``(n_faces,)``
    """
    n_total = len(cell_wse)
    n_faces = face_cell_indexes.shape[0]
    face_wse = np.zeros(n_faces)

    left = face_cell_indexes[:, 0].astype(int)
    right = face_cell_indexes[:, 1].astype(int)

    l_real = (left >= 0) & (left < n_total)
    r_real = (right >= 0) & (right < n_total)

    # Interior faces: interpolate at the face's position between cell centres
    both = l_real & r_real
    if np.any(both):
        fi = np.where(both)[0]
        l_idx = left[fi]
        r_idx = right[fi]
        d_left = np.linalg.norm(face_coords[fi] - cell_coords[l_idx], axis=1)
        d_right = np.linalg.norm(face_coords[fi] - cell_coords[r_idx], axis=1)
        total = d_left + d_right
        # fallback to 0.5 for degenerate geometry (cells at same location).
        # Use a safe denominator to avoid division-by-zero in the numpy
        # expression before np.where applies its mask.
        safe_total = np.where(total > 1e-12, total, 1.0)
        t = np.where(total > 1e-12, d_left / safe_total, 0.5)
        face_wse[fi] = (1.0 - t) * cell_wse[l_idx] + t * cell_wse[r_idx]

    # Boundary faces: use the single real neighbour's WSE
    left_only = l_real & ~r_real
    face_wse[left_only] = cell_wse[left[left_only]]

    right_only = ~l_real & r_real
    face_wse[right_only] = cell_wse[right[right_only]]

    return face_wse


# ---------------------------------------------------------------------------
# Vectorised batch computation (main entry point)
# ---------------------------------------------------------------------------


def compute_all_cell_velocities(
    n_cells: int,
    cell_face_info: np.ndarray,
    cell_face_values: np.ndarray,
    face_normals: np.ndarray,
    face_cell_indexes: np.ndarray,
    face_ae_info: np.ndarray,
    face_ae_values: np.ndarray,
    face_vel: np.ndarray,
    cell_wse: np.ndarray,
    method: str = "area_weighted",
    face_flow: np.ndarray | None = None,
    wse_interp: str = "average",
    cell_coords: np.ndarray | None = None,
    face_coords: np.ndarray | None = None,
) -> np.ndarray:
    """Compute WLS cell-centre velocity vectors for real cells.

    Ghost cells are not reconstructed: HEC-RAS stores no face normal
    velocities for ghost-only faces, so the WLS system for a ghost cell
    would be severely underdetermined (one face, two unknowns).  Ghost-cell
    WSE values in *cell_wse* are still used by the face-WSE estimators to
    improve accuracy at boundary faces of real cells.

    Parameters
    ----------
    n_cells : int
        Number of real computational cells.
    cell_face_info : ndarray, shape ``(>= n_cells, 2)``
        ``[start_index, count]`` into *cell_face_values*.
    cell_face_values : ndarray, shape ``(total, 2)``
        ``[face_index, orientation]`` for each cell-face pair.
    face_normals : ndarray, shape ``(n_faces, 3)``
        ``[nx, ny, face_length]``.
    face_cell_indexes : ndarray, shape ``(n_faces, 2)``
        Left and right cell indices per face.
    face_ae_info : ndarray, shape ``(n_faces, 2)``
        Area-elevation table index for each face.
    face_ae_values : ndarray, shape ``(total, 4)``
        Area-elevation table values.
    face_vel : ndarray, shape ``(n_faces,)``
        Signed face normal velocities for this timestep.
    cell_wse : ndarray, shape ``(n_cells,)`` or ``(n_cells + n_ghost,)``
        Cell-centre water-surface elevations.  Ghost-cell WSEs (indices
        ``>= n_cells``) are used by the face-WSE estimators but are not
        reconstructed as velocity vectors.
    method : str
        ``"area_weighted"`` (default, matches HEC-RAS),
        ``"length_weighted"`` (simpler, no table lookup), or
        ``"flow_ratio"`` (requires *face_flow*; back-calculates flow area
        as |Q|/|V_n|, exactly as HEC-RAS computed internally).
    face_flow : ndarray, shape ``(n_faces,)``, optional
        Volumetric face flows.  Required when ``method="flow_ratio"``.
    wse_interp : str
        How to estimate face WSE when ``method="area_weighted"``.
        ``"average"`` (default) — simple mean of the two adjacent cell WSEs.
        ``"sloped"`` — distance-weighted linear interpolation at the face's
        actual position between the two cell centres (requires *cell_coords*
        and *face_coords*).
        ``"max"`` — maximum of the two adjacent cell WSEs (conservative;
        tends to increase flow area at partially-wet faces).
    cell_coords : ndarray, shape ``(n_cells + n_ghost, 2)``, optional
        X, Y coordinates of cell centres.  Must cover ghost rows when
        ``wse_interp="sloped"`` so boundary faces are interpolated
        correctly.  Required when ``wse_interp="sloped"``.
    face_coords : ndarray, shape ``(n_faces, 2)``, optional
        X, Y coordinates of face centroids.  Required when
        ``wse_interp="sloped"``.

    Returns
    -------
    ndarray, shape ``(n_cells, 2)``
        ``[Vx, Vy]`` depth-averaged velocity at each real cell centre.
    """
    if method not in {"area_weighted", "length_weighted", "flow_ratio"}:
        raise ValueError(
            f"method must be 'area_weighted', 'length_weighted', or "
            f"'flow_ratio'; got {method!r}"
        )
    if method == "flow_ratio" and face_flow is None:
        raise ValueError("face_flow is required when method='flow_ratio'")
    if wse_interp not in {"average", "sloped", "max"}:
        raise ValueError(
            f"wse_interp must be 'average', 'sloped', or 'max'; got {wse_interp!r}"
        )
    if wse_interp == "sloped" and (cell_coords is None or face_coords is None):
        raise ValueError(
            "cell_coords and face_coords are required when wse_interp='sloped'"
        )

    # Pre-compute face WSE for area_weighted (used once, shared across cells).
    # cell_wse may include ghost rows; the estimators use len(cell_wse) as the
    # bound so ghost-cell WSE contributes to boundary face estimates.
    if method == "area_weighted":
        if wse_interp == "sloped":
            face_wse = _estimate_face_wse_sloped(
                face_cell_indexes, cell_wse, cell_coords, face_coords
            )
        elif wse_interp == "max":
            face_wse = _estimate_face_wse_max(face_cell_indexes, cell_wse)
        else:
            face_wse = _estimate_face_wse_average(face_cell_indexes, cell_wse)
    else:
        face_wse = None

    velocities = np.zeros((n_cells, 2), dtype=np.float64)

    for c in range(n_cells):
        start = int(cell_face_info[c, 0])
        count = int(cell_face_info[c, 1])
        vals = cell_face_values[start : start + count]
        face_idxs = vals[:, 0].astype(int)

        normals = face_normals[face_idxs, :2]  # (k, 2)
        lengths = face_normals[face_idxs, 2]  # (k,)
        vn = face_vel[face_idxs]  # (k,)

        if method == "area_weighted":
            weights = np.array(
                [
                    _interpolate_face_flow_area(
                        fi, face_wse[fi], face_ae_info, face_ae_values
                    )
                    for fi in face_idxs
                ]
            )
        elif method == "length_weighted":
            weights = lengths
        else:  # flow_ratio
            qf = face_flow[face_idxs]
            weights = np.where(np.abs(vn) > 1e-10, np.abs(qf / vn), 0.0)

        velocities[c] = _wls_velocity(vn, weights, normals)

    return velocities


# ---------------------------------------------------------------------------
# Corner (facepoint) velocity — average adjacent double-C face velocities
# ---------------------------------------------------------------------------


def average_face_velocities_at_facepoints(
    face_facepoint_indexes: np.ndarray,
    face_vel_2d: np.ndarray,
    wet_face: np.ndarray,
) -> np.ndarray:
    """Average full 2D face velocities at each mesh corner (facepoint).

    For each facepoint *p*, collects the full 2D velocity vectors of all
    adjacent *wet* faces (computed via the double-C stencil by
    :func:`compute_all_face_velocities`) and returns their unweighted mean:

    .. code-block:: text

        V_p = mean( face_vel_2d[k]  for each wet face k incident to p )

    This is physically grounded because the double-C stencil already
    preserves the exact measured face-normal velocity and estimates the
    tangential component from the two adjacent cell WLS velocities.
    Averaging those full vectors at the shared corner is a simple, consistent
    linear interpolation from the nearest data locations (face midpoints).

    Parameters
    ----------
    face_facepoint_indexes : ndarray, shape ``(n_faces, 2)``
        Start and end facepoint index for each face.
    face_vel_2d : ndarray, shape ``(n_faces, 2)``
        Full ``[Vx, Vy]`` velocity at each face midpoint, as returned by
        :func:`compute_all_face_velocities`.
    wet_face : ndarray, shape ``(n_faces,)``, bool
        ``True`` for faces adjacent to at least one wet cell.

    Returns
    -------
    ndarray, shape ``(n_facepoints, 2)``
        Mean ``[Vx, Vy]`` at each facepoint.
        Dry facepoints (no adjacent wet face) receive ``[0, 0]``.
    """
    n_facepoints = int(face_facepoint_indexes.max()) + 1

    fp0 = face_facepoint_indexes[:, 0]
    fp1 = face_facepoint_indexes[:, 1]
    wet_idx = np.where(wet_face)[0]

    result = np.zeros((n_facepoints, 2), dtype=np.float64)
    count  = np.zeros(n_facepoints,     dtype=np.int64)

    np.add.at(result, fp0[wet_idx], face_vel_2d[wet_idx])
    np.add.at(result, fp1[wet_idx], face_vel_2d[wet_idx])
    np.add.at(count,  fp0[wet_idx], 1)
    np.add.at(count,  fp1[wet_idx], 1)

    nonzero = count > 0
    result[nonzero] /= count[nonzero, np.newaxis]

    return result


# ---------------------------------------------------------------------------
# Face velocity reconstruction (double-C stencil)
# ---------------------------------------------------------------------------


def compute_all_face_velocities(
    face_normals: np.ndarray,
    face_normal_velocity: np.ndarray,
    face_cell_indexes: np.ndarray,
    cell_velocity: np.ndarray,
    dry_mask: np.ndarray,
) -> np.ndarray:
    """Reconstruct full 2D face velocity using the HEC-RAS double-C stencil.

    HEC-RAS stores only the face-normal velocity ``vn``.  The tangential
    component is recovered by projecting each adjacent cell's WLS velocity
    vector onto the face tangential direction, then arithmetically averaging
    the two sides (left and right).  This mirrors the double-C stencil
    described in the HEC-RAS 2D Technical Reference Manual:

    .. code-block:: text

        vt_L = cell_velocity[L] · t_hat
        vt_R = cell_velocity[R] · t_hat
        vt   = (vt_L + vt_R) / 2          (both cells hydraulically connected)
        V_face = vn * n_hat + vt * t_hat

    The normal component ``vn`` is kept exactly as stored in the HDF; only
    the tangential component is estimated from the WLS reconstruction.

    Ghost-cell support: when *cell_velocity* and *dry_mask* include ghost
    cells (length ``n_cells + n_ghost``), boundary faces automatically use
    the ghost-cell WLS velocity for the tangential component, improving
    accuracy at the domain perimeter.

    Parameters
    ----------
    face_normals : ndarray, shape ``(n_faces, 3)``
        ``[nx, ny, face_length]``; ``(nx, ny)`` must be a unit normal vector.
    face_normal_velocity : ndarray, shape ``(n_faces,)``
        Signed face-normal velocities for this timestep.
    face_cell_indexes : ndarray, shape ``(n_faces, 2)``
        Left and right cell indices; ``-1`` = no neighbour.
    cell_velocity : ndarray, shape ``(n_total, 2)``
        WLS velocity vectors ``[Vx, Vy]`` at each cell centre.  May include
        ghost cells (indices ``n_cells_real .. n_total-1``).
    dry_mask : ndarray, shape ``(n_total,)``, bool
        ``True`` where a cell is dry; its WLS velocity is excluded.
        Must have the same length as *cell_velocity*.

    Returns
    -------
    ndarray, shape ``(n_faces, 2)``
        Full ``[Vx, Vy]`` velocity at each face midpoint.
        Faces where *both* adjacent cells are dry receive ``[0, 0]``.
    """
    n_total = len(cell_velocity)
    n_faces = len(face_normals)
    nx_ny = face_normals[:, :2]                            # (n_faces, 2)
    t_hat = np.column_stack([-nx_ny[:, 1], nx_ny[:, 0]])  # (n_faces, 2) 90° CCW

    left  = face_cell_indexes[:, 0]
    right = face_cell_indexes[:, 1]

    # Clamp indices for safe array lookup (out-of-range cases handled by masks).
    left_safe  = np.where((left  >= 0) & (left  < n_total), left,  0)
    right_safe = np.where((right >= 0) & (right < n_total), right, 0)

    valid_left  = (left  >= 0) & (left  < n_total) & ~dry_mask[left_safe]
    valid_right = (right >= 0) & (right < n_total) & ~dry_mask[right_safe]

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
    # Faces with no valid neighbour: vt stays 0 — normal component still correct.

    # Compose: preserve measured vn on the normal axis; add estimated vt.
    return face_normal_velocity[:, np.newaxis] * nx_ny + vt[:, np.newaxis] * t_hat
