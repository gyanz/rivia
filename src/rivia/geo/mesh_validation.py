"""Geometric validation of HEC-RAS 2D mesh cells.

HEC-RAS enforces the following rules on 2D mesh cells:

1. Each cell is a polygon with 3 to 8 sides.
2. Each cell must be strictly convex.
3. No two adjacent faces may be collinear.
4. No duplicate computation points (cell centres or facepoints).
5. Every cell centre must lie inside the 2D flow area boundary.

:func:`check_mesh_cells` checks these rules from the raw HDF arrays and
returns a structured report.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger("rivia.geo")

# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------


def _reconstruct_polygon(
    cell_idx: int,
    cell_face_info: np.ndarray,
    cell_face_values: np.ndarray,
    face_facepoint_indexes: np.ndarray,
    facepoint_coordinates: np.ndarray,
    face_perimeter_info: np.ndarray | None = None,
    face_perimeter_values: np.ndarray | None = None,
) -> tuple[np.ndarray, list[int]] | None:
    """Return ``(polygon_xy, ordered_fp_indices)`` for *cell_idx*.

    Recover the ordered polygon for a single mesh cell by walking the
    facepoint adjacency graph.  Each face contributes its two corner
    facepoint endpoints plus any interior perimeter points along curved
    faces; because every corner facepoint in a valid cell belongs to
    exactly two faces, the edges form a single closed cycle.  The function
    follows that cycle to produce vertices in polygon order, independent of
    the orientation flags stored in the HDF file.

    When *face_perimeter_info* and *face_perimeter_values* are supplied
    (from ``Faces Perimeter Info`` and ``Faces Perimeter Values``), interior
    points along curved faces are inserted between the corner facepoints in
    the correct traversal direction.  This is needed for correct convexity
    checking of cells that have curved face edges.

    Returns ``None`` for malformed cells where the graph cannot be
    traversed as a closed cycle.

    Returns
    -------
    polygon_xy : ndarray, shape ``(n_vertices, 2)``
        All polygon vertices in order (corner facepoints + any interior
        perimeter points on curved faces).
    fp_order : list[int]
        Indices into *facepoint_coordinates* for the corner facepoints only,
        in polygon order.
    """
    start = int(cell_face_info[cell_idx, 0])
    count = int(cell_face_info[cell_idx, 1])
    face_idxs = cell_face_values[start : start + count, 0]

    # Build facepoint adjacency graph and edge→face lookup.
    # Each corner facepoint appears in exactly two faces.
    adj: dict[int, list[int]] = defaultdict(list)
    edge_to_face: dict[tuple[int, int], int] = {}
    for f in face_idxs:
        f = int(f)
        fp0, fp1 = int(face_facepoint_indexes[f, 0]), int(face_facepoint_indexes[f, 1])
        adj[fp0].append(fp1)
        adj[fp1].append(fp0)
        edge_to_face[(fp0, fp1)] = f
        edge_to_face[(fp1, fp0)] = f

    # Validate: every node must have exactly 2 neighbours.
    if any(len(v) != 2 for v in adj.values()):
        return None

    use_perimeter = (
        face_perimeter_info is not None and face_perimeter_values is not None
    )

    # Follow the cycle, inserting interior perimeter points for curved faces.
    fp_order: list[int] = []
    all_coords: list[np.ndarray] = []
    all_fps = list(adj.keys())
    current = all_fps[0]
    prev: int | None = None
    for _ in range(count):
        fp_order.append(current)
        all_coords.append(facepoint_coordinates[current])
        nb = adj[current]
        nxt = nb[1] if nb[0] == prev else nb[0]

        # Insert interior perimeter points between current and nxt.
        if use_perimeter:
            face_idx = edge_to_face.get((current, nxt))
            if face_idx is not None:
                peri_start = int(face_perimeter_info[face_idx, 0])
                n_interior = int(face_perimeter_info[face_idx, 1])
                if n_interior > 0:
                    interior = face_perimeter_values[
                        peri_start : peri_start + n_interior
                    ]
                    # Perimeter points are stored fp0→fp1 in canonical HDF order.
                    # Reverse if we are traversing fp1→fp0.
                    fp0_canonical = int(face_facepoint_indexes[face_idx, 0])
                    if current != fp0_canonical:
                        interior = interior[::-1]
                    all_coords.extend(interior)

        prev = current
        current = nxt

    if current != fp_order[0]:          # cycle did not close
        return None

    return np.array(all_coords), fp_order


def _cross_products(polygon: np.ndarray) -> np.ndarray:
    """Signed cross product at every vertex of *polygon* (n, 2).

    Positive → left turn (CCW), negative → right turn (CW), zero → collinear.

    Example
    -------
    A unit square traversed counter-clockwise produces all-positive cross
    products (all left turns). The turn at vertex ``i`` is the signed rotation
    from incoming edge ``(i-1 -> i)`` to outgoing edge ``(i -> i+1)`` with
    wraparound. For ``i=0``, incoming is ``(n-1 -> 0)`` and outgoing is
    ``(0 -> 1)``::

        >>> sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        >>> _cross_products(sq)
        array([1., 1., 1., 1.])

    A collinear triplet yields a zero at the middle vertex::

        >>> tri = np.array([[0, 0], [1, 0], [2, 0], [2, 1], [0, 1]], dtype=float)
        >>> _cross_products(tri)   # vertex 1 is collinear
        array([ 1.,  0.,  2.,  2.,  1.])
    """
    n = len(polygon)
    v1 = polygon - np.roll(polygon, 1, axis=0)  # edge arriving at vertex i
    v2 = np.roll(polygon, -1, axis=0) - polygon  # edge leaving vertex i
    return v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test.  Works for any simple polygon.

    Example
    -------
    ::

        >>> sq = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)
        >>> _point_in_polygon(np.array([2.0, 2.0]), sq)   # inside
        True
        >>> _point_in_polygon(np.array([5.0, 5.0]), sq)   # outside
        False
    """
    x, y = float(point[0]), float(point[1])
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i, 0]), float(polygon[i, 1])
        xj, yj = float(polygon[j, 0]), float(polygon[j, 1])
        if ((yi > y) != (yj > y)) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _find_exact_duplicates(
    coords: np.ndarray,
) -> list[tuple[int, int]]:
    """Return pairs ``(i, j)`` where ``coords[i] == coords[j]`` exactly.

    Example
    -------
    Given::

        coords = np.array([
            [0.0, 0.0],  # idx 0
            [1.0, 2.0],  # idx 1
            [3.0, 4.0],  # idx 2
            [1.0, 2.0],  # idx 3 (duplicate of idx 1)
            [0.0, 0.0],  # idx 4 (duplicate of idx 0)
        ])

    the function returns pairs equivalent to ``[(0, 4), (1, 3)]``
    (ordering may vary). Comparisons are exact, not tolerance-based.
    """
    # Round-trip through structured array for lexicographic uniqueness.
    n = len(coords)
    # lexsort expects keys in (minor, ..., major) order; reversing columns to
    # [y, x] and transposing makes the primary row order x then y, and equal rows become adjacent.
    order = np.lexsort(coords[:, ::-1].T)
    sorted_c = coords[order]
    pairs: list[tuple[int, int]] = []
    for k in range(n - 1):
        # TODO: np.isclose with a tolerance may be more appropriate here
        if np.array_equal(sorted_c[k], sorted_c[k + 1]):
            pairs.append((int(order[k]), int(order[k + 1])))
    return pairs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_mesh_cells(
    cell_centers: np.ndarray,
    facepoint_coordinates: np.ndarray,
    face_facepoint_indexes: np.ndarray,
    cell_face_info: np.ndarray,
    cell_face_values: np.ndarray,
    *,
    face_perimeter_info: np.ndarray | None = None,
    face_perimeter_values: np.ndarray | None = None,
    boundary_polygon: np.ndarray | None = None,
    tol: float = 1e-10,
) -> dict:
    """Check HEC-RAS 2D mesh cells against geometric validity rules.

    Parameters
    ----------
    cell_centers : ndarray, shape ``(n_cells, 2)``
        Cell-centre x, y coordinates.
    facepoint_coordinates : ndarray, shape ``(n_facepoints, 2)``
        Mesh vertex x, y coordinates.
    face_facepoint_indexes : ndarray, shape ``(n_faces, 2)``
        Start/end facepoint index for each face.
    face_perimeter_info : ndarray, shape ``(n_faces, 2)``, optional
        ``[start_index, count]`` into *face_perimeter_values* for interior
        perimeter points on curved faces (``Faces Perimeter Info`` in HDF).
        When supplied together with *face_perimeter_values*, interior points
        are included in the polygon used for convexity checking.
    face_perimeter_values : ndarray, shape ``(total, 2)``, optional
        x, y coordinates of interior perimeter points along curved faces
        (``Faces Perimeter Values`` in HDF).
    cell_face_info : ndarray, shape ``(>= n_cells, 2)``
        ``[start_index, count]`` into *cell_face_values* for each cell.
    cell_face_values : ndarray, shape ``(total, 2)``
        ``[face_index, orientation]`` for each cell-face association.
    boundary_polygon : ndarray, shape ``(n_vertices, 2)``, optional
        Ordered vertices of the 2D flow area outer boundary.  When supplied,
        each cell centre is also tested against this perimeter (rule 5).
        Omitting it skips the boundary test.
    tol : float
        Absolute tolerance used only for near-collinear edge detection
        (``|cross_product| < tol * edge_length²``).  Exact-duplicate checks
        use bitwise equality.

    Returns
    -------
    dict with keys:

    ``n_cells``, ``n_facepoints``
        Dataset sizes.
    ``duplicate_cell_centers`` : list of ``(i, j)`` index pairs
        Cell centre pairs that share identical coordinates (rule 4).
    ``duplicate_facepoints`` : list of ``(i, j)`` index pairs
        Facepoint pairs that share identical coordinates (rule 4).
    ``cells`` : list of per-cell dicts
        One entry per cell.  Keys:

        - ``cell_idx`` — 0-based cell index.
        - ``n_faces`` — number of bounding faces.
        - ``face_count_ok`` — ``3 <= n_faces <= 8`` (rule 1).
        - ``strictly_convex`` — all cross products share the same sign and are
          all non-zero (rules 2 & 3).
        - ``collinear_vertex_indices`` — vertex positions in the ordered
          polygon where the cross product is ~zero (rule 3).
        - ``reflex_vertex_indices`` — vertex positions where the polygon turns
          the wrong way (breaks strict convexity, rule 2).
        - ``center_inside_polygon`` — cell centre is inside its own polygon.
        - ``center_inside_boundary`` — cell centre is inside *boundary_polygon*
          (only present when *boundary_polygon* is supplied; rule 5).
        - ``malformed`` — ``True`` when the face adjacency graph could not be
          traversed as a closed cycle.
    ``summary`` : dict
        Aggregate violation counts.
    """
    cell_centers = np.asarray(cell_centers, dtype=np.float64)
    facepoint_coordinates = np.asarray(facepoint_coordinates, dtype=np.float64)
    face_facepoint_indexes = np.asarray(face_facepoint_indexes, dtype=np.int64)
    cell_face_info = np.asarray(cell_face_info, dtype=np.int64)
    cell_face_values = np.asarray(cell_face_values, dtype=np.int64)

    n_cells = len(cell_centers)
    n_facepoints = len(facepoint_coordinates)

    # ── Global: duplicate point checks (rule 4) ────────────────────────────
    dup_cc = _find_exact_duplicates(cell_centers)
    dup_fp = _find_exact_duplicates(facepoint_coordinates)

    # ── Per-cell checks ────────────────────────────────────────────────────
    cell_results: list[dict] = []

    n_face_count_bad = 0
    n_non_convex = 0
    n_collinear = 0
    n_center_outside_polygon = 0
    n_center_outside_boundary = 0
    n_malformed = 0

    for c in range(n_cells):
        count = int(cell_face_info[c, 1])
        result: dict = {"cell_idx": c, "n_faces": count}

        # Rule 1: face count
        face_count_ok = 3 <= count <= 8
        result["face_count_ok"] = face_count_ok
        if not face_count_ok:
            n_face_count_bad += 1

        # Reconstruct ordered polygon (includes curved-face interior points
        # when face_perimeter_info/values are provided).
        poly_result = _reconstruct_polygon(
            c, cell_face_info, cell_face_values,
            face_facepoint_indexes, facepoint_coordinates,
            face_perimeter_info, face_perimeter_values,
        )

        if poly_result is None:
            result["malformed"] = True
            result["strictly_convex"] = False
            result["collinear_vertex_indices"] = []
            result["reflex_vertex_indices"] = []
            result["center_inside_polygon"] = False
            if boundary_polygon is not None:
                result["center_inside_boundary"] = False
            n_malformed += 1
            cell_results.append(result)
            continue

        result["malformed"] = False
        polygon, _ = poly_result

        # Rules 2 & 3: strict convexity and collinearity
        # convex of all values in cross of same sign excluding zero 
        cross = _cross_products(polygon)

        # Normalize |cross| by local edge size before comparing with tol.
        # Raw cross magnitude scales with edge lengths, so the same small bend
        # can look large on big polygons and tiny on small polygons.
        # Using edge_sq keeps the collinearity threshold approximately
        # scale-independent and makes the "near collinear" test more consistent.
        # edge_sq = |v_arriving|^2 (one of the two edges).
        arriving = polygon - np.roll(polygon, 1, axis=0)
        edge_sq = np.einsum("ij,ij->i", arriving, arriving)
        near_zero = edge_sq > 0  # avoid divide-by-zero for degenerate edges
        normalised_cross = np.where(near_zero, np.abs(cross) / np.maximum(edge_sq, 1e-300), 0.0)
        is_collinear = normalised_cross < tol

        collinear_verts = [int(i) for i, c_ in enumerate(is_collinear) if c_]
        strictly_convex_signs = cross[~is_collinear]
        if len(strictly_convex_signs) == 0:
            # All edges are collinear — degenerate polygon
            strictly_convex = False
            reflex_verts = []
        else:
            dominant_sign = np.sign(strictly_convex_signs.mean())
            reflex_mask = (np.sign(cross) != dominant_sign) & ~is_collinear
            reflex_verts = [int(i) for i, r in enumerate(reflex_mask) if r]
            strictly_convex = len(collinear_verts) == 0 and len(reflex_verts) == 0

        result["strictly_convex"] = strictly_convex
        result["collinear_vertex_indices"] = collinear_verts
        result["reflex_vertex_indices"] = reflex_verts

        if not strictly_convex:
            n_non_convex += 1
        if collinear_verts:
            n_collinear += 1

        # Rule 5a: cell centre inside its own polygon
        centre = cell_centers[c]
        inside_poly = _point_in_polygon(centre, polygon)
        result["center_inside_polygon"] = inside_poly
        if not inside_poly:
            n_center_outside_polygon += 1

        # Rule 5b: cell centre inside the 2D area boundary (optional)
        if boundary_polygon is not None:
            inside_bnd = _point_in_polygon(centre, np.asarray(boundary_polygon))
            result["center_inside_boundary"] = inside_bnd
            if not inside_bnd:
                n_center_outside_boundary += 1

        cell_results.append(result)

    # ── Summary ────────────────────────────────────────────────────────────
    summary: dict = {
        "n_face_count_violations": n_face_count_bad,
        "n_non_convex": n_non_convex,
        "n_collinear_edges": n_collinear,
        "n_center_outside_polygon": n_center_outside_polygon,
        "n_duplicate_cell_centers": len(dup_cc),
        "n_duplicate_facepoints": len(dup_fp),
        "n_malformed": n_malformed,
    }
    if boundary_polygon is not None:
        summary["n_center_outside_boundary"] = n_center_outside_boundary

    summary["n_total_violations"] = sum(
        v for k, v in summary.items() if k != "n_total_violations"
    )

    return {
        "n_cells": n_cells,
        "n_facepoints": n_facepoints,
        "duplicate_cell_centers": dup_cc,
        "duplicate_facepoints": dup_fp,
        "cells": cell_results,
        "summary": summary,
    }


def print_mesh_report(report: dict, max_cells: int = 20) -> None:
    """Print a human-readable summary of a :func:`check_mesh_cells` report.

    Parameters
    ----------
    report :
        Dict returned by :func:`check_mesh_cells`.
    max_cells :
        Maximum number of per-cell violation rows to print.
    """
    s = report["summary"]
    print(f"Mesh validation report")
    print(f"  Cells      : {report['n_cells']:,}")
    print(f"  Facepoints : {report['n_facepoints']:,}")
    print()
    print(f"  Rule 1 — face count not in [3, 8]  : {s['n_face_count_violations']:,}")
    print(f"  Rule 2 — non-strictly-convex        : {s['n_non_convex']:,}")
    print(f"  Rule 3 — collinear adjacent edges   : {s['n_collinear_edges']:,}")
    print(f"  Rule 4 — duplicate cell centres     : {s['n_duplicate_cell_centers']:,}")
    print(f"           duplicate facepoints       : {s['n_duplicate_facepoints']:,}")
    print(f"  Rule 5 — centre outside own polygon : {s['n_center_outside_polygon']:,}")
    if "n_center_outside_boundary" in s:
        print(f"           centre outside boundary   : {s['n_center_outside_boundary']:,}")
    print(f"  Malformed face graph                : {s['n_malformed']:,}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Total violations                    : {s['n_total_violations']:,}")

    # Per-cell detail
    bad_cells = [r for r in report["cells"] if _cell_has_violation(r)]
    if not bad_cells:
        print("\nNo per-cell violations found.")
        return

    shown = bad_cells[:max_cells]
    print(f"\nPer-cell violations ({len(bad_cells)} total, showing {len(shown)}):")
    for r in shown:
        flags = []
        if not r["face_count_ok"]:
            flags.append(f"faces={r['n_faces']}")
        if r.get("malformed"):
            flags.append("malformed")
        if not r["strictly_convex"]:
            if r["collinear_vertex_indices"]:
                flags.append(f"collinear@{r['collinear_vertex_indices']}")
            if r["reflex_vertex_indices"]:
                flags.append(f"reflex@{r['reflex_vertex_indices']}")
        if not r["center_inside_polygon"]:
            flags.append("centre_outside_polygon")
        if not r.get("center_inside_boundary", True):
            flags.append("centre_outside_boundary")
        print(f"  cell {r['cell_idx']:>8,}  →  {', '.join(flags)}")

    if len(bad_cells) > max_cells:
        print(f"  ... and {len(bad_cells) - max_cells} more")


def _cell_has_violation(r: dict) -> bool:
    """Return ``True`` if the per-cell result dict contains any validity violation.

    Example
    -------
    ::

        >>> good = {
        ...     'face_count_ok': True, 'malformed': False,
        ...     'strictly_convex': True, 'center_inside_polygon': True,
        ... }
        >>> _cell_has_violation(good)
        False
        >>> _cell_has_violation({**good, 'strictly_convex': False})
        True
    """
    return (
        not r["face_count_ok"]
        or r.get("malformed", False)
        or not r["strictly_convex"]
        or not r["center_inside_polygon"]
        or not r.get("center_inside_boundary", True)
    )




