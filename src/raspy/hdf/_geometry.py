"""GeometryHdf — read HEC-RAS geometry HDF5 files (.g*.hdf).

Provides structured access to 2-D flow-area mesh data:
cell centres, face connectivity, hydraulic property tables, etc.

Also provides access to storage areas and boundary condition lines.

Derived from archive/ras_tools/r2d/ras_io.py and
archive/ras_tools/r2d/ras2d_cell_velocity.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ._base import _HdfFile

if TYPE_CHECKING:
    import h5py


# ---------------------------------------------------------------------------
# HDF path constants
# ---------------------------------------------------------------------------
_GEOM_2D_ROOT = "Geometry/2D Flow Areas"
_GEOM_2D_ATTRS = f"{_GEOM_2D_ROOT}/Attributes"
_SA_ROOT = "Geometry/Storage Areas"
_BC_ROOT = "Geometry/Boundary Condition Lines"
_STRUCT_ROOT = "Geometry/Structures"


# ---------------------------------------------------------------------------
# Private geometry utilities
# ---------------------------------------------------------------------------


def _point_in_polygon(px: float, py: float, polygon: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test for a simple polygon.

    Parameters
    ----------
    px, py:
        Query point coordinates.
    polygon:
        Ordered vertices, shape ``(n, 2)``.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i, 0]), float(polygon[i, 1])
        xj, yj = float(polygon[j, 0]), float(polygon[j, 1])
        if ((yi > py) != (yj > py)) and px < (xj - xi) * (py - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# FlowArea — geometry for a single 2-D flow area
# ---------------------------------------------------------------------------


class FlowArea:
    """Geometry data for one named 2-D flow area.

    All properties are loaded eagerly on first access and cached; they are
    small static arrays compared with the time-series results.

    Parameters
    ----------
    group:
        The ``h5py.Group`` at ``Geometry/2D Flow Areas/<name>``.
    name:
        Human-readable name of the flow area.
    n_cells:
        Number of *real* computational cells (ghost cells excluded).
    """

    def __init__(self, group: "h5py.Group", name: str, n_cells: int) -> None:
        self._g = group
        self._name = name
        self._n_cells = n_cells
        # lazy-loaded cache
        self._cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load(self, key: str) -> np.ndarray:
        if key not in self._cache:
            self._cache[key] = np.array(self._g[key])
        return self._cache[key]

    # ------------------------------------------------------------------
    # Basic info
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Name of the 2-D flow area."""
        return self._name

    @property
    def n_cells(self) -> int:
        """Number of real computational cells (ghost cells excluded)."""
        return self._n_cells

    @property
    def n_faces(self) -> int:
        """Total number of cell faces."""
        return self.face_normals.shape[0]

    # ------------------------------------------------------------------
    # Cell geometry
    # ------------------------------------------------------------------

    @property
    def cell_centers(self) -> np.ndarray:
        """x, y coordinates of real cell centres.  Shape ``(n_cells, 2)``."""
        return self._load("Cells Center Coordinate")[: self._n_cells]

    @property
    def ghost_cell_centers(self) -> np.ndarray:
        """x, y coordinates of ghost (boundary) cell centres.

        Ghost cells occupy indices ``n_cells .. n_total-1`` in the HDF
        datasets.  They hold boundary-condition data and are needed when
        reconstructing WSE or velocity at perimeter faces.

        Returns
        -------
        ndarray, shape ``(n_ghost, 2)``
            May be empty (shape ``(0, 2)``) if the HDF stores no ghost rows.
        """
        return self._load("Cells Center Coordinate")[self._n_cells :]

    @property
    def cell_min_elevation(self) -> np.ndarray:
        """Minimum bed elevation per cell.  Shape ``(n_cells,)``."""
        return self._load("Cells Minimum Elevation")[: self._n_cells]

    @property
    def cell_mannings_n(self) -> np.ndarray:
        """Manning's n at each cell centre.  Shape ``(n_cells,)``."""
        return self._load("Cells Center Manning's n")[: self._n_cells]

    @property
    def cell_surface_area(self) -> np.ndarray:
        """Plan-view surface area per cell.  Shape ``(n_cells,)``."""
        return self._load("Cells Surface Area")[: self._n_cells]

    @property
    def cell_volume_elevation(self) -> tuple[np.ndarray, np.ndarray]:
        """Volume-elevation table index and values.

        Returns
        -------
        info : ndarray, shape ``(n_cells, 2)``
            Columns: ``[start_index, count]`` into *values*.
        values : ndarray, shape ``(total, 2)``
            Columns: ``[elevation, volume]``.
        """
        return (
            self._load("Cells Volume Elevation Info")[: self._n_cells],
            self._load("Cells Volume Elevation Values"),
        )

    @property
    def cell_face_info(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell-to-face connectivity index and values.

        Returns
        -------
        info : ndarray, shape ``(n_cells, 2)``
            Columns: ``[start_index, count]`` into *values*.
        values : ndarray, shape ``(total, 2)``
            Columns: ``[face_index, orientation]``.
            Orientation ``+1`` means the stored face normal points outward
            from this cell; ``-1`` means inward.

        Example
        -------
        Given::

            info = np.array([[0, 3], [3, 4]])
            values = np.array([
                [10, +1], [11, -1], [12, +1],
                [20, +1], [21, +1], [22, -1], [23, +1],
            ])

        Cell ``0`` uses ``values[0:3]`` (faces ``10, 11, 12``), and cell ``1``
        uses ``values[3:7]`` (faces ``20, 21, 22, 23``).
        """
        return (
            self._load("Cells Face and Orientation Info"),
            self._load("Cells Face and Orientation Values"),
        )

    @property
    def cell_facepoint_indexes(self) -> np.ndarray:
        """Corner facepoint indices per cell in polygon order.

        Shape ``(n_cells, 8)``.  Each row contains up to 8 facepoint
        indices (into :attr:`facepoint_coordinates`) in polygon-traversal
        order; unused slots are padded with ``-1``.

        Example
        -------
        A triangular cell and a quadrilateral cell::

            cell_facepoint_indexes = np.array([
                [ 5, 12, 18, -1, -1, -1, -1, -1],  # triangle
                [ 0,  3,  7, 11, -1, -1, -1, -1],  # quad
            ])
        """
        return self._load("Cells FacePoint Indexes")[: self._n_cells]

    @property
    def cell_polygons(self) -> list[np.ndarray]:
        """Polygon vertices for every cell in counter-clockwise order.

        Returns a list of length ``n_cells``.  Each element is an
        ``ndarray`` of shape ``(n_vertices, 2)`` containing the ``(x, y)``
        coordinates of the cell polygon in **counter-clockwise** winding
        (GeoJSON / OGC / Shapely exterior-ring convention).

        For cells that border curved faces the interior perimeter points
        from :attr:`face_perimeter` are inserted between the corner
        facepoints, giving an accurate boundary representation.  Cells
        whose face adjacency graph cannot be traversed as a closed cycle
        emit an empty ``(0, 2)`` array.

        Computed once and cached.

        Examples
        --------
        Create Shapely geometries::

            from shapely.geometry import Polygon
            polys = [Polygon(pts) for pts in fa.cell_polygons]

        Create a GeoDataFrame (requires geopandas)::

            import geopandas as gpd
            from shapely.geometry import Polygon
            gdf = gpd.GeoDataFrame(
                geometry=[Polygon(pts) for pts in fa.cell_polygons],
                crs="EPSG:…",
            )
        """
        cache_key = "_cell_polygons"
        if cache_key in self._cache:
            return self._cache[cache_key]

        from collections import defaultdict

        fp_idx = self.face_facepoint_indexes     # (n_faces, 2)
        fp_coords = self.facepoint_coordinates   # (n_facepoints, 2)
        peri_info, peri_vals = self.face_perimeter
        cell_fp_idx = self.cell_facepoint_indexes  # (n_cells, 8)
        cfi, cfv = self.cell_face_info

        n = self._n_cells

        # Cells that border at least one curved face need the slow path so
        # that interior perimeter points are inserted correctly.
        curved_face_idxs = np.where(peri_info[:, 1] > 0)[0]
        if len(curved_face_idxs) > 0:
            face_cell_idx = self.face_cell_indexes
            curved_cells: set[int] = set(
                face_cell_idx[curved_face_idxs].flatten().tolist()
            )
            curved_cells.discard(-1)
        else:
            curved_cells = set()

        polygons: list[np.ndarray] = [np.empty((0, 2), dtype=np.float64) for x in range(n)]

        for c in range(n):
            if c not in curved_cells:
                # Fast path: polygon-order corners are already in HDF.
                corners = cell_fp_idx[c]
                valid = corners[corners >= 0]
                poly: np.ndarray = fp_coords[valid].copy()
            else:
                # Slow path: adjacency graph walk, inserting interior points.
                start = int(cfi[c, 0])
                count = int(cfi[c, 1])
                face_idxs = cfv[start : start + count, 0]

                adj: dict[int, list[int]] = defaultdict(list)
                edge_to_face: dict[tuple[int, int], int] = {}
                for f in face_idxs:
                    f = int(f)
                    fp0 = int(fp_idx[f, 0])
                    fp1 = int(fp_idx[f, 1])
                    adj[fp0].append(fp1)
                    adj[fp1].append(fp0)
                    edge_to_face[(fp0, fp1)] = f
                    edge_to_face[(fp1, fp0)] = f

                all_pts: list[np.ndarray] = []
                fps_list = list(adj.keys())
                current = fps_list[0]
                prev_fp = None
                for _ in range(count):
                    all_pts.append(fp_coords[current])
                    nb = adj[current]
                    nxt = nb[1] if nb[0] == prev_fp else nb[0]

                    face_i = edge_to_face.get((current, nxt))
                    if face_i is not None:
                        pstart = int(peri_info[face_i, 0])
                        n_int = int(peri_info[face_i, 1])
                        if n_int > 0:
                            pts = peri_vals[pstart : pstart + n_int]
                            if current != int(fp_idx[face_i, 0]):
                                pts = pts[::-1]
                            all_pts.extend(pts)

                    prev_fp = current
                    current = nxt

                if current != fps_list[0]:  # cycle did not close — malformed
                    continue

                poly = np.array(all_pts)

            if len(poly) < 3:
                continue

            # Enforce CCW winding (shoelace signed area: positive → CCW).
            x, y = poly[:, 0], poly[:, 1]
            signed_area = float(
                np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)
            )
            if signed_area < 0:
                poly = poly[::-1]

            polygons[c] = poly

        self._cache[cache_key] = polygons
        return polygons

    # ------------------------------------------------------------------
    # Face geometry
    # ------------------------------------------------------------------

    @property
    def face_normals(self) -> np.ndarray:
        """Face unit normal vectors and plan-view lengths.

        Shape ``(n_faces, 3)``.  Columns: ``[nx, ny, length]``.
        The normal points from the *left* cell to the *right* cell as
        defined by :attr:`face_cell_indexes`.

        Example
        -------
        Given::

            face_normals = np.array([
                [1.0, 0.0, 12.5],
                [0.0, -1.0, 8.0],
            ])

        Face ``0`` has a unit normal pointing in ``+nx`` with length ``12.5``.
        Face ``1`` has a unit normal pointing in ``-ny`` with length ``8.0``.
        """
        return self._load("Faces NormalUnitVector and Length")

    @property
    def face_cell_indexes(self) -> np.ndarray:
        """Left and right cell index for each face.

        Shape ``(n_faces, 2)``.  A value of ``-1`` indicates a boundary
        face with no neighbour on that side.

        Example
        -------
        Given::

            face_cell_indexes = np.array([
                [0, 1],
                [1, 2],
                [2, -1],
                [-1, 0],
            ])

        Face ``0`` is interior between cells ``0`` (left) and ``1`` (right).
        Face ``2`` is a boundary face for cell ``2`` on the left side, and
        face ``3`` is a boundary face for cell ``0`` on the right side.
        """
        return self._load("Faces Cell Indexes")

    @property
    def face_min_elevation(self) -> np.ndarray:
        """Minimum bed elevation at each face centroid.  Shape ``(n_faces,)``."""
        return self._load("Faces Minimum Elevation")

    @property
    def face_invert_station(self) -> np.ndarray:
        """Station of the invert centroid of each face.

        HEC-RAS stores the station distance (from one end of the face) to the
        centroid of the bottom 5% of the face cross-sectional area — the invert
        region used internally for flow calculations.

        Shape ``(n_faces,)``, dtype float32.
        """
        return self._load("Faces Low Elevation Centroid")

    @property
    def face_facepoint_indexes(self) -> np.ndarray:
        """Start and end face-point indices for each face.

        Shape ``(n_faces, 2)``.

        Example
        -------
        Given::

            face_facepoint_indexes = np.array([
                [12, 18],
                [18, 24],
                [24, 12],
            ])

        Face ``0`` connects facepoints ``12`` and ``18``; face ``1`` connects
        ``18`` and ``24``; face ``2`` connects ``24`` and ``12``.
        """
        return self._load("Faces FacePoint Indexes")

    @property
    def face_perimeter(self) -> tuple[np.ndarray, np.ndarray]:
        """Interior perimeter points along curved faces.

        Most faces are straight lines between their two endpoint facepoints.
        For curved faces, one or more intermediate ``(x, y)`` coordinate
        points are stored here.  These points are **not** indexed facepoints;
        they do not appear in :attr:`facepoint_coordinates`.

        Returns
        -------
        info : ndarray, shape ``(n_faces, 2)``
            ``[start_index, count]`` into *values*.  ``count == 0`` for
            straight faces.
        values : ndarray, shape ``(total_interior_pts, 2)``
            ``(x, y)`` coordinates of interior perimeter points in order
            along the face polyline.
        """
        return (
            self._load("Faces Perimeter Info"),
            self._load("Faces Perimeter Values"),
        )

    @property
    def face_centroids(self) -> np.ndarray:
        """Centroid coordinate of each face.  Shape ``(n_faces, 2)``.

        For straight faces (no interior perimeter points) this equals the
        midpoint of the chord between the two endpoint facepoints.  For
        curved faces this is the **arc midpoint** — the point on the face
        polyline at exactly half the total arc length — which guarantees the
        result lies on the face itself.  An arc-length-weighted average of
        segment midpoints is *not* used because it can float off the polyline
        for non-convex shapes (V-notches, loops).

        Computed once and cached; subsequent accesses return the same array.
        """
        cache_key = "_face_centroids"
        if cache_key in self._cache:
            return self._cache[cache_key]

        fp_idx = self._load("Faces FacePoint Indexes")    # (n_faces, 2)
        fp_coords = self._load("FacePoints Coordinate")   # (n_facepoints, 2)
        peri_info = self._load("Faces Perimeter Info")    # (n_faces, 2)
        peri_vals = self._load("Faces Perimeter Values")  # (total, 2)

        fp0 = fp_coords[fp_idx[:, 0]]   # (n_faces, 2)
        fp1 = fp_coords[fp_idx[:, 1]]   # (n_faces, 2)

        # Default: straight-face midpoint for all faces.
        # For a two-point segment the arc midpoint equals the chord midpoint.
        centroids = 0.5 * (fp0 + fp1)

        # Override for curved faces (interior perimeter points present).
        # Use the arc midpoint — the point on the polyline at exactly half the
        # total arc length.  This guarantees the centroid lies ON the face
        # polyline, unlike an arc-length-weighted average which can float off
        # the curve for non-convex shapes (V-notches, loops, etc.).
        curved = np.where(peri_info[:, 1] > 0)[0]
        for fi in curved:
            start = int(peri_info[fi, 0])
            count = int(peri_info[fi, 1])
            pts = np.vstack([fp0[fi], peri_vals[start : start + count], fp1[fi]])
            seg_len = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
            total = seg_len.sum()
            if total == 0:
                centroids[fi] = pts.mean(axis=0)
                continue
            cumlen = np.concatenate([[0.0], np.cumsum(seg_len)])
            half = total / 2.0
            seg_idx = int(np.searchsorted(cumlen[1:], half, side="right"))
            seg_idx = min(seg_idx, len(seg_len) - 1)
            t = (half - cumlen[seg_idx]) / seg_len[seg_idx]
            centroids[fi] = pts[seg_idx] + t * (pts[seg_idx + 1] - pts[seg_idx])

        self._cache[cache_key] = centroids
        return centroids

    @property
    def facepoint_coordinates(self) -> np.ndarray:
        """x, y coordinates of all face endpoints.  Shape ``(n_facepoints, 2)``."""
        return self._load("FacePoints Coordinate")

    @property
    def face_area_elevation(self) -> tuple[np.ndarray, np.ndarray]:
        """Hydraulic property table for each face.

        Returns
        -------
        info : ndarray, shape ``(n_faces, 2)``
            Columns: ``[start_index, count]`` into *values*.
        values : ndarray, shape ``(total, 4)``
            Columns: ``[elevation, flow_area, wetted_perimeter, mannings_n]``.

        Example
        -------
        Given::

            info = np.array([[0, 2], [2, 3]])
            values = np.array([
                [100.0, 0.0, 10.0, 0.030],
                [101.0, 5.0, 11.0, 0.030],
                [100.0, 0.0, 12.0, 0.035],
                [101.0, 6.0, 13.0, 0.035],
                [102.0, 9.0, 14.0, 0.035],
            ])

        Face ``0`` uses ``values[0:2]`` and face ``1`` uses ``values[2:5]``.
        For face ``0`` specifically:
        ``[100.0, 0.0, 10.0, 0.030]`` means at elevation ``100.0`` the flow
        area is ``0.0`` (dry threshold), wetted perimeter is ``10.0``, and
        Manning's ``n`` is ``0.030``; ``[101.0, 5.0, 11.0, 0.030]`` is the
        next stage row with larger flow area/perimeter at elevation ``101.0``.
        """
        return (
            self._load("Faces Area Elevation Info"),
            self._load("Faces Area Elevation Values"),
        )

    # ------------------------------------------------------------------
    # Perimeter
    # ------------------------------------------------------------------

    @property
    def perimeter(self) -> np.ndarray:
        """Boundary polygon of the 2-D flow area.  Shape ``(n_pts, 2)``."""
        return self._load("Perimeter")

    def check_cells(
        self,
        *,
        check_boundary: bool = True,
        tol: float = 1e-10,
    ) -> dict:
        """Check all mesh cells against HEC-RAS geometric validity rules.

        Calls :func:`raspy.geo.mesh_validation.check_mesh_cells` with the
        HDF arrays for this flow area.

        Parameters
        ----------
        check_boundary :
            When ``True`` (default) each cell centre is also tested against
            the 2D flow area perimeter polygon (rule 5).
        tol :
            Absolute tolerance for near-collinear edge detection.

        Returns
        -------
        dict
            Full validation report.  Pass to
            :func:`raspy.geo.mesh_validation.print_mesh_report` for a
            human-readable summary.
        """
        from raspy.geo.mesh_validation import check_mesh_cells

        cfi, cfv = self.cell_face_info
        peri_info, peri_vals = self.face_perimeter
        return check_mesh_cells(
            cell_centers=self.cell_centers,
            facepoint_coordinates=self.facepoint_coordinates,
            face_facepoint_indexes=self.face_facepoint_indexes,
            cell_face_info=cfi,
            cell_face_values=cfv,
            face_perimeter_info=peri_info,
            face_perimeter_values=peri_vals,
            boundary_polygon=self.perimeter if check_boundary else None,
            tol=tol,
        )

    # ------------------------------------------------------------------
    # Derived topology and geometry helpers (computed from HDF arrays)
    # ------------------------------------------------------------------

    @property
    def facepoint_to_faces(self) -> list[np.ndarray]:
        """Face indices adjacent to each facepoint.

        Returns a list of length ``n_facepoints``.  Element ``i`` is an
        ``int64`` array of face indices whose endpoints include facepoint
        ``i``.  Most interior facepoints are shared by exactly two faces;
        boundary facepoints may appear in more.

        Computed once and cached.
        """
        cache_key = "_facepoint_to_faces"
        if cache_key in self._cache:
            return self._cache[cache_key]  # type: ignore[return-value]

        fp_idx = self.face_facepoint_indexes  # (n_faces, 2)
        n_fp = len(self.facepoint_coordinates)
        n_faces = len(fp_idx)

        buckets: list[list[int]] = [[] for _ in range(n_fp)]
        for fi in range(n_faces):
            buckets[int(fp_idx[fi, 0])].append(fi)
            buckets[int(fp_idx[fi, 1])].append(fi)

        result = [np.array(b, dtype=np.int64) for b in buckets]
        self._cache[cache_key] = result  # type: ignore[assignment]
        return result

    @property
    def facepoint_to_cells(self) -> list[np.ndarray]:
        """Real cell indices surrounding each facepoint.

        Returns a list of length ``n_facepoints``.  Element ``i`` is a
        sorted ``int64`` array of cell indices (``< n_cells``) that include
        facepoint ``i`` as a polygon corner.  Ghost cells and the sentinel
        value ``-1`` are excluded.

        Computed once and cached.
        """
        cache_key = "_facepoint_to_cells"
        if cache_key in self._cache:
            return self._cache[cache_key]  # type: ignore[return-value]

        fp_to_faces = self.facepoint_to_faces
        fci = self.face_cell_indexes  # (n_faces, 2)
        n = self._n_cells

        result: list[np.ndarray] = []
        for faces in fp_to_faces:
            cells: set[int] = set()
            for fi in faces:
                c0, c1 = int(fci[fi, 0]), int(fci[fi, 1])
                if 0 <= c0 < n:
                    cells.add(c0)
                if 0 <= c1 < n:
                    cells.add(c1)
            result.append(np.array(sorted(cells), dtype=np.int64))

        self._cache[cache_key] = result  # type: ignore[assignment]
        return result

    @property
    def cell_neighbors(self) -> list[np.ndarray]:
        """Adjacent real cell indices for each cell.

        Returns a list of length ``n_cells``.  Element ``c`` is an
        ``int64`` array of real cell indices (``< n_cells``) that share a
        face with cell ``c``.  Boundary faces (where the neighbour index is
        ``-1``) contribute no entry.  Order follows the face order in
        :attr:`cell_face_info`.

        Computed once and cached.
        """
        cache_key = "_cell_neighbors"
        if cache_key in self._cache:
            return self._cache[cache_key]  # type: ignore[return-value]

        cfi, cfv = self.cell_face_info
        fci = self.face_cell_indexes
        n = self._n_cells

        neighbors: list[np.ndarray] = []
        for c in range(n):
            start = int(cfi[c, 0])
            count = int(cfi[c, 1])
            nb: list[int] = []
            for fi in cfv[start : start + count, 0]:
                c0, c1 = int(fci[fi, 0]), int(fci[fi, 1])
                other = c1 if c0 == c else c0
                if 0 <= other < n:
                    nb.append(other)
            neighbors.append(np.array(nb, dtype=np.int64))

        self._cache[cache_key] = neighbors  # type: ignore[assignment]
        return neighbors

    @property
    def boundary_face_mask(self) -> np.ndarray:
        """Boolean mask of faces on the 2D flow area perimeter.

        Shape ``(n_faces,)``.  ``True`` for faces where one side has no
        neighbouring cell (``face_cell_indexes`` value is ``-1``).

        Computed once and cached.
        """
        cache_key = "_boundary_face_mask"
        if cache_key in self._cache:
            return self._cache[cache_key]
        fci = self.face_cell_indexes
        mask = (fci[:, 0] < 0) | (fci[:, 1] < 0)
        self._cache[cache_key] = mask
        return mask

    @property
    def boundary_cell_mask(self) -> np.ndarray:
        """Boolean mask of cells that touch the flow area perimeter.

        Shape ``(n_cells,)``.  ``True`` for any real cell that has at least
        one boundary face (see :attr:`boundary_face_mask`).

        Computed once and cached.
        """
        cache_key = "_boundary_cell_mask"
        if cache_key in self._cache:
            return self._cache[cache_key]

        bfm = self.boundary_face_mask
        fci = self.face_cell_indexes
        n = self._n_cells

        bfi = np.where(bfm)[0]
        bc = fci[bfi].flatten()
        bc = bc[(bc >= 0) & (bc < n)]
        mask = np.zeros(n, dtype=bool)
        mask[bc] = True
        self._cache[cache_key] = mask
        return mask

    @property
    def face_polylines(self) -> list[np.ndarray]:
        """Full vertex sequence for each face.

        Returns a list of length ``n_faces``.  Each element is an
        ``ndarray`` of shape ``(n_pts, 2)`` in canonical HDF order
        (facepoint 0 → interior points → facepoint 1).  Straight faces
        yield 2-point arrays; curved faces have 3 or more points.

        Computed once and cached.
        """
        cache_key = "_face_polylines"
        if cache_key in self._cache:
            return self._cache[cache_key]  # type: ignore[return-value]

        fp_idx = self.face_facepoint_indexes    # (n_faces, 2)
        fp_coords = self.facepoint_coordinates  # (n_facepoints, 2)
        peri_info, peri_vals = self.face_perimeter
        n_faces = len(fp_idx)

        fp0 = fp_coords[fp_idx[:, 0]]  # (n_faces, 2)
        fp1 = fp_coords[fp_idx[:, 1]]  # (n_faces, 2)

        polylines: list[np.ndarray] = [
            np.array([fp0[fi], fp1[fi]]) for fi in range(n_faces)
        ]
        for fi in np.where(peri_info[:, 1] > 0)[0]:
            start = int(peri_info[fi, 0])
            n_int = int(peri_info[fi, 1])
            polylines[fi] = np.vstack(
                [fp0[fi], peri_vals[start : start + n_int], fp1[fi]]
            )

        self._cache[cache_key] = polylines  # type: ignore[assignment]
        return polylines

    @property
    def face_lengths(self) -> np.ndarray:
        """Arc length of each face polyline.  Shape ``(n_faces,)``.

        Straight faces: Euclidean chord length (equals column 2 of
        :attr:`face_normals`).  Curved faces: sum of segment lengths along
        the full polyline from :attr:`face_perimeter`.

        Computed once and cached.
        """
        cache_key = "_face_lengths"
        if cache_key in self._cache:
            return self._cache[cache_key]

        fp_idx = self.face_facepoint_indexes
        fp_coords = self.facepoint_coordinates
        peri_info, peri_vals = self.face_perimeter

        fp0 = fp_coords[fp_idx[:, 0]]
        fp1 = fp_coords[fp_idx[:, 1]]
        lengths = np.linalg.norm(fp1 - fp0, axis=1).copy()

        for fi in np.where(peri_info[:, 1] > 0)[0]:
            start = int(peri_info[fi, 0])
            n_int = int(peri_info[fi, 1])
            pts = np.vstack([fp0[fi], peri_vals[start : start + n_int], fp1[fi]])
            lengths[fi] = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())

        self._cache[cache_key] = lengths
        return lengths

    @property
    def cell_bbox(self) -> np.ndarray:
        """Axis-aligned bounding box per cell.  Shape ``(n_cells, 4)``.

        Columns: ``[xmin, ymin, xmax, ymax]``.  Derived from the polygon
        corner facepoints in :attr:`cell_facepoint_indexes`.  Interior
        perimeter points on curved faces are not included; the error is
        negligible for the near-convex cells HEC-RAS requires.

        Computed once and cached.
        """
        cache_key = "_cell_bbox"
        if cache_key in self._cache:
            return self._cache[cache_key]

        cell_fp_idx = self.cell_facepoint_indexes  # (n_cells, 8)
        fp_coords = self.facepoint_coordinates     # (n_facepoints, 2)

        valid = cell_fp_idx >= 0
        safe = np.where(valid, cell_fp_idx, 0)
        coords = fp_coords[safe]  # (n_cells, 8, 2)

        cx = np.where(valid, coords[:, :, 0], np.nan)
        cy = np.where(valid, coords[:, :, 1], np.nan)
        bbox = np.column_stack([
            np.nanmin(cx, axis=1),
            np.nanmin(cy, axis=1),
            np.nanmax(cx, axis=1),
            np.nanmax(cy, axis=1),
        ])
        self._cache[cache_key] = bbox
        return bbox

    @property
    def mesh_bbox(self) -> np.ndarray:
        """Overall bounding box of the flow area.  Shape ``(4,)``.

        Returns ``[xmin, ymin, xmax, ymax]`` covering all cells.
        """
        b = self.cell_bbox
        return np.array([b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()])

    @property
    def cell_aspect_ratio(self) -> np.ndarray:
        """Aspect ratio of each cell.  Shape ``(n_cells,)``.

        Ratio of the longest face to the shortest face (by arc length from
        :attr:`face_lengths`).  Value of ``1.0`` for a regular cell; large
        values indicate elongated cells.

        Computed once and cached.
        """
        cache_key = "_cell_aspect_ratio"
        if cache_key in self._cache:
            return self._cache[cache_key]

        face_len = self.face_lengths
        cfi, cfv = self.cell_face_info
        n = self._n_cells

        ratios = np.ones(n)
        for c in range(n):
            start = int(cfi[c, 0])
            count = int(cfi[c, 1])
            lens = face_len[cfv[start : start + count, 0]]
            mn = float(lens.min())
            if mn > 0.0:
                ratios[c] = float(lens.max()) / mn

        self._cache[cache_key] = ratios
        return ratios

    @property
    def cell_compactness(self) -> np.ndarray:
        """Isoperimetric compactness of each cell.  Shape ``(n_cells,)``.

        ``compactness = 4π × area / perimeter²``.  A circle scores ``1.0``;
        elongated or irregular cells score lower.  Uses :attr:`cell_surface_area`
        for area and :attr:`face_lengths` for the perimeter.

        Computed once and cached.
        """
        cache_key = "_cell_compactness"
        if cache_key in self._cache:
            return self._cache[cache_key]

        area = self.cell_surface_area
        face_len = self.face_lengths
        cfi, cfv = self.cell_face_info
        n = self._n_cells

        perimeter = np.zeros(n)
        for c in range(n):
            start = int(cfi[c, 0])
            count = int(cfi[c, 1])
            perimeter[c] = face_len[cfv[start : start + count, 0]].sum()

        with np.errstate(invalid="ignore", divide="ignore"):
            compactness = np.where(
                perimeter > 0.0,
                4.0 * np.pi * area / perimeter**2,
                np.nan,
            )
        self._cache[cache_key] = compactness
        return compactness

    def cells_containing_points(self, xy: np.ndarray) -> np.ndarray:
        """Return the cell index containing each query point.

        Parameters
        ----------
        xy : ndarray, shape ``(m, 2)`` or ``(2,)``
            Query ``(x, y)`` coordinates.

        Returns
        -------
        ndarray, shape ``(m,)``, dtype ``int64``
            0-based cell index for each point, or ``-1`` when outside all
            cells (or the cell polygon is malformed).

        Notes
        -----
        Uses :attr:`cell_bbox` for a fast bounding-box pre-filter then
        :attr:`cell_polygons` for exact ray-casting containment.  No
        spatial index is built; for very large query sets consider
        building an external RTree over :attr:`cell_bbox`.
        """
        xy = np.asarray(xy, dtype=np.float64)
        scalar = xy.ndim == 1
        if scalar:
            xy = xy[np.newaxis]
        m = len(xy)

        bbox = self.cell_bbox      # (n_cells, 4)
        polygons = self.cell_polygons

        result = np.full(m, -1, dtype=np.int64)
        for qi in range(m):
            px, py = float(xy[qi, 0]), float(xy[qi, 1])
            candidates = np.where(
                (px >= bbox[:, 0]) & (px <= bbox[:, 2]) &
                (py >= bbox[:, 1]) & (py <= bbox[:, 3])
            )[0]
            for ci in candidates:
                poly = polygons[ci]
                if len(poly) >= 3 and _point_in_polygon(px, py, poly):
                    result[qi] = int(ci)
                    break

        return int(result[0]) if scalar else result

    def wse_at_facepoints(self, cell_wse: np.ndarray) -> np.ndarray:
        """Interpolate water-surface elevation to facepoints.

        For each facepoint the WSE is the mean of the WSEs of all wet
        (non-``NaN``) cells that share that facepoint, using
        :attr:`cell_facepoint_indexes` so each adjacent cell is counted
        exactly once.  A facepoint with no adjacent wet cells returns
        ``NaN``.

        Parameters
        ----------
        cell_wse : ndarray, shape ``(n_cells,)`` or ``(n_cells + n_ghost,)``
            Water-surface elevation per cell.  Pass ``NaN`` for dry cells.
            When the array is extended to include ghost-cell WSEs (indices
            ``n_cells_real .. n_total-1``), perimeter facepoints receive
            contributions from ghost cells, improving boundary accuracy.

        Returns
        -------
        ndarray, shape ``(n_facepoints,)``
            WSE at each facepoint.  ``NaN`` where all adjacent cells are dry.
        """
        cell_wse = np.asarray(cell_wse, dtype=np.float64)
        n_cells = len(cell_wse)
        # Load the raw dataset (not the [:n_cells]-clipped property) so that
        # ghost-cell rows are included when the caller passes an extended
        # cell_wse array (length = n_cells_real + n_ghost).
        cfi = self._load("Cells FacePoint Indexes")[:n_cells]  # (n_cells, 8)
        n_fp = len(self.facepoint_coordinates)

        wet = ~np.isnan(cell_wse)
        # accumulators for each fp
        fp_sum = np.zeros(n_fp)
        fp_count = np.zeros(n_fp, dtype=np.int64)
        # loop over the 8 facepoint slots of every cell
        for j in range(cfi.shape[1]):
            # gives jth facepoint index for each cell
            fp_col = cfi[:, j]
            # For each wet cell with a real facepoint in that slot, add that cell’s 
            # WSE into the bucket for that facepoint index
            valid = (fp_col >= 0) & wet
            np.add.at(fp_sum, fp_col[valid], cell_wse[valid])
            np.add.at(fp_count, fp_col[valid], 1)

        result = np.full(n_fp, np.nan)
        mask = fp_count > 0
        np.divide(fp_sum, fp_count, out=result, where=mask)
        return result

    def wse_at_facecentroids(self, cell_wse: np.ndarray) -> np.ndarray:
        """Average water-surface elevation at the centroid of each face.

        For each face the WSE is the mean of the WSEs of its adjacent cells
        (left and right as defined by :attr:`face_cell_indexes`), ignoring
        dry (``NaN``) cells.  Boundary faces have only one adjacent cell, so
        that cell's WSE is returned directly.  A face whose adjacent cells are
        all dry returns ``NaN``.

        Parameters
        ----------
        cell_wse : ndarray, shape ``(n_cells,)``
            Water-surface elevation per cell.  Pass ``NaN`` for dry cells.

        Returns
        -------
        ndarray, shape ``(n_faces,)``
            WSE at each face centroid.  ``NaN`` where all adjacent cells are dry.
        """
        cell_wse = np.asarray(cell_wse, dtype=np.float64)
        n_cells = len(cell_wse)
        fci = self.face_cell_indexes  # (n_faces, 2), -1 = boundary side

        # Append a NaN sentinel so that boundary/ghost-cell indices map to NaN.
        # face_cell_indexes can contain positive ghost-cell indices >= n_cells
        # in addition to -1 boundary markers; both must be remapped to the sentinel.
        padded = np.append(cell_wse, np.nan)                       # index n_cells → NaN
        # sentinel both cases: boundary (-1) and ghost cells (>= n_cells)
        safe = np.where((fci < 0) | (fci >= n_cells), n_cells, fci)

        # maps invalid cells to NaN which at index n_cells in cell_wse
        vals = padded[safe]                          # (n_faces, 2)
        with np.errstate(all="ignore"):              # suppress all-NaN warning
            return np.nanmean(vals, axis=1)


# ---------------------------------------------------------------------------
# FlowAreaCollection — dict-like access to all flow areas in the HDF file
# ---------------------------------------------------------------------------


class FlowAreaCollection:
    """Access all 2-D flow areas stored in an HDF geometry or plan file.

    Parameters
    ----------
    hdf:
        Open ``h5py.File`` handle.
    """

    def __init__(self, hdf: "h5py.File") -> None:
        self._hdf = hdf
        self._cache: dict[str, FlowArea] = {}

    # ------------------------------------------------------------------
    # Attributes table
    # ------------------------------------------------------------------

    @property
    def summary(self) -> pd.DataFrame:
        """One row per 2-D flow area with geometry attributes.

        Columns include ``name``, ``cell_count``, and all other fields from
        the ``Geometry/2D Flow Areas/Attributes`` structured dataset.
        """
        attrs_ds = self._hdf[_GEOM_2D_ATTRS]
        data = np.array(attrs_ds)
        df = pd.DataFrame(data)
        # Decode byte-string columns
        for col in df.columns:
            if df[col].dtype.kind in ("S", "O"):
                df[col] = df[col].str.decode("utf-8").str.strip()
        # Normalise the name column to lower-case key 'name'
        name_col = next((c for c in df.columns if c.lower() == "name"), None)
        if name_col and name_col != "name":
            df = df.rename(columns={name_col: "name"})
        return df

    @property
    def names(self) -> list[str]:
        """Names of all 2-D flow areas in the file."""
        import h5py

        root = self._hdf[_GEOM_2D_ROOT]
        return [
            k
            for k, v in root.items()
            if isinstance(v, h5py.Group) and k != "Attributes"
        ]

    # ------------------------------------------------------------------
    # Item access
    # ------------------------------------------------------------------

    def __getitem__(self, name: str) -> FlowArea:
        if name not in self._cache:
            root = self._hdf[_GEOM_2D_ROOT]
            if name not in root:
                raise KeyError(
                    f"2D flow area {name!r} not found. Available: {self.names}"
                )
            n_cells = self._get_real_cell_count(name)
            self._cache[name] = FlowArea(root[name], name, n_cells)
        return self._cache[name]

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def __iter__(self):
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_real_cell_count(self, area_name: str) -> int:
        """Return the number of real (non-ghost) cells for *area_name*.

        Reads from ``Geometry/2D Flow Areas/Attributes`` structured array.
        Falls back to the Water Surface result shape if Attributes lookup
        fails, then to the coordinate array length as a last resort.

        Derived from archive/ras_tools/r2d/ras2d_cell_velocity.py
        ``RAS2DCellVelocity._get_real_cell_count``.

        TODO: this may be uncessary single the 'Cell Count' in summary/attributes
          is reliably populated by HEC-RAS.
        """
        attrs_ds = self._hdf.get(_GEOM_2D_ATTRS)
        if attrs_ds is not None:
            attrs = np.array(attrs_ds)
            dtype_names = [f.lower() for f in attrs_ds.dtype.names]
            name_idx = next((i for i, f in enumerate(dtype_names) if "name" in f), None)
            count_idx = next(
                (i for i, f in enumerate(dtype_names) if "cell" in f and "count" in f),
                None,
            )
            if name_idx is not None and count_idx is not None:
                field_name = attrs_ds.dtype.names[name_idx]
                field_count = attrs_ds.dtype.names[count_idx]
                for row in attrs:
                    row_name = row[field_name]
                    if isinstance(row_name, bytes):
                        row_name = row_name.decode().strip()
                    if row_name == area_name:
                        return int(row[field_count])

        # Fallback: coordinate array length (may include ghost cells)
        coord = self._hdf[f"{_GEOM_2D_ROOT}/{area_name}/Cells Center Coordinate"]
        return coord.shape[0]


# ---------------------------------------------------------------------------
# StorageArea / StorageAreaCollection
# ---------------------------------------------------------------------------


def _decode(value: bytes | str) -> str:
    """Decode a byte-string field from an HDF structured array."""
    if isinstance(value, bytes):
        return value.decode("utf-8").strip()
    return str(value).strip()


@dataclass
class StorageArea:
    """Geometry for a single HEC-RAS storage area (reservoir, pond, etc.).

    Attributes
    ----------
    name:
        Name of the storage area.
    mode:
        Storage mode string from HEC-RAS (e.g. ``"Elev Vol RC"`` for an
        elevation-volume rating curve, or ``"Normal"`` for a flat-pool).
    boundary:
        x, y coordinates of the storage area boundary polygon.
        Shape ``(n_pts, 2)``.
    volume_elevation:
        Elevation-volume rating curve.  Shape ``(n_pairs, 2)`` with columns
        ``[elevation, volume]``.  Empty array (shape ``(0, 2)``) when the
        storage area has no rating curve (e.g. flat-pool mode).
    """

    name: str
    mode: str
    boundary: np.ndarray  # (n_pts, 2)
    volume_elevation: np.ndarray  # (n_pairs, 2): [elevation, volume]

    @property
    def elevations(self) -> np.ndarray:
        """Elevation column of the rating curve.  Shape ``(n_pairs,)``."""
        return self.volume_elevation[:, 0]

    @property
    def volumes(self) -> np.ndarray:
        """Volume column of the rating curve.  Shape ``(n_pairs,)``."""
        return self.volume_elevation[:, 1]

    def volume_at_elevation(self, wse: float) -> float:
        """Return interpolated stored volume at *wse*.

        Uses linear interpolation via ``numpy.interp``.  Values outside the
        rating-curve range are clamped to the curve endpoints.

        Raises
        ------
        ValueError
            If the storage area has no volume-elevation rating curve.
        """
        if len(self.volume_elevation) == 0:
            raise ValueError(
                f"Storage area {self.name!r} has no volume-elevation rating curve "
                f"(mode={self.mode!r})."
            )
        return float(np.interp(wse, self.elevations, self.volumes))


class StorageAreaCollection:
    """Access all storage areas stored in an HDF geometry file.

    Parameters
    ----------
    hdf:
        Open ``h5py.File`` handle.
    """

    def __init__(self, hdf: "h5py.File") -> None:
        self._hdf = hdf
        self._items: dict[str, StorageArea] | None = None

    def _load(self) -> dict[str, StorageArea]:
        if self._items is not None:
            return self._items

        if _SA_ROOT not in self._hdf:
            self._items = {}
            return self._items

        root = self._hdf[_SA_ROOT]
        attrs = np.array(root["Attributes"])
        poly_info = np.array(root["Polygon Info"])  # (n, 4): [start_pt, n_pts, ...]
        poly_pts = np.array(root["Polygon Points"])  # (total, 2)
        ve_info = np.array(root["Volume Elevation Info"])  # (n, 2): [start, count]
        ve_vals = np.array(root["Volume Elevation Values"])  # (total, 2)

        items: dict[str, StorageArea] = {}
        for i, row in enumerate(attrs):
            name = _decode(row["Name"])
            mode = _decode(row["Mode"])

            start_pt = int(poly_info[i, 0])
            n_pts = int(poly_info[i, 1])
            boundary = poly_pts[start_pt : start_pt + n_pts].astype(float)

            ve_start = int(ve_info[i, 0])
            ve_count = int(ve_info[i, 1])
            if ve_count > 0:
                vol_elev = ve_vals[ve_start : ve_start + ve_count].astype(float)
            else:
                vol_elev = np.empty((0, 2), dtype=float)

            items[name] = StorageArea(
                name=name,
                mode=mode,
                boundary=boundary,
                volume_elevation=vol_elev,
            )

        self._items = items
        return self._items

    @property
    def names(self) -> list[str]:
        """Names of all storage areas in the file."""
        return list(self._load().keys())

    @property
    def summary(self) -> pd.DataFrame:
        """One row per storage area with basic attributes.

        Columns: ``name``, ``mode``, ``n_boundary_points``,
        ``n_elev_vol_pairs``.
        """
        rows = [
            {
                "name": sa.name,
                "mode": sa.mode,
                "n_boundary_points": len(sa.boundary),
                "n_elev_vol_pairs": len(sa.volume_elevation),
            }
            for sa in self._load().values()
        ]
        return pd.DataFrame(rows)

    def __getitem__(self, name: str) -> StorageArea:
        items = self._load()
        if name not in items:
            raise KeyError(
                f"Storage area {name!r} not found. Available: {self.names}"
            )
        return items[name]

    def __contains__(self, name: str) -> bool:
        return name in self._load()

    def __iter__(self):
        return iter(self._load())

    def __len__(self) -> int:
        return len(self._load())


# ---------------------------------------------------------------------------
# BoundaryConditionLine / BoundaryConditionCollection
# ---------------------------------------------------------------------------


@dataclass
class BoundaryConditionLine:
    """A single HEC-RAS boundary condition line.

    Attributes
    ----------
    name:
        Name of the BC line (e.g. ``"DSNormalDepth1"``).
    connected_area:
        Name of the 2-D flow area or storage area this line belongs to
        (the ``SA-2D`` field in the HDF attributes table).
    bc_type:
        Type string (e.g. ``"External"`` or ``"Internal"``).
    polyline:
        x, y coordinates of the BC line.  Shape ``(n_pts, 2)``.
    """

    name: str
    connected_area: str
    bc_type: str
    polyline: np.ndarray  # (n_pts, 2)


class BoundaryConditionCollection:
    """Access all boundary condition lines in an HDF geometry file.

    Parameters
    ----------
    hdf:
        Open ``h5py.File`` handle.
    """

    def __init__(self, hdf: "h5py.File") -> None:
        self._hdf = hdf
        self._items: dict[str, BoundaryConditionLine] | None = None

    def _load(self) -> dict[str, BoundaryConditionLine]:
        if self._items is not None:
            return self._items

        if _BC_ROOT not in self._hdf:
            self._items = {}
            return self._items

        root = self._hdf[_BC_ROOT]
        attrs = np.array(root["Attributes"])
        poly_info = np.array(root["Polyline Info"])  # (n, 4): [start_pt, n_pts, ...]
        poly_pts = np.array(root["Polyline Points"])  # (total, 2)

        # Resolve the connected-area field name (varies: "SA-2D" or "SA/2D Area")
        sa_field = next(
            (f for f in attrs.dtype.names if "2D" in f or "2d" in f),
            None,
        )

        items: dict[str, BoundaryConditionLine] = {}
        for i, row in enumerate(attrs):
            name = _decode(row["Name"])
            connected = _decode(row[sa_field]) if sa_field else ""
            bc_type = _decode(row["Type"])

            start_pt = int(poly_info[i, 0])
            n_pts = int(poly_info[i, 1])
            polyline = poly_pts[start_pt : start_pt + n_pts].astype(float)

            items[name] = BoundaryConditionLine(
                name=name,
                connected_area=connected,
                bc_type=bc_type,
                polyline=polyline,
            )

        self._items = items
        return self._items

    @property
    def names(self) -> list[str]:
        """Names of all boundary condition lines in the file."""
        return list(self._load().keys())

    @property
    def summary(self) -> pd.DataFrame:
        """One row per BC line with basic attributes.

        Columns: ``name``, ``connected_area``, ``type``, ``n_points``.
        """
        rows = [
            {
                "name": bc.name,
                "connected_area": bc.connected_area,
                "type": bc.bc_type,
                "n_points": len(bc.polyline),
            }
            for bc in self._load().values()
        ]
        return pd.DataFrame(rows)

    def __getitem__(self, name: str) -> BoundaryConditionLine:
        items = self._load()
        if name not in items:
            raise KeyError(
                f"Boundary condition line {name!r} not found. "
                f"Available: {self.names}"
            )
        return items[name]

    def __contains__(self, name: str) -> bool:
        return name in self._load()

    def __iter__(self):
        return iter(self._load())

    def __len__(self) -> int:
        return len(self._load())


# ---------------------------------------------------------------------------
# Structure / SA2DConnection / StructureCollection
# ---------------------------------------------------------------------------


@dataclass
class Weir:
    """Overflow weir parameters read from ``Geometry/Structures/Attributes``.

    Present on :class:`Bridge`, :class:`Inline`, and :class:`Lateral`
    structures when *mode* is ``'Weir/Gate/Culverts'``.

    Attributes
    ----------
    width:
        Weir crest length perpendicular to flow (HDF ``Weir Width``).
    coefficient:
        Discharge coefficient (HDF ``Weir Coef``).
    shape:
        Crest shape: ``'Broad Crested'``, ``'Ogee'``, etc.
        (HDF ``Weir Shape``).
    max_submergence:
        Maximum submergence ratio above which flow is fully submerged
        (HDF ``Weir Max Submergence``).
    min_elevation:
        Minimum crest elevation; ``nan`` when not set
        (HDF ``Weir Min Elevation``).
    us_slope:
        Upstream face slope H:V (HDF ``Weir US Slope``).
    ds_slope:
        Downstream face slope H:V (HDF ``Weir DS Slope``).
    skew:
        Skew angle in degrees (HDF ``Weir Skew``).
    use_water_surface:
        When ``True`` the water-surface elevation is used as the weir
        reference head; when ``False`` the energy grade line is used
        (HDF ``Use WS for Weir Reference``).
    """

    width: float
    coefficient: float
    shape: str
    max_submergence: float
    min_elevation: float
    us_slope: float
    ds_slope: float
    skew: float
    use_water_surface: bool


@dataclass
class GateOpening:
    """One physical opening within a :class:`GateGroup`.

    Attributes
    ----------
    name:
        Opening label (HDF ``Name`` in ``Gate Groups/Openings/Attributes``).
    station:
        Lateral station of this opening along the structure centreline
        (HDF ``Station``).
    """

    name: str
    station: float


@dataclass
class GateGroup:
    """One gate group from ``Geometry/Structures/Gate Groups/Attributes``.

    A gate group defines a set of identical gate openings.  Each opening
    in the group shares the same geometry (width, height, invert, coefficients)
    but is placed at a different station along the structure.

    Attributes
    ----------
    name:
        Gate group label (e.g. ``'Gate #1'``).
    width:
        Gate opening width (ft or m).
    height:
        Gate opening height (ft or m).
    invert:
        Gate sill elevation.
    sluice_coefficient:
        Sluice gate discharge coefficient.
    radial_coefficient:
        Radial (Tainter) gate discharge coefficient.
    weir_coefficient:
        Overflow weir coefficient for the gate crest.
    spillway_shape:
        Crest shape used when gate overflows (e.g. ``'Broad Crested'``).
    openings:
        Individual openings in this group, one per physical gate bay.
    """

    name: str
    width: float
    height: float
    invert: float
    sluice_coefficient: float
    radial_coefficient: float
    weir_coefficient: float
    spillway_shape: str
    openings: list[GateOpening] = field(default_factory=list)


@dataclass
class Structure:
    """Base class for one HEC-RAS structure from ``Geometry/Structures/Attributes``.

    Attributes
    ----------
    mode:
        HDF ``Mode`` field (e.g. ``'Weir/Gate/Culverts'``).  Empty string
        when the field is blank.
    upstream_type:
        HDF ``US Type`` field: ``'XS'`` (1-D cross section), ``'SA'``
        (storage area), ``'2D'`` (2-D flow area), or ``'--'`` (unspecified /
        treated as storage area by HEC-RAS).
    downstream_type:
        HDF ``DS Type`` field — same vocabulary as *upstream_type*.
    centerline:
        x, y coordinates of the structure centreline.  Shape ``(n_pts, 2)``.
    """

    mode: str
    upstream_type: str
    downstream_type: str
    centerline: np.ndarray  # (n_pts, 2)


@dataclass
class Bridge(Structure):
    """Bridge structure embedded in a 1-D HEC-RAS reach.

    Both sides are always ``'XS'``.

    Attributes
    ----------
    location:
        ``(river, reach, rs)`` of this bridge in the 1-D geometry
        (HDF ``River`` / ``Reach`` / ``RS`` fields).
    upstream_node:
        ``(river, reach, rs)`` of the adjacent upstream cross section
        (HDF ``US River`` / ``US Reach`` / ``US RS``).
    downstream_node:
        ``(river, reach, rs)`` of the adjacent downstream cross section
        (HDF ``DS River`` / ``DS Reach`` / ``DS RS``).
    weir:
        Weir overflow parameters; ``None`` when *mode* is empty (pure bridge,
        no overflow weir modelled).
    gate_groups:
        Gate groups attached to this structure (empty list when none).
    """

    location: tuple[str, str, str] = ("", "", "")
    upstream_node: tuple[str, str, str] = ("", "", "")
    downstream_node: tuple[str, str, str] = ("", "", "")
    weir: Weir | None = None
    gate_groups: list[GateGroup] = field(default_factory=list)


@dataclass
class Inline(Structure):
    """Inline structure (e.g. inline weir/dam) embedded in a 1-D HEC-RAS reach.

    Both sides are always ``'XS'``.

    Attributes
    ----------
    location:
        ``(river, reach, rs)`` of this structure in the 1-D geometry.
    upstream_node:
        ``(river, reach, rs)`` of the adjacent upstream cross section.
    downstream_node:
        ``(river, reach, rs)`` of the adjacent downstream cross section.
    weir:
        Weir overflow parameters; ``None`` when *mode* is empty.
    gate_groups:
        Gate groups attached to this structure (empty list when none).
    """

    location: tuple[str, str, str] = ("", "", "")
    upstream_node: tuple[str, str, str] = ("", "", "")
    downstream_node: tuple[str, str, str] = ("", "", "")
    weir: Weir | None = None
    gate_groups: list[GateGroup] = field(default_factory=list)


@dataclass
class Lateral(Structure):
    """Lateral structure connecting a 1-D reach to a Storage Area or 2-D Flow Area.

    The upstream side is always ``'XS'`` (the 1-D reach).  The downstream
    side connects to a Storage Area (``'SA'``), a 2-D Flow Area (``'2D'``),
    or nothing when flow exits the system (empty *downstream_type*).

    Attributes
    ----------
    location:
        ``(river, reach, rs)`` of this structure in the 1-D geometry.
    upstream_node:
        ``(river, reach, rs)`` of the adjacent upstream cross section.
    downstream_node:
        Name of the connected Storage Area or 2-D Flow Area
        (HDF ``DS SA/2D``).  Empty string when flow exits the system.
    weir:
        Weir overflow parameters; ``None`` when *mode* is empty.
    gate_groups:
        Gate groups attached to this structure (empty list when none).
    """

    location: tuple[str, str, str] = ("", "", "")
    upstream_node: tuple[str, str, str] = ("", "", "")
    downstream_node: str = ""
    weir: Weir | None = None
    gate_groups: list[GateGroup] = field(default_factory=list)


@dataclass
class SA2DConnection(Structure):
    """Connection structure linking two Storage Areas or 2-D Flow Areas.

    Both sides are ``'SA'``, ``'2D'``, or ``'--'`` (treated as SA by
    HEC-RAS).  Common examples: dam breach connection, levee between two
    2-D domains, SA-to-SA link.

    Attributes
    ----------
    name:
        User-given connection name from the HDF ``Connection`` field
        (e.g. ``"Dam"``, ``"Lower Levee"``).
    upstream_node:
        Name of the upstream Storage Area or 2-D Flow Area
        (HDF ``US SA/2D``).
    downstream_node:
        Name of the downstream Storage Area or 2-D Flow Area
        (HDF ``DS SA/2D``).

    Notes
    -----
    Plan-result groups (see :class:`~raspy.hdf.SA2DConnectionResults`) may
    use a different naming convention: for 2D↔2D connections HEC-RAS prefixes
    the flow area name (e.g. geometry ``"Lower Levee"`` → plan result
    ``"BaldEagleCr Lower Levee"``).
    """

    name: str = ""
    upstream_node: str = ""
    downstream_node: str = ""


class StructureCollection:
    """Access all structures stored in ``Geometry/Structures/Attributes``.

    The collection is keyed by a string identifier:

    * :class:`SA2DConnection` — the HDF ``Connection`` field (user-given name).
    * :class:`Bridge`, :class:`Inline`, :class:`Lateral` — ``"River Reach RS"``
      built from the HDF ``River`` / ``Reach`` / ``RS`` fields.

    Use the typed filter properties (:attr:`connections`, :attr:`bridges`,
    :attr:`laterals`, :attr:`inlines`) to narrow by structure subclass.

    Parameters
    ----------
    hdf:
        Open ``h5py.File`` handle.
    """

    def __init__(self, hdf: "h5py.File") -> None:
        self._hdf = hdf
        self._items: dict[str, Structure] | None = None

    # ------------------------------------------------------------------
    # Internal loader
    # ------------------------------------------------------------------

    def _load_gate_groups(self) -> dict[int, list[GateGroup]]:
        """Return gate groups keyed by structure row index (0-based)."""
        gg_root = f"{_STRUCT_ROOT}/Gate Groups"
        if gg_root not in self._hdf:
            return {}

        gg_ds = self._hdf[f"{gg_root}/Attributes"]
        gg_arr = np.array(gg_ds)
        gg_fn = gg_ds.dtype.names

        def _gg(row, f: str) -> str:
            return _decode(row[f]) if f in gg_fn else ""

        def _ggf(row, f: str) -> float:
            return float(row[f]) if f in gg_fn else float("nan")

        # Build openings dict: (struct_id, gate_group_local_id) → [GateOpening]
        openings_map: dict[tuple[int, int], list[GateOpening]] = {}
        op_path = f"{gg_root}/Openings/Attributes"
        if op_path in self._hdf:
            op_ds = self._hdf[op_path]
            op_arr = np.array(op_ds)
            op_fn = op_ds.dtype.names
            for op_row in op_arr:
                sid = int(op_row["Structure ID"]) if "Structure ID" in op_fn else -1
                gid = int(op_row["Gate Group ID"]) if "Gate Group ID" in op_fn else 0
                name = _decode(op_row["Name"]) if "Name" in op_fn else ""
                station = float(op_row["Station"]) if "Station" in op_fn else float("nan")
                key = (sid, gid)
                openings_map.setdefault(key, []).append(GateOpening(name=name, station=station))

        # Build gate groups, tracking local index per structure
        gate_groups_map: dict[int, list[GateGroup]] = {}
        local_count: dict[int, int] = {}
        for gg_row in gg_arr:
            sid = int(gg_row["Structure ID"]) if "Structure ID" in gg_fn else -1
            local_id = local_count.get(sid, 0)
            local_count[sid] = local_id + 1
            gg = GateGroup(
                name=_gg(gg_row, "Name"),
                width=_ggf(gg_row, "Width"),
                height=_ggf(gg_row, "Height"),
                invert=_ggf(gg_row, "Invert"),
                sluice_coefficient=_ggf(gg_row, "Sluice Coef"),
                radial_coefficient=_ggf(gg_row, "Radial Coef"),
                weir_coefficient=_ggf(gg_row, "Weir Coef"),
                spillway_shape=_gg(gg_row, "Spillway Shape"),
                openings=openings_map.get((sid, local_id), []),
            )
            gate_groups_map.setdefault(sid, []).append(gg)

        return gate_groups_map

    def _load(self) -> dict[str, Structure]:
        if self._items is not None:
            return self._items

        if _STRUCT_ROOT not in self._hdf:
            self._items = {}
            return self._items

        root = self._hdf[_STRUCT_ROOT]
        attrs_ds = root["Attributes"]
        attrs = np.array(attrs_ds)
        cl_info = np.array(root["Centerline Info"])   # (n, 4): [start, count, ...]
        cl_pts = np.array(root["Centerline Points"])  # (total, 2)

        fn = attrs_ds.dtype.names  # available field names

        def _get(row, f: str) -> str:
            return _decode(row[f]) if f in fn else ""

        def _getf(row, f: str) -> float:
            return float(row[f]) if f in fn else float("nan")

        def _xs_node(r: str, rc: str, rstation: str) -> tuple[str, str, str]:
            return (r, rc, rstation)

        def _build_weir(row) -> Weir:
            return Weir(
                width=_getf(row, "Weir Width"),
                coefficient=_getf(row, "Weir Coef"),
                shape=_get(row, "Weir Shape"),
                max_submergence=_getf(row, "Weir Max Submergence"),
                min_elevation=_getf(row, "Weir Min Elevation"),
                us_slope=_getf(row, "Weir US Slope"),
                ds_slope=_getf(row, "Weir DS Slope"),
                skew=_getf(row, "Weir Skew"),
                use_water_surface=bool(int(row["Use WS for Weir Reference"]))
                if "Use WS for Weir Reference" in fn else False,
            )

        gate_groups_map = self._load_gate_groups()

        items: dict[str, Structure] = {}
        for i, row in enumerate(attrs):
            typ  = _get(row, "Type")
            mode = _get(row, "Mode")
            us_t = _get(row, "US Type")
            ds_t = _get(row, "DS Type")

            start_pt = int(cl_info[i, 0])
            n_pts    = int(cl_info[i, 1])
            centerline = cl_pts[start_pt : start_pt + n_pts].astype(float)

            base = dict(mode=mode, upstream_type=us_t, downstream_type=ds_t, centerline=centerline)

            if typ == "Connection":
                conn_name = _get(row, "Connection") or f"Connection_{i}"
                key = conn_name
                if key in items:
                    key = f"{key}_{i}"
                items[key] = SA2DConnection(
                    **base,
                    name=conn_name,
                    upstream_node=_get(row, "US SA/2D"),
                    downstream_node=_get(row, "DS SA/2D"),
                )
            else:
                river = _get(row, "River")
                reach = _get(row, "Reach")
                rs    = _get(row, "RS")
                key = f"{river} {reach} {rs}".strip() if (river or reach or rs) else f"{typ}_{i}"
                if key in items:
                    key = f"{key}_{i}"
                location = (river, reach, rs)
                us_xs = _xs_node(_get(row, "US River"), _get(row, "US Reach"), _get(row, "US RS"))
                weir = _build_weir(row) if mode else None
                gate_groups = gate_groups_map.get(i, [])
                if typ == "Bridge":
                    items[key] = Bridge(
                        **base,
                        location=location,
                        upstream_node=us_xs,
                        downstream_node=_xs_node(_get(row, "DS River"), _get(row, "DS Reach"), _get(row, "DS RS")),
                        weir=weir,
                        gate_groups=gate_groups,
                    )
                elif typ == "Inline":
                    items[key] = Inline(
                        **base,
                        location=location,
                        upstream_node=us_xs,
                        downstream_node=_xs_node(_get(row, "DS River"), _get(row, "DS Reach"), _get(row, "DS RS")),
                        weir=weir,
                        gate_groups=gate_groups,
                    )
                elif typ == "Lateral":
                    items[key] = Lateral(
                        **base,
                        location=location,
                        upstream_node=us_xs,
                        downstream_node=_get(row, "DS SA/2D"),
                        weir=weir,
                        gate_groups=gate_groups,
                    )
                else:
                    items[key] = Structure(**base)  # unknown type, store as base

        self._items = items
        return self._items

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        """Keys of all structures in the collection."""
        return list(self._load().keys())

    @property
    def connections(self) -> dict[str, SA2DConnection]:
        """All :class:`SA2DConnection` instances keyed by connection name."""
        return {k: v for k, v in self._load().items() if isinstance(v, SA2DConnection)}

    @property
    def bridges(self) -> dict[str, Bridge]:
        """All :class:`Bridge` instances keyed by ``"River Reach RS"``."""
        return {k: v for k, v in self._load().items() if isinstance(v, Bridge)}

    @property
    def laterals(self) -> dict[str, Lateral]:
        """All :class:`Lateral` instances keyed by ``"River Reach RS"``."""
        return {k: v for k, v in self._load().items() if isinstance(v, Lateral)}

    @property
    def inlines(self) -> dict[str, Inline]:
        """All :class:`Inline` instances keyed by ``"River Reach RS"``."""
        return {k: v for k, v in self._load().items() if isinstance(v, Inline)}

    @property
    def summary(self) -> pd.DataFrame:
        """One row per structure with basic attributes.

        Columns: ``key``, ``subclass``, ``mode``, ``upstream_type``,
        ``upstream_node``, ``downstream_type``, ``downstream_node``,
        ``n_centerline_points``.

        ``upstream_node`` / ``downstream_node`` are ``(river, reach, rs)``
        tuples for :class:`Bridge` and :class:`Inline` sides, and plain
        strings (area names) for :class:`SA2DConnection` and
        :class:`Lateral` downstream sides.
        """
        rows = []
        for key, s in self._load().items():
            rows.append({
                "key": key,
                "subclass": type(s).__name__,
                "mode": s.mode,
                "upstream_type": s.upstream_type,
                "upstream_node": getattr(s, "upstream_node", None),
                "downstream_type": s.downstream_type,
                "downstream_node": getattr(s, "downstream_node", None),
                "n_centerline_points": len(s.centerline),
            })
        return pd.DataFrame(rows)

    def __getitem__(self, key: str) -> Structure:
        items = self._load()
        if key not in items:
            raise KeyError(f"Structure {key!r} not found. Available: {self.names}")
        return items[key]

    def __contains__(self, key: str) -> bool:
        return key in self._load()

    def __iter__(self):
        return iter(self._load())

    def __len__(self) -> int:
        return len(self._load())


# ---------------------------------------------------------------------------
# GeometryHdf — public entry point
# ---------------------------------------------------------------------------


class GeometryHdf(_HdfFile):
    """Read HEC-RAS geometry HDF5 output files (``*.g*.hdf``).

    Parameters
    ----------
    filename:
        Path to the geometry HDF file.  The ``.hdf`` suffix is appended
        automatically if absent.

    Examples
    --------
    ::

        with GeometryHdf("MyModel.g01") as g:
            print(g.flow_areas.summary)
            centers = g.flow_areas["spillway"].cell_centers
    """

    def __init__(self, filename: str | Path) -> None:
        super().__init__(filename)
        self._flow_areas: FlowAreaCollection | None = None
        self._storage_areas: StorageAreaCollection | None = None
        self._boundary_condition_lines: BoundaryConditionCollection | None = None
        self._structures: StructureCollection | None = None

    # ------------------------------------------------------------------
    # Collections
    # ------------------------------------------------------------------

    @property
    def flow_areas(self) -> FlowAreaCollection:
        """Access 2-D flow areas stored in the geometry HDF."""
        if self._flow_areas is None:
            self._flow_areas = FlowAreaCollection(self._hdf)
        return self._flow_areas

    @property
    def storage_areas(self) -> StorageAreaCollection:
        """Access storage areas (reservoirs, ponds) stored in the geometry HDF."""
        if self._storage_areas is None:
            self._storage_areas = StorageAreaCollection(self._hdf)
        return self._storage_areas

    @property
    def boundary_condition_lines(self) -> BoundaryConditionCollection:
        """Access boundary condition lines stored in the geometry HDF."""
        if self._boundary_condition_lines is None:
            self._boundary_condition_lines = BoundaryConditionCollection(self._hdf)
        return self._boundary_condition_lines

    @property
    def structures(self) -> StructureCollection:
        """Access all structures (connections, bridges, laterals, inline weirs)."""
        if self._structures is None:
            self._structures = StructureCollection(self._hdf)
        return self._structures







