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
        return check_mesh_cells(
            cell_centers=self.cell_centers,
            facepoint_coordinates=self.facepoint_coordinates,
            face_facepoint_indexes=self.face_facepoint_indexes,
            cell_face_info=cfi,
            cell_face_values=cfv,
            boundary_polygon=self.perimeter if check_boundary else None,
            tol=tol,
        )


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







