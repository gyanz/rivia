"""GeometryHdf — read HEC-RAS geometry HDF5 files (.g*.hdf).

Provides structured access to 2-D flow-area mesh data:
cell centres, face connectivity, hydraulic property tables, etc.

Derived from archive/ras_tools/r2d/ras_io.py and
archive/ras_tools/r2d/ras2d_cell_velocity.py.
"""
from __future__ import annotations

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
        """
        return self._load("Faces NormalUnitVector and Length")

    @property
    def face_cell_indexes(self) -> np.ndarray:
        """Left and right cell index for each face.

        Shape ``(n_faces, 2)``.  A value of ``-1`` indicates a boundary
        face with no neighbour on that side.
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
        name_col = next(
            (c for c in df.columns if c.lower() == "name"), None
        )
        if name_col and name_col != "name":
            df = df.rename(columns={name_col: "name"})
        return df

    @property
    def names(self) -> list[str]:
        """Names of all 2-D flow areas in the file."""
        import h5py
        root = self._hdf[_GEOM_2D_ROOT]
        return [
            k for k, v in root.items()
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
                    f"2D flow area {name!r} not found. "
                    f"Available: {self.names}"
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
            name_idx = next(
                (i for i, f in enumerate(dtype_names) if "name" in f), None
            )
            count_idx = next(
                (
                    i for i, f in enumerate(dtype_names)
                    if "cell" in f and "count" in f
                ),
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

    # ------------------------------------------------------------------
    # Collections
    # ------------------------------------------------------------------

    @property
    def flow_areas(self) -> FlowAreaCollection:
        """Access 2-D flow areas stored in the geometry HDF."""
        if self._flow_areas is None:
            self._flow_areas = FlowAreaCollection(self._hdf)
        return self._flow_areas
