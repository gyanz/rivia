"""PlanHdf - read HEC-RAS plan HDF5 files (.p*.hdf).

Plan HDF files embed the same ``Geometry/`` group as geometry HDF files
*plus* ``Results/Unsteady/...`` time-series and summary output.

``PlanHdf`` inherits ``GeometryHdf`` so all geometry accessors are available.
``FlowAreaResults`` extends ``FlowArea`` with lazy time-series properties,
summary DataFrames, and computed depth / velocity methods.  Raster export
methods delegate to ``raspy.geo`` via a deferred import so this module is
fully usable without rasterio or scipy installed.

Derived from archive/ras_tools/r2d/ras_io.py and
archive/ras_tools/r2d/ras2d_cell_velocity.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from raspy.utils import log_call, timed

from ._base import _HdfFile
from ._geometry import (
    _SA_ROOT,
    Bridge,
    FlowArea,
    FlowAreaCollection,
    GeometryHdf,
    Inline,
    Lateral,
    SA2DConnection,
    StorageArea,
    StorageAreaCollection,
    Structure,
    StructureCollection,
    _decode,
)

if TYPE_CHECKING:
    import h5py
    import rasterio.io

logger = logging.getLogger("raspy.hdf")


# ---------------------------------------------------------------------------
# HDF path constants
# ---------------------------------------------------------------------------
_TS_ROOT = "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series"
_SUM_ROOT = "Results/Unsteady/Output/Output Blocks/Base Output/Summary Output"
_TS_2D = f"{_TS_ROOT}/2D Flow Areas"
_SUM_2D = f"{_SUM_ROOT}/2D Flow Areas"

_TS_SA = f"{_TS_ROOT}/Storage Areas"
_SUM_SA = f"{_SUM_ROOT}/Storage Areas"
_TS_SA_CONN = f"{_TS_ROOT}/SA 2D Area Conn"

_TIME_DS = f"{_TS_ROOT}/Time"
_TIME_STAMP_DS = f"{_TS_ROOT}/Time Date Stamp"

_DSS_ROOT = (
    "Results/Unsteady/Output/Output Blocks"
    "/DSS Hydrograph Output/Unsteady Time Series"
)
_DSS_INLINE = f"{_DSS_ROOT}/Inline Structures"
_DSS_LATERAL = f"{_DSS_ROOT}/Lateral Structures"
_DSS_BRIDGE = f"{_DSS_ROOT}/Bridge"
_DSS_TIME_STAMP_DS = f"{_DSS_ROOT}/Time Date Stamp"


# Timestamp format written by HEC-RAS (e.g. "03Jan2000 00:00:00")
_RAS_TS_FMT = "%d%b%Y %H:%M:%S"


# ---------------------------------------------------------------------------
# FlowAreaResults - extends FlowArea with plan results
# ---------------------------------------------------------------------------


class FlowAreaResults(FlowArea):
    """Geometry *and* time-series results for one named 2-D flow area.

    Inherits all geometry properties from :class:`FlowArea`.

    Time-series properties return raw ``h5py.Dataset`` objects so the caller
    controls how much data is loaded::

        area.water_surface[10]          # one timestep -> ndarray (n_cells,)
        area.water_surface[10:20]       # slice -> ndarray (10, n_cells)
        area.water_surface[:]           # all -> ndarray (n_t, n_cells)

    Parameters
    ----------
    geom_group:
        ``h5py.Group`` at ``Geometry/2D Flow Areas/<name>``.
    ts_group:
        ``h5py.Group`` at the time-series result path for this area.
    sum_group:
        ``h5py.Group`` at the summary result path for this area.
    name:
        Flow area name.
    n_cells:
        Number of real computational cells.
    """

    def __init__(
        self,
        geom_group: "h5py.Group",
        ts_group: "h5py.Group",
        sum_group: "h5py.Group",
        name: str,
        n_cells: int,
    ) -> None:
        super().__init__(geom_group, name, n_cells)
        self._ts = ts_group
        self._sum = sum_group

    # ------------------------------------------------------------------
    # Lazy time-series (h5py.Dataset - slice to control memory)
    # ------------------------------------------------------------------

    @property
    def water_surface(self) -> "h5py.Dataset":
        """Water-surface elevation time series.

        ``h5py.Dataset``, shape ``(n_timesteps, n_cells + n_ghost)``.
        HEC-RAS stores ghost cell WSE (boundary condition stages) in the
        trailing columns.  Slice with ``[:self.n_cells]`` for real cells only,
        or ``[:]`` for all including ghost cells.
        Slice to read: ``area.water_surface[10]``.
        """
        return self._ts["Water Surface"]

    @property
    def face_velocity(self) -> "h5py.Dataset":
        """Signed face-normal velocity time series.

        ``h5py.Dataset``, shape ``(n_timesteps, n_faces)``.
        """
        return self._ts["Face Velocity"]

    @property
    def face_flow(self) -> "h5py.Dataset | None":
        """Volumetric face-flow time series, or ``None`` if not output.

        ``h5py.Dataset``, shape ``(n_timesteps, n_faces)``.
        """
        return self._ts.get("Face Flow")

    @property
    def cell_velocity(self) -> "h5py.Dataset | None":
        """HEC-RAS cell-velocity *speed* scalar, or ``None`` if not output.

        ``h5py.Dataset``, shape ``(n_timesteps, n_cells)``.
        This is the optional output enabled in the HDF Write Parameters;
        see :meth:`cell_velocity_vectors` for the derived vector field.
        """
        return self._ts.get("Cell Velocity")

    # ------------------------------------------------------------------
    # Eager summary results (small arrays, loaded once per access)
    # ------------------------------------------------------------------

    def _load_summary(self, key: str, n: int | None = None) -> pd.DataFrame:
        """Load a ``(2, n_*)`` summary dataset as a tidy DataFrame.

        The HDF summary datasets have shape ``(2, n_elements)`` where
        ``[0, :]`` = maximum/minimum values and ``[1, :]`` = elapsed-time
        (in days) at which the extremum occurred.  HEC-RAS stores entries
        for ghost cells as well; pass *n* to clip to the first *n* entries.

        Returns a DataFrame with columns ``['value', 'time']`` and
        integer index corresponding to cell or face index.
        """
        raw = np.array(self._sum[key])  # shape (2, n_elements)
        if n is not None:
            raw = raw[:, :n]
        return pd.DataFrame(
            {"value": raw[0], "time": raw[1]},
        )

    @property
    def max_water_surface(self) -> pd.DataFrame:
        """Maximum WSE per cell.

        DataFrame with columns ``['value', 'time']``.
        ``value``: maximum water-surface elevation (model units).
        ``time``: elapsed simulation time (days) when max occurred.
        Index: 0-based cell index.  Real cells only (ghost rows excluded).
        """
        return self._load_summary("Maximum Water Surface", n=self.n_cells)

    @property
    def _max_water_surface(self) -> pd.DataFrame:
        """Maximum WSE including ghost cell rows.  Same layout as
        :attr:`max_water_surface` but shape ``(n_cells + n_ghost,)``.
        Required when indexing with raw ``face_cell_indexes`` values.
        """
        return self._load_summary("Maximum Water Surface")

    @property
    def min_water_surface(self) -> pd.DataFrame:
        """Minimum WSE per cell.  Same column layout as :attr:`max_water_surface`.
        Real cells only (ghost rows excluded).
        """
        return self._load_summary("Minimum Water Surface", n=self.n_cells)

    @property
    def _min_water_surface(self) -> pd.DataFrame:
        """Minimum WSE including ghost cell rows.  Same layout as
        :attr:`min_water_surface` but shape ``(n_cells + n_ghost,)``.
        Required when indexing with raw ``face_cell_indexes`` values.
        """
        return self._load_summary("Minimum Water Surface")

    @property
    def max_face_velocity(self) -> pd.DataFrame:
        """Maximum face velocity per face.

        DataFrame with columns ``['value', 'time']``.
        Index: 0-based face index.
        """
        return self._load_summary("Maximum Face Velocity", n=self.n_faces)

    # ------------------------------------------------------------------
    # Computed results - pure numpy, no geo dependency
    # ------------------------------------------------------------------

    def wse(self, timestep: int) -> np.ndarray:
        """Water-surface elevation at each real cell for one timestep.

        Parameters
        ----------
        timestep:
            0-based index into the time dimension.

        Returns
        -------
        ndarray, shape ``(n_cells,)``
            Water-surface elevation in model units.  Real cells only.
        """
        return np.array(self.water_surface[timestep, : self.n_cells])

    def _wse(self, timestep: int) -> np.ndarray:
        """Water-surface elevation including ghost cell rows for one timestep.

        Shape ``(n_cells + n_ghost,)``.  Required when indexing with raw
        ``face_cell_indexes`` values which contain ghost cell indices.
        """
        return np.array(self.water_surface[timestep, :])

    def depth(self, timestep: int) -> np.ndarray:
        """Water depth at each cell centre for one timestep.

        Depth = max(0, WSE - bed_elevation), where bed elevation is the
        cell minimum elevation from the geometry.

        Parameters
        ----------
        timestep:
            0-based index into the time dimension.

        Returns
        -------
        ndarray, shape ``(n_cells,)``
            Real cells only (ghost rows excluded).
        """
        wse = np.array(self.water_surface[timestep, : self.n_cells])
        return np.maximum(0.0, wse - self.cell_min_elevation)

    def max_depth(self) -> pd.DataFrame:
        """Maximum depth per cell using the time of maximum WSE.

        ``value = max(0, max_WSE - bed_elevation)``.
        ``time`` is the elapsed time of maximum WSE (days); this is an
        approximation - maximum depth may not coincide with maximum WSE.

        Returns
        -------
        DataFrame with columns ``['value', 'time']``, index = cell index.
        """
        df = self.max_water_surface
        df["value"] = np.maximum(0.0, df["value"].to_numpy() - self.cell_min_elevation)
        return df

    def cell_velocity_vectors(
        self,
        timestep: int,
        method: Literal[
            "area_weighted", "length_weighted", "flow_ratio"
        ] = "area_weighted",
        wse_interp: Literal["average", "sloped", "max"] = "average",
        face_velocity_location: Literal[
            "centroid", "normal_intercept"
        ] = "normal_intercept",
    ) -> np.ndarray:
        """Reconstruct cell-centre velocity vectors via weighted least-squares.

        Uses the WLS method prescribed by the HEC-RAS Technical Reference
        Manual (Section: 2D Unsteady Flow - Cell Velocity).

        Parameters
        ----------
        timestep:
            0-based index into the time dimension.
        method:
            ``"area_weighted"`` (default, matches HEC-RAS): weights are
            wetted face flow areas from the hydraulic property tables.
            ``"length_weighted"``: weights are face plan-view lengths.
            ``"flow_ratio"``: requires ``Face Flow`` output; back-calculates
            flow area as |Q|/|V_n|.
        wse_interp:
            How to estimate face WSE when ``method="area_weighted"``.
            ``"average"`` (default): simple mean of the two adjacent cell WSEs.
            ``"sloped"``: distance-weighted interpolation at the face's actual
            position - see *face_velocity_location*.
            ``"max"``: maximum of the two adjacent cell WSEs.
        face_velocity_location:
            Position used as the face normal velocity measurement point when
            ``wse_interp="sloped"``.
            ``"normal_intercept"`` (default): the point where the
            cell-centre connecting line crosses the face polyline, matching
            how HEC-RAS locates its finite-difference gradient.
            ``"centroid"``: the geometric centroid of the face polyline.
            Has no effect when ``wse_interp`` is ``"average"`` or ``"max"``.

        Returns
        -------
        ndarray, shape ``(n_cells, 2)``
            ``[Vx, Vy]`` depth-averaged velocity components for real cells.
        """
        from ._velocity import compute_all_cell_velocities

        if method == "flow_ratio" and self.face_flow is None:
            raise KeyError(
                "Face Flow is not present in this HDF file. "
                "Enable 'Face Flow' in HEC-RAS HDF5 Write Parameters "
                "before running the simulation, or use a different method."
            )

        face_normal_velocity = np.array(self.face_velocity[timestep, :])
        # Read all rows (real + ghost) so boundary face WSE benefits from
        # ghost-cell WSE (boundary condition stage).
        cell_wse = np.array(self.water_surface[timestep, :])
        face_flow = (
            np.array(self.face_flow[timestep, :]) if method == "flow_ratio" else None
        )

        cell_face_info, cell_face_values = self.cell_face_info
        face_ae_info, face_ae_values = self.face_area_elevation

        if wse_interp == "sloped":
            # Stack real + ghost cell coordinates so the sloped face-WSE
            # estimator can distance-weight boundary faces correctly.
            cell_centers = np.vstack([self.cell_centers, self.ghost_cell_centers])
            face_velocity_coords = (
                self.face_normal_intercept
                if face_velocity_location == "normal_intercept"
                else self.face_centroids
            )
        else:
            cell_centers = None
            face_velocity_coords = None

        return compute_all_cell_velocities(
            n_cells=self.n_cells,
            cell_face_info=cell_face_info,
            cell_face_values=cell_face_values,
            face_normals=self.face_normals,
            face_cell_indexes=self.face_cell_indexes,
            face_ae_info=face_ae_info,
            face_ae_values=face_ae_values,
            face_normal_velocity=face_normal_velocity,
            cell_wse=cell_wse,
            method=method,
            face_flow=face_flow,
            wse_interp=wse_interp,
            cell_centers=cell_centers,
            face_velocity_coords=face_velocity_coords,
        )

    def cell_speed(
        self,
        timestep: int,
        method: Literal[
            "area_weighted", "length_weighted", "flow_ratio"
        ] = "area_weighted",
        wse_interp: Literal["average", "sloped", "max"] = "average",
        face_velocity_location: Literal[
            "centroid", "normal_intercept"
        ] = "normal_intercept",
    ) -> np.ndarray:
        """Velocity magnitude at each cell centre for one timestep.

        Parameters
        ----------
        timestep:
            0-based index into the time dimension.
        method:
            Passed to :meth:`cell_velocity_vectors`.
        wse_interp:
            Passed to :meth:`cell_velocity_vectors`.
        face_velocity_location:
            Passed to :meth:`cell_velocity_vectors`.

        Returns
        -------
        ndarray, shape ``(n_cells,)``
        """
        vecs = self.cell_velocity_vectors(
            timestep, method=method, wse_interp=wse_interp,
            face_velocity_location=face_velocity_location,
        )
        return np.sqrt(vecs[:, 0] ** 2 + vecs[:, 1] ** 2)

    def cell_velocity_angle(
        self,
        timestep: int,
        method: Literal[
            "area_weighted", "length_weighted", "flow_ratio"
        ] = "area_weighted",
        wse_interp: Literal["average", "sloped", "max"] = "average",
        face_velocity_location: Literal[
            "centroid", "normal_intercept"
        ] = "normal_intercept",
    ) -> np.ndarray:
        """Flow direction at each cell centre for one timestep.

        Angle is measured in degrees clockwise from north (the conventional
        "flow-to" bearing used in hydraulics and GIS).

        Parameters
        ----------
        timestep:
            0-based index into the time dimension.
        method:
            Passed to :meth:`cell_velocity_vectors`.
        wse_interp:
            Passed to :meth:`cell_velocity_vectors`.
        face_velocity_location:
            Passed to :meth:`cell_velocity_vectors`.

        Returns
        -------
        ndarray, shape ``(n_cells,)``
            Direction the flow is heading in degrees clockwise from north
            (0 = north, 90 = east, 180 = south, 270 = west).
            Cells whose speed is below 1e-10 return ``nan``.
        """
        vecs = self.cell_velocity_vectors(
            timestep, method=method, wse_interp=wse_interp,
            face_velocity_location=face_velocity_location,
        )
        vx = vecs[:, 0]
        vy = vecs[:, 1]
        speed = np.sqrt(vx**2 + vy**2)
        angle = (90.0 - np.degrees(np.arctan2(vy, vx))) % 360.0
        angle[speed < 1e-10] = np.nan
        return angle

    def face_velocity_vectors(self, timestep: int) -> np.ndarray:
        """Full 2D velocity ``[Vx, Vy]`` at each face midpoint.

        Implements the RASMapper-exact C-stencil least-squares reconstruction
        (Step A + Step 2 from ``geo/_rasmap.py``):

        * **Step A** — hydraulic connectivity: determines which faces are
          actively conveying flow based on adjacent-cell WSE and bed elevation.
        * **Step 2** — for each face, a 3-face C-stencil (the face itself plus
          its clockwise and counter-clockwise neighbors within each adjacent
          cell) solves a 2×2 WLS system to recover the full ``(Vx, Vy)``
          vector from the stored face-normal scalar.  The result from each
          adjacent cell is averaged to give a single face vector.

        Replicates ``ReconstructFaceVelocitiesLeastSquares`` from
        ``archive/DLLs/RasMapperLib/MeshFV2D.cs``.

        Parameters
        ----------
        timestep:
            0-based index into the time dimension.

        Returns
        -------
        ndarray, shape ``(n_faces, 2)``
            ``[Vx, Vy]`` velocity at each face midpoint.
            Disconnected (dry) faces receive ``[0, 0]``.
        """
        from raspy.geo import _rasmap

        cell_wse = np.array(self.water_surface[timestep, :])
        face_normal_vel = np.array(self.face_velocity[timestep, :])
        cell_face_info, cell_face_values = self.cell_face_info
        _cell_face_count = cell_face_info[:, 1].astype(np.int32)

        _, _, face_hconn = _rasmap.compute_face_wss(
            cell_wse, self._cell_min_elevation, self.face_min_elevation,
            self.face_cell_indexes, _cell_face_count,
        )
        face_connected = (face_hconn >= _rasmap.HC_BACKFILL) & (face_hconn <= _rasmap.HC_DOWNHILL_SHALLOW)
        face_vel_A, _ = _rasmap.reconstruct_face_velocities(
            face_normal_vel, self.face_normals[:, :2],
            face_connected, self.face_cell_indexes,
            cell_face_info, cell_face_values,
        )
        return face_vel_A

    def facepoint_velocity_vectors(self, timestep: int) -> np.ndarray:
        """Full 2D velocity ``[Vx, Vy]`` at each mesh facepoint (corner).

        Implements the RASMapper-exact pipeline (Steps A + 2 + 3):

        * **Step A** — hydraulic connectivity
          (:func:`~raspy.geo._rasmap.compute_face_wss`).
        * **Step 2** — C-stencil face velocity reconstruction
          (:func:`~raspy.geo._rasmap.reconstruct_face_velocities`).
        * **Step 3** — inverse-face-length weighted facepoint averaging
          (:func:`~raspy.geo._rasmap.compute_facepoint_velocities`).
          Each facepoint has one arc-context velocity vector per adjacent
          face; these are averaged to produce a single ``[Vx, Vy]`` per
          facepoint.

        Replicates ``ComputeVertexVelocities`` from
        ``archive/DLLs/RasMapperLib/MeshFV2D.cs``.

        Parameters
        ----------
        timestep:
            0-based index into the time dimension.

        Returns
        -------
        ndarray, shape ``(n_facepoints, 2)``
            ``[Vx, Vy]`` velocity at each mesh corner.
            Facepoints adjacent only to dry faces receive ``[0, 0]``.
        """
        from raspy.geo import _rasmap

        cell_wse = np.array(self.water_surface[timestep, :])
        face_normal_vel = np.array(self.face_velocity[timestep, :])
        cell_face_info, cell_face_values = self.cell_face_info
        _cell_face_count = cell_face_info[:, 1].astype(np.int32)

        face_value_a, face_value_b, face_hconn = _rasmap.compute_face_wss(
            cell_wse, self._cell_min_elevation, self.face_min_elevation,
            self.face_cell_indexes, _cell_face_count,
        )
        face_connected = (face_hconn >= _rasmap.HC_BACKFILL) & (face_hconn <= _rasmap.HC_DOWNHILL_SHALLOW)
        face_vel_A, face_vel_B = _rasmap.reconstruct_face_velocities(
            face_normal_vel, self.face_normals[:, :2],
            face_connected, self.face_cell_indexes,
            cell_face_info, cell_face_values,
        )
        fp_face_info, fp_face_values = self.facepoint_face_orientation
        fp_velocities, _ = _rasmap.compute_facepoint_velocities(
            face_vel_A, face_vel_B, face_connected,
            self.face_lengths,
            self.face_facepoint_indexes, self.face_cell_indexes,
            cell_wse, fp_face_info, fp_face_values,
            face_value_a, face_value_b,
        )
        # Collapse one vector per adjacent-face context → single vector per facepoint.
        n_fp = len(fp_velocities)
        result = np.zeros((n_fp, 2), dtype=np.float64)
        for fp in range(n_fp):
            vecs = fp_velocities[fp]
            if len(vecs):
                result[fp] = vecs.mean(axis=0)
        return result

    def velocity_plot(
        self,
        timestep: int,
        cell_index: int,
        *,
        render_mode: Literal["horizontal", "sloping", "hybrid"] = "sloping",
        buffer: int = 1,
        reference_raster: str | Path | None = None,
        use_depth_weights: bool = False,
        shallow_to_flat: bool = False,
        pixel_size: float | None = None,
        n_arrows: int = 200,
        ax: Any | None = None,
    ) -> Any:
        """Quiver plot of rasterized velocity vectors around a target cell.

        Rasterizes the neighbourhood of *cell_index* using the RASMapper-exact
        pipeline (``rasmap_raster`` with ``variable="velocity"``), then draws
        quiver arrows at the pixel centres of wet pixels.  Arrows show the
        final, fully interpolated ``[Vx, Vy]`` value at each pixel — the same
        values that appear in RASMapper's velocity map.

        Mesh polygon outlines and cell index labels are drawn as context.

        Requires ``matplotlib`` (``pip install matplotlib``).

        Parameters
        ----------
        timestep:
            0-based time index.
        cell_index:
            Target cell.  The neighbourhood is expanded by BFS from this cell.
        render_mode:
            ``"horizontal"``, ``"sloping"`` (default), or ``"hybrid"`` —
            passed to :meth:`export_raster`.
        buffer:
            Number of face-adjacency hops to expand from *cell_index*.
            ``1`` = immediate neighbours; ``2`` = two rings out.
        reference_raster:
            Optional path to a terrain DEM GeoTIFF.  When supplied, the
            pixel size and CRS are inherited from the DEM and
            ``use_depth_weights=True`` becomes available.  When ``None``
            (default), the pixel size is auto-derived from the median face
            length in the neighbourhood.
        use_depth_weights:
            Passed to :meth:`export_raster`.  ``hybrid`` mode only.
            Requires *reference_raster*.
        shallow_to_flat:
            Passed to :meth:`export_raster`.  ``hybrid`` mode only.
        pixel_size:
            Raster pixel size in model coordinate units.  Smaller values
            produce a finer grid and more potential arrow positions.
            Ignored when *reference_raster* is supplied (the DEM pixel size
            is used instead).  Defaults to ``local_cell_size / 3``, giving
            ~9 pixels per face-length unit so *n_arrows* has room to work.
        n_arrows:
            Target number of quiver arrows.  Wet pixels are subsampled with
            stride ``ceil(sqrt(n_wet / n_arrows))`` so the actual count is
            approximately *n_arrows*.  Increase for denser plots (e.g.
            ``n_arrows=800``); set to a very large number to show every
            wet pixel.  Default ``200``.
        ax:
            Existing ``matplotlib.Axes`` to draw on.  If ``None`` a new
            figure/axes is created.

        Returns
        -------
        matplotlib.axes.Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for velocity_plot(). "
                "Install it with:  pip install matplotlib"
            ) from exc

        from raspy.geo import raster as _raster

        # -- 1. BFS neighbourhood (with ring tracking) -------------------
        cell_face_info, cell_face_values = self.cell_face_info
        face_cell_indexes = self.face_cell_indexes
        n_cells = self.n_cells

        # ring_of[c] = BFS hop distance from cell_index (0 = focus cell)
        ring_of: dict[int, int] = {cell_index: 0}
        neighbors: set[int] = {cell_index}
        frontier: set[int] = {cell_index}
        for hop in range(buffer):
            next_frontier: set[int] = set()
            for c in frontier:
                start = int(cell_face_info[c, 0])
                count = int(cell_face_info[c, 1])
                for k in range(count):
                    fi = int(cell_face_values[start + k, 0])
                    for nb in (
                        int(face_cell_indexes[fi, 0]),
                        int(face_cell_indexes[fi, 1]),
                    ):
                        if 0 <= nb < n_cells and nb not in neighbors:
                            ring_of[nb] = hop + 1
                            next_frontier.add(nb)
            neighbors |= next_frontier
            frontier = next_frontier

        # -- 2. Bounding box and auto cell_size --------------------------
        cell_polys = self.cell_polygons
        cell_centers = self.cell_centers

        all_verts = np.vstack([
            cell_polys[c] for c in neighbors if len(cell_polys[c]) > 0
        ])
        x_min = float(all_verts[:, 0].min())
        x_max = float(all_verts[:, 0].max())
        y_min = float(all_verts[:, 1].min())
        y_max = float(all_verts[:, 1].max())

        # Local cell size: median face length in the neighbourhood.
        # Always computed — used for arrow sizing and (when no reference_raster)
        # as the output pixel size.
        nb_face_idx: list[int] = []
        for c in neighbors:
            start = int(cell_face_info[c, 0])
            count = int(cell_face_info[c, 1])
            for k in range(count):
                nb_face_idx.append(int(cell_face_values[start + k, 0]))
        face_lengths = self.face_normals[:, 2]
        local_cell_size = float(np.median(face_lengths[sorted(set(nb_face_idx))]))

        if reference_raster is None:
            # pixel_size overrides the auto value; default is local_cell_size / 3
            # so each cell contains ~9 pixels and n_arrows has room to work.
            _cell_size: float | None = (
                pixel_size if pixel_size is not None else local_cell_size / 3.0
            )
        else:
            _cell_size = None  # reference_raster provides the pixel grid
        _margin = local_cell_size

        # Rectangular perimeter with a small margin
        bbox_perim = np.array([
            [x_min - _margin, y_min - _margin],
            [x_max + _margin, y_min - _margin],
            [x_max + _margin, y_max + _margin],
            [x_min - _margin, y_max + _margin],
        ])

        # -- 3. Rasterize velocity via full rasmap pipeline ---------------
        # reference_raster enables pixel-level dry masking (WSE < terrain →
        # nodata) in addition to the coarser cell-level wet check.
        # "velocity_vector" → 4-band [Vx, Vy, speed, direction]; quiver
        # needs all four bands.
        fp_face_info, fp_face_values = self.facepoint_face_orientation
        ds = _raster.rasmap_raster(
            variable="velocity_vector",
            cell_wse=np.array(self.water_surface[timestep, :]),
            cell_min_elevation=self._cell_min_elevation,
            face_min_elevation=self.face_min_elevation,
            face_cell_indexes=face_cell_indexes,
            cell_face_info=cell_face_info,
            cell_face_values=cell_face_values,
            face_facepoint_indexes=self.face_facepoint_indexes,
            fp_coords=self.facepoint_coordinates,
            face_normals=self.face_normals,
            fp_face_info=fp_face_info,
            fp_face_values=fp_face_values,
            cell_polygons=cell_polys,
            face_normal_velocity=np.array(self.face_velocity[timestep, :]),
            output_path=None,
            cell_centers=cell_centers,
            cell_surface_area=self.cell_surface_area,
            reference_raster=reference_raster,
            cell_size=_cell_size,
            render_mode=render_mode,
            use_depth_weights=use_depth_weights,
            shallow_to_flat=shallow_to_flat,
            tight_extent=False,
            perimeter=bbox_perim,
        )

        # -- 4. Read bands and pixel coordinates -------------------------
        from matplotlib.colors import Normalize
        from matplotlib.path import Path as _MPath

        nodata_val = float(ds.nodata)
        vx_grid = ds.read(1).astype(np.float64)
        vy_grid = ds.read(2).astype(np.float64)
        speed_grid = ds.read(3).astype(np.float64)
        transform = ds.transform
        ds.close()

        height, width = vx_grid.shape
        rows_idx, cols_idx = np.mgrid[0:height, 0:width]
        qx_grid = transform.c + (cols_idx + 0.5) * transform.a
        qy_grid = transform.f + (rows_idx + 0.5) * transform.e  # e < 0

        wet = (speed_grid > 0) & (speed_grid != nodata_val)

        # Assign each pixel to the innermost neighbourhood ring it belongs to.
        # Pixels outside all neighbourhood cells are excluded from quiver.
        pts = np.column_stack([qx_grid.ravel(), qy_grid.ravel()])
        pixel_ring = np.full(height * width, buffer + 1, dtype=np.int32)
        for c in neighbors:
            poly = cell_polys[c]
            if len(poly) < 3:
                continue
            ring = ring_of.get(c, buffer)
            inside = _MPath(poly).contains_points(pts)
            # Lower ring number = closer to focus cell → higher priority
            pixel_ring = np.where(inside & (ring < pixel_ring), ring, pixel_ring)
        pixel_ring = pixel_ring.reshape(height, width)

        wet = wet & (pixel_ring <= buffer)

        # -- 5. Arrow scaling and subsampling ----------------------------
        # Arrow length is mapped linearly from [sp_min, sp_max] speed to
        # [0.30, 0.85] × local_cell_size so every arrow is visible.
        # Outer rings are scaled down slightly to emphasise the focus area.
        wet_r, wet_c = np.where(wet)

        if len(wet_r) == 0:
            # No wet pixels in neighbourhood — draw polygons only
            if ax is None:
                _, ax = plt.subplots()
        else:
            speeds = speed_grid[wet_r, wet_c]
            rings = pixel_ring[wet_r, wet_c]
            sp_min, sp_max = float(speeds.min()), float(speeds.max())

            arrow_min = local_cell_size * 0.30
            arrow_max = local_cell_size * 0.85
            if sp_max > sp_min + 1e-12:
                t = (speeds - sp_min) / (sp_max - sp_min)
                arrow_len = arrow_min + t * (arrow_max - arrow_min)
            else:
                arrow_len = np.full(len(speeds), arrow_max)

            # Outer rings get 15 % shorter arrows per hop from the focus cell
            ring_weight = np.maximum(0.55, 1.0 - rings * 0.15)
            arrow_len *= ring_weight

            # Unit direction × scaled length (data-coordinate arrows)
            eps = 1e-12
            u_norm = np.where(speeds > eps, vx_grid[wet_r, wet_c] / speeds, 0.0)
            v_norm = np.where(speeds > eps, vy_grid[wet_r, wet_c] / speeds, 0.0)
            u_draw = u_norm * arrow_len
            v_draw = v_norm * arrow_len

            # Subsample: target ~200 arrows (subsampled from neighbourhood only)
            stride = max(1, int(np.ceil(np.sqrt(len(wet_r) / n_arrows))))
            if stride > 1:
                sel = np.arange(0, len(wet_r), stride)
                wet_r, wet_c = wet_r[sel], wet_c[sel]
                u_draw, v_draw, speeds = u_draw[sel], v_draw[sel], speeds[sel]

            norm = Normalize(vmin=sp_min, vmax=sp_max)

            # -- 6. Draw -------------------------------------------------
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.get_figure()

            for c in neighbors:
                poly = cell_polys[c]
                if len(poly) == 0:
                    continue
                is_target = c == cell_index
                ring_x = np.append(poly[:, 0], poly[0, 0])
                ring_y = np.append(poly[:, 1], poly[0, 1])
                fc = "steelblue" if is_target else "lightgray"
                lw = 1.8 if is_target else 0.8
                ax.fill(poly[:, 0], poly[:, 1], fc=fc, alpha=0.18, ec="none")
                ax.plot(ring_x, ring_y, color="dimgray", lw=lw)
                cx, cy = float(cell_centers[c, 0]), float(cell_centers[c, 1])
                ax.text(cx, cy, str(c), ha="center", va="center",
                        fontsize=7, color="black", clip_on=True)

            Q = ax.quiver(
                qx_grid[wet_r, wet_c], qy_grid[wet_r, wet_c],
                u_draw, v_draw, speeds,
                cmap="plasma", norm=norm,
                scale=1.0, scale_units="xy", angles="xy",
                pivot="middle",
            )
            fig.colorbar(Q, ax=ax, label="Speed")

            ax.set_aspect("equal")
            ax.set_title(
                f"{self.name}  t={timestep}  cell={cell_index}  "
                f"mode={render_mode}  buf={buffer}"
            )
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            return ax

        # Fallback: no wet pixels — still draw polygons
        if ax is None:
            _, ax = plt.subplots()
        for c in neighbors:
            poly = cell_polys[c]
            if len(poly) == 0:
                continue
            is_target = c == cell_index
            ring_x = np.append(poly[:, 0], poly[0, 0])
            ring_y = np.append(poly[:, 1], poly[0, 1])
            fc = "steelblue" if is_target else "lightgray"
            lw = 1.8 if is_target else 0.8
            ax.fill(poly[:, 0], poly[:, 1], fc=fc, alpha=0.18, ec="none")
            ax.plot(ring_x, ring_y, color="dimgray", lw=lw)
            cx, cy = float(cell_centers[c, 0]), float(cell_centers[c, 1])
            ax.text(cx, cy, str(c), ha="center", va="center",
                    fontsize=7, color="black", clip_on=True)
        ax.set_aspect("equal")
        ax.set_title(
            f"{self.name}  t={timestep}  cell={cell_index}  "
            f"mode={render_mode}  buf={buffer}  [dry]"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        return ax

    # ------------------------------------------------------------------
    # Derived results helpers (computed from HDF result arrays)
    # ------------------------------------------------------------------

    def water_surface_at_facepoints(self, timestep: int) -> np.ndarray:
        """WSE interpolated to facepoints for one timestep.

        Convenience wrapper around :meth:`~FlowArea.wse_at_facepoints` that
        reads the water-surface time series internally.

        Parameters
        ----------
        timestep:
            0-based index into the time dimension.

        Returns
        -------
        ndarray, shape ``(n_facepoints,)``
            Facepoint WSE.  ``nan`` where all adjacent cells are dry.
        """
        # Include ghost-cell WSE so perimeter facepoints receive a value
        # from the adjacent boundary cell rather than returning NaN.
        cell_wse = np.array(self.water_surface[timestep, :])
        return self.wse_at_facepoints(cell_wse)

    def wet_cells(self, timestep: int, depth_min: float = 0.0) -> np.ndarray:
        """Boolean mask of wet cells for one timestep.

        A cell is wet when ``WSE - cell_min_elevation > depth_min``.

        Parameters
        ----------
        timestep:
            0-based index into the time dimension.
        depth_min:
            Minimum depth threshold in model units.  Default ``0.0``.

        Returns
        -------
        ndarray, shape ``(n_cells,)``, dtype bool
        """
        wse = np.array(self.water_surface[timestep, : self.n_cells])
        return (wse - self.cell_min_elevation) > depth_min

    def wet_faces(self, timestep: int, depth_min: float = 0.0) -> np.ndarray:
        """Boolean mask of wet faces for one timestep.

        A face is wet when at least one of its adjacent cells is wet
        (see :meth:`wet_cells`).  Boundary faces are wet when their single
        adjacent real cell is wet.

        Parameters
        ----------
        timestep:
            0-based index into the time dimension.
        depth_min:
            Minimum cell depth threshold passed to :meth:`wet_cells`.

        Returns
        -------
        ndarray, shape ``(n_faces,)``, dtype bool
        """
        wc = self.wet_cells(timestep, depth_min)
        fci = self.face_cell_indexes  # (n_faces, 2)
        n = self.n_cells
        c0_wet = np.where(fci[:, 0] >= 0, wc[np.clip(fci[:, 0], 0, n - 1)], False)
        c1_wet = np.where(fci[:, 1] >= 0, wc[np.clip(fci[:, 1], 0, n - 1)], False)
        return c0_wet | c1_wet



    # ------------------------------------------------------------------
    # Raster export - delegates to raspy.geo (deferred import)
    # ------------------------------------------------------------------

    @log_call(logging.INFO)
    @timed()
    def export_raster(
        self,
        variable: Literal["wse", "water_surface", "depth", "velocity", "velocity_vector"],
        timestep: int | None = None,
        output_path: str | Path | None = None,
        *,
        reference_raster: str | Path | None = None,
        cell_size: float | None = None,
        crs: Any | None = None,
        nodata: float = -9999.0,
        render_mode: Literal["horizontal", "sloping", "hybrid"] = "sloping",
        use_depth_weights: bool = False,
        shallow_to_flat: bool = False,
        depth_threshold: float = 0.001,
        tight_extent: bool = True,
    ) -> Path | rasterio.io.DatasetReader:
        """Rasterize a result variable using the RASMapper-exact algorithm.

        Implements the pixel-perfect pipeline reverse-engineered from
        ``archive/DLLs/RasMapperLib/`` (decompiled C# source, HEC-RAS 6.6),
        validated against RASMapper VRT exports — median |diff| = 0.000000.

        Parameters
        ----------
        variable:
            ``"wse"`` / ``"water_surface"`` — water-surface elevation.
            ``"depth"``    — water depth (WSE minus terrain); requires
                             *reference_raster*.
            ``"velocity"``        — 1-band speed raster ``sqrt(Vx²+Vy²)``; requires
                                    an explicit *timestep*.
            ``"velocity_vector"`` — 4-band raster ``[Vx, Vy, speed, direction_deg]``;
                                    requires an explicit *timestep*.
        timestep:
            0-based time index.  Pass ``None`` to use the time of maximum
            water-surface elevation (``"wse"``/``"water_surface"`` and
            ``"depth"`` only; raises ``ValueError`` for ``"velocity"``).
        output_path:
            Destination ``.tif`` file path.  ``None`` returns an open
            in-memory ``rasterio.DatasetReader``; the caller must close it.
        reference_raster:
            Existing GeoTIFF whose transform and CRS are inherited.
            Also used as the terrain DEM for depth computation.
            **Required** when ``variable="depth"``.
            Mutually exclusive with *cell_size*.
        cell_size:
            Output pixel size in model coordinate units.  Used when no
            *reference_raster* is supplied; grid origin is derived from
            the flow-area perimeter bounding box.
            Mutually exclusive with *reference_raster*.
        crs:
            Output CRS.  Inherited from *reference_raster* when ``None``.
        nodata:
            Fill value for dry / out-of-domain pixels (default ``-9999``).
        render_mode:
            ``"sloping"`` (default) — RASMapper "Sloping Cell Corners";
            uses corner facepoints only.  RasMapperLib hardcodes
            ``shallow_to_flat=True`` for this mode; the user-supplied value
            is overridden.  Matches ``store_map(render_mode="sloping")``.
            ``"hybrid"`` — "Sloping Cell Corners + Face Centers";
            ``use_depth_weights`` and ``shallow_to_flat`` are honoured.
            Matches ``store_map(render_mode="hybrid")``.
            ``"horizontal"`` — flat per-cell value; facepoint interpolation
            is skipped.  Matches ``store_map(render_mode="horizontal")``.
        use_depth_weights:
            Weight face contributions by water depth.  **``hybrid`` only**;
            forced ``False`` for other modes.  Requires *reference_raster*.
        shallow_to_flat:
            Render cells with no hydraulically-connected faces flat.
            **``hybrid`` only** (user-configurable); forced ``True`` for
            ``"sloping"`` per RasMapperLib, ``False`` for ``"horizontal"``.
        depth_threshold:
            Minimum depth for a pixel to be considered wet (default
            ``0.001``).  Matches ``RASResults.MinWSPlotTolerance``.
        tight_extent:
            When ``True`` (default), pixels outside the flow-area boundary
            polygon are set to *nodata*.

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
            If ``variable="depth"`` and *reference_raster* is not provided.
            If ``variable="velocity"`` or ``variable="velocity_vector"`` and ``timestep=None``.
            If neither *reference_raster* nor *cell_size* is provided.
        """
        from raspy.geo import raster as _raster

        if variable in ("velocity", "velocity_vector") and timestep is None:
            raise ValueError(
                "timestep=None is not supported for velocity. "
                "Provide an explicit timestep index."
            )
        if reference_raster is None and cell_size is None:
            # Default: median face length (same heuristic as export_raster)
            cell_size = float(np.median(self.face_normals[:, 2]))

        # ---- Read HDF arrays ------------------------------------------------
        if timestep is None:
            cell_wse = self._max_water_surface["value"].to_numpy()
        else:
            cell_wse = self._wse(timestep)

        face_normal_velocity: np.ndarray | None = None
        if variable in ("velocity", "velocity_vector"):
            face_normal_velocity = np.array(self.face_velocity[timestep, :])

        cell_face_info, cell_face_values = self.cell_face_info
        fp_face_info, fp_face_values = self.facepoint_face_orientation

        # ---- Delegate to rasmap_raster -------------------------------------
        return _raster.rasmap_raster(
            variable=variable,
            cell_wse=cell_wse,
            cell_min_elevation=self._cell_min_elevation,
            face_min_elevation=self.face_min_elevation,
            face_cell_indexes=self.face_cell_indexes,
            cell_face_info=cell_face_info,
            cell_face_values=cell_face_values,
            face_facepoint_indexes=self.face_facepoint_indexes,
            fp_coords=self.facepoint_coordinates,
            face_normals=self.face_normals,
            fp_face_info=fp_face_info,
            fp_face_values=fp_face_values,
            cell_polygons=self.cell_polygons,
            face_normal_velocity=face_normal_velocity,
            output_path=output_path,
            cell_centers=self.cell_centers,
            cell_surface_area=self.cell_surface_area,
            reference_raster=reference_raster,
            cell_size=cell_size,
            crs=crs,
            nodata=nodata,
            render_mode=render_mode,
            use_depth_weights=use_depth_weights,
            shallow_to_flat=shallow_to_flat,
            depth_threshold=depth_threshold,
            tight_extent=tight_extent,
            perimeter=self.perimeter,
        )

    @log_call(logging.INFO)
    @timed()
    def export_hydraulic_rasters(
        self,
        timestep: int,
        reference_raster: str | Path,
        *,
        wse_path: str | Path | None = None,
        depth_path: str | Path | None = None,
        velocity_path: str | Path | None = None,
        nodata: float = -9999.0,
        render_mode: Literal["horizontal", "sloping", "hybrid"] = "sloping",
        use_depth_weights: bool = False,
        shallow_to_flat: bool = False,
        depth_threshold: float = 0.001,
        tight_extent: bool = True,
    ) -> dict[str, Path | "rasterio.io.DatasetReader"]:
        """Export water-surface elevation, depth, and velocity rasters in one call.

        Convenience wrapper that calls :meth:`export_raster` three times — once
        each for ``"water_surface"``, ``"depth"``, and ``"velocity"`` — sharing
        the same render settings.

        Parameters
        ----------
        timestep:
            0-based time index.  Required — all three outputs need a specific
            timestep (velocity has no max-value fallback).
        reference_raster:
            Path to the terrain DEM GeoTIFF.  Required — used to derive depth
            (WSE minus DEM) and to inherit the output CRS and transform.
        wse_path:
            Destination ``.tif`` for the water-surface elevation raster.
            ``None`` returns an open in-memory ``rasterio.DatasetReader``.
        depth_path:
            Destination ``.tif`` for the depth raster.
            ``None`` returns an open in-memory ``rasterio.DatasetReader``.
        velocity_path:
            Destination ``.tif`` for the velocity magnitude raster.
            ``None`` returns an open in-memory ``rasterio.DatasetReader``.
        nodata:
            Fill value for dry / out-of-domain pixels (default ``-9999``).
        render_mode:
            Passed to :meth:`export_raster` for all three variables.
        use_depth_weights:
            Passed to :meth:`export_raster`.  ``hybrid`` only.
        shallow_to_flat:
            Passed to :meth:`export_raster`.  ``hybrid`` only.
        depth_threshold:
            Minimum depth for a pixel to be considered wet (default ``0.001``).
        tight_extent:
            When ``True`` (default), pixels outside the flow-area boundary are
            set to *nodata*.

        Returns
        -------
        dict with keys ``"water_surface"``, ``"depth"``, ``"velocity"``.
        Each value is the written ``Path`` (when the corresponding ``*_path``
        argument is given) or an open in-memory ``rasterio.DatasetReader``
        (when ``None``).  The caller must close any in-memory datasets.
        """
        common = dict(
            timestep=timestep,
            reference_raster=reference_raster,
            nodata=nodata,
            render_mode=render_mode,
            use_depth_weights=use_depth_weights,
            shallow_to_flat=shallow_to_flat,
            depth_threshold=depth_threshold,
            tight_extent=tight_extent,
        )
        return {
            "water_surface": self.export_raster(
                "water_surface", output_path=wse_path, **common
            ),
            "depth": self.export_raster(
                "depth", output_path=depth_path, **common
            ),
            "velocity": self.export_raster(
                "velocity", output_path=velocity_path, **common
            ),
        }



# ---------------------------------------------------------------------------
# FlowAreaResultsCollection
# ---------------------------------------------------------------------------


class FlowAreaResultsCollection(FlowAreaCollection):
    """Collection of :class:`FlowAreaResults` objects backed by a plan HDF file.

    Overrides :class:`FlowAreaCollection` to return ``FlowAreaResults``
    instead of plain ``FlowArea`` instances.
    """

    def __getitem__(self, name: str) -> FlowAreaResults:
        if name not in self._cache:
            root = self._hdf["Geometry/2D Flow Areas"]
            if name not in root:
                raise KeyError(
                    f"2D flow area {name!r} not found. Available: {self.names}"
                )
            n_cells = self._get_real_cell_count(name)

            # Time-series group for this area
            ts_path = f"{_TS_2D}/{name}"
            if ts_path not in self._hdf:
                raise KeyError(
                    f"No time-series results found for flow area {name!r} "
                    f"at '{ts_path}'. Has the plan been computed?"
                )

            # Summary group (may be absent for steady-flow plans)
            sum_path = f"{_SUM_2D}/{name}"
            sum_group = self._hdf.get(sum_path)
            if sum_group is None:
                raise KeyError(
                    f"No summary results found for flow area {name!r} at '{sum_path}'."
                )

            self._cache[name] = FlowAreaResults(
                geom_group=root[name],
                ts_group=self._hdf[ts_path],
                sum_group=sum_group,
                name=name,
                n_cells=n_cells,
            )
        return self._cache[name]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# StorageAreaResults - extends StorageArea geometry with plan results
# ---------------------------------------------------------------------------


class StorageAreaResults(StorageArea):
    """Geometry *and* time-series results for one storage area.

    Inherits all geometry properties from :class:`~raspy.hdf.StorageArea`
    (:attr:`boundary`, :attr:`volume_elevation`, :meth:`volume_at_elevation`, etc.).

    Time-series properties return ``numpy`` arrays (storage areas are scalar
    entities so their result datasets are small and eager loading is appropriate).

    Parameters
    ----------
    sa:
        Parent geometry object whose fields are copied into this instance.
    sa_index:
        0-based column index of this SA in the flat ``(n_t, n_sa)`` datasets
        (``Water Surface``, ``Flow``) stored under ``Storage Areas/``.
    ts_sa_group:
        ``h5py.Group`` at ``-/Unsteady Time Series/Storage Areas``, or ``None``
        when the plan has no SA results.
    sum_sa_group:
        ``h5py.Group`` at ``-/Summary Output/Storage Areas``, or ``None``.
    """

    def __init__(
        self,
        sa: StorageArea,
        sa_index: int,
        ts_sa_group: "h5py.Group | None",
        sum_sa_group: "h5py.Group | None",
    ) -> None:
        super().__init__(
            name=sa.name,
            mode=sa.mode,
            boundary=sa.boundary,
            volume_elevation=sa.volume_elevation,
        )
        self._i = sa_index
        self._ts = ts_sa_group
        self._sum = sum_sa_group
        # per-SA subgroup: -/Storage Areas/<name>/
        self._sub = ts_sa_group.get(sa.name) if ts_sa_group else None
        self._cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_flat(self, key: str) -> np.ndarray:
        """Load column *i* from a flat ``(n_t, n_sa)`` dataset."""
        if key not in self._cache:
            if self._ts is None:
                raise KeyError(
                    f"No time-series results for storage area {self.name!r}. "
                    "Has the plan been computed?"
                )
            self._cache[key] = np.array(self._ts[key])[:, self._i]
        return self._cache[key]

    def _load_vars(self) -> np.ndarray:
        """Load and cache the ``(n_t, 6)`` Storage Area Variables array."""
        if "_vars" not in self._cache:
            if self._sub is None or "Storage Area Variables" not in self._sub:
                raise KeyError(
                    f"'Storage Area Variables' not found for storage area "
                    f"{self.name!r}."
                )
            self._cache["_vars"] = np.array(self._sub["Storage Area Variables"])
        return self._cache["_vars"]

    # ------------------------------------------------------------------
    # Flat time-series (one value per timestep)
    # ------------------------------------------------------------------

    @property
    def water_surface(self) -> np.ndarray:
        """Water-surface elevation time series.  Shape ``(n_t,)``."""
        return self._load_flat("Water Surface")

    @property
    def flow(self) -> np.ndarray:
        """Net inflow rate (positive = into SA).  Shape ``(n_t,)``."""
        return self._load_flat("Flow")

    # ------------------------------------------------------------------
    # Storage Area Variables columns (Stage, flows, area, volume)
    # ------------------------------------------------------------------

    @property
    def stage(self) -> np.ndarray:
        """Stage (WSE) from Storage Area Variables.  Shape ``(n_t,)``."""
        return self._load_vars()[:, 0]

    @property
    def net_inflow(self) -> np.ndarray:
        """Net inflow rate.  Shape ``(n_t,)``."""
        return self._load_vars()[:, 1]

    @property
    def total_inflow(self) -> np.ndarray:
        """Total inflow rate (sum of all inflow sources).  Shape ``(n_t,)``."""
        return self._load_vars()[:, 2]

    @property
    def total_outflow(self) -> np.ndarray:
        """Total outflow rate (sum of all outflow sinks).  Shape ``(n_t,)``."""
        return self._load_vars()[:, 3]

    @property
    def surface_area_ts(self) -> np.ndarray:
        """Water-surface area time series (model area units).  Shape ``(n_t,)``."""
        return self._load_vars()[:, 4]

    @property
    def volume_ts(self) -> np.ndarray:
        """Stored volume time series (model volume units).  Shape ``(n_t,)``."""
        return self._load_vars()[:, 5]

    # ------------------------------------------------------------------
    # Connection inflows
    # ------------------------------------------------------------------

    @property
    def connections(self) -> np.ndarray | None:
        """Inflow from each named connection.

        Shape ``(n_t, n_conns)``, or ``None`` when no connection data is stored.
        Column names are in :attr:`connection_names`.
        """
        if "_conns" not in self._cache:
            if self._sub is None:
                return None
            ds = self._sub.get("Connections to Storage Area")
            if ds is None:
                return None
            self._cache["_conns"] = np.array(ds)
        return self._cache["_conns"]

    @property
    def connection_names(self) -> list[str]:
        """Names of the inflow connection sources (from HDF ``Connections`` attribute).

        Falls back to index-based names if the attribute is absent.
        """
        if self._sub is None:
            return []
        ds = self._sub.get("Connections to Storage Area")
        if ds is None:
            return []
        attr = ds.attrs.get("Connections")
        if attr is None:
            n = ds.shape[1] if ds.ndim > 1 else 1
            return [f"connection_{i}" for i in range(n)]
        return [_decode(v) for v in attr]

    # ------------------------------------------------------------------
    # Summary results
    # ------------------------------------------------------------------

    def _load_summary(self, key: str) -> pd.DataFrame:
        """Load a ``(2, n_sa)`` summary dataset as a single-row DataFrame."""
        if self._sum is None:
            raise KeyError(
                f"No summary results for storage area {self.name!r}. "
                "Has the plan been computed?"
            )
        raw = np.array(self._sum[key])  # shape (2, n_sa)
        return pd.DataFrame({"value": [float(raw[0, self._i])],
                             "time":  [float(raw[1, self._i])]})

    @property
    def max_water_surface(self) -> pd.DataFrame:
        """Maximum WSE.

        DataFrame with columns ``['value', 'time']``.
        ``value``: maximum WSE in model units.
        ``time``: elapsed simulation time (days) when maximum occurred.
        """
        return self._load_summary("Maximum Water Surface")

    @property
    def min_water_surface(self) -> pd.DataFrame:
        """Minimum WSE.  Same column layout as :attr:`max_water_surface`."""
        return self._load_summary("Minimum Water Surface")


# ---------------------------------------------------------------------------
# StorageAreaResultsCollection
# ---------------------------------------------------------------------------


class StorageAreaResultsCollection(StorageAreaCollection):
    """Collection of :class:`StorageAreaResults` backed by a plan HDF file.

    Overrides :class:`~raspy.hdf.StorageAreaCollection` to return
    ``StorageAreaResults`` with both geometry *and* plan results.
    """

    def _load(self) -> dict[str, StorageAreaResults]:  # type: ignore[override]
        if self._items is not None:
            return self._items  # type: ignore[return-value]

        if _SA_ROOT not in self._hdf:
            self._items = {}
            return self._items  # type: ignore[return-value]

        # Re-read geometry flat arrays (same logic as StorageAreaCollection._load)
        root = self._hdf[_SA_ROOT]
        attrs = np.array(root["Attributes"])
        poly_info = np.array(root["Polygon Info"])
        poly_pts = np.array(root["Polygon Points"])
        ve_info = np.array(root["Volume Elevation Info"])
        ve_vals = np.array(root["Volume Elevation Values"])

        ts_sa_group = self._hdf.get(_TS_SA)
        sum_sa_group = self._hdf.get(_SUM_SA)

        items: dict[str, StorageAreaResults] = {}
        for i, row in enumerate(attrs):
            name = _decode(row["Name"])
            mode = _decode(row["Mode"])

            start_pt = int(poly_info[i, 0])
            n_pts = int(poly_info[i, 1])
            boundary = poly_pts[start_pt : start_pt + n_pts].astype(float)

            ve_start = int(ve_info[i, 0])
            ve_count = int(ve_info[i, 1])
            vol_elev = (
                ve_vals[ve_start : ve_start + ve_count].astype(float)
                if ve_count > 0
                else np.empty((0, 2), dtype=float)
            )

            sa = StorageArea(
                name=name, mode=mode, boundary=boundary, volume_elevation=vol_elev
            )
            items[name] = StorageAreaResults(
                sa=sa,
                sa_index=i,
                ts_sa_group=ts_sa_group,
                sum_sa_group=sum_sa_group,
            )

        self._items = items  # type: ignore[assignment]
        return items

    def __getitem__(self, name: str) -> StorageAreaResults:  # type: ignore[override]
        items = self._load()
        if name not in items:
            raise KeyError(
                f"Storage area {name!r} not found. Available: {self.names}"
            )
        return items[name]


# ---------------------------------------------------------------------------
# _StructureResultsMixin - shared HDF result access for all structure types
# ---------------------------------------------------------------------------


class _StructureResultsMixin:
    """Mixin that adds ``Structure Variables`` HDF access to geometry dataclasses.

    All four structure result classes inherit this so ``variable_names``,
    ``total_flow``, ``stage_hw``, ``stage_tw``, ``weir_variables``, and
    ``gate_flow`` are implemented once and shared.

    Concrete subclasses must set ``self._g`` (the result ``h5py.Group``) and
    ``self._cache`` (empty ``dict``) in their ``__init__``.
    """

    if TYPE_CHECKING:
        _g: h5py.Group
        _cache: dict[str, np.ndarray]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, key: str) -> np.ndarray:
        if key not in self._cache:
            self._cache[key] = np.array(self._g[key])
        return self._cache[key]

    def _col_index(self, *candidates: str) -> int:
        """Index of first column whose name contains any candidate (case-insensitive).

        Candidates are tried in order; the first column whose lowercased name
        contains a lowercased candidate wins.  Raises ``KeyError`` if nothing
        matches.
        """
        names_lower = [n.lower() for n in self.variable_names]
        for cand in candidates:
            cand_l = cand.lower()
            for i, n in enumerate(names_lower):
                if cand_l in n:
                    return i
        raise KeyError(
            f"No column matching {candidates!r} in {self.variable_names!r}"
        )

    # ------------------------------------------------------------------
    # Structure Variables
    # ------------------------------------------------------------------

    @property
    def variable_names(self) -> list[str]:
        """Column names from the ``Structure Variables`` ``Variable_Unit`` attribute.

        Falls back to ``col_0``, ``col_1``, ... when the attribute is absent.
        """
        ds = self._g["Structure Variables"]
        attr = ds.attrs.get("Variable_Unit")
        if attr is None:
            attr = ds.attrs.get("Variables")
        if attr is not None:
            return [_decode(v[0]) for v in attr]
        return [f"col_{i}" for i in range(ds.shape[1])]

    @property
    def structure_variables(self) -> "h5py.Dataset":
        """All structure variables as a lazy ``h5py.Dataset``, shape ``(n_t, n_vars)``.

        Column names are in :attr:`variable_names`.
        """
        return self._g["Structure Variables"]

    @property
    def total_flow(self) -> np.ndarray:
        """Total flow through the structure.  Shape ``(n_t,)``."""
        return self._load("Structure Variables")[
            :, self._col_index("total flow", "flow")
        ]

    @property
    def stage_hw(self) -> np.ndarray:
        """Headwater stage (upstream side).  Shape ``(n_t,)``."""
        return self._load("Structure Variables")[
            :, self._col_index("stage hw", "hw", "headwater")
        ]

    @property
    def stage_tw(self) -> np.ndarray:
        """Tailwater stage (downstream side).  Shape ``(n_t,)``."""
        return self._load("Structure Variables")[
            :, self._col_index("stage tw", "tw", "tailwater")
        ]

    # ------------------------------------------------------------------
    # Optional datasets (shared by Inline, Lateral, Bridge, SA2DConnection)
    # ------------------------------------------------------------------

    @property
    def weir_variables(self) -> "h5py.Dataset | None":
        """Detailed weir hydraulics time series, or ``None`` if absent."""
        return self._g.get("Weir Variables")

    def gate_flow(self, gate_number: int) -> "h5py.Dataset":
        """Gate operation dataset for gate *gate_number* (1-based).

        Returns a lazy ``h5py.Dataset``, shape ``(n_t, n_vars)``.

        Raises
        ------
        KeyError
            If the gate number does not exist for this structure.
        """
        path = f"Gate Groups/Gate #{gate_number}"
        if path not in self._g:
            gates_grp = self._g.get("Gate Groups")
            available = list(gates_grp.keys()) if gates_grp is not None else []
            raise KeyError(
                f"Gate #{gate_number} not found. Available: {available}"
            )
        return self._g[path]


# ---------------------------------------------------------------------------
# SA2DConnectionResults - one connection between two hydraulic areas
# ---------------------------------------------------------------------------


class SA2DConnectionResults(_StructureResultsMixin, SA2DConnection):
    """Geometry *and* time-series results for one HEC-RAS SA/2D connection.

    Inherits geometry from :class:`~raspy.hdf.SA2DConnection` and shared HDF
    result access (``structure_variables``, ``total_flow``, ``stage_hw``,
    ``stage_tw``, ``weir_variables``, ``gate_flow``) from
    :class:`_StructureResultsMixin`.

    Parameters
    ----------
    geom:
        Geometry object from :class:`~raspy.hdf.StructureCollection`.
    group:
        ``h5py.Group`` at ``-/SA 2D Area Conn/<plan_name>``.
    """

    def __init__(self, geom: SA2DConnection, group: h5py.Group) -> None:
        SA2DConnection.__init__(
            self,
            mode=geom.mode,
            upstream_type=geom.upstream_type,
            downstream_type=geom.downstream_type,
            centerline=geom.centerline,
            name=geom.name,
            upstream_node=geom.upstream_node,
            downstream_node=geom.downstream_node,
        )
        # Plan result group name may differ from geometry name for 2D↔2D connections
        # (HEC-RAS prefixes the flow-area name, e.g. "Lower Levee" →
        #  "BaldEagleCr Lower Levee").
        self._plan_name: str = group.name.split("/")[-1]
        self._g = group
        self._cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # SA2D-specific result datasets
    # ------------------------------------------------------------------

    @property
    def weir_flow(self) -> np.ndarray:
        """Weir overflow component.  Shape ``(n_t,)``."""
        return self._load("Structure Variables")[
            :, self._col_index("weir flow")
        ]

    @property
    def breaching_variables(self) -> "h5py.Dataset | None":
        """Breach geometry and flow time series, or ``None`` if not breach-capable.

        Lazy ``h5py.Dataset``, shape ``(n_t, 10)``.
        Columns: Stage HW, Stage TW, Bottom Width, Bottom Elevation,
        Left Side Slope, Right Side Slope, Breach Flow, Breach Velocity,
        Breach Flow Area, Top Elevation.
        """
        return self._g.get("Breaching Variables")

    @property
    def headwater_cells(self) -> np.ndarray | None:
        """2-D mesh cell indices on the headwater side, or ``None`` if absent.

        Shape ``(n_faces,)``.
        """
        if "Headwater Cells" not in self._g:
            return None
        return self._load("Headwater Cells")

    @property
    def tailwater_cells(self) -> np.ndarray | None:
        """2-D mesh cell indices on the tailwater side, or ``None`` if absent.

        For 2D↔2D connections (levees) these are stored as a flat ``int32``
        dataset at the group root.  For SA↔2D connections (e.g. a dam with a
        storage-area headwater) they are stored as fixed-width byte strings in
        ``HW TW Segments/Tailwater Cells`` and are decoded here.

        Shape ``(n_cells,)``.
        """
        # 2D↔2D: flat int32 at group root
        if "Tailwater Cells" in self._g:
            return self._load("Tailwater Cells")
        # SA↔2D: string-encoded cell indices in HW TW Segments subgroup
        seg = self._g.get("HW TW Segments")
        if seg is not None and "Tailwater Cells" in seg:
            raw = seg["Tailwater Cells"][:]
            return np.array(
                [int(v.decode().strip()) for v in raw], dtype=np.int32
            )
        return None


# ---------------------------------------------------------------------------
# InlineResults - inline structure geometry + plan results
# ---------------------------------------------------------------------------


class InlineResults(_StructureResultsMixin, Inline):
    """Geometry *and* time-series results for one HEC-RAS inline structure.

    Inherits geometry from :class:`~raspy.hdf.Inline` and shared HDF result
    access from :class:`_StructureResultsMixin`.

    The HDF group is at
    ``Results/.../DSS Hydrograph Output/.../Inline Structures/<river reach rs>``.

    Parameters
    ----------
    geom:
        Geometry object from :class:`~raspy.hdf.StructureCollection`.
    group:
        ``h5py.Group`` at the inline structure result path.
    """

    def __init__(self, geom: Inline, group: h5py.Group) -> None:
        Inline.__init__(
            self,
            mode=geom.mode,
            upstream_type=geom.upstream_type,
            downstream_type=geom.downstream_type,
            centerline=geom.centerline,
            location=geom.location,
            upstream_node=geom.upstream_node,
            downstream_node=geom.downstream_node,
            weir=geom.weir,
            gate_groups=geom.gate_groups,
        )
        self._g = group
        self._cache: dict[str, np.ndarray] = {}


# ---------------------------------------------------------------------------
# LateralResults - lateral structure geometry + plan results
# ---------------------------------------------------------------------------


class LateralResults(_StructureResultsMixin, Lateral):
    """Geometry *and* time-series results for one HEC-RAS lateral structure.

    Inherits geometry from :class:`~raspy.hdf.Lateral` and shared HDF result
    access from :class:`_StructureResultsMixin`.

    The HDF group is at
    ``Results/.../DSS Hydrograph Output/.../Lateral Structures/<river reach rs>``.

    ``downstream_node`` is the name of the connected Storage Area or 2-D Flow
    Area, or an empty string when flow exits the system.

    Parameters
    ----------
    geom:
        Geometry object from :class:`~raspy.hdf.StructureCollection`.
    group:
        ``h5py.Group`` at the lateral structure result path.
    """

    def __init__(self, geom: Lateral, group: h5py.Group) -> None:
        Lateral.__init__(
            self,
            mode=geom.mode,
            upstream_type=geom.upstream_type,
            downstream_type=geom.downstream_type,
            centerline=geom.centerline,
            location=geom.location,
            upstream_node=geom.upstream_node,
            downstream_node=geom.downstream_node,
            weir=geom.weir,
            gate_groups=geom.gate_groups,
        )
        self._g = group
        self._cache: dict[str, np.ndarray] = {}


# ---------------------------------------------------------------------------
# BridgeResults - bridge geometry + plan results
# ---------------------------------------------------------------------------


class BridgeResults(_StructureResultsMixin, Bridge):
    """Geometry *and* time-series results for one HEC-RAS bridge structure.

    Inherits geometry from :class:`~raspy.hdf.Bridge` and shared HDF result
    access from :class:`_StructureResultsMixin`.

    The HDF group is at
    ``Results/.../DSS Hydrograph Output/.../Bridge/<river reach rs>``.

    Parameters
    ----------
    geom:
        Geometry object from :class:`~raspy.hdf.StructureCollection`.
    group:
        ``h5py.Group`` at the bridge result path.
    """

    def __init__(self, geom: Bridge, group: h5py.Group) -> None:
        Bridge.__init__(
            self,
            mode=geom.mode,
            upstream_type=geom.upstream_type,
            downstream_type=geom.downstream_type,
            centerline=geom.centerline,
            location=geom.location,
            upstream_node=geom.upstream_node,
            downstream_node=geom.downstream_node,
            weir=geom.weir,
            gate_groups=geom.gate_groups,
        )
        self._g = group
        self._cache: dict[str, np.ndarray] = {}


# ---------------------------------------------------------------------------
# PlanStructureCollection - plan-enriched StructureCollection
# ---------------------------------------------------------------------------


class PlanStructureCollection(StructureCollection):
    """Plan-enriched structure collection: all structure types with results.

    Overrides :class:`~raspy.hdf.StructureCollection` so each item carries
    both geometry attributes *and* time-series result access:

    * :class:`SA2DConnection` → :class:`SA2DConnectionResults`
      (Base Output ``SA 2D Area Conn``)
    * :class:`Inline` → :class:`InlineResults`
      (DSS Hydrograph Output ``Inline Structures``)
    * :class:`Lateral` → :class:`LateralResults`
      (DSS Hydrograph Output ``Lateral Structures``)
    * :class:`Bridge` → :class:`BridgeResults`
      (DSS Hydrograph Output ``Bridge``)

    When no plan result group is found for a structure (e.g. DSS output was
    not requested for that type), the plain geometry object is kept unchanged.
    """

    def _load(self) -> dict[str, Structure]:  # type: ignore[override]
        if self._items is not None:
            return self._items

        import h5py as _h5

        # Build geometry items first (parent caches in self._items).
        geom_items = StructureCollection._load(self)

        # Helper: collect sub-groups from an HDF path (returns {} when absent).
        def _groups(path: str) -> dict[str, h5py.Group]:
            root = self._hdf.get(path)
            if root is None:
                return {}
            return {k: v for k, v in root.items() if isinstance(v, _h5.Group)}

        conn_groups = _groups(_TS_SA_CONN)
        inline_groups = _groups(_DSS_INLINE)
        lateral_groups = _groups(_DSS_LATERAL)
        bridge_groups = _groups(_DSS_BRIDGE)

        items: dict[str, Structure] = {}
        for key, geom in geom_items.items():

            if isinstance(geom, SA2DConnection):
                # Derive plan result group name from geometry fields:
                #   2D↔2D (levee): "{upstream_2d_area} {connection}"
                #   SA↔2D / SA↔SA (one end is SA or '--'): Connection name
                if (
                    geom.upstream_type == "2D"
                    and geom.downstream_type == "2D"
                ):
                    plan_key = f"{geom.upstream_node} {geom.name}"
                else:
                    plan_key = geom.name
                grp = conn_groups.get(plan_key)
                items[key] = (
                    SA2DConnectionResults(geom, grp) if grp is not None else geom
                )

            elif isinstance(geom, Inline):
                plan_key = " ".join(geom.location)
                grp = inline_groups.get(plan_key)
                items[key] = InlineResults(geom, grp) if grp is not None else geom

            elif isinstance(geom, Lateral):
                plan_key = " ".join(geom.location)
                grp = lateral_groups.get(plan_key)
                items[key] = LateralResults(geom, grp) if grp is not None else geom

            elif isinstance(geom, Bridge):
                plan_key = " ".join(geom.location)
                grp = bridge_groups.get(plan_key)
                items[key] = BridgeResults(geom, grp) if grp is not None else geom

            else:
                items[key] = geom

        self._items = items
        return self._items


# ---------------------------------------------------------------------------
# PlanHdf - public entry point
# ---------------------------------------------------------------------------


class PlanHdf(GeometryHdf):
    """Read HEC-RAS plan HDF5 output files (``*.p*.hdf``).

    A plan HDF file contains the same ``Geometry/`` data as a geometry HDF
    file, *plus* ``Results/Unsteady/...`` time-series and summary output.

    Parameters
    ----------
    filename:
        Path to the plan HDF file.  The ``.hdf`` suffix is appended
        automatically if absent.

    Examples
    --------
    ::

        with PlanHdf("MyModel.p01") as hdf:
            ts   = hdf.time_stamps_map
            area = hdf.flow_areas["spillway"]

            wse   = area.water_surface[10]    # one timestep
            depth = area.depth(10)
            speed = area.cell_speed(10)
            max_d = area.max_depth()

            # requires rasterio + scipy:
            area.export_raster("depth", "depth.tif", timestep=None,
                               cell_size=5.0, crs="EPSG:26910")
            # or get an in-memory dataset:
            ds = area.export_raster("depth", timestep=None, cell_size=5.0)
            arr = ds.read(1)
            ds.close()
    """

    def __init__(self, filename: str | Path, program_directory: str | Path | None = None) -> None:
        super().__init__(filename)
        self._program_directory = Path(program_directory) if program_directory else None
        self._plan_flow_areas: FlowAreaResultsCollection | None = None
        self._plan_storage_areas: StorageAreaResultsCollection | None = None
        self._plan_structures: PlanStructureCollection | None = None

    # ------------------------------------------------------------------
    # File metadata
    # ------------------------------------------------------------------

    @property
    def ras_version(self) -> str:
        """HEC-RAS version string from the plan HDF root attribute.

        Returns the ``File Version`` root attribute, e.g.
        ``'HEC-RAS 6.6 September 2024'``.
        """
        raw = self._hdf.attrs["File Version"]
        return raw.decode() if isinstance(raw, (bytes, np.bytes_)) else str(raw)

    @staticmethod
    def _parse_ras_time_interval(raw: bytes | np.bytes_ | str) -> float:
        """Parse a HEC-RAS time-interval string to seconds.

        HEC-RAS concatenates the numeric value and unit without spaces,
        e.g. ``b'20SEC'``, ``b'5MIN'``, ``b'1HR'``, ``b'1HOUR'``,
        ``b'1DAY'``.  Returns the equivalent number of seconds.
        """
        if isinstance(raw, (bytes, np.bytes_)):
            text = raw.decode().strip()
        else:
            text = str(raw).strip()
        i = 0
        while i < len(text) and (text[i].isdigit() or text[i] in ".+-"):
            i += 1
        value, unit = float(text[:i]), text[i:].upper()
        _to_seconds = {"SEC": 1, "MIN": 60, "HR": 3600, "HOUR": 3600, "DAY": 86400}
        multiplier = next((v for k, v in _to_seconds.items() if unit.startswith(k)), 1)
        return value * multiplier

    def _plan_info_attr(self, name: str) -> bytes | np.bytes_ | str | None:
        """Return a ``Plan Data/Plan Information`` attribute by name, or ``None``."""
        grp = self._hdf.get("Plan Data/Plan Information")
        return None if grp is None else grp.attrs.get(name)

    @property
    def time_step_computation(self) -> float | None:
        """Computation time step in seconds, or ``None`` if absent.

        Read from ``Plan Data/Plan Information`` attribute
        ``Computation Time Step Base``.  HEC-RAS stores the value as a
        concatenated number-unit string, e.g. ``'20SEC'``, ``'5MIN'``,
        ``'1HR'``, ``'1HOUR'``, ``'1DAY'``.
        """
        raw = self._plan_info_attr("Computation Time Step Base")
        return None if raw is None else self._parse_ras_time_interval(raw)

    @property
    def time_step_map(self) -> float | None:
        """Mapping output interval in seconds, or ``None`` if absent.

        Read from ``Plan Data/Plan Information`` attribute
        ``Base Output Interval``.  HEC-RAS stores the value as a
        concatenated number-unit string, e.g. ``'5MIN'``.
        """
        raw = self._plan_info_attr("Base Output Interval")
        return None if raw is None else self._parse_ras_time_interval(raw)

    @property
    def projection(self) -> str | None:
        """WKT projection string stored in the plan HDF root, or ``None``.

        HEC-RAS writes the model CRS as a WKT string in the root attribute
        ``Projection``.  Returns ``None`` when the attribute is absent (older
        files or models without a defined projection).

        The raw WKT string can be converted to a ``pyproj.CRS`` or a
        ``rasterio.crs.CRS`` object if needed::

            import pyproj
            crs = pyproj.CRS.from_wkt(hdf.projection)
        """
        raw = self._hdf.attrs.get("Projection")
        if raw is None:
            return None
        return raw.decode() if isinstance(raw, (bytes, np.bytes_)) else str(raw)

    # ------------------------------------------------------------------
    # Time stamps
    # ------------------------------------------------------------------

    @property
    def time_stamps_map(self) -> pd.DatetimeIndex:
        """Simulation output time stamps as a ``pd.DatetimeIndex``.

        Parsed from the ``Time Date Stamp`` dataset written by HEC-RAS.
        Format: ``DD Mon YYYY HH:MM:SS`` (e.g. ``03Jan2000 00:00:00``).
        """
        ds = self._hdf.get(_TIME_STAMP_DS)
        if ds is None:
            raise KeyError(
                f"Time Date Stamp dataset not found at '{_TIME_STAMP_DS}'. "
                "Ensure this is an unsteady-flow plan HDF file."
            )
        raw = np.array(ds).astype(str)
        return pd.to_datetime(raw, format=_RAS_TS_FMT)

    @property
    def time_stamps_hydrograph(self) -> pd.DatetimeIndex:
        """DSS hydrograph output time stamps as a ``pd.DatetimeIndex``.

        Parsed from ``Results/.../DSS Hydrograph Output/Unsteady Time
        Series/Time Date Stamp``.  Format: ``DD Mon YYYY HH:MM:SS``.
        """
        ds = self._hdf.get(_DSS_TIME_STAMP_DS)
        if ds is None:
            raise KeyError(
                f"Time Date Stamp dataset not found at '{_DSS_TIME_STAMP_DS}'. "
                "Ensure DSS hydrograph output was written for this plan."
            )
        raw = np.array(ds).astype(str)
        return pd.to_datetime(raw, format=_RAS_TS_FMT)

    @property
    def n_timesteps_map(self) -> int | None:
        """Number of output time steps, or ``None`` for steady-flow plans."""
        ds = self._hdf.get(_TIME_STAMP_DS)
        if ds is None:
            return None
        return len(ds)

    # ------------------------------------------------------------------
    # Collections (override GeometryHdf equivalents with results-aware types)
    # ------------------------------------------------------------------

    @property
    def flow_areas(self) -> FlowAreaResultsCollection:
        """Access 2-D flow areas with both geometry and results data."""
        if self._plan_flow_areas is None:
            self._plan_flow_areas = FlowAreaResultsCollection(self._hdf)
        return self._plan_flow_areas

    @property
    def storage_areas(self) -> StorageAreaResultsCollection:
        """Access storage areas with both geometry and plan results data."""
        if self._plan_storage_areas is None:
            self._plan_storage_areas = StorageAreaResultsCollection(self._hdf)
        return self._plan_storage_areas

    @property
    def structures(self) -> PlanStructureCollection:
        """Access all structures with geometry *and* plan results.

        Returns a :class:`PlanStructureCollection` where each item is
        upgraded to the matching results class when plan output is present:

        * :class:`SA2DConnectionResults` — SA/2D connections
        * :class:`InlineResults` — inline structures
        * :class:`LateralResults` — lateral structures
        * :class:`BridgeResults` — bridge structures

        Use :attr:`~raspy.hdf.StructureCollection.connections`,
        :attr:`~raspy.hdf.StructureCollection.inlines`,
        :attr:`~raspy.hdf.StructureCollection.laterals`, and
        :attr:`~raspy.hdf.StructureCollection.bridges` for filtered access.
        """
        if self._plan_structures is None:
            self._plan_structures = PlanStructureCollection(self._hdf)
        return self._plan_structures

    @property
    def sa2d_connections(self) -> dict[str, SA2DConnection]:
        """SA/2D hydraulic connections keyed by geometry name.

        Convenience alias for ``hdf.structures.connections``.
        Items are :class:`SA2DConnectionResults` when plan output is present,
        plain :class:`SA2DConnection` otherwise.
        """
        return self.structures.connections
