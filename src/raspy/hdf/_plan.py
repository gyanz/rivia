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
    FlowArea,
    FlowAreaCollection,
    GeometryHdf,
    StorageArea,
    StorageAreaCollection,
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

    def face_velocity_vectors(
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
        """Full 2D velocity ``[Vx, Vy]`` at each face midpoint via the double-C stencil.

        HEC-RAS stores only the face-normal component.  The tangential
        component is estimated by projecting the WLS cell-centre velocity
        vectors of the two adjacent cells onto the face tangential direction
        and averaging (double-C stencil, HEC-RAS 2D Technical Reference
        Manual).

        Parameters
        ----------
        timestep:
            0-based index into the time dimension.
        method:
            WLS weight scheme passed to :meth:`cell_velocity_vectors`.
        wse_interp:
            Face WSE interpolation method passed to :meth:`cell_velocity_vectors`.
        face_velocity_location:
            Passed to :meth:`cell_velocity_vectors`.

        Returns
        -------
        ndarray, shape ``(n_faces, 2)``
            ``[Vx, Vy]`` velocity at each face midpoint.
            Faces where both adjacent cells are dry receive ``[0, 0]``.
        """
        from ._velocity import compute_all_face_velocities

        # cell_vel has shape (n_cells + n_ghost, 2); dry_mask matches.
        cell_vel = self.cell_velocity_vectors(
            timestep, method=method, wse_interp=wse_interp,
            face_velocity_location=face_velocity_location,
        )
        vel_mag = np.linalg.norm(cell_vel, axis=1)
        dry_mask = (vel_mag == 0.0) | ~np.isfinite(vel_mag)
        return compute_all_face_velocities(
            face_normals=self.face_normals,
            face_normal_velocity=np.array(self.face_velocity[timestep, :]),
            face_cell_indexes=self.face_cell_indexes,
            cell_velocity=cell_vel,
            dry_mask=dry_mask,
        )

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

    def _max_velocity(
        self,
        method: Literal[
            "area_weighted", "length_weighted", "flow_ratio"
        ] = "area_weighted",
        wse_interp: Literal["average", "sloped", "max"] = "average",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-cell maximum speed and corresponding velocity vector.

        Iterates over every timestep, computing WLS velocity vectors and
        tracking the element-wise maximum speed along with the vector at
        that peak-speed timestep.

        Returns
        -------
        max_speed : ndarray, shape ``(n_cells,)``
        max_vecs  : ndarray, shape ``(n_cells, 2)``
            ``[Vx, Vy]`` at the timestep of each cell's peak speed.
        """
        n_t = self.water_surface.shape[0]
        max_speed = np.full(self.n_cells, -np.inf)
        max_vecs = np.zeros((self.n_cells, 2))

        for t in range(n_t):
            vecs = self.cell_velocity_vectors(t, method=method, wse_interp=wse_interp)
            speed = np.sqrt(vecs[:, 0] ** 2 + vecs[:, 1] ** 2)
            faster = speed > max_speed
            max_speed[faster] = speed[faster]
            max_vecs[faster] = vecs[faster]

        return max_speed, max_vecs

    @log_call(logging.INFO)
    @timed(logging.INFO)
    def export_raster(
        self,
        variable: Literal["wse", "water_surface", "depth", "velocity"],
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
            ``"velocity"`` — velocity magnitude ``sqrt(Vx²+Vy²)``; requires
                             an explicit *timestep*.
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
            If ``variable="velocity"`` and ``timestep=None``.
            If neither *reference_raster* nor *cell_size* is provided.
        """
        from raspy.geo import raster as _raster

        if variable == "velocity" and timestep is None:
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
        if variable == "velocity":
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
    @timed(logging.INFO)
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
# SA2DConnectionResults - one connection between two hydraulic areas
# ---------------------------------------------------------------------------


class SA2DConnectionResults:
    """Time-series results for one HEC-RAS hydraulic connection.

    In HEC-RAS, the "SA/2D Area Conn" group holds connections between
    any two of: a Storage Area, a 2-D Flow Area, or another Storage Area.
    Common examples: dam, levee, inline weir, gate structure.

    Parameters
    ----------
    name:
        Name of the connection (group key in the HDF file).
    group:
        ``h5py.Group`` at ``-/SA 2D Area Conn/<name>``.
    """

    def __init__(self, name: str, group: "h5py.Group") -> None:
        self.name = name
        self._g = group
        self._cache: dict[str, np.ndarray] = {}

    def _load(self, key: str) -> np.ndarray:
        if key not in self._cache:
            self._cache[key] = np.array(self._g[key])
        return self._cache[key]

    # ------------------------------------------------------------------
    # Structure Variables (always present)
    # ------------------------------------------------------------------

    @property
    def variable_names(self) -> list[str]:
        """Column names from the ``Structure Variables`` HDF attribute.

        Typical columns: ``Total Flow``, ``Weir Flow``, ``Stage HW``,
        ``Stage TW`` [, ``Total Gate Flow``].  Falls back to
        ``col_0``, ``col_1``, - when the attribute is absent.
        """
        ds = self._g["Structure Variables"]
        attr = ds.attrs.get("Variable_Unit")
        if attr is None:
            attr = ds.attrs.get("Variables")
        if attr is not None:
            # attr shape is (n_vars, 2): column 0 = variable name, column 1 = unit
            return [_decode(v[0]) for v in attr]
        return [f"col_{i}" for i in range(ds.shape[1])]

    @property
    def structure_variables(self) -> "h5py.Dataset":
        """All structure variables.

        Lazy ``h5py.Dataset``, shape ``(n_t, n_vars)``.
        Slice to read: ``conn.structure_variables[:]``.
        Column names are in :attr:`variable_names`.
        """
        return self._g["Structure Variables"]

    # Convenience column accessors (always present for all connection types)

    @property
    def total_flow(self) -> np.ndarray:
        """Total flow through the connection.  Shape ``(n_t,)``."""
        return self._load("Structure Variables")[:, 0]

    @property
    def weir_flow(self) -> np.ndarray:
        """Weir flow component.  Shape ``(n_t,)``."""
        return self._load("Structure Variables")[:, 1]

    @property
    def stage_hw(self) -> np.ndarray:
        """Headwater stage (upstream side).  Shape ``(n_t,)``."""
        return self._load("Structure Variables")[:, 2]

    @property
    def stage_tw(self) -> np.ndarray:
        """Tailwater stage (downstream side).  Shape ``(n_t,)``."""
        return self._load("Structure Variables")[:, 3]

    # ------------------------------------------------------------------
    # Optional datasets
    # ------------------------------------------------------------------

    @property
    def breaching_variables(self) -> "h5py.Dataset | None":
        """Breach geometry and flow time series, or ``None`` if the connection
        is not breach-capable.

        Lazy ``h5py.Dataset``, shape ``(n_t, 10)``.
        Columns: Stage HW, Stage TW, Bottom Width, Bottom Elevation,
        Left Side Slope, Right Side Slope, Breach Flow, Breach Velocity,
        Breach Flow Area, Top Elevation.
        """
        return self._g.get("Breaching Variables")

    @property
    def weir_variables(self) -> "h5py.Dataset | None":
        """Detailed weir hydraulics time series, or ``None`` if absent.

        Lazy ``h5py.Dataset``, shape ``(n_t, 9)``.
        """
        return self._g.get("Weir Variables")

    def gate_flow(self, gate_number: int) -> "h5py.Dataset":
        """Gate operation dataset for the specified gate (1-based numbering).

        Returns a lazy ``h5py.Dataset``, shape ``(n_t, n_vars)``.

        Raises
        ------
        KeyError
            If the gate number does not exist for this connection.
        """
        path = f"Gate Groups/Gate #{gate_number}"
        if path not in self._g:
            gates_grp = self._g.get("Gate Groups")
            available = list(gates_grp.keys()) if gates_grp is not None else []
            raise KeyError(
                f"Gate #{gate_number} not found for connection {self.name!r}. "
                f"Available: {available}"
            )
        return self._g[path]

    # ------------------------------------------------------------------
    # Static cell connectivity
    # ------------------------------------------------------------------

    @property
    def headwater_cells(self) -> np.ndarray | None:
        """Cell indices on the headwater side, or ``None`` if not stored.

        Shape ``(n_faces,)``.  Absent for some structure types (e.g. gates
        without explicit face-level connectivity).
        """
        if "Headwater Cells" not in self._g:
            return None
        return self._load("Headwater Cells")

    @property
    def tailwater_cells(self) -> np.ndarray | None:
        """Cell indices on the tailwater side, or ``None`` if not stored.

        Shape ``(n_faces,)``.
        """
        if "Tailwater Cells" not in self._g:
            return None
        return self._load("Tailwater Cells")


# ---------------------------------------------------------------------------
# SA2DConnectionCollection
# ---------------------------------------------------------------------------


class SA2DConnectionCollection:
    """Access all SA/2D hydraulic connections in a plan HDF file.

    Connections can link a Storage Area to a 2-D Flow Area, two Storage Areas
    to each other, or two 2-D Flow Areas.  Each connection is a named
    ``h5py.Group`` under ``-/SA 2D Area Conn/``.

    Parameters
    ----------
    hdf:
        Open ``h5py.File`` handle.
    """

    def __init__(self, hdf: "h5py.File") -> None:
        self._hdf = hdf
        self._items: dict[str, SA2DConnectionResults] | None = None

    def _load(self) -> dict[str, SA2DConnectionResults]:
        if self._items is not None:
            return self._items

        if _TS_SA_CONN not in self._hdf:
            self._items = {}
            return self._items

        import h5py as _h5

        root = self._hdf[_TS_SA_CONN]
        self._items = {
            k: SA2DConnectionResults(k, root[k])
            for k, v in root.items()
            if isinstance(v, _h5.Group)
        }
        return self._items

    @property
    def names(self) -> list[str]:
        """Names of all connections in the file."""
        return list(self._load().keys())

    @property
    def summary(self) -> pd.DataFrame:
        """One row per connection with basic attributes.

        Columns: ``name``, ``n_variables``, ``variable_names``,
        ``has_breaching``, ``has_weir``.
        """
        rows = [
            {
                "name": conn.name,
                "n_variables": conn.structure_variables.shape[1],
                "variable_names": conn.variable_names,
                "has_breaching": conn.breaching_variables is not None,
                "has_weir": conn.weir_variables is not None,
            }
            for conn in self._load().values()
        ]
        return pd.DataFrame(rows)

    def __getitem__(self, name: str) -> SA2DConnectionResults:
        items = self._load()
        if name not in items:
            raise KeyError(
                f"SA/2D connection {name!r} not found. Available: {self.names}"
            )
        return items[name]

    def __contains__(self, name: str) -> bool:
        return name in self._load()

    def __iter__(self):
        return iter(self._load())

    def __len__(self) -> int:
        return len(self._load())


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
            ts   = hdf.time_stamps
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
        self._sa_connections: SA2DConnectionCollection | None = None

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

    @property
    def time_step(self) -> float | None:
        """Output time step in seconds, or ``None`` for steady-flow plans.

        Derived from the difference between the first two time stamps.
        Returns ``None`` when the time-stamp dataset is absent (steady plans
        have no unsteady output block).
        """
        ds = self._hdf.get(_TIME_STAMP_DS)
        if ds is None:
            return None
        raw = np.array(ds[:2]).astype(str)
        ts = pd.to_datetime(raw, format=_RAS_TS_FMT)
        return (ts[1] - ts[0]).total_seconds()

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
    def time_stamps(self) -> pd.DatetimeIndex:
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
    def n_timesteps(self) -> int | None:
        """Number of output time steps, or ``None`` for steady-flow plans."""
        ds = self._hdf.get(_TIME_STAMP_DS)
        if ds is None:
            return None
        return len(ds)

    # ------------------------------------------------------------------
    # Collections (override GeometryHdf.flow_areas)
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
    def storage_area_connections(self) -> SA2DConnectionCollection:
        """Access SA/2D hydraulic connections (levees, dams, gates, etc.).

        Each connection can link a Storage Area to a 2-D Flow Area, two
        Storage Areas, or two 2-D Flow Areas.
        """
        if self._sa_connections is None:
            self._sa_connections = SA2DConnectionCollection(self._hdf)
        return self._sa_connections
