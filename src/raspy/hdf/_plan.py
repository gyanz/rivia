"""PlanHdf — read HEC-RAS plan HDF5 files (.p*.hdf).

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

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from ._base import _HdfFile
from ._geometry import FlowArea, FlowAreaCollection, GeometryHdf

if TYPE_CHECKING:
    import h5py
    import rasterio.io


# ---------------------------------------------------------------------------
# HDF path constants
# ---------------------------------------------------------------------------
_TS_ROOT  = (
    "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series"
)
_SUM_ROOT = (
    "Results/Unsteady/Output/Output Blocks/Base Output/Summary Output"
)
_TS_2D    = f"{_TS_ROOT}/2D Flow Areas"
_SUM_2D   = f"{_SUM_ROOT}/2D Flow Areas"

_TIME_DS         = f"{_TS_ROOT}/Time"
_TIME_STAMP_DS   = f"{_TS_ROOT}/Time Date Stamp"

# Timestamp format written by HEC-RAS (e.g. "03Jan2000 00:00:00")
_RAS_TS_FMT = "%d%b%Y %H:%M:%S"


# ---------------------------------------------------------------------------
# FlowAreaResults — extends FlowArea with plan results
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
        self._ts  = ts_group
        self._sum = sum_group

    # ------------------------------------------------------------------
    # Lazy time-series (h5py.Dataset — slice to control memory)
    # ------------------------------------------------------------------

    @property
    def water_surface(self) -> "h5py.Dataset":
        """Water-surface elevation time series.

        ``h5py.Dataset``, shape ``(n_timesteps, n_cells)``.
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
        raw = np.array(self._sum[key])   # shape (2, n_elements)
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
        Index: 0-based cell index.
        """
        return self._load_summary("Maximum Water Surface", n=self.n_cells)

    @property
    def min_water_surface(self) -> pd.DataFrame:
        """Minimum WSE per cell.  Same column layout as :attr:`max_water_surface`."""
        return self._load_summary("Minimum Water Surface", n=self.n_cells)

    @property
    def max_face_velocity(self) -> pd.DataFrame:
        """Maximum face velocity per face.

        DataFrame with columns ``['value', 'time']``.
        Index: 0-based face index.
        """
        return self._load_summary("Maximum Face Velocity", n=self.n_faces)

    # ------------------------------------------------------------------
    # Computed results — pure numpy, no geo dependency
    # ------------------------------------------------------------------

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
        """
        wse = np.array(self.water_surface[timestep, : self.n_cells])
        return np.maximum(0.0, wse - self.cell_min_elevation)

    def max_depth(self) -> pd.DataFrame:
        """Maximum depth per cell using the time of maximum WSE.

        ``value = max(0, max_WSE - bed_elevation)``.
        ``time`` is the elapsed time of maximum WSE (days); this is an
        approximation — maximum depth may not coincide with maximum WSE.

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
        method: Literal["area_weighted", "length_weighted", "flow_ratio"] = "area_weighted",
        wse_interp: Literal["average", "sloped"] = "average",
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
            position between the two cell centres.

        Returns
        -------
        ndarray, shape ``(n_cells, 2)``
            ``[Vx, Vy]`` depth-averaged velocity components.
        """
        from ._velocity import compute_all_cell_velocities

        if method == "flow_ratio" and self.face_flow is None:
            raise KeyError(
                "Face Flow is not present in this HDF file. "
                "Enable 'Face Flow' in HEC-RAS HDF5 Write Parameters "
                "before running the simulation, or use a different method."
            )

        face_vel  = np.array(self.face_velocity[timestep, :])
        cell_wse  = np.array(self.water_surface[timestep, : self.n_cells])
        face_flow = (
            np.array(self.face_flow[timestep, :])
            if method == "flow_ratio"
            else None
        )

        cell_face_info, cell_face_values = self.cell_face_info
        face_ae_info, face_ae_values = self.face_area_elevation

        if wse_interp == "sloped":
            cell_coords = self.cell_centers
            fp_idx      = self.face_facepoint_indexes   # (n_faces, 2)
            fp_xy       = self.facepoint_coordinates    # (n_facepoints, 2)
            face_coords = 0.5 * (fp_xy[fp_idx[:, 0]] + fp_xy[fp_idx[:, 1]])
        else:
            cell_coords = None
            face_coords = None

        return compute_all_cell_velocities(
            n_cells           = self.n_cells,
            cell_face_info    = cell_face_info,
            cell_face_values  = cell_face_values,
            face_normals      = self.face_normals,
            face_cell_indexes = self.face_cell_indexes,
            face_ae_info      = face_ae_info,
            face_ae_values    = face_ae_values,
            face_vel          = face_vel,
            cell_wse          = cell_wse,
            method            = method,
            face_flow         = face_flow,
            wse_interp        = wse_interp,
            cell_coords       = cell_coords,
            face_coords       = face_coords,
        )

    def cell_speed(
        self,
        timestep: int,
        method: Literal["area_weighted", "length_weighted", "flow_ratio"] = "area_weighted",
        wse_interp: Literal["average", "sloped"] = "average",
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

        Returns
        -------
        ndarray, shape ``(n_cells,)``
        """
        vecs = self.cell_velocity_vectors(
            timestep, method=method, wse_interp=wse_interp
        )
        return np.sqrt(vecs[:, 0] ** 2 + vecs[:, 1] ** 2)

    def cell_velocity_angle(
        self,
        timestep: int,
        method: Literal["area_weighted", "length_weighted", "flow_ratio"] = "area_weighted",
        wse_interp: Literal["average", "sloped"] = "average",
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

        Returns
        -------
        ndarray, shape ``(n_cells,)``
            Direction the flow is heading in degrees clockwise from north
            (0 = north, 90 = east, 180 = south, 270 = west).
            Cells whose speed is below 1e-10 return ``nan``.
        """
        vecs = self.cell_velocity_vectors(
            timestep, method=method, wse_interp=wse_interp
        )
        vx = vecs[:, 0]
        vy = vecs[:, 1]
        speed = np.sqrt(vx ** 2 + vy ** 2)
        angle = (90.0 - np.degrees(np.arctan2(vy, vx))) % 360.0
        angle[speed < 1e-10] = np.nan
        return angle

    def debug_cell_velocity(
        self,
        cell_idx: int,
        timestep: int,
        wse_interp: Literal["average", "sloped"] = "average",
    ) -> None:
        """Print a detailed per-face breakdown of velocity reconstruction.

        Displays the WLS input data for each face of the specified cell, then
        shows the reconstructed velocity for all available weight schemes.
        Optionally compares against the HEC-RAS stored ``Cell Velocity``
        scalar when that dataset is present in the HDF file.

        Parameters
        ----------
        cell_idx:
            0-based cell index.
        timestep:
            0-based index into the time dimension.
        wse_interp:
            Face WSE interpolation method used for ``area_weighted`` weights.
        """
        from ._velocity import (
            _estimate_face_wse_average,
            _estimate_face_wse_sloped,
            _interpolate_face_flow_area,
            _wls_velocity,
        )

        if cell_idx < 0 or cell_idx >= self.n_cells:
            raise IndexError(
                f"cell_idx {cell_idx} is out of range [0, {self.n_cells})"
            )

        face_vel = np.array(self.face_velocity[timestep, :])
        cell_wse = np.array(self.water_surface[timestep, : self.n_cells])

        cell_face_info, cell_face_values = self.cell_face_info
        start = int(cell_face_info[cell_idx, 0])
        count = int(cell_face_info[cell_idx, 1])
        vals  = cell_face_values[start : start + count]
        face_idxs    = vals[:, 0].astype(int)
        orientations = vals[:, 1].astype(int)

        normals = self.face_normals[face_idxs, :2]   # (k, 2)
        lengths = self.face_normals[face_idxs, 2]    # (k,)
        vn      = face_vel[face_idxs]                # (k,)

        face_ci = self.face_cell_indexes
        if wse_interp == "sloped":
            fp_idx      = self.face_facepoint_indexes
            fp_xy       = self.facepoint_coordinates
            face_coords = 0.5 * (fp_xy[fp_idx[:, 0]] + fp_xy[fp_idx[:, 1]])
            face_wse_all = _estimate_face_wse_sloped(
                face_ci, cell_wse, self.n_cells, self.cell_centers, face_coords
            )
        else:
            face_wse_all = _estimate_face_wse_average(face_ci, cell_wse, self.n_cells)

        ae_info, ae_values = self.face_area_elevation
        areas = np.array([
            _interpolate_face_flow_area(fi, face_wse_all[fi], ae_info, ae_values)
            for fi in face_idxs
        ])

        # --- header ---
        print(
            f"\n=== debug_cell_velocity  area={self.name}  "
            f"cell={cell_idx}  timestep={timestep} ==="
        )
        print(f"Cell WSE   : {cell_wse[cell_idx]:.4f}")
        print(f"wse_interp : {wse_interp}")
        print()

        hdr = (
            f"{'Face':>7}  {'Orient':>6}  {'nx':>8}  {'ny':>8}  "
            f"{'Length':>9}  {'V_n':>9}  {'face_WSE':>9}  {'A_face':>9}"
        )
        print(hdr)
        print("-" * len(hdr))
        for fi, ori, (nx, ny), L, v, fwse, a in zip(
            face_idxs, orientations, normals, lengths, vn,
            face_wse_all[face_idxs], areas,
            strict=False,
        ):
            print(
                f"{fi:>7}  {ori:>6}  {nx:>8.4f}  {ny:>8.4f}  "
                f"{L:>9.2f}  {v:>9.4f}  {fwse:>9.4f}  {a:>9.4f}"
            )

        # --- velocity results for each method ---
        print()
        vel_aw = _wls_velocity(vn, areas, normals)
        vel_lw = _wls_velocity(vn, lengths, normals)
        print(
            f"area_weighted  : Vx={vel_aw[0]:+.4f}  Vy={vel_aw[1]:+.4f}"
            f"  speed={np.linalg.norm(vel_aw):.4f}"
        )
        print(
            f"length_weighted: Vx={vel_lw[0]:+.4f}  Vy={vel_lw[1]:+.4f}"
            f"  speed={np.linalg.norm(vel_lw):.4f}"
        )

        if self.face_flow is not None:
            face_flow = np.array(self.face_flow[timestep, :])
            qf = face_flow[face_idxs]
            weights_fr = np.where(np.abs(vn) > 1e-10, np.abs(qf / vn), 0.0)
            vel_fr = _wls_velocity(vn, weights_fr, normals)
            print(
                f"flow_ratio     : Vx={vel_fr[0]:+.4f}  Vy={vel_fr[1]:+.4f}"
                f"  speed={np.linalg.norm(vel_fr):.4f}"
            )
            print(f"  Face Flows at cell faces : {face_flow[face_idxs]}")
        else:
            print("  (Face Flow not in HDF - flow_ratio unavailable)")

        if self.cell_velocity is not None:
            ras_speed = float(self.cell_velocity[timestep, cell_idx])
            print(f"\nHEC-RAS stored Cell Velocity : {ras_speed:.4f}")
        else:
            print("\n  (Cell Velocity scalar not stored in HDF)")

    # ------------------------------------------------------------------
    # Raster export — delegates to raspy.geo (deferred import)
    # ------------------------------------------------------------------

    def _max_velocity(
        self,
        method: Literal[
            "area_weighted", "length_weighted", "flow_ratio"
        ] = "area_weighted",
        wse_interp: Literal["average", "sloped"] = "average",
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
        max_vecs  = np.zeros((self.n_cells, 2))

        for t in range(n_t):
            vecs  = self.cell_velocity_vectors(t, method=method, wse_interp=wse_interp)
            speed = np.sqrt(vecs[:, 0] ** 2 + vecs[:, 1] ** 2)
            faster = speed > max_speed
            max_speed[faster] = speed[faster]
            max_vecs[faster]  = vecs[faster]

        return max_speed, max_vecs

    def export_raster(
        self,
        variable: Literal["water_surface", "depth", "cell_speed", "cell_velocity"],
        timestep: int | None = None,
        output_path: str | Path | None = None,
        *,
        cell_size: float | None = None,
        transform: Any | None = None,
        reference_raster: str | Path | None = None,
        crs: Any | None = None,
        nodata: float = -9999.0,
        interp_method: Literal["linear", "nearest", "cubic"] = "linear",
        min_value: float | None = None,
        vel_method: Literal[
            "area_weighted", "length_weighted", "flow_ratio"
        ] = "area_weighted",
        wse_interp: Literal["average", "sloped"] = "average",
    ) -> Path | rasterio.io.DatasetReader:
        """Interpolate a scalar or vector field to a GeoTIFF raster.

        Requires ``scipy`` and ``rasterio`` (available via ``pip install
        raspy[geo]``).  When these libraries are absent a clear
        ``ImportError`` is raised; all other ``raspy.hdf`` functionality
        remains available.

        Parameters
        ----------
        variable:
            ``"water_surface"`` — water-surface elevation.
            ``"depth"``         — water depth (WSE - bed elevation, >= 0).
            ``"cell_speed"``    — WLS-reconstructed velocity magnitude.
            ``"cell_velocity"`` — WLS-reconstructed velocity vector
                                  (writes 4 bands: Vx, Vy, Speed, Direction).
        output_path:
            Destination ``.tif`` file path.  When ``None`` (default), the
            raster is written to an in-memory buffer and an open
            ``rasterio.DatasetReader`` is returned; the caller must close it.
        timestep:
            0-based time index.  Pass ``None`` (default) to use maximum values.
            For ``"water_surface"`` and ``"depth"``, pre-computed HDF summary
            arrays are used.  For ``"cell_speed"`` and ``"cell_velocity"``,
            all timesteps are iterated to find the per-cell peak speed.
        cell_size:
            Output pixel size in model coordinate units.  Ignored when
            *transform* or *reference_raster* is supplied.  Defaults to the
            median face length of this flow area.
        transform:
            ``rasterio.transform.Affine`` reference transform.  The output
            grid is snapped to this pixel grid so the result aligns exactly
            with an existing raster.  Mutually exclusive with
            *reference_raster*.
        reference_raster:
            Path to an existing GeoTIFF.  The transform *and* CRS are read
            from this file.  Mutually exclusive with *transform*.  If *crs*
            is also supplied it overrides the file CRS.
        crs:
            Output CRS (e.g. ``"EPSG:26910"`` or an integer EPSG code).
            When *reference_raster* is given and *crs* is ``None``, the
            reference raster's CRS is inherited.
        nodata:
            Fill value for pixels outside the mesh convex hull.
        interp_method:
            SciPy ``griddata`` interpolation method: ``"linear"`` (default),
            ``"nearest"``, or ``"cubic"``.
        min_value:
            Source cells whose value is below this threshold are excluded
            from interpolation and set to *nodata*.  Useful for masking
            near-dry cells (e.g. ``min_value=0.01`` for depth in metres).
        vel_method:
            Velocity reconstruction weight scheme.  Passed to
            :meth:`cell_velocity_vectors`; ignored for ``"water_surface"``
            and ``"depth"``.
        wse_interp:
            Face WSE interpolation method.  Passed to
            :meth:`cell_velocity_vectors`; ignored for ``"water_surface"``
            and ``"depth"``.

        Returns
        -------
        Path
            Absolute path to the written GeoTIFF (when *output_path* is given).
        rasterio.io.DatasetReader
            Open in-memory dataset (when *output_path* is ``None``).
            The caller must close it when done.
        """
        from raspy.geo import raster as _raster  # deferred — geo not required

        # ── 1. Resolve values array (numpy only) ──────────────────────
        if timestep is None:
            if variable == "water_surface":
                values = self.max_water_surface["value"].to_numpy()
            elif variable == "depth":
                values = self.max_depth()["value"].to_numpy()
            elif variable == "cell_speed":
                values, _ = self._max_velocity(
                    method=vel_method, wse_interp=wse_interp
                )
            elif variable == "cell_velocity":
                _, values = self._max_velocity(
                    method=vel_method, wse_interp=wse_interp
                )
            else:
                raise ValueError(f"Unknown variable: {variable!r}")
        else:
            if variable == "water_surface":
                values = np.array(self.water_surface[timestep, : self.n_cells])
            elif variable == "depth":
                values = self.depth(timestep)
            elif variable == "cell_speed":
                values = self.cell_speed(
                    timestep, method=vel_method, wse_interp=wse_interp
                )
            elif variable == "cell_velocity":
                values = self.cell_velocity_vectors(  # (n_cells, 2)
                    timestep, method=vel_method, wse_interp=wse_interp
                )
            else:
                raise ValueError(f"Unknown variable: {variable!r}")

        # ── 2. Default cell size (median face length) — computed here  ──
        #       so raspy.geo does not need access to mesh geometry.
        resolved_cell_size = cell_size
        if cell_size is None and transform is None and reference_raster is None:
            resolved_cell_size = float(np.median(self.face_normals[:, 2]))

        # ── 3. Delegate all rasterio logic to raspy.geo ────────────────
        return _raster.points_to_raster(
            self.cell_centers,
            values,
            output_path,
            cell_size        = resolved_cell_size,
            transform        = transform,
            reference_raster = reference_raster,
            crs              = crs,
            nodata           = nodata,
            interp_method    = interp_method,
            min_value        = min_value,
        )


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
                    f"2D flow area {name!r} not found. "
                    f"Available: {self.names}"
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
                    f"No summary results found for flow area {name!r} "
                    f"at '{sum_path}'."
                )

            self._cache[name] = FlowAreaResults(
                geom_group = root[name],
                ts_group   = self._hdf[ts_path],
                sum_group  = sum_group,
                name       = name,
                n_cells    = n_cells,
            )
        return self._cache[name]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# PlanHdf — public entry point
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

    def __init__(self, filename: str | Path) -> None:
        super().__init__(filename)
        self._plan_flow_areas: FlowAreaResultsCollection | None = None

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

    # ------------------------------------------------------------------
    # Collections (override GeometryHdf.flow_areas)
    # ------------------------------------------------------------------

    @property
    def flow_areas(self) -> FlowAreaResultsCollection:
        """Access 2-D flow areas with both geometry and results data."""
        if self._plan_flow_areas is None:
            self._plan_flow_areas = FlowAreaResultsCollection(self._hdf)
        return self._plan_flow_areas
