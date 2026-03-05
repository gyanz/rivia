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

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

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
        self._ts = ts_group
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
        method: Literal[
            "area_weighted", "length_weighted", "flow_ratio"
        ] = "area_weighted",
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

        face_vel = np.array(self.face_velocity[timestep, :])
        cell_wse = np.array(self.water_surface[timestep, : self.n_cells])
        face_flow = (
            np.array(self.face_flow[timestep, :]) if method == "flow_ratio" else None
        )

        cell_face_info, cell_face_values = self.cell_face_info
        face_ae_info, face_ae_values = self.face_area_elevation

        if wse_interp == "sloped":
            cell_coords = self.cell_centers
            fp_idx = self.face_facepoint_indexes  # (n_faces, 2)
            fp_xy = self.facepoint_coordinates  # (n_facepoints, 2)
            face_coords = 0.5 * (fp_xy[fp_idx[:, 0]] + fp_xy[fp_idx[:, 1]])
        else:
            cell_coords = None
            face_coords = None

        return compute_all_cell_velocities(
            n_cells=self.n_cells,
            cell_face_info=cell_face_info,
            cell_face_values=cell_face_values,
            face_normals=self.face_normals,
            face_cell_indexes=self.face_cell_indexes,
            face_ae_info=face_ae_info,
            face_ae_values=face_ae_values,
            face_vel=face_vel,
            cell_wse=cell_wse,
            method=method,
            face_flow=face_flow,
            wse_interp=wse_interp,
            cell_coords=cell_coords,
            face_coords=face_coords,
        )

    def cell_speed(
        self,
        timestep: int,
        method: Literal[
            "area_weighted", "length_weighted", "flow_ratio"
        ] = "area_weighted",
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
        method: Literal[
            "area_weighted", "length_weighted", "flow_ratio"
        ] = "area_weighted",
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
        speed = np.sqrt(vx**2 + vy**2)
        angle = (90.0 - np.degrees(np.arctan2(vy, vx))) % 360.0
        angle[speed < 1e-10] = np.nan
        return angle

    def debug_cell_velocity(
        self,
        cell_idx: int,
        timestep: int,
        wse_interp: Literal["average", "sloped"] = "sloped",
        vel_method: Literal[
            "area_weighted", "length_weighted", "flow_ratio"
        ] = "length_weighted",
    ) -> None:
        """Print a detailed per-face breakdown of velocity reconstruction.

        Displays the WLS input data for each face of the specified cell, then
        shows the reconstructed velocity for all available weight schemes.
        Optionally compares against the HEC-RAS stored ``Cell Velocity``
        scalar when that dataset is present in the HDF file.  Also prints a
        double-C stencil face velocity table using *vel_method* WLS vectors.

        Parameters
        ----------
        cell_idx:
            0-based cell index.
        timestep:
            0-based index into the time dimension.
        wse_interp:
            Face WSE interpolation method used for ``area_weighted`` weights.
        vel_method:
            WLS weight scheme used as V_L / V_R in the double-C stencil
            reconstruction.  Defaults to ``"area_weighted"``.
        """
        from ._velocity import (
            _estimate_face_wse_average,
            _estimate_face_wse_sloped,
            _interpolate_face_flow_area,
            _wls_velocity,
        )

        if cell_idx < 0 or cell_idx >= self.n_cells:
            raise IndexError(f"cell_idx {cell_idx} is out of range [0, {self.n_cells})")

        face_vel = np.array(self.face_velocity[timestep, :])
        cell_wse = np.array(self.water_surface[timestep, : self.n_cells])

        cell_face_info, cell_face_values = self.cell_face_info
        start = int(cell_face_info[cell_idx, 0])
        count = int(cell_face_info[cell_idx, 1])
        vals = cell_face_values[start : start + count]
        face_idxs = vals[:, 0].astype(int)
        orientations = vals[:, 1].astype(int)

        normals = self.face_normals[face_idxs, :2]  # (k, 2)
        lengths = self.face_normals[face_idxs, 2]  # (k,)
        vn = face_vel[face_idxs]  # (k,)

        face_ci = self.face_cell_indexes
        if wse_interp == "sloped":
            fp_idx = self.face_facepoint_indexes
            fp_xy = self.facepoint_coordinates
            face_coords = 0.5 * (fp_xy[fp_idx[:, 0]] + fp_xy[fp_idx[:, 1]])
            face_wse_all = _estimate_face_wse_sloped(
                face_ci, cell_wse, self.n_cells, self.cell_centers, face_coords
            )
        else:
            face_wse_all = _estimate_face_wse_average(face_ci, cell_wse, self.n_cells)

        ae_info, ae_values = self.face_area_elevation
        areas = np.array(
            [
                _interpolate_face_flow_area(fi, face_wse_all[fi], ae_info, ae_values)
                for fi in face_idxs
            ]
        )

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
            face_idxs,
            orientations,
            normals,
            lengths,
            vn,
            face_wse_all[face_idxs],
            areas,
            strict=False,
        ):
            print(
                f"{fi:>7}  {ori:>6}  {nx:>8.4f}  {ny:>8.4f}  "
                f"{L:>9.2f}  {v:>9.4f}  {fwse:>9.4f}  {a:>9.4f}"
            )

        def _angle(v: np.ndarray) -> str:
            spd = np.linalg.norm(v)
            if spd < 1e-10:
                return "     n/a"
            return f"{(90.0 - np.degrees(np.arctan2(v[1], v[0]))) % 360.0:>8.2f}"

        # --- velocity results for each method ---
        print()
        vel_aw = _wls_velocity(vn, areas, normals)
        vel_lw = _wls_velocity(vn, lengths, normals)
        print(
            f"area_weighted  : Vx={vel_aw[0]:+.4f}  Vy={vel_aw[1]:+.4f}"
            f"  speed={np.linalg.norm(vel_aw):.4f}  dir={_angle(vel_aw)}°"
        )
        print(
            f"length_weighted: Vx={vel_lw[0]:+.4f}  Vy={vel_lw[1]:+.4f}"
            f"  speed={np.linalg.norm(vel_lw):.4f}  dir={_angle(vel_lw)}°"
        )

        if self.face_flow is not None:
            face_flow = np.array(self.face_flow[timestep, :])
            qf = face_flow[face_idxs]
            weights_fr = np.where(np.abs(vn) > 1e-10, np.abs(qf / vn), 0.0)
            vel_fr = _wls_velocity(vn, weights_fr, normals)
            print(
                f"flow_ratio     : Vx={vel_fr[0]:+.4f}  Vy={vel_fr[1]:+.4f}"
                f"  speed={np.linalg.norm(vel_fr):.4f}  dir={_angle(vel_fr)}°"
            )
            print(f"  Face Flows at cell faces : {face_flow[face_idxs]}")
        else:
            print("  (Face Flow not in HDF - flow_ratio unavailable)")

        if self.cell_velocity is not None:
            ras_speed = float(self.cell_velocity[timestep, cell_idx])
            print(f"\nHEC-RAS stored Cell Velocity : {ras_speed:.4f}")
        else:
            print("\n  (Cell Velocity scalar not stored in HDF)")

        # --- double-C stencil: per-face tangential velocity reconstruction ---
        print()
        print(
            f"--- Double-C stencil face velocity reconstruction"
            f" (vel_method={vel_method!r}) ---"
        )
        all_cell_vecs = self.cell_velocity_vectors(
            timestep, method=vel_method, wse_interp=wse_interp
        )
        n_cells = self.n_cells
        face_ci = self.face_cell_indexes  # (n_faces, 2)

        hdr2 = (
            f"{'Face':>7}  {'tx':>8}  {'ty':>8}  "
            f"{'vt_L':>9}  {'vt_R':>9}  {'vt_avg':>9}  "
            f"{'Vfx':>9}  {'Vfy':>9}  {'|Vf|':>9}  {'side':>6}"
        )
        print(hdr2)
        print("-" * len(hdr2))
        for fi, (nx, ny), v in zip(face_idxs, normals, vn, strict=False):
            t_hat = np.array([-ny, nx])
            left  = int(face_ci[fi, 0])
            right = int(face_ci[fi, 1])

            valid_left  = 0 <= left  < n_cells
            valid_right = 0 <= right < n_cells

            _nan = float("nan")
            vt_L = float(np.dot(all_cell_vecs[left],  t_hat)) if valid_left  else _nan
            vt_R = float(np.dot(all_cell_vecs[right], t_hat)) if valid_right else _nan

            if valid_left and valid_right:
                vt_avg = 0.5 * (vt_L + vt_R)
                side_label = "both"
            elif valid_left:
                vt_avg = vt_L
                side_label = "L only"
            elif valid_right:
                vt_avg = vt_R
                side_label = "R only"
            else:
                vt_avg = 0.0
                side_label = "none"

            vf = v * np.array([nx, ny]) + vt_avg * t_hat
            vt_L_str = f"{vt_L:>9.4f}" if valid_left  else f"{'n/a':>9}"
            vt_R_str = f"{vt_R:>9.4f}" if valid_right else f"{'n/a':>9}"
            print(
                f"{fi:>7}  {t_hat[0]:>8.4f}  {t_hat[1]:>8.4f}  "
                f"{vt_L_str}  {vt_R_str}  {vt_avg:>9.4f}  "
                f"{vf[0]:>9.4f}  {vf[1]:>9.4f}  {np.linalg.norm(vf):>9.4f}  "
                f"{side_label:>6}"
            )

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
        max_vecs = np.zeros((self.n_cells, 2))

        for t in range(n_t):
            vecs = self.cell_velocity_vectors(t, method=method, wse_interp=wse_interp)
            speed = np.sqrt(vecs[:, 0] ** 2 + vecs[:, 1] ** 2)
            faster = speed > max_speed
            max_speed[faster] = speed[faster]
            max_vecs[faster] = vecs[faster]

        return max_speed, max_vecs

    def _export_raster(
        self,
        variable: Literal["water_surface", "depth", "cell_speed", "cell_velocity"],
        timestep: int | None = None,
        output_path: str | Path | None = None,
        *,
        cell_size: float | None = None,
        reference_transform: Any | None = None,
        reference_raster: str | Path | None = None,
        snap_to_reference_extent: bool = True,
        crs: Any | None = None,
        nodata: float = -9999.0,
        interp_method: Literal["linear", "nearest", "cubic"] = "linear",
        min_value: float | None = None,
        use_adjacency: bool = True,
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
            ``"depth"``         — water depth from DEM subtraction (requires
                                  *reference_raster*): WSE is interpolated then
                                  the DEM pixel value is subtracted; negative
                                  values are clamped to 0.
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
            *reference_transform* or *reference_raster* is supplied.
            Defaults to the median face length of this flow area.
        reference_transform:
            ``rasterio.transform.Affine`` reference transform.  The output
            grid is snapped to this pixel grid so the result aligns exactly
            with an existing raster.  Mutually exclusive with
            *reference_raster*.
        reference_raster:
            Path to an existing GeoTIFF.  The transform *and* CRS are read
            from this file.  Mutually exclusive with *transform*.  If *crs*
            is also supplied it overrides the file CRS.  **Required** when
            ``variable="depth"``.
        snap_to_reference_extent:
            Passed to :func:`~raspy.geo.raster.points_to_raster`.  Only
            relevant when *reference_raster* is supplied.  When ``True``
            (default) the output covers the full reference raster extent;
            when ``False`` it is cropped to the point cloud.
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
            For ``"depth"``, filtering is applied using HDF cell depths
            (WSE minus cell minimum elevation) before WSE interpolation.
        use_adjacency:
            When ``True`` (default), mesh face connectivity is used to
            prevent spurious interpolation across gaps between disconnected
            wet areas.  Set to ``False`` to fall back to standard convex-hull
            ``griddata`` interpolation (faster, no gap masking).
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

        # ── 0. Guard: depth requires a reference DEM ───────────────────
        if variable == "depth" and reference_raster is None:
            raise ValueError(
                "reference_raster is required when variable='depth'. "
                "Provide a path to a terrain DEM GeoTIFF."
            )

        # ── 1. Resolve values array (numpy only) ──────────────────────
        if variable == "depth":
            # Interpolate WSE; subtract DEM after interpolation.
            # HDF depth (WSE - cell_min_elevation) is used only for
            # min_value filtering so that the threshold applies to depth,
            # not to raw WSE values.
            if timestep is None:
                wse_values      = self.max_water_surface["value"].to_numpy()
                depth_at_cells  = self.max_depth()["value"].to_numpy()
            else:
                wse_values     = np.array(self.water_surface[timestep, : self.n_cells])
                depth_at_cells = self.depth(timestep)

            interp_points = self.cell_centers
            if min_value is not None:
                mask          = depth_at_cells >= min_value
                interp_points = self.cell_centers[mask]
                wse_values    = wse_values[mask]

        elif timestep is None:
            if variable == "water_surface":
                values = self.max_water_surface["value"].to_numpy()
            elif variable == "cell_speed":
                values, _ = self._max_velocity(method=vel_method, wse_interp=wse_interp)
            elif variable == "cell_velocity":
                _, values = self._max_velocity(method=vel_method, wse_interp=wse_interp)
            else:
                raise ValueError(f"Unknown variable: {variable!r}")
        else:
            if variable == "water_surface":
                values = np.array(self.water_surface[timestep, : self.n_cells])
            elif variable == "cell_speed":
                values = self.cell_speed(
                    timestep, method=vel_method, wse_interp=wse_interp
                )
            elif variable == "cell_velocity":
                values = self.cell_velocity_vectors(
                    timestep, method=vel_method, wse_interp=wse_interp
                )
            else:
                raise ValueError(f"Unknown variable: {variable!r}")

        # ── 2. Default cell size (median face length) — computed here  ──
        #       so raspy.geo does not need access to mesh geometry.
        resolved_cell_size = cell_size
        no_grid = reference_transform is None and reference_raster is None
        if cell_size is None and no_grid:
            resolved_cell_size = float(np.median(self.face_normals[:, 2]))

        # ── 3. Delegate all rasterio logic to raspy.geo ────────────────
        resolved_adjacency = self.face_cell_indexes if use_adjacency else None

        if variable == "depth":
            # Remap adjacency to the filtered interp_points index space when
            # min_value pre-filtering has already reduced cell_centers to a subset.
            fci = resolved_adjacency  # (n_faces, 2), -1 = boundary
            if min_value is not None:
                orig_to_new = np.full(self.n_cells, -1, dtype=np.int64)
                orig_to_new[np.where(mask)[0]] = np.arange(int(mask.sum()))
                depth_adjacency = np.where(
                    fci >= 0, orig_to_new[np.maximum(fci, 0)], -1
                )
            else:
                depth_adjacency = fci

            wse_ds = _raster.points_to_raster(
                interp_points,
                wse_values,
                output_path=None,  # always in-memory; written after DEM subtraction
                cell_size=resolved_cell_size,
                reference_transform=reference_transform,
                reference_raster=reference_raster,
                crs=crs,
                nodata=nodata,
                interp_method=interp_method,
                min_value=None,
                snap_to_reference_extent=snap_to_reference_extent,
                adjacency=depth_adjacency,
            )
            result = _raster._depth_from_wse_and_dem(
                wse_ds, reference_raster, nodata, output_path, min_value=min_value
            )
            wse_ds.close()
            return result

        return _raster.points_to_raster(
            self.cell_centers,
            values,
            output_path,
            cell_size=resolved_cell_size,
            reference_transform=reference_transform,
            reference_raster=reference_raster,
            crs=crs,
            nodata=nodata,
            interp_method=interp_method,
            min_value=min_value,
            snap_to_reference_extent=snap_to_reference_extent,
            adjacency=resolved_adjacency,
        )

    def export_raster(
        self,
        variable: Literal["water_surface", "depth", "cell_speed", "cell_velocity"],
        timestep: int | None = None,
        output_path: str | Path | None = None,
        *,
        cell_size: float | None = None,
        reference_transform: Any | None = None,
        reference_raster: str | Path | None = None,
        snap_to_reference_extent: bool = True,
        crs: Any | None = None,
        nodata: float = -9999.0,
        depth_min: float | None = None,
        vel_min: float | None = None,
        vel_method: Literal[
            "area_weighted", "length_weighted", "flow_ratio"
        ] = "area_weighted",
        vel_wse_method: Literal["average", "sloped"] = "sloped",
        render_mode: Literal["sloping", "horizontal", "hybrid"] = "sloping",
        face_active_threshold: float = 0.0,
        vel_interp: bool = True,
    ) -> Path | rasterio.io.DatasetReader:
        """Interpolate a field to a GeoTIFF using mesh-conforming triangulation.

        Uses :func:`~raspy.geo.raster.mesh_to_raster` which sub-divides each
        mesh cell into triangles from the cell centre to each bounding face,
        then performs linear (barycentric) interpolation within those triangles.
        This prevents spurious fill across dry gaps or disconnected wet islands,
        matching the interpolation used by HEC-RAS RASMapper.

        Requires ``rasterio`` and ``matplotlib`` (``pip install raspy[geo]``).

        Parameters
        ----------
        variable:
            ``"water_surface"`` — water-surface elevation.
            ``"depth"``         — water depth (requires *reference_raster*):
                                  WSE is interpolated then the DEM pixel value
                                  is subtracted; negative depths are clamped
                                  to 0.
            ``"cell_speed"``    — WLS-reconstructed velocity magnitude.
            ``"cell_velocity"`` — WLS-reconstructed velocity vector
                                  (4 bands: Vx, Vy, Speed, Direction).
        timestep:
            0-based time index.  Pass ``None`` (default) to use maximum values
            (only valid for ``"water_surface"`` and ``"depth"``).
        output_path:
            Destination ``.tif`` file path.  ``None`` returns an in-memory
            ``rasterio.DatasetReader``; the caller must close it.
        cell_size:
            Output pixel size in model coordinate units.  Defaults to the
            median face length when no reference grid is supplied.
        reference_transform:
            ``rasterio.transform.Affine`` for pixel-perfect alignment.
            Mutually exclusive with *reference_raster*.
        reference_raster:
            Existing GeoTIFF whose transform and CRS are inherited.
            **Required** when ``variable="depth"``.
        snap_to_reference_extent:
            When *reference_raster* is given, extend the output to its full
            extent (default ``True``).
        crs:
            Output CRS.  When *reference_raster* is given and *crs* is
            ``None``, the reference raster CRS is inherited.
        nodata:
            Fill value for pixels outside the wet mesh.
        depth_min:
            Minimum water depth (m).  Only used when ``variable="depth"``.
            Cells whose HDF depth (WSE minus cell minimum elevation) is
            below this value are excluded before WSE interpolation, and
            output pixels shallower than this value are set to *nodata*
            after DEM subtraction.
        vel_min:
            Minimum speed (m/s).  Only used when ``variable="cell_speed"``
            or ``"cell_velocity"``.  Cells whose WLS speed is below this
            threshold are excluded from the WSE wet-extent render and from
            the final velocity output.
        vel_method:
            Velocity reconstruction scheme passed to
            :meth:`cell_velocity_vectors`.
        vel_wse_method:
            Face WSE interpolation method passed to
            :meth:`cell_velocity_vectors`.
        render_mode:
            Water-surface rendering mode — ``"sloping"`` (default),
            ``"horizontal"``, or ``"hybrid"``.  See
            :func:`~raspy.geo.raster.mesh_to_raster` for full description.
        face_active_threshold:
            Velocity magnitude threshold (m/s) used to classify each face
            as active (wet) for ``render_mode="hybrid"``.  A face is
            considered active when
            ``|face_velocity| > face_active_threshold``.  Default ``0.0``
            treats any non-zero velocity as active.  Ignored for other
            render modes.
        vel_interp:
            When ``True`` and *variable* is ``"cell_velocity"`` or
            ``"cell_speed"``, use spatially varying interpolation within
            each mesh cell (HEC-RAS double-C stencil).  Each pixel's
            velocity is a barycentric blend of the WLS cell-centre vector
            and the reconstructed face vector, so the field varies
            smoothly within a cell while remaining strictly confined to
            its boundaries.  When ``False`` (default), the uniform
            cell-centre WLS velocity is painted over all wet pixels
            inside the cell, matching the original behaviour.

        Returns
        -------
        Path
            Written GeoTIFF path (when *output_path* is given).
        rasterio.io.DatasetReader
            Open in-memory dataset (when *output_path* is ``None``).

        Raises
        ------
        ImportError
            If ``rasterio`` or ``matplotlib`` are not installed.
        ValueError
            If ``variable="depth"`` and *reference_raster* is not provided,
            or if ``timestep=None`` is used with ``"cell_speed"`` or
            ``"cell_velocity"``.
        """
        from raspy.geo import raster as _raster  # deferred — geo not required

        # ── 0. Guards ──────────────────────────────────────────────────
        if variable == "depth" and reference_raster is None:
            raise ValueError(
                "reference_raster is required when variable='depth'. "
                "Provide a path to a terrain DEM GeoTIFF."
            )
        if variable in ("cell_speed", "cell_velocity") and timestep is None:
            raise ValueError(
                "timestep=None is not supported for cell_speed / cell_velocity. "
                "Provide an explicit timestep index."
            )

        # ── 0b. Hybrid face-active mask ────────────────────────────────
        # Hybrid mode requires per-timestep face velocities.  When timestep
        # is None (max-value maps) no such data exists; fall back to sloping.
        _render_mode = render_mode
        face_active: np.ndarray | None = None
        if _render_mode == "hybrid":
            if timestep is None:
                logging.warning(
                    "render_mode='hybrid' is not supported when timestep=None "
                    "(no per-timestep face data for max-value maps). "
                    "Falling back to render_mode='sloping'."
                )
                _render_mode = "sloping"
            else:
                face_active = (
                    np.abs(np.array(self.face_velocity[timestep, :]))
                    > face_active_threshold
                )

        # ── 1. Resolve values array ────────────────────────────────────
        if variable == "depth":
            if timestep is None:
                wse_values = self.max_water_surface["value"].to_numpy()
                depth_at_cells = self.max_depth()["value"].to_numpy()
            else:
                wse_values = np.array(self.water_surface[timestep, : self.n_cells])
                depth_at_cells = self.depth(timestep)
            # Pre-mask dry cells by depth so mesh_to_raster sees NaN at those
            # cell centres and excludes the corresponding triangles.
            cell_wse = wse_values.copy()
            if depth_min is not None:
                cell_wse[depth_at_cells < depth_min] = np.nan
        elif variable == "water_surface":
            if timestep is None:
                values = self.max_water_surface["value"].to_numpy()
            else:
                values = np.array(self.water_surface[timestep, : self.n_cells])
        elif variable in ("cell_speed", "cell_velocity"):
            # Compute WLS velocity vectors and cell WSE for the velocity raster.
            # mesh_to_velocity_raster renders WSE to determine wet extent, then
            # assigns velocity per-cell (no spatial interpolation across cells).
            cell_vel_wse = np.array(self.water_surface[timestep, : self.n_cells])
            cell_vel_vecs = self.cell_velocity_vectors(
                timestep, method=vel_method, wse_interp=vel_wse_method
            )
        else:
            raise ValueError(f"Unknown variable: {variable!r}")

        # ── 2. Default cell size ───────────────────────────────────────
        resolved_cell_size = cell_size
        no_grid = reference_transform is None and reference_raster is None
        if cell_size is None and no_grid:
            resolved_cell_size = float(np.median(self.face_normals[:, 2]))

        # ── 3. Common mesh topology keyword arguments ──────────────────
        _cfi, _cfv = self.cell_face_info  # property returns (info, values) tuple
        mesh_kw: dict[str, Any] = dict(
            cell_centers=self.cell_centers,
            facepoint_coordinates=self.facepoint_coordinates,
            face_facepoint_indexes=self.face_facepoint_indexes,
            face_cell_indexes=self.face_cell_indexes,
            cell_face_info=_cfi,
            cell_face_values=_cfv,
            cell_size=resolved_cell_size,
            reference_transform=reference_transform,
            reference_raster=reference_raster,
            crs=crs,
            nodata=nodata,
            snap_to_reference_extent=snap_to_reference_extent,
            render_mode=_render_mode,
            face_active=face_active,
        )

        # ── 4. Delegate to mesh_to_raster / mesh_to_velocity_raster ──────
        if variable == "depth":
            wse_ds = _raster.mesh_to_raster(
                **mesh_kw,
                cell_values=cell_wse,
                output_path=None,   # in-memory; depth written after DEM subtraction
                min_value=None,     # dry cells already NaN-masked above
                min_above_ref=depth_min,
            )
            result = _raster._depth_from_wse_and_dem(
                wse_ds, reference_raster, nodata, output_path, min_value=depth_min
            )
            wse_ds.close()
            return result

        if variable == "cell_velocity":
            if vel_interp:
                # Spatially varying interpolation within each mesh cell using
                # the HEC-RAS double-C stencil (barycentric blend of WLS
                # cell-centre velocity and reconstructed face velocity).
                face_vel_arr = np.array(self.face_velocity[timestep, :])
                return _raster.mesh_to_velocity_raster_interp(
                    **mesh_kw,
                    face_normals=self.face_normals,
                    face_vel=face_vel_arr,
                    cell_wse=cell_vel_wse,
                    cell_velocity=cell_vel_vecs,
                    output_path=output_path,
                    vel_min=vel_min,
                    depth_min=depth_min,
                )
            # WSE-based wet extent; velocity assigned per-cell without spatial
            # interpolation across cell boundaries.
            return _raster.mesh_to_velocity_raster(
                **mesh_kw,
                cell_wse=cell_vel_wse,
                cell_velocity=cell_vel_vecs,
                output_path=output_path,
                vel_min=vel_min,
                depth_min=depth_min,
            )

        if variable == "cell_speed":
            # Same rendering as cell_velocity; extract the speed band only.
            if vel_interp:
                face_vel_arr = np.array(self.face_velocity[timestep, :])
                vel_ds = _raster.mesh_to_velocity_raster_interp(
                    **mesh_kw,
                    face_normals=self.face_normals,
                    face_vel=face_vel_arr,
                    cell_wse=cell_vel_wse,
                    cell_velocity=cell_vel_vecs,
                    output_path=None,
                    vel_min=vel_min,
                    depth_min=depth_min,
                )
            else:
                vel_ds = _raster.mesh_to_velocity_raster(
                    **mesh_kw,
                    cell_wse=cell_vel_wse,
                    cell_velocity=cell_vel_vecs,
                    output_path=None,
                    vel_min=vel_min,
                    depth_min=depth_min,
                )
            return _raster._velocity_raster_to_speed(vel_ds, output_path, nodata)

        # water_surface: min_above_ref controls the wet/dry threshold relative
        # to the DEM (when reference_raster is given); no scalar min_value needed.
        return _raster.mesh_to_raster(
            **mesh_kw,
            cell_values=values,
            output_path=output_path,
            min_value=None,
            min_above_ref=depth_min,
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
# StorageAreaResults — extends StorageArea geometry with plan results
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
        ``h5py.Group`` at ``…/Unsteady Time Series/Storage Areas``, or ``None``
        when the plan has no SA results.
    sum_sa_group:
        ``h5py.Group`` at ``…/Summary Output/Storage Areas``, or ``None``.
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
        # per-SA subgroup: …/Storage Areas/<name>/
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
# SA2DConnectionResults — one connection between two hydraulic areas
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
        ``h5py.Group`` at ``…/SA 2D Area Conn/<name>``.
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
        ``col_0``, ``col_1``, … when the attribute is absent.
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
    ``h5py.Group`` under ``…/SA 2D Area Conn/``.

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
        self._plan_storage_areas: StorageAreaResultsCollection | None = None
        self._sa_connections: SA2DConnectionCollection | None = None

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
