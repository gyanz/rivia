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

from raspy.utils import timed

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
        vel_weight_method: Literal[
            "area_weighted", "length_weighted", "flow_ratio"
        ] = "length_weighted",
    ) -> None:
        """Print a detailed per-face breakdown of velocity reconstruction.

        Displays the WLS input data for each face of the specified cell, then
        shows the reconstructed velocity for all available weight schemes.
        Optionally compares against the HEC-RAS stored ``Cell Velocity``
        scalar when that dataset is present in the HDF file.

        Prints three tables:

        1. **WLS inputs** — per-face normal, length, V_n, face WSE, flow area.
        2. **Double-C stencil face velocities** — full 2D face velocity
           ``vn·n̂ + vt·t̂`` using the *vel_weight_method* WLS vectors for
           the tangential component.  This is what all intra-cell interpolation
           methods (``triangle_blend``, ``face_idw``, ``face_gradient``,
           ``facepoint_blend``) use at the face midpoints.
        3. **Facepoint velocities** — velocity assigned to each polygon vertex
           by averaging the full 2D velocities of all adjacent wet faces.
           This is what ``facepoint_blend`` interpolates between.

        Parameters
        ----------
        cell_idx:
            0-based cell index.
        timestep:
            0-based index into the time dimension.
        wse_interp:
            Face WSE interpolation method used for ``area_weighted`` weights.
        vel_weight_method:
            WLS weight scheme used as V_L / V_R in the double-C stencil
            reconstruction.  Defaults to ``"length_weighted"``.
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
            f" (vel_weight_method={vel_weight_method!r}) ---"
        )
        all_cell_vecs = self.cell_velocity_vectors(
            timestep, method=vel_weight_method, wse_interp=wse_interp
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

        # Collect full 2D face velocity for each of this cell's faces so we
        # can reuse it for the facepoint section below.
        face_vel_2d: dict[int, np.ndarray] = {}

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
            face_vel_2d[fi] = vf

            vt_L_str = f"{vt_L:>9.4f}" if valid_left  else f"{'n/a':>9}"
            vt_R_str = f"{vt_R:>9.4f}" if valid_right else f"{'n/a':>9}"
            print(
                f"{fi:>7}  {t_hat[0]:>8.4f}  {t_hat[1]:>8.4f}  "
                f"{vt_L_str}  {vt_R_str}  {vt_avg:>9.4f}  "
                f"{vf[0]:>9.4f}  {vf[1]:>9.4f}  {np.linalg.norm(vf):>9.4f}  "
                f"{side_label:>6}"
            )

        # --- facepoint velocities (facepoint_blend method) ---
        # For each unique facepoint of this cell, find all faces in the ENTIRE
        # mesh that share it, average their full 2D face velocities (wet faces
        # only), and display the result.  This is exactly the value that
        # facepoint_blend would assign at each polygon vertex.
        print()
        print("--- Facepoint velocities (facepoint_blend) ---")

        fp_idx_all = self.face_facepoint_indexes   # (n_faces, 2)
        fp_coords  = self.facepoint_coordinates    # (n_fp, 2)

        # Collect all wet-face 2D velocities for the full mesh so we can
        # average over them per facepoint.  We reuse all_cell_vecs for the
        # double-C stencil rather than re-computing per-face here.
        dry_mask_full = np.isnan(cell_wse)

        # Build full face_vel_2d for all faces via the double-C stencil.
        face_vel_full = face_vel   # signed normal velocities for all faces
        n_faces_all   = len(face_vel_full)
        face_vel_2d_all = np.zeros((n_faces_all, 2), dtype=np.float64)
        for gfi in range(n_faces_all):
            gnx = float(self.face_normals[gfi, 0])
            gny = float(self.face_normals[gfi, 1])
            gt = np.array([-gny, gnx])
            gl = int(face_ci[gfi, 0])
            gr = int(face_ci[gfi, 1])
            gl_ok = 0 <= gl < n_cells and not dry_mask_full[gl]
            gr_ok = 0 <= gr < n_cells and not dry_mask_full[gr]
            if gl_ok and gr_ok:
                gvt = 0.5 * (
                    float(np.dot(all_cell_vecs[gl], gt))
                    + float(np.dot(all_cell_vecs[gr], gt))
                )
            elif gl_ok:
                gvt = float(np.dot(all_cell_vecs[gl], gt))
            elif gr_ok:
                gvt = float(np.dot(all_cell_vecs[gr], gt))
            else:
                gvt = 0.0
            face_vel_2d_all[gfi] = (
                face_vel_full[gfi] * np.array([gnx, gny]) + gvt * gt
            )

        # Wet-face mask: at least one wet neighbour.
        l_idx = face_ci[:, 0]
        r_idx = face_ci[:, 1]
        l_safe = np.where((l_idx >= 0) & (l_idx < n_cells), l_idx, 0)
        r_safe = np.where((r_idx >= 0) & (r_idx < n_cells), r_idx, 0)
        wet_face_mask = (
            ((l_idx >= 0) & (l_idx < n_cells) & ~dry_mask_full[l_safe])
            | ((r_idx >= 0) & (r_idx < n_cells) & ~dry_mask_full[r_safe])
        )

        # Unique facepoints for this cell, ordered by face.
        seen: set[int] = set()
        cell_fp_ordered: list[int] = []
        for fi in face_idxs:
            for fp in (int(fp_idx_all[fi, 0]), int(fp_idx_all[fi, 1])):
                if fp not in seen:
                    seen.add(fp)
                    cell_fp_ordered.append(fp)

        hdr3 = (
            f"{'FP':>7}  {'x':>11}  {'y':>11}  "
            f"{'Vfp_x':>9}  {'Vfp_y':>9}  {'|Vfp|':>9}  {'n_faces':>7}"
        )
        print(hdr3)
        print("-" * len(hdr3))
        for fp in cell_fp_ordered:
            # All mesh faces that have this facepoint as an endpoint.
            touching = np.where(
                (fp_idx_all[:, 0] == fp) | (fp_idx_all[:, 1] == fp)
            )[0]
            wet_touching = touching[wet_face_mask[touching]]
            if len(wet_touching) == 0:
                vfp = np.zeros(2)
            else:
                vfp = face_vel_2d_all[wet_touching].mean(axis=0)
            x, y = fp_coords[fp]
            print(
                f"{fp:>7}  {x:>11.3f}  {y:>11.3f}  "
                f"{vfp[0]:>9.4f}  {vfp[1]:>9.4f}  "
                f"{np.linalg.norm(vfp):>9.4f}  {len(wet_touching):>7}"
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

    @timed(logging.INFO)
    def export_raster(
        self,
        variable: Literal["water_surface", "depth", "cell_speed", "cell_velocity"],
        timestep: int | None = None,
        output_path: str | Path | None = None,
        *,
        cell_size: float | None = None,
        reference_transform: Any | None = None,
        reference_raster: str | Path | None = None,
        snap_to_reference_extent: bool = False,
        crs: Any | None = None,
        nodata: float = -9999.0,
        render_mode: Literal[
            "horizontal",
            "sloping_corners",
            "sloping_corners_faces",
            "sloping_corners_faces_shallow",
        ] = "sloping_corners",
        depth_min: float | None = 0.001,
        vel_min: float | None = 0.0001,
        vel_weight_method: Literal[
            "area_weighted", "length_weighted", "flow_ratio"
        ] = "area_weighted",
        vel_wse_method: Literal["average", "sloped"] = "sloped",
        vel_interp_method: Literal[
            "flat_cell_center",
            "triangle_blend", "face_idw", "face_gradient",
            "facepoint_blend", "scatter_interp", "scatter_interp2",
        ] = "scatter_interp2",
        scatter_interp_method: Literal["nearest", "linear", "cubic"] = "linear",
        fix_triangulation: bool = True,
        clip_to_perimeter: bool = True,
    ) -> Path | rasterio.io.DatasetReader:
        """Interpolate a field to a GeoTIFF using mesh-conforming triangulation.

        Uses :func:`~raspy.geo.raster.mesh_to_wse_raster` which sub-divides each
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
            Minimum speed (m/s).  Cells whose WLS speed is below this
            threshold are excluded from the WSE wet-extent render and from
            the final velocity output.  Default ``0.001``.
        vel_weight_method:
            Velocity reconstruction scheme passed to
            :meth:`cell_velocity_vectors`.
        vel_wse_method:
            Face WSE interpolation method passed to
            :meth:`cell_velocity_vectors`.
        render_mode:
            Water-surface rendering mode — ``"sloping_corners"`` (default),
            ``"horizontal"``, ``"sloping_corners_faces"``, or
            ``"sloping_corners_faces_shallow"``.  See
            :func:`~raspy.geo.raster.mesh_to_wse_raster` for full description.
        vel_interp_method:
            Intra-cell velocity interpolation method for ``"cell_velocity"``
            and ``"cell_speed"``:

            ``"flat_cell_center"`` — paint the flat WLS cell-centre velocity
            over all pixels inside the cell (fastest; no spatial interpolation).

            ``"triangle_blend"`` *(default)* — barycentric blend of WLS
            cell-centre velocity and reconstructed face velocity within each
            fan-triangle.

            ``"face_idw"`` — inverse-distance-weighted average of all
            face-midpoint 2D velocities within the owning cell.

            ``"face_gradient"`` — least-squares linear gradient fit inside
            each cell from face-midpoint velocities.

            ``"facepoint_blend"`` — average face velocities onto each polygon
            vertex (facepoint), then full 3-vertex barycentric interpolation
            within each fan-triangle.  C0-continuous across all cell faces;
            best choice for smooth flow-arrow rendering.

            ``"scatter_interp"`` — global ``scipy.griddata`` over wet cell
            centres and wet face midpoints.

            ``"scatter_interp2"`` — same but face midpoints only; avoids
            discontinuities originating at cell centres.
        scatter_interp_method:
            ``scipy.interpolate.griddata`` *method* used by
            ``"scatter_interp"`` and ``"scatter_interp2"``.  One of
            ``"nearest"``, ``"linear"`` *(default)*, ``"cubic"``.  Ignored
            for all other *vel_interp_method* values.
        fix_triangulation:
            When ``True`` (default), deduplicate coincident mesh vertices and
            remove zero-area triangles before building the triangulation.
            Prevents ``RuntimeError: Triangulation is invalid`` on meshes with
            unusual cell geometry.  Set ``False`` to skip on large, clean meshes.
        clip_to_perimeter:
            When ``True``, the output extent is restricted to the bounding
            box of the 2-D flow area boundary polygon (``FlowArea.perimeter``)
            snapped outward to the nearest reference pixel boundaries, and
            pixels outside the polygon are set to *nodata*.  Useful when
            malformed boundary cells cause KDTree noise outside the model
            domain.  **Mutually exclusive with** ``snap_to_reference_extent``
            — a ``ValueError`` is raised if both are ``True``.
            Default ``False``.
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
        if clip_to_perimeter and snap_to_reference_extent:
            raise ValueError(
                "clip_to_perimeter and snap_to_reference_extent are mutually "
                "exclusive: clip_to_perimeter restricts the output to the "
                "perimeter bounding box, while snap_to_reference_extent expands "
                "it to the full reference raster extent.  Set "
                "snap_to_reference_extent=False when using clip_to_perimeter=True."
            )

        # ── 1. Resolve values array ────────────────────────────────────
        if variable == "depth":
            if timestep is None:
                wse_values = self.max_water_surface["value"].to_numpy()
                depth_at_cells = self.max_depth()["value"].to_numpy()
            else:
                wse_values = np.array(self.water_surface[timestep, : self.n_cells])
                depth_at_cells = self.depth(timestep)
            # Pre-mask dry cells by depth so mesh_to_wse_raster sees NaN at those
            # cell centres and excludes the corresponding triangles.
            cell_wse = wse_values.copy()
            if depth_min is not None:
                cell_wse[depth_at_cells < depth_min] = np.nan
        elif variable == "water_surface":
            if timestep is None:
                values = self.max_water_surface["value"].to_numpy()
                depth_at_cells = self.max_depth()["value"].to_numpy()
            else:
                values = np.array(self.water_surface[timestep, : self.n_cells])
                depth_at_cells = self.depth(timestep)
            if depth_min is not None:
                values[depth_at_cells < depth_min] = np.nan
        elif variable in ("cell_speed", "cell_velocity"):
            # Compute WLS velocity vectors and cell WSE for the velocity raster.
            # mesh_to_velocity_raster renders WSE to determine wet extent, then
            # assigns velocity per-cell (no spatial interpolation across cells).
            cell_vel_wse = np.array(self.water_surface[timestep, : self.n_cells])
            cell_vel_vecs = self.cell_velocity_vectors(
                timestep, method=vel_weight_method, wse_interp=vel_wse_method
            )
        else:
            raise ValueError(f"Unknown variable: {variable!r}")

        # ── 2. Default cell size ───────────────────────────────────────
        resolved_cell_size = cell_size
        no_grid = reference_transform is None and reference_raster is None
        if cell_size is None and no_grid:
            resolved_cell_size = float(np.median(self.face_normals[:, 2]))

        # ── 2b. Facepoint WSE for sloping render ───────────────────────
        # Compute facepoint values from the dry-masked cell WSE so
        # mesh_to_wse_raster receives them for the griddata sloping path.
        # For velocity variables the vel_min mask is applied first.
        _fp_wse: np.ndarray | None = None
        _fp_wse_vel: np.ndarray | None = None
        _fc_wse: np.ndarray | None = None
        _fc_wse_vel: np.ndarray | None = None
        _use_facecenters = render_mode in (
            "sloping_corners_faces", "sloping_corners_faces_shallow"
        )
        if render_mode != "horizontal":
            if variable == "water_surface":
                _fp_wse = self.wse_at_facepoints(values)
                if _use_facecenters:
                    _fc_wse = self.wse_at_facecentroids(values)
            elif variable == "depth":
                _fp_wse = self.wse_at_facepoints(cell_wse)
                if _use_facecenters:
                    _fc_wse = self.wse_at_facecentroids(cell_wse)
            elif variable in ("cell_velocity", "cell_speed"):
                _vel_wse_masked = cell_vel_wse.copy()
                if vel_min is not None:
                    with np.errstate(invalid="ignore"):
                        _speed = np.linalg.norm(cell_vel_vecs, axis=1)
                    _vel_wse_masked[_speed < vel_min] = np.nan
                _fp_wse_vel = self.wse_at_facepoints(_vel_wse_masked)
                if _use_facecenters:
                    _fc_wse_vel = self.wse_at_facecentroids(_vel_wse_masked)

        # ── 3. Common mesh topology keyword arguments ──────────────────
        _cfi, _cfv = self.cell_face_info  # property returns (info, values) tuple
        # When clipping to the perimeter, use the perimeter polygon bounding
        # box as the output extent.  _tight_pixel_bounds (inside mesh_to_wse_raster)
        # snaps this bbox outward to reference pixel boundaries, preserving
        # grid alignment while keeping the extent compact and consistent with
        # the polygon mask applied after.
        _perimeter_bbox: tuple[float, float, float, float] | None = None
        if clip_to_perimeter:
            _perim = self.perimeter  # (n_pts, 2)
            _perimeter_bbox = (
                float(_perim[:, 0].min()), float(_perim[:, 1].min()),
                float(_perim[:, 0].max()), float(_perim[:, 1].max()),
            )
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
            render_mode=render_mode,
            fix_triangulation=fix_triangulation,
            extent_bbox=_perimeter_bbox,
            facepoint_values=_fp_wse,
            scatter_interp_method=scatter_interp_method,
            cell_polygons=self.cell_polygons,
            facecenter_coordinates=(
                self.face_centroids if _use_facecenters else None
            ),
            facecenter_values=_fc_wse,
            face_min_elevation=(
                self.face_min_elevation
                if render_mode == "sloping_corners_faces_shallow" else None
            ),
        )
        _cell_facepoint_indexes = self.cell_facepoint_indexes
        # Exclude facecenter_coordinates: not a parameter of mesh_to_velocity_raster
        # (face_centroids serves that role there).  Override facepoint_values and
        # facecenter_values with the vel_min-masked versions for the velocity render.
        _vel_mesh_kw = {k: v for k, v in mesh_kw.items()
                        if k != "facecenter_coordinates"}
        _vel_mesh_kw["facepoint_values"] = _fp_wse_vel
        _vel_mesh_kw["facecenter_values"] = _fc_wse_vel

        # ── 4. Delegate to mesh_to_wse_raster / mesh_to_velocity_raster ──────
        if variable == "depth":
            wse_ds = _raster.mesh_to_wse_raster(
                **mesh_kw,
                cell_values=cell_wse,
                output_path=None,   # in-memory; depth written after DEM subtraction
                min_value=None,     # dry cells already NaN-masked above
                min_above_ref=depth_min,
            )
            depth_ds = _raster._depth_from_wse_and_dem(
                wse_ds, reference_raster, nodata,
                None if clip_to_perimeter else output_path,
                min_value=depth_min,
            )
            wse_ds.close()
            if clip_to_perimeter:
                result = _raster._mask_outside_polygon(
                    depth_ds, self.perimeter, nodata, output_path
                )
                depth_ds.close()
                return result
            return depth_ds

        elif variable == "cell_velocity":
            _face_vel_arr = (
                None if vel_interp_method == "flat_cell_center"
                else np.array(self.face_velocity[timestep, :])
            )
            vel_ds = _raster.mesh_to_velocity_raster(
                **_vel_mesh_kw,
                cell_wse=cell_vel_wse,
                cell_velocity=cell_vel_vecs,
                output_path=None if clip_to_perimeter else output_path,
                vel_min=vel_min,
                depth_min=depth_min,
                cell_facepoint_indexes=_cell_facepoint_indexes,
                method=vel_interp_method,
                face_normals=self.face_normals,
                face_vel=_face_vel_arr,
                face_centroids=self.face_centroids,
            )
            if clip_to_perimeter:
                result = _raster._mask_outside_polygon(
                    vel_ds, self.perimeter, nodata, output_path
                )
                vel_ds.close()
                return result
            return vel_ds

        elif variable == "cell_speed":
            _face_vel_arr = (
                None if vel_interp_method == "flat_cell_center"
                else np.array(self.face_velocity[timestep, :])
            )
            vel_ds = _raster.mesh_to_velocity_raster(
                **_vel_mesh_kw,
                cell_wse=cell_vel_wse,
                cell_velocity=cell_vel_vecs,
                output_path=None,
                vel_min=vel_min,
                depth_min=depth_min,
                cell_facepoint_indexes=_cell_facepoint_indexes,
                method=vel_interp_method,
                face_normals=self.face_normals,
                face_vel=_face_vel_arr,
                face_centroids=self.face_centroids,
            )
            speed_ds = _raster._velocity_raster_to_speed(
                vel_ds, None if clip_to_perimeter else output_path, nodata
            )
            if clip_to_perimeter:
                result = _raster._mask_outside_polygon(
                    speed_ds, self.perimeter, nodata, output_path
                )
                speed_ds.close()
                return result
            return speed_ds

        else:
            # water_surface: min_above_ref controls the wet/dry threshold relative
            # to the DEM (when reference_raster is given); no scalar min_value needed.
            wse_ds = _raster.mesh_to_wse_raster(
                **mesh_kw,
                cell_values=values,
                output_path=None if clip_to_perimeter else output_path,
                min_value=None,
                min_above_ref=depth_min,
            )
            if clip_to_perimeter:
                result = _raster._mask_outside_polygon(
                    wse_ds, self.perimeter, nodata, output_path
                )
                wse_ds.close()
                return result
            return wse_ds

    @timed(logging.INFO)
    def export_hydraulic_rasters(
        self,
        timestep: int,
        reference_raster: str | Path,
        *,
        wse_path: str | Path | None = None,
        depth_path: str | Path | None = None,
        speed_path: str | Path | None = None,
        snap_to_reference_extent: bool = False,
        nodata: float = -9999.0,
        render_mode: Literal[
            "horizontal",
            "sloping_corners",
            "sloping_corners_faces",
            "sloping_corners_faces_shallow",
        ] = "horizontal",
        depth_min: float | None = 0.001,
        vel_min: float | None = 0.0001,
        vel_weight_method: Literal[
            "area_weighted", "length_weighted", "flow_ratio"
        ] = "area_weighted",
        vel_wse_method: Literal["average", "sloped"] = "sloped",
        vel_interp_method: Literal[
            "flat_cell_center",
            "triangle_blend", "face_idw", "face_gradient",
            "facepoint_blend", "scatter_interp", "scatter_interp2",
        ] = "scatter_interp2",
        scatter_interp_method: Literal["nearest", "linear", "cubic"] = "linear",
        fix_triangulation: bool = True,
        clip_to_perimeter: bool = True,
    ) -> dict[str, Path | rasterio.io.DatasetReader]:
        """Export water-surface elevation, depth, and speed rasters in one pass.

        Computes all three hydraulic output rasters for a single timestep while
        sharing intermediate results: the ``water_surface`` HDF dataset is read
        once, ``cell_velocity_vectors()`` (WLS reconstruction) is called once,
        and the ``face_velocity`` HDF dataset is read once.  The in-memory WSE
        raster produced for the WSE output is reused directly for the depth
        DEM-subtraction step.

        Parameters
        ----------
        timestep:
            0-based time index.  Required — all three outputs need a specific
            timestep (speed has no max-value fallback).
        reference_raster:
            Path to the terrain DEM GeoTIFF.  Required — used to derive depth
            (WSE minus DEM) and to inherit the output CRS and transform.
        wse_path:
            Destination ``.tif`` for the water-surface elevation raster.
            ``None`` returns an open in-memory ``rasterio.DatasetReader``.
        depth_path:
            Destination ``.tif`` for the depth raster.
            ``None`` returns an open in-memory ``rasterio.DatasetReader``.
        speed_path:
            Destination ``.tif`` for the cell speed raster.
            ``None`` returns an open in-memory ``rasterio.DatasetReader``.
        snap_to_reference_extent:
            Extend the output to the full extent of *reference_raster*
            (default ``True``).
        nodata:
            Fill value for pixels outside the wet mesh.
        render_mode:
            Water-surface rendering mode — ``"horizontal"`` (default),
            ``"sloping_corners"``, ``"sloping_corners_faces"``, or
            ``"sloping_corners_faces_shallow"``.
        depth_min:
            Minimum water depth.  Cells shallower than this are excluded from
            WSE interpolation; output depth pixels below this are set to
            *nodata*.
        vel_min:
            Minimum speed threshold for velocity reconstruction and wet-extent
            classification.
        vel_weight_method:
            Velocity reconstruction scheme passed to
            :meth:`cell_velocity_vectors`.
        vel_wse_method:
            Face WSE interpolation method passed to
            :meth:`cell_velocity_vectors`.
        vel_interp_method:
            Intra-cell velocity interpolation method.  See
            :meth:`export_raster` for full description of each option.
        scatter_interp_method:
            ``scipy.interpolate.griddata`` *method* used by
            ``"scatter_interp"`` / ``"scatter_interp2"``.
        fix_triangulation:
            When ``True`` (default), deduplicate coincident mesh vertices and
            remove zero-area triangles before building the triangulation.
            Prevents ``RuntimeError: Triangulation is invalid`` on meshes with
            unusual cell geometry.  Set ``False`` to skip on large, clean meshes.
        clip_to_perimeter:
            When ``True``, the output extent for all three rasters is
            restricted to the bounding box of the 2-D flow area boundary
            polygon snapped outward to the nearest reference pixel boundaries,
            and pixels outside the polygon are set to *nodata*.  **Mutually
            exclusive with** ``snap_to_reference_extent`` — a ``ValueError``
            is raised if both are ``True``.  Default ``False``.
        Returns
        -------
        dict with keys ``"water_surface"``, ``"depth"``, ``"speed"``.
        Each value is the written ``Path`` (when the corresponding ``*_path``
        argument is given) or an open in-memory ``rasterio.DatasetReader``
        (when the argument is ``None``).  The caller must close any in-memory
        datasets.

        Raises
        ------
        ImportError
            If ``rasterio`` or ``matplotlib`` are not installed.
        ValueError
            If both *clip_to_perimeter* and *snap_to_reference_extent* are
            ``True``.
        """
        from raspy.geo import raster as _raster  # deferred — geo not required

        # ── 0. Guards ──────────────────────────────────────────────────
        if clip_to_perimeter and snap_to_reference_extent:
            raise ValueError(
                "clip_to_perimeter and snap_to_reference_extent are mutually "
                "exclusive: clip_to_perimeter restricts the output to the "
                "perimeter bounding box, while snap_to_reference_extent expands "
                "it to the full reference raster extent.  Set "
                "snap_to_reference_extent=False when using clip_to_perimeter=True."
            )

        # ── 2. Read HDF data once ──────────────────────────────────────
        wse_values = np.array(self.water_surface[timestep, : self.n_cells])
        depth_at_cells = self.depth(timestep)
        face_vel_arr = (
            None if vel_interp_method == "flat_cell_center"
            else np.array(self.face_velocity[timestep, :])
        )

        # ── 3. WLS velocity vectors once ──────────────────────────────
        cell_vel_vecs = self.cell_velocity_vectors(
            timestep, method=vel_weight_method, wse_interp=vel_wse_method
        )

        # ── 3. Shared mesh topology kwargs ────────────────────────────
        _cfi, _cfv = self.cell_face_info
        _perimeter_bbox: tuple[float, float, float, float] | None = None
        if clip_to_perimeter:
            _perim = self.perimeter  # (n_pts, 2)
            _perimeter_bbox = (
                float(_perim[:, 0].min()), float(_perim[:, 1].min()),
                float(_perim[:, 0].max()), float(_perim[:, 1].max()),
            )
        mesh_kw: dict[str, Any] = dict(
            cell_centers=self.cell_centers,
            facepoint_coordinates=self.facepoint_coordinates,
            face_facepoint_indexes=self.face_facepoint_indexes,
            face_cell_indexes=self.face_cell_indexes,
            cell_face_info=_cfi,
            cell_face_values=_cfv,
            reference_raster=reference_raster,
            nodata=nodata,
            snap_to_reference_extent=snap_to_reference_extent,
            render_mode=render_mode,
            fix_triangulation=fix_triangulation,
            extent_bbox=_perimeter_bbox,
            face_min_elevation=(
                self.face_min_elevation
                if render_mode == "sloping_corners_faces_shallow" else None
            ),
        )

        # ── 5. WSE raster in-memory (shared for WSE output + depth) ───
        logging.info("Building water-surface raster (shared for depth output)...")

        cell_wse_masked = wse_values.copy()
        if depth_min is not None:
            cell_wse_masked[depth_at_cells < depth_min] = np.nan

        # Facepoint WSE for sloping render — computed once for WSE/depth;
        # velocity uses a separate vel_min-masked array.
        _fp_wse: np.ndarray | None = None
        _fp_wse_vel: np.ndarray | None = None
        _fc_wse: np.ndarray | None = None
        _fc_wse_vel: np.ndarray | None = None
        _use_facecenters = render_mode in (
            "sloping_corners_faces", "sloping_corners_faces_shallow"
        )
        if render_mode != "horizontal":
            _fp_wse = self.wse_at_facepoints(cell_wse_masked)
            if _use_facecenters:
                _fc_wse = self.wse_at_facecentroids(cell_wse_masked)
            _vel_wse_masked = wse_values.copy()
            if vel_min is not None:
                with np.errstate(invalid="ignore"):
                    _speed = np.linalg.norm(cell_vel_vecs, axis=1)
                _vel_wse_masked[_speed < vel_min] = np.nan
            _fp_wse_vel = self.wse_at_facepoints(_vel_wse_masked)
            if _use_facecenters:
                _fc_wse_vel = self.wse_at_facecentroids(_vel_wse_masked)

        mesh_kw["facepoint_values"] = _fp_wse
        mesh_kw["scatter_interp_method"] = scatter_interp_method
        mesh_kw["cell_polygons"] = self.cell_polygons
        mesh_kw["facecenter_coordinates"] = (
            self.face_centroids if _use_facecenters else None
        )
        mesh_kw["facecenter_values"] = _fc_wse
        _cell_facepoint_indexes = self.cell_facepoint_indexes
        # Exclude facecenter_coordinates: not a parameter of mesh_to_velocity_raster
        # (face_centroids serves that role there).
        _vel_mesh_kw = {k: v for k, v in mesh_kw.items()
                        if k != "facecenter_coordinates"}

        wse_ds = _raster.mesh_to_wse_raster(
            **mesh_kw,
            cell_values=cell_wse_masked,
            output_path=None,
            min_value=None,
            min_above_ref=depth_min,
        )

        # ── 6. WSE output ──────────────────────────────────────────────
        if clip_to_perimeter:
            wse_result: Path | rasterio.io.DatasetReader = (
                _raster._mask_outside_polygon(wse_ds, self.perimeter, nodata, wse_path)
            )
        elif wse_path is not None:
            wse_result = _raster._write_dataset(wse_ds, wse_path)
        else:
            wse_result = wse_ds

        # ── 7. Depth output ────────────────────────────────────────────
        logging.info("Building depth raster from WSE raster and DEM...")

        depth_ds = _raster._depth_from_wse_and_dem(
            wse_ds, reference_raster, nodata,
            None if clip_to_perimeter else depth_path,
            min_value=depth_min,
        )
        if clip_to_perimeter:
            depth_result: Path | rasterio.io.DatasetReader = (
                _raster._mask_outside_polygon(
                    depth_ds, self.perimeter, nodata, depth_path
                )
            )
            depth_ds.close()
        else:
            depth_result = depth_ds

        # ── 8. Speed output ────────────────────────────────────────────
        # Pass wse_ds directly so mesh_to_velocity_raster reuses the
        # already-rendered wet extent instead of re-running the WSE render.
        logging.info("Building velocity raster for speed output...")

        vel_ds = _raster.mesh_to_velocity_raster(
            **_vel_mesh_kw,
            wse_raster=wse_ds,
            cell_wse=wse_values,
            cell_velocity=cell_vel_vecs,
            output_path=None,
            vel_min=vel_min,
            depth_min=depth_min,
            cell_facepoint_indexes=_cell_facepoint_indexes,
            method=vel_interp_method,
            face_normals=self.face_normals,
            face_vel=face_vel_arr,
            face_centroids=self.face_centroids,
        )

        # Close shared in-memory WSE dataset now that all consumers are done.
        # When clip_to_perimeter=True or wse_path is set, wse_result is a
        # separate object so wse_ds can always be closed.  Otherwise
        # wse_result IS wse_ds and must stay open for the caller.
        if clip_to_perimeter or wse_path is not None:
            wse_ds.close()
        speed_ds = _raster._velocity_raster_to_speed(
            vel_ds, None if clip_to_perimeter else speed_path, nodata
        )
        if clip_to_perimeter:
            speed_result: Path | rasterio.io.DatasetReader = (
                _raster._mask_outside_polygon(
                    speed_ds, self.perimeter, nodata, speed_path
                )
            )
            speed_ds.close()
        else:
            speed_result = speed_ds

        return {
            "water_surface": wse_result,
            "depth": depth_result,
            "speed": speed_result,
        }

    def debug_raster_export(
        self,
        timestep: int | None = None,
        *,
        render_mode: Literal[
            "horizontal",
            "sloping_corners",
            "sloping_corners_faces",
            "sloping_corners_faces_shallow",
        ] = "horizontal",
        depth_min: float | None = 0.001,
        check_boundary: bool = True,
        verbose: bool = True,
    ) -> dict:
        """Diagnose why ``export_raster`` / ``export_hydraulic_rasters`` may fail.

        Runs two complementary checks and combines them into one report:

        1. **Mesh geometry** (:func:`~raspy.geo.mesh_validation.check_mesh_cells`)
           — validates every cell against the HEC-RAS rules (convexity, face
           count, duplicate points, cell centre location).
        2. **Triangulation probe** — actually builds the fan-triangulation used
           by :func:`~raspy.geo.raster.mesh_to_wse_raster` and attempts to
           initialise ``matplotlib``'s ``TrapezoidMapTriFinder``, both with and
           without the ``fix_triangulation`` pre-processing step, so you can see
           whether the fix resolves the problem or whether the KDTree fallback
           will be needed.

        Parameters
        ----------
        timestep :
            0-based time index used to build the dry-cell mask (cells whose
            WSE is below *depth_min* are excluded from the triangulation probe).
            ``None`` uses the max-WSE summary and always treats every cell as wet
            (conservative: more triangles are included).
        render_mode :
            Render mode to probe.  ``"horizontal"`` (default) is the mode most
            likely to trigger the triangulation error.
        depth_min :
            Depth threshold for the dry-cell mask (same default as
            :meth:`export_raster`).
        check_boundary :
            Pass ``True`` (default) to include the flow-area perimeter check
            (rule 5) in the mesh validation step.
        verbose :
            Print the report to stdout as well as returning it.

        Returns
        -------
        dict with keys:

        ``mesh_report``
            Full output of :func:`~raspy.geo.mesh_validation.check_mesh_cells`.
        ``triangulation``
            Sub-dict describing the triangulation probe result:

            - ``n_triangles_raw`` — triangles before fix.
            - ``n_triangles_after_fix`` — triangles after dedup + zero-area filter.
            - ``n_removed_by_fix`` — triangles removed.
            - ``trifinder_without_fix`` — ``"ok"`` / ``"failed"`` / ``"not_tested"``.
            - ``trifinder_with_fix`` — ``"ok"`` / ``"failed"``.
            - ``fallback_needed`` — ``True`` when even the fixed triangulation
              cannot initialise the trifinder (KDTree fallback will be used).
        """
        import matplotlib.tri as mtri

        from raspy.geo.mesh_validation import print_mesh_report

        # ── Step 1: Mesh geometry validation ──────────────────────────────
        mesh_report = self.check_cells(check_boundary=check_boundary)

        if verbose:
            print(f"=== debug_raster_export: {self.name} ===\n")
            print_mesh_report(mesh_report)

        # ── Step 2: Build dry-cell mask ───────────────────────────────────
        if timestep is not None:
            wse = np.array(self.water_surface[timestep, : self.n_cells])
            if depth_min is not None:
                depth = self.depth(timestep)
                dry_mask = depth < depth_min
            else:
                dry_mask = np.zeros(self.n_cells, dtype=bool)
        else:
            wse = self.max_water_surface["value"].to_numpy()
            dry_mask = np.zeros(self.n_cells, dtype=bool)  # treat all as wet

        cv = wse.copy()
        cv[dry_mask] = np.nan
        n_wet = int((~dry_mask).sum())

        # ── Step 3: Build the fan-triangulation ───────────────────────────
        cfi, cfv = self.cell_face_info
        cfi = np.asarray(cfi, dtype=np.int64)[: self.n_cells]
        cfv = np.asarray(cfv, dtype=np.int64)
        fp_coords = np.asarray(self.facepoint_coordinates, dtype=np.float64)
        fp_idx = np.asarray(self.face_facepoint_indexes, dtype=np.int64)
        cc = np.asarray(self.cell_centers, dtype=np.float64)
        n_cells = self.n_cells

        all_pts = np.vstack([cc, fp_coords])

        counts = cfi[:, 1]
        starts = cfi[:, 0]
        total = int(counts.sum())
        idx_step = np.ones(total, dtype=np.int64)
        if n_cells > 1:
            bp = np.cumsum(counts[:-1])
            idx_step[bp] = starts[1:] - starts[:-1] - counts[:-1] + 1
        idx_step[0] = starts[0]
        entry_indices = np.cumsum(idx_step)

        face_idx_arr = cfv[entry_indices, 0]
        cell_for_entry = np.repeat(np.arange(n_cells, dtype=np.int64), counts)
        fp0 = fp_idx[face_idx_arr, 0]
        fp1 = fp_idx[face_idx_arr, 1]
        valid = fp0 != fp1
        triangles_raw = np.column_stack([
            cell_for_entry[valid],
            n_cells + fp0[valid],
            n_cells + fp1[valid],
        ])
        tri_to_cell_raw = cell_for_entry[valid]

        if render_mode == "horizontal":
            tri_mask_raw = dry_mask[tri_to_cell_raw]
        else:
            tri_mask_raw = np.zeros(len(triangles_raw), dtype=bool)

        n_raw = len(triangles_raw)

        # ── Step 4: Probe trifinder WITHOUT fix ───────────────────────────
        try:
            triang_raw = mtri.Triangulation(
                all_pts[:, 0], all_pts[:, 1], triangles_raw
            )
            triang_raw.set_mask(tri_mask_raw)
            triang_raw.get_trifinder()
            result_without_fix: str = "ok"
        except RuntimeError:
            result_without_fix = "failed"
        except Exception as exc:
            result_without_fix = f"error: {exc}"

        # ── Step 5: Apply fix — track exactly what is removed and why ────────
        _, _uniq_idx, _inv_idx = np.unique(
            all_pts, axis=0, return_index=True, return_inverse=True
        )
        all_pts_fixed = all_pts[_uniq_idx]
        triangles_fixed = _inv_idx[triangles_raw]

        t0, t1, t2 = triangles_fixed[:, 0], triangles_fixed[:, 1], triangles_fixed[:, 2]
        nondegen = (t0 != t1) & (t1 != t2) & (t0 != t2)
        p0, p1, p2 = all_pts_fixed[t0], all_pts_fixed[t1], all_pts_fixed[t2]
        cross_z = (
            (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
            - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
        )
        nonzero = cross_z != 0.0
        keep = nondegen & nonzero

        # Classify removed triangles.
        removed_degenerate = ~nondegen          # collapsed to point/line after dedup
        removed_zero_area  = nondegen & ~nonzero  # collinear but distinct vertices

        def _removed_detail(mask: np.ndarray, reason: str) -> list[dict]:
            """Build a list of location records for removed triangles."""
            records = []
            for i in np.where(mask)[0]:
                cell_idx = int(tri_to_cell_raw[i])
                cx_, cy_ = float(cc[cell_idx, 0]), float(cc[cell_idx, 1])
                records.append({
                    "reason": reason,
                    "cell_idx": cell_idx,
                    "cell_x": cx_,
                    "cell_y": cy_,
                    # original (pre-dedup) vertex indices into all_pts
                    "vertex_indices": triangles_raw[i].tolist(),
                    "vertex_coords": all_pts[triangles_raw[i]].tolist(),
                })
            return records

        removed_locations = (
            _removed_detail(removed_degenerate, "duplicate_vertex")
            + _removed_detail(removed_zero_area,  "zero_area")
        )

        # Duplicate facepoints (global — cause duplicate vertices in all_pts).
        dup_fp_locations: list[dict] = []
        for fp_a, fp_b in mesh_report["duplicate_facepoints"]:
            coord = fp_coords[fp_a].tolist()
            dup_fp_locations.append({
                "facepoint_a": fp_a,
                "facepoint_b": fp_b,
                "x": coord[0],
                "y": coord[1],
            })

        # Duplicate cell centres (global).
        dup_cc_locations: list[dict] = []
        for cc_a, cc_b in mesh_report["duplicate_cell_centers"]:
            coord = cc[cc_a].tolist()
            dup_cc_locations.append({
                "cell_a": cc_a,
                "cell_b": cc_b,
                "x": coord[0],
                "y": coord[1],
            })

        triangles_fixed = triangles_fixed[keep]
        tri_mask_fixed = tri_mask_raw[keep]
        n_fixed = len(triangles_fixed)
        n_removed = n_raw - n_fixed

        try:
            triang_fixed = mtri.Triangulation(
                all_pts_fixed[:, 0], all_pts_fixed[:, 1], triangles_fixed
            )
            triang_fixed.set_mask(tri_mask_fixed)
            triang_fixed.get_trifinder()
            result_with_fix: str = "ok"
        except RuntimeError:
            result_with_fix = "failed"
        except Exception as exc:
            result_with_fix = f"error: {exc}"

        fallback_needed = result_with_fix != "ok"

        tri_report = {
            "render_mode": render_mode,
            "n_cells_total": n_cells,
            "n_cells_wet": n_wet,
            "n_triangles_raw": n_raw,
            "n_triangles_after_fix": n_fixed,
            "n_removed_by_fix": n_removed,
            "n_removed_duplicate_vertex": int(removed_degenerate.sum()),
            "n_removed_zero_area": int(removed_zero_area.sum()),
            "trifinder_without_fix": result_without_fix,
            "trifinder_with_fix": result_with_fix,
            "fallback_needed": fallback_needed,
            "removed_triangles": removed_locations,
            "duplicate_facepoints": dup_fp_locations,
            "duplicate_cell_centers": dup_cc_locations,
        }

        if verbose:
            print()
            print("Triangulation probe")
            print(f"  render_mode            : {render_mode}")
            print(f"  wet cells              : {n_wet:,} / {n_cells:,}")
            print(f"  triangles (raw)        : {n_raw:,}")
            print(f"  triangles (after fix)  : {n_fixed:,}  ({n_removed:,} removed)")
            if n_removed:
                print(f"    duplicate vertex     : {int(removed_degenerate.sum()):,}")
                print(f"    zero area            : {int(removed_zero_area.sum()):,}")
            print(f"  trifinder without fix  : {result_without_fix}")
            print(f"  trifinder with fix     : {result_with_fix}")
            if fallback_needed:
                print("  *** KDTree fallback will be used for horizontal rendering ***")
            else:
                print("  fix_triangulation=True is sufficient — no fallback needed")

            if dup_fp_locations:
                print(f"\n  Duplicate facepoints ({len(dup_fp_locations)}):")
                for d in dup_fp_locations:
                    print(
                        f"    fp {d['facepoint_a']:>8,} == fp {d['facepoint_b']:>8,}"
                        f"  @ ({d['x']:.3f}, {d['y']:.3f})"
                    )

            if dup_cc_locations:
                print(f"\n  Duplicate cell centres ({len(dup_cc_locations)}):")
                for d in dup_cc_locations:
                    print(
                        f"    cell {d['cell_a']:>8,} == cell {d['cell_b']:>8,}"
                        f"  @ ({d['x']:.3f}, {d['y']:.3f})"
                    )

            if removed_locations:
                print(f"\n  Removed triangles ({len(removed_locations)}):")
                for r in removed_locations:
                    print(
                        f"    cell {r['cell_idx']:>8,}"
                        f"  centre ({r['cell_x']:.3f}, {r['cell_y']:.3f})"
                        f"  reason: {r['reason']}"
                    )

        return {
            "mesh_report": mesh_report,
            "triangulation": tri_report,
        }

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
        cell_wse = np.array(self.water_surface[timestep, : self.n_cells])
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
