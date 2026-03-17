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
    # Computed results - pure numpy, no geo dependency
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
        vel_wse_method: Literal["average", "sloped", "max"] = "sloped",
        vel_interp_method: Literal[
            "flat_cell_center",
            "triangle_blend", "face_idw", "face_gradient",
            "facepoint_blend", "scatter_cell_face", "scatter_face",
            "scatter_corners", "scatter_corners_face", "scatter_cell_corners_face",
            "scatter_face_normal",
        ] = "scatter_face",
        scatter_interp_method: Literal["nearest", "linear", "cubic"] = "linear",
        fix_triangulation: bool = True,
        clip_to_perimeter: bool = True,
        face_velocity_location: Literal[
            "centroid", "normal_intercept"
        ] = "normal_intercept",
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
            ``"water_surface"`` - water-surface elevation.
            ``"depth"``         - water depth (requires *reference_raster*):
                                  WSE is interpolated then the DEM pixel value
                                  is subtracted; negative depths are clamped
                                  to 0.
            ``"cell_speed"``    - WLS-reconstructed velocity magnitude.
            ``"cell_velocity"`` - WLS-reconstructed velocity vector
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
            extent (default ``False``).
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
            the final velocity output.  Default ``0.0001``.
        vel_weight_method:
            Velocity reconstruction scheme passed to
            :meth:`cell_velocity_vectors`.
        vel_wse_method:
            Face WSE interpolation method passed to
            :meth:`cell_velocity_vectors`.
        render_mode:
            Water-surface rendering mode - ``"sloping_corners"`` (default),
            ``"horizontal"``, ``"sloping_corners_faces"``, or
            ``"sloping_corners_faces_shallow"``.  See
            :func:`~raspy.geo.raster.mesh_to_wse_raster` for full description.
        vel_interp_method:
            Intra-cell velocity interpolation method for ``"cell_velocity"``
            and ``"cell_speed"``:

            ``"flat_cell_center"`` - paint the flat WLS cell-centre velocity
            over all pixels inside the cell (fastest; no spatial interpolation).

            ``"triangle_blend"`` - barycentric blend of WLS
            cell-centre velocity and reconstructed face velocity within each
            fan-triangle.

            ``"face_idw"`` - inverse-distance-weighted average of all
            face-midpoint 2D velocities within the owning cell.

            ``"face_gradient"`` - least-squares linear gradient fit inside
            each cell from face-midpoint velocities.

            ``"facepoint_blend"`` - average face velocities onto each polygon
            vertex (facepoint), then full 3-vertex barycentric interpolation
            within each fan-triangle.  C0-continuous across all cell faces;
            best choice for smooth flow-arrow rendering.

            ``"scatter_cell_face"`` - global ``scipy.griddata`` over wet cell
            centres and wet face midpoints.

            ``"scatter_face"`` *(default)* - same but face midpoints only;
            avoids discontinuities originating at cell centres.

            ``"scatter_corners"`` - global ``scipy.griddata`` over wet mesh
            corners; velocity at each corner is the mean of the double-C
            stencil face velocities from all adjacent wet faces.

            ``"scatter_corners_face"`` - combined scatter from wet mesh
            corners (double-C stencil mean) and wet face midpoints (2D face
            velocity).

            ``"scatter_cell_corners_face"`` - maximum-density scatter combining
            wet cell centres (WLS), wet mesh corners (double-C averaged), and
            wet face midpoints (double-C); union of all other scatter sources.

            ``"scatter_face_normal"`` - global ``scipy.griddata`` over wet
            face midpoints using only the stored normal component (``vn * n_hat``).
            No double-C tangential estimation; cell WLS velocities do not
            influence the result.  Avoids tangential contamination near
            wet/dry boundaries.
        scatter_interp_method:
            ``scipy.interpolate.griddata`` *method* used by
            ``"scatter_cell_face"``, ``"scatter_face"``, ``"scatter_corners"``,
            ``"scatter_corners_face"``, ``"scatter_cell_corners_face"``, and
            ``"scatter_face_normal"``.
            One of ``"nearest"``, ``"linear"`` *(default)*, ``"cubic"``.
            Ignored for all other *vel_interp_method* values.
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
            - a ``ValueError`` is raised if both are ``True``.
            Default ``True``.
        face_velocity_location:
            Position used as the face normal velocity measurement point when
            ``vel_wse_method="sloped"``.
            ``"normal_intercept"`` (default): where the cell-centre connecting
            line crosses the face polyline.
            ``"centroid"``: geometric centroid of the face polyline.
            Passed to :meth:`cell_velocity_vectors`.
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
        from raspy.geo import raster as _raster  # deferred - geo not required

        logger.info("Exporting hydraulic raster %s at timestep %d", variable, timestep)
        # -- 0. Guards --------------------------------------------------
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

        # -- 1. Resolve values array ------------------------------------
        if variable == "depth":
            if timestep is None:
                wse_values = self.max_water_surface["value"].to_numpy()
                depth_at_cells = (
                    self.max_depth()["value"].to_numpy()
                    if depth_min is not None else None
                )
            else:
                wse_values = np.array(self.water_surface[timestep, : self.n_cells])
                depth_at_cells = self.depth(timestep) if depth_min is not None else None
            # Pre-mask dry cells so mesh_to_wse_raster excludes those triangles.
            cell_wse = (
                np.where(depth_at_cells < depth_min, np.nan, wse_values)
                if depth_min is not None else wse_values
            )
        elif variable == "water_surface":
            if timestep is None:
                values = self.max_water_surface["value"].to_numpy()
                depth_at_cells = (
                    self.max_depth()["value"].to_numpy()
                    if depth_min is not None else None
                )
            else:
                values = np.array(self.water_surface[timestep, : self.n_cells])
                depth_at_cells = self.depth(timestep) if depth_min is not None else None
            if depth_min is not None:
                values = np.where(depth_at_cells < depth_min, np.nan, values)
        elif variable in ("cell_speed", "cell_velocity"):
            # Compute WLS velocity vectors and cell WSE for the velocity raster.
            # mesh_to_velocity_raster renders WSE to determine wet extent, then
            # assigns velocity per-cell (no spatial interpolation across cells).
            cell_vel_wse = np.array(self.water_surface[timestep, : self.n_cells])
            cell_vel_vecs = self.cell_velocity_vectors(
                timestep, method=vel_weight_method, wse_interp=vel_wse_method,
                face_velocity_location=face_velocity_location,
            )
        else:
            raise ValueError(f"Unknown variable: {variable!r}")

        # -- 2. Default cell size ---------------------------------------
        resolved_cell_size = cell_size
        no_grid = reference_transform is None and reference_raster is None
        if cell_size is None and no_grid:
            resolved_cell_size = float(np.median(self.face_normals[:, 2]))

        # -- 2b. Facepoint WSE for sloping render -----------------------
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
                if vel_min is not None:
                    with np.errstate(invalid="ignore"):
                        _speed = np.linalg.norm(cell_vel_vecs[: self.n_cells], axis=1)
                    _vel_wse_masked = np.where(_speed < vel_min, np.nan, cell_vel_wse)
                else:
                    _vel_wse_masked = cell_vel_wse
                _fp_wse_vel = self.wse_at_facepoints(_vel_wse_masked)
                if _use_facecenters:
                    _fc_wse_vel = self.wse_at_facecentroids(_vel_wse_masked)

        # -- 3. Common mesh topology keyword arguments ------------------
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
            facepoint_wse=_fp_wse,
            scatter_interp_method=scatter_interp_method,
            cell_polygons=self.cell_polygons,
            face_centers=(
                self.face_centroids if _use_facecenters else None
            ),
            face_center_wse=_fc_wse,
            face_min_elevation=(
                self.face_min_elevation
                if render_mode == "sloping_corners_faces_shallow" else None
            ),
        )
        # Exclude face_centers: not shared with mesh_to_velocity_raster via mesh_kw
        # (face_centers is passed explicitly in the velocity call).  Override
        # facepoint_wse and face_center_wse with vel_min-masked versions.
        _vel_mesh_kw = {k: v for k, v in mesh_kw.items()
                        if k not in ("face_centers",
                                     "facepoint_wse", "face_center_wse")}
        _vel_mesh_kw["facepoint_wse"] = _fp_wse_vel
        _vel_mesh_kw["face_center_wse"] = _fc_wse_vel

        # -- 4. Delegate to mesh_to_wse_raster / mesh_to_velocity_raster ------
        if variable == "depth":
            wse_ds = _raster.mesh_to_wse_raster(
                **mesh_kw,
                cell_wse=cell_wse,
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
                    depth_ds, _perim, nodata, output_path
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
                method=vel_interp_method,
                face_normals=self.face_normals,
                face_normal_velocity=_face_vel_arr,
                face_centers=self.face_centroids,
                face_velocity_coords=(
                    self.face_normal_intercept
                    if face_velocity_location == "normal_intercept"
                    else self.face_centroids
                ),
            )
            if clip_to_perimeter:
                result = _raster._mask_outside_polygon(
                    vel_ds, _perim, nodata, output_path
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
                method=vel_interp_method,
                face_normals=self.face_normals,
                face_normal_velocity=_face_vel_arr,
                face_centers=self.face_centroids,
                face_velocity_coords=(
                    self.face_normal_intercept
                    if face_velocity_location == "normal_intercept"
                    else self.face_centroids
                ),
            )
            speed_ds = _raster._velocity_raster_to_speed(
                vel_ds, None if clip_to_perimeter else output_path, nodata
            )
            if clip_to_perimeter:
                result = _raster._mask_outside_polygon(
                    speed_ds, _perim, nodata, output_path
                )
                speed_ds.close()
                return result
            return speed_ds

        else:
            # water_surface: min_above_ref controls the wet/dry threshold relative
            # to the DEM (when reference_raster is given); no scalar min_value needed.
            wse_ds = _raster.mesh_to_wse_raster(
                **mesh_kw,
                cell_wse=values,
                output_path=None if clip_to_perimeter else output_path,
                min_value=None,
                min_above_ref=depth_min,
            )
            if clip_to_perimeter:
                result = _raster._mask_outside_polygon(
                    wse_ds, _perim, nodata, output_path
                )
                wse_ds.close()
                return result
            return wse_ds

    @log_call(logging.INFO)
    @timed(logging.INFO)
    def export_raster2(
        self,
        variable: Literal["water_surface", "depth", "speed", "velocity"],
        timestep: int | None = None,
        output_path: str | Path | None = None,
        *,
        reference_raster: str | Path | None = None,
        cell_size: float | None = None,
        crs: Any | None = None,
        nodata: float = -9999.0,
        interp_mode: Literal["flat", "sloping"] = "sloping",
        depth_threshold: float = 0.001,
        clip_to_perimeter: bool = True,
    ) -> Path | rasterio.io.DatasetReader:
        """Rasterize a result variable using the RASMapper-exact algorithm.

        Implements the pixel-perfect pipeline reverse-engineered from
        ``RasMapperLib.dll`` (CLB Engineering, 2026), validated against
        RASMapper VRT exports — median |diff| = 0.000000.  Replaces the
        older :meth:`export_raster` triangulation-based approach for
        ``water_surface``, ``depth``, ``speed``, and ``velocity``.

        Parameters
        ----------
        variable:
            ``"water_surface"`` — water-surface elevation.
            ``"depth"``         — water depth; requires *reference_raster*.
            ``"speed"``         — velocity magnitude.
            ``"velocity"``      — 4-band raster ``[Vx, Vy, speed, dir_deg]``.
        timestep:
            0-based time index.  Pass ``None`` to use the time of maximum
            water-surface elevation (supported for ``"water_surface"`` and
            ``"depth"`` only; raises ``ValueError`` for ``"speed"`` /
            ``"velocity"``).
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
        interp_mode:
            ``"sloping"`` (default) — full RASMapper pipeline with per-pixel
            WSE interpolation and sloped wet/dry masking.
            ``"flat"`` — cell value painted directly over each owned pixel.
        depth_threshold:
            Minimum depth for a pixel to be considered wet (default
            ``0.001``).  Matches ``RASResults.MinWSPlotTolerance``.
        clip_to_perimeter:
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
            If ``variable="speed"`` or ``"velocity"`` and ``timestep=None``.
            If neither *reference_raster* nor *cell_size* is provided.
        """
        from raspy.geo import raster as _raster

        if variable in ("speed", "velocity") and timestep is None:
            raise ValueError(
                "timestep=None is not supported for speed / velocity. "
                "Provide an explicit timestep index."
            )
        if reference_raster is None and cell_size is None:
            # Default: median face length (same heuristic as export_raster)
            cell_size = float(np.median(self.face_normals[:, 2]))

        # ---- Read HDF arrays ------------------------------------------------
        if timestep is None:
            cell_wse = self.max_water_surface["value"].to_numpy()
        else:
            cell_wse = np.array(self.water_surface[timestep, : self.n_cells])

        face_normal_velocity: np.ndarray | None = None
        if variable in ("speed", "velocity"):
            face_normal_velocity = np.array(self.face_velocity[timestep, :])

        cell_face_info, cell_face_values = self.cell_face_info
        fp_face_info, fp_face_values = self.facepoint_face_orientation

        # ---- Delegate to rasmap_raster -------------------------------------
        return _raster.rasmap_raster(
            variable=variable,
            cell_wse=cell_wse,
            cell_min_elevation=self.cell_min_elevation,
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
            reference_raster=reference_raster,
            cell_size=cell_size,
            crs=crs,
            nodata=nodata,
            interp_mode=interp_mode,
            depth_threshold=depth_threshold,
            clip_to_perimeter=clip_to_perimeter,
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
        speed_path: str | Path | None = None,
        snap_to_reference_extent: bool = False,
        nodata: float = -9999.0,
        render_mode: Literal[
            "horizontal",
            "sloping_corners",
            "sloping_corners_faces",
            "sloping_corners_faces_shallow",
        ] = "horizontal",
        depth_min: float | None = 0.0,
        vel_min: float | None = 0.0,
        vel_weight_method: Literal[
            "area_weighted", "length_weighted", "flow_ratio"
        ] = "length_weighted",
        vel_wse_method: Literal["average", "sloped", "max"] = "sloped",
        vel_interp_method: Literal[
            "flat_cell_center",
            "triangle_blend", "face_idw", "face_gradient",
            "facepoint_blend", "scatter_cell_face", "scatter_face",
            "scatter_corners", "scatter_corners_face", "scatter_cell_corners_face",
            "scatter_face_normal",
        ] = "scatter_face",
        scatter_interp_method: Literal["nearest", "linear", "cubic"] = "linear",
        fix_triangulation: bool = True,
        clip_to_perimeter: bool = True,
        face_velocity_location: Literal[
            "centroid", "normal_intercept"
        ] = "normal_intercept",
        wse_path_compare: str | Path | None = None,
        depth_path_compare: str | Path | None = None,
        vel_path_compare: str | Path | None = None,
        threshold_pct_compare: float = 5.0,
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
            0-based time index.  Required - all three outputs need a specific
            timestep (speed has no max-value fallback).
        reference_raster:
            Path to the terrain DEM GeoTIFF.  Required - used to derive depth
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
            (default ``False``).
        nodata:
            Fill value for pixels outside the wet mesh.
        render_mode:
            Water-surface rendering mode - ``"horizontal"`` (default),
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
            ``"scatter_cell_face"``, ``"scatter_face"``, ``"scatter_corners"``,
            ``"scatter_corners_face"``, ``"scatter_cell_corners_face"``, and
            ``"scatter_face_normal"``.
            One of ``"nearest"``, ``"linear"`` *(default)*, ``"cubic"``.
            Ignored for all other *vel_interp_method* values.
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
            exclusive with** ``snap_to_reference_extent`` - a ``ValueError``
            is raised if both are ``True``.  Default ``True``.
        face_velocity_location:
            Position used as the face normal velocity measurement point when
            ``vel_wse_method="sloped"``.
            ``"normal_intercept"`` (default): where the cell-centre connecting
            line crosses the face polyline.
            ``"centroid"``: geometric centroid of the face polyline.
            Passed to :meth:`cell_velocity_vectors`.
        wse_path_compare:
            Optional path to a RasMapper WSE raster for debug comparison.
            When provided, pixel-level difference statistics between the raspy
            output and this raster are logged at INFO level after export.
        depth_path_compare:
            Optional path to a RasMapper depth raster for debug comparison.
        vel_path_compare:
            Optional path to a RasMapper speed raster for debug comparison.
        threshold_pct_compare:
            Percentage-difference threshold used to classify outlier pixels in
            the debug comparison shapefiles (default ``5.0``).  Pixels where
            ``|raspy - reference| / |reference| * 100`` exceeds this value are
            written to a point shapefile in the system temp directory.
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
        from raspy.geo import raster as _raster  # deferred - geo not required

        logger.info(
            "Exporting hydraulic rasters: Water Surface Elevation, Depth and Velocity"
        )
        # -- 0. Guards --------------------------------------------------
        if clip_to_perimeter and snap_to_reference_extent:
            raise ValueError(
                "clip_to_perimeter and snap_to_reference_extent are mutually "
                "exclusive: clip_to_perimeter restricts the output to the "
                "perimeter bounding box, while snap_to_reference_extent expands "
                "it to the full reference raster extent.  Set "
                "snap_to_reference_extent=False when using clip_to_perimeter=True."
            )

        # -- 1. Read HDF data once --------------------------------------
        wse_values = np.array(self.water_surface[timestep, : self.n_cells])
        depth_at_cells = self.depth(timestep) if depth_min is not None else None
        face_vel_arr = (
            None if vel_interp_method == "flat_cell_center"
            else np.array(self.face_velocity[timestep, :])
        )

        # -- 2. WLS velocity vectors once ------------------------------
        cell_vel_vecs = self.cell_velocity_vectors(
            timestep, method=vel_weight_method, wse_interp=vel_wse_method,
            face_velocity_location=face_velocity_location,
        )

        # -- 3. Mesh topology kwargs ------------------------------------
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

        # -- 4. WSE raster in-memory (shared for WSE output + depth) ---
        logger.info("Building water-surface raster (shared for depth output)...")

        cell_wse_masked = (
            np.where(depth_at_cells < depth_min, np.nan, wse_values)
            if depth_min is not None else wse_values
        )

        # Facepoint WSE for sloping render - computed once for WSE/depth;
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
            if vel_min is not None:
                with np.errstate(invalid="ignore"):
                    _speed = np.linalg.norm(cell_vel_vecs[: self.n_cells], axis=1)
                _vel_wse_masked = np.where(_speed < vel_min, np.nan, wse_values)
            else:
                _vel_wse_masked = wse_values
            _fp_wse_vel = self.wse_at_facepoints(_vel_wse_masked)
            if _use_facecenters:
                _fc_wse_vel = self.wse_at_facecentroids(_vel_wse_masked)

        mesh_kw["facepoint_wse"] = _fp_wse
        mesh_kw["scatter_interp_method"] = scatter_interp_method
        mesh_kw["cell_polygons"] = self.cell_polygons
        mesh_kw["face_centers"] = (
            self.face_centroids if _use_facecenters else None
        )
        mesh_kw["face_center_wse"] = _fc_wse
        # Exclude face_centers: not shared with mesh_to_velocity_raster via mesh_kw
        # (face_centers is passed explicitly in the velocity call).  Override
        # facepoint_wse and face_center_wse with vel_min-masked versions.
        _vel_mesh_kw = {k: v for k, v in mesh_kw.items()
                        if k not in ("face_centers",
                                     "facepoint_wse", "face_center_wse")}
        _vel_mesh_kw["facepoint_wse"] = _fp_wse_vel
        _vel_mesh_kw["face_center_wse"] = _fc_wse_vel

        wse_ds = _raster.mesh_to_wse_raster(
            **mesh_kw,
            cell_wse=cell_wse_masked,
            output_path=None,
            min_value=None,
            min_above_ref=depth_min,
        )

        # -- 5. WSE output ----------------------------------------------
        if clip_to_perimeter:
            wse_result: Path | rasterio.io.DatasetReader = (
                _raster._mask_outside_polygon(wse_ds, _perim, nodata, wse_path)
            )
        elif wse_path is not None:
            wse_result = _raster._write_dataset(wse_ds, wse_path)
        else:
            wse_result = wse_ds

        # -- 6. Depth output --------------------------------------------
        logger.info("Building depth raster from WSE raster and DEM...")

        depth_ds = _raster._depth_from_wse_and_dem(
            wse_ds, reference_raster, nodata,
            None if clip_to_perimeter else depth_path,
            min_value=depth_min,
        )
        if clip_to_perimeter:
            depth_result: Path | rasterio.io.DatasetReader = (
                _raster._mask_outside_polygon(
                    depth_ds, _perim, nodata, depth_path
                )
            )
            depth_ds.close()
        else:
            depth_result = depth_ds

        # -- 7. Speed output --------------------------------------------
        # Pass wse_ds directly so mesh_to_velocity_raster reuses the
        # already-rendered wet extent instead of re-running the WSE render.
        logger.info("Building velocity raster for speed output...")

        vel_ds = _raster.mesh_to_velocity_raster(
            **_vel_mesh_kw,
            wse_raster=wse_ds,
            cell_wse=wse_values,
            cell_velocity=cell_vel_vecs,
            output_path=None,
            vel_min=vel_min,
            depth_min=depth_min,
            method=vel_interp_method,
            face_normals=self.face_normals,
            face_normal_velocity=face_vel_arr,
            face_centers=self.face_centroids,
            face_velocity_coords=(
                self.face_normal_intercept
                if face_velocity_location == "normal_intercept"
                else self.face_centroids
            ),
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
                    speed_ds, _perim, nodata, speed_path
                )
            )
            speed_ds.close()
        else:
            speed_result = speed_ds

        # -- 8. Debug comparison vs RasMapper (optional) ----------------
        if wse_path_compare is not None:
            _raster._compare_rasters_debug(
                "WSE", wse_result, wse_path_compare, nodata, threshold_pct_compare
            )
        if depth_path_compare is not None:
            _raster._compare_rasters_debug(
                "Depth", depth_result, depth_path_compare, nodata, threshold_pct_compare
            )
        if vel_path_compare is not None:
            _raster._compare_rasters_debug(
                "Speed", speed_result, vel_path_compare, nodata, threshold_pct_compare
            )

        return {
            "water_surface": wse_result,
            "depth": depth_result,
            "speed": speed_result,
        }

    def plot_cell_velocity(
        self,
        cell_idx: int,
        timestep: int,
        methods: list[str] | None = None,
        wse_interp: Literal["average", "sloped", "max"] = "average",
        vel_weight_method: Literal[
            "area_weighted", "length_weighted", "flow_ratio"
        ] = "area_weighted",
        scatter_interp_method: Literal["nearest", "linear", "cubic"] = "linear",
        sample_density: int = 20,
        min_arrow_fraction: float = 0.10,
        face_velocity_location: Literal[
            "centroid", "normal_intercept"
        ] = "normal_intercept",
        colorbar: bool = True,
    ) -> tuple:
        """Plot velocity decomposition and scatter-method comparison for a cell.

        Creates two matplotlib figures:

        **Figure 1 - Velocity Components** shows the focus cell and its
        immediate neighbours.  For each face of the focus cell the face-normal
        velocity ``V_n*n_hat``, the double-C-stencil tangential component
        ``V_t*t_hat``, and the full 2-D face velocity are drawn as arrows.
        WLS cell-centre velocity vectors are shown for every cell in the
        neighbourhood.

        **Figure 2 - Scatter Method Comparison** plots one sub-panel per
        scatter interpolation method.  Each panel shows the velocity values at
        the source points used by that method (cell centres, face midpoints,
        or polygon-corner facepoints) as quiver arrows, overlaid on the
        neighbourhood cell polygons.  This lets you compare how each method's
        scatter set differs in spatial density and origin.

        **Figure 3 - Interpolated Velocity Field** shows one sub-panel per
        scatter method (matching *methods*).  Each panel interpolates Vx and
        Vy onto a regular sample grid (masked to neighbourhood cell polygons)
        using ``scipy.interpolate.griddata`` and renders the result as quiver
        arrows coloured by speed.

        Parameters
        ----------
        cell_idx:
            0-based index of the focus cell.
        timestep:
            0-based index into the HDF time dimension.
        methods:
            Scatter methods to include in Figures 2 and 3.  Defaults to all
            six: ``"scatter_face"``, ``"scatter_corners"``,
            ``"scatter_cell_face"``, ``"scatter_corners_face"``,
            ``"scatter_cell_corners_face"``, ``"scatter_face_normal"``.
        wse_interp:
            Face WSE interpolation scheme passed to
            :meth:`cell_velocity_vectors`.
        vel_weight_method:
            WLS weight scheme passed to :meth:`cell_velocity_vectors`.
        scatter_interp_method:
            ``scipy.interpolate.griddata`` *method* used in Figure 3.
            One of ``"nearest"``, ``"linear"`` *(default)*, ``"cubic"``.
        sample_density:
            Number of sample points along each axis of the neighbourhood
            bounding box before polygon masking.  Default ``20``.
        face_velocity_location:
            Position of the face normal velocity measurement point used as
            scatter coordinates and passed to :meth:`cell_velocity_vectors`.
            ``"normal_intercept"`` (default) or ``"centroid"``.
        colorbar:
            Whether to draw the shared speed colorbar on Figure 3.
            ``True`` (default) shows it; ``False`` hides it, which can be
            useful when all subplot slots are filled and the bar would
            otherwise overlap the panels.

        Returns
        -------
        fig1 : matplotlib.figure.Figure
            Velocity-components figure.
        fig2 : matplotlib.figure.Figure
            Scatter source-points comparison figure.
        fig3 : matplotlib.figure.Figure
            Interpolated velocity field figure (one panel per method).

        Raises
        ------
        ImportError
            If ``matplotlib`` or ``scipy`` are not installed.
        IndexError
            If *cell_idx* is out of range.
        ValueError
            If any entry in *methods* is not a recognised scatter method.
        """
        try:
            import matplotlib.patches as mpatches
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon as MplPolygon
        except ImportError as exc:
            raise ImportError(
                "plot_cell_velocity requires matplotlib. "
                "Install it with: pip install matplotlib"
            ) from exc

        from ._velocity import (
            average_face_velocities_at_facepoints,
            compute_all_face_velocities,
        )

        _SCATTER_METHODS = [
            "scatter_face",
            "scatter_corners",
            "scatter_cell_face",
            "scatter_corners_face",
            "scatter_cell_corners_face",
            "scatter_face_normal",
        ]
        if methods is None:
            methods = _SCATTER_METHODS
        else:
            bad = [m for m in methods if m not in _SCATTER_METHODS]
            if bad:
                raise ValueError(
                    f"Unknown methods: {bad}. "
                    f"plot_cell_velocity only supports scatter methods: "
                    f"{_SCATTER_METHODS}"
                )
        if scatter_interp_method not in ("nearest", "linear", "cubic"):
            raise ValueError(
                f"scatter_interp_method must be 'nearest', 'linear', or 'cubic'; "
                f"got {scatter_interp_method!r}."
            )
        if sample_density < 2:
            raise ValueError("sample_density must be >= 2.")

        if cell_idx < 0 or cell_idx >= self.n_cells:
            raise IndexError(
                f"cell_idx {cell_idx} is out of range [0, {self.n_cells})"
            )

        # -- Raw geometry and result data -------------------------------
        n_cells     = self.n_cells
        fv_raw      = np.array(self.face_velocity[timestep, :])
        cfi, cfv    = self.cell_face_info
        face_ci     = self.face_cell_indexes        # (n_faces, 2)
        fn_all      = self.face_normals             # (n_faces, 3): nx, ny, L
        fc_all      = (                              # (n_faces, 2)
            self.face_normal_intercept
            if face_velocity_location == "normal_intercept"
            else self.face_centroids
        )
        fp_idx      = self.face_facepoint_indexes   # (n_faces, 2)
        fp_xy       = self.facepoint_coordinates    # (n_fp, 2)
        cc_all      = self.cell_centers             # (n_cells, 2)
        polys       = self.cell_polygons            # list[(n_v, 2)]

        # -- Focus-cell faces and facepoint-adjacent neighbourhood -----
        s0 = int(cfi[cell_idx, 0])
        k0 = int(cfi[cell_idx, 1])
        focus_fv = cfv[s0:s0 + k0]
        focus_fi = focus_fv[:, 0].astype(int)

        # All facepoints belonging to the focus cell
        focus_fp_set = set(fp_idx[focus_fi].ravel().tolist())

        # Every cell sharing at least one facepoint with the focus cell
        # (includes face-adjacent and corner/diagonal neighbours)
        touches = (
            np.isin(fp_idx[:, 0], list(focus_fp_set))
            | np.isin(fp_idx[:, 1], list(focus_fp_set))
        )
        touched_ci = face_ci[touches].ravel()
        adj = [
            int(ci) for ci in np.unique(touched_ci)
            if 0 <= ci < n_cells and ci != cell_idx
        ]
        nbr = [cell_idx] + adj  # neighbourhood (focus first)

        # Collect all faces / facepoints belonging to neighbourhood cells
        nbr_face_set: set[int] = set()
        for ci in nbr:
            s, k = int(cfi[ci, 0]), int(cfi[ci, 1])
            nbr_face_set.update(map(int, cfv[s:s + k, 0]))
        nbr_fi = np.array(sorted(nbr_face_set), dtype=int)

        nbr_fp_set: set[int] = set()
        for fi in nbr_fi:
            nbr_fp_set.add(int(fp_idx[fi, 0]))
            nbr_fp_set.add(int(fp_idx[fi, 1]))
        nbr_fpi = np.array(sorted(nbr_fp_set), dtype=int)

        # -- Velocity reconstruction ------------------------------------
        # cell_vecs has shape (n_cells + n_ghost, 2); dry_mask matches.
        cell_vecs = self.cell_velocity_vectors(
            timestep,
            method=vel_weight_method,
            wse_interp=wse_interp,
            face_velocity_location=face_velocity_location,
        )
        _vel_mag = np.linalg.norm(cell_vecs, axis=1)
        dry_mask = (_vel_mag == 0.0) | ~np.isfinite(_vel_mag)
        n_total = len(dry_mask)

        L  = face_ci[:, 0]
        R  = face_ci[:, 1]
        Ls = np.where((L >= 0) & (L < n_total), L, 0)
        Rs = np.where((R >= 0) & (R < n_total), R, 0)
        wet = (
            ((L >= 0) & (L < n_total) & ~dry_mask[Ls])
            | ((R >= 0) & (R < n_total) & ~dry_mask[Rs])
        )

        face_2d = compute_all_face_velocities(
            face_normals=fn_all,
            face_normal_velocity=fv_raw,
            face_cell_indexes=face_ci,
            cell_velocity=cell_vecs,
            dry_mask=dry_mask,
        )

        fp_vel = average_face_velocities_at_facepoints(
            face_facepoint_indexes=fp_idx,
            face_vel_2d=face_2d,
            wet_face=wet,
        )

        # -- Source-point arrays for Figure 2 --------------------------
        nbr_wet_fi  = nbr_fi[wet[nbr_fi]]
        wet_fp_mask = np.zeros(len(fp_xy), dtype=bool)
        wet_fp_mask[fp_idx[wet, 0]] = True
        wet_fp_mask[fp_idx[wet, 1]] = True
        nbr_wet_fpi = nbr_fpi[wet_fp_mask[nbr_fpi]]

        pts_cc  = cc_all[nbr]
        vel_cc  = cell_vecs[nbr]
        pts_fc  = fc_all[nbr_wet_fi]
        vel_fc  = face_2d[nbr_wet_fi]
        pts_fp  = fp_xy[nbr_wet_fpi]
        vel_fp  = fp_vel[nbr_wet_fpi]
        # scatter_face_normal: vn * n_hat at face centroids (not normal-intercept).
        # vn is a face-integrated quantity - centroid is always the correct position
        # regardless of face_velocity_location.
        pts_fn  = self.face_centroids[nbr_wet_fi]
        vel_fn  = fv_raw[nbr_wet_fi][:, None] * fn_all[nbr_wet_fi, :2]

        # Typical cell diameter for arrow scaling
        poly_f    = np.asarray(polys[cell_idx])
        cell_diam = float(np.ptp(poly_f, axis=0).mean()) or 1.0

        # Shared axis limits for all figures - derived from neighbourhood
        # polygon extents so every panel shows the same spatial window.
        all_poly_pts = np.vstack([np.asarray(polys[ci]) for ci in nbr])
        _nbr_xmin, _nbr_ymin = all_poly_pts.min(axis=0)
        _nbr_xmax, _nbr_ymax = all_poly_pts.max(axis=0)
        _xpad = (_nbr_xmax - _nbr_xmin) * 0.05 or 1.0
        _ypad = (_nbr_ymax - _nbr_ymin) * 0.05 or 1.0
        _nbr_xlim = (_nbr_xmin - _xpad, _nbr_xmax + _xpad)
        _nbr_ylim = (_nbr_ymin - _ypad, _nbr_ymax + _ypad)

        # -- Shared helpers ---------------------------------------------
        def _draw_polys(ax) -> None:
            for ci in nbr:
                poly = np.asarray(polys[ci])
                foc  = ci == cell_idx
                patch = MplPolygon(
                    poly, closed=True,
                    facecolor="steelblue" if foc else "lightgray",
                    edgecolor="navy"      if foc else "gray",
                    linewidth=1.8 if foc else 0.7,
                    alpha=0.28 if foc else 0.12,
                    zorder=1,
                )
                ax.add_patch(patch)
                cx, cy = cc_all[ci]
                ax.text(cx, cy, str(ci), ha="center", va="center",
                        fontsize=7,
                        color="navy" if foc else "dimgray", zorder=4)

        def _arrow(ax, x, y, dx, dy, color, lw=1.4, zo=7, num=None) -> None:
            if abs(dx) + abs(dy) < 1e-12:
                return
            ax.annotate(
                "",
                xy=(x + dx, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
                zorder=zo,
            )
            if num is not None:
                ax.text(
                    x + dx, y + dy, str(num),
                    fontsize=5.5, color=color, fontweight="bold",
                    ha="center", va="center", zorder=zo + 2,
                    bbox=dict(boxstyle="round,pad=0.08", fc="white",
                              ec="none", alpha=0.75),
                )

        def _quiver(ax, pts, vels, color, label, scale, min_length=0.0) -> None:
            """Quiver arrows with pre-scaled length in data coordinates.

            Arrows shorter than *min_length* (data units) are boosted to that
            length and drawn at half opacity so they remain visible while
            clearly signalling that their length is not to scale.
            """
            if len(pts) == 0:
                return
            u_raw = vels[:, 0] * scale
            v_raw = vels[:, 1] * scale
            lengths = np.hypot(u_raw, v_raw)
            boosted = (lengths > 1e-12) & (lengths < min_length)
            boost_f = np.where(boosted, min_length / np.maximum(lengths, 1e-12), 1.0)
            u = u_raw * boost_f
            v = v_raw * boost_f
            normal = ~boosted
            kw = dict(angles="xy", scale_units="xy", scale=1,
                      width=0.004, headwidth=4, headlength=5, zorder=5)
            if normal.any():
                ax.quiver(pts[normal, 0], pts[normal, 1], u[normal], v[normal],
                          color=color, label=label, **kw)
            if boosted.any():
                ax.quiver(pts[boosted, 0], pts[boosted, 1], u[boosted], v[boosted],
                          color=color, alpha=0.45,
                          label=label if not normal.any() else None, **kw)
            ax.scatter(pts[:, 0], pts[:, 1], color=color,
                       s=14, zorder=6, alpha=0.85)

        # --------------------------------------------------------------
        # Figure 1: Velocity Components
        # --------------------------------------------------------------
        # GridSpec: left panel (ax1) = velocity components (wide);
        # right panel (ax_idx) = cell & face index map (narrow).
        # width_ratios [3, 1] gives ~75 % / 25 % split.
        from matplotlib.gridspec import GridSpec as _GS
        fig1 = plt.figure(figsize=(15, 9))
        _gs1 = _GS(1, 2, figure=fig1, width_ratios=[3, 1], wspace=0.28)
        ax1    = fig1.add_subplot(_gs1[0, 0])
        ax_idx = fig1.add_subplot(_gs1[0, 1])
        _draw_polys(ax1)

        # Common arrow scale (largest vector = 38 % of cell diameter)
        sp_cells = np.linalg.norm(cell_vecs[nbr], axis=1)
        sp_faces = np.linalg.norm(face_2d[focus_fi], axis=1)
        max_sp1  = max(
            float(np.nanmax(sp_cells)) if len(sp_cells) else 0.0,
            float(np.nanmax(sp_faces)) if len(sp_faces) else 0.0,
            1e-9,
        )
        sc1 = cell_diam * 0.38 / max_sp1

        # Numbered-arrow tracking for the speed table (Figure 1 only)
        _arrow_ctr: int = 0
        _arrow_tbl: list[tuple[int, str, float]] = []   # (num, label, speed)
        _wls_nums:  dict[int, int] = {}                 # ci ? arrow num
        _face_nums: dict[tuple[int, str], int] = {}     # (fi, type) ? num

        # WLS cell-centre velocity arrows
        for ci in nbr:
            cx, cy = cc_all[ci]
            vx, vy = cell_vecs[ci]
            spd = float(np.hypot(vx, vy))
            is_foc = ci == cell_idx
            color = "darkred" if is_foc else "indigo"
            lw    = 2.2       if is_foc else 1.5
            if spd > 1e-12:
                _arrow_ctr += 1
                _wls_nums[ci] = _arrow_ctr
                _arrow_tbl.append((_arrow_ctr, f"WLS cell {ci}", spd))
            _arrow(ax1, cx, cy, vx * sc1, vy * sc1, color=color, lw=lw,
                   zo=8, num=_wls_nums.get(ci))

        # Face velocity decomposition for the focus cell's faces
        for fi in focus_fi:
            fcx, fcy = fc_all[fi]
            nx, ny = float(fn_all[fi, 0]), float(fn_all[fi, 1])
            tx, ty = -ny, nx                    # tangential unit vector
            vn_s = float(fv_raw[fi])
            v2d  = face_2d[fi]
            vt_s = float(v2d[0] * tx + v2d[1] * ty)
            v2d_spd = float(np.hypot(v2d[0], v2d[1]))

            eps = cell_diam * 0.018

            if abs(vn_s) > 1e-12:
                _arrow_ctr += 1
                _face_nums[(fi, "vn")] = _arrow_ctr
                _arrow_tbl.append((_arrow_ctr, f"Face {fi} Vn", abs(vn_s)))
            _arrow(ax1, fcx - eps, fcy,
                   nx * vn_s * sc1, ny * vn_s * sc1,
                   "green", lw=1.6, num=_face_nums.get((fi, "vn")))

            if abs(vt_s) > 1e-12:
                _arrow_ctr += 1
                _face_nums[(fi, "vt")] = _arrow_ctr
                _arrow_tbl.append((_arrow_ctr, f"Face {fi} Vt", abs(vt_s)))
            _arrow(ax1, fcx, fcy - eps,
                   tx * vt_s * sc1, ty * vt_s * sc1,
                   "darkorange", lw=1.6, num=_face_nums.get((fi, "vt")))

            if v2d_spd > 1e-12:
                _arrow_ctr += 1
                _face_nums[(fi, "v2d")] = _arrow_ctr
                _arrow_tbl.append((_arrow_ctr, f"Face {fi} V2D", v2d_spd))
            _arrow(ax1, fcx + eps, fcy + eps,
                   v2d[0] * sc1, v2d[1] * sc1,
                   "dodgerblue", lw=1.9, zo=9, num=_face_nums.get((fi, "v2d")))
            ax1.plot(fcx, fcy, "k.", ms=5, zorder=10)

        legend1 = [
            mpatches.Patch(facecolor="steelblue", alpha=0.4,
                           edgecolor="navy", label="Focus cell"),
            mpatches.Patch(facecolor="lightgray", alpha=0.3,
                           edgecolor="gray", label="Adjacent cell"),
            mpatches.FancyArrow(0, 0, 1, 0, color="darkred",
                                label=f"WLS vel - focus ({vel_weight_method})"),
            mpatches.FancyArrow(0, 0, 1, 0, color="indigo",
                                label="WLS vel - adjacent"),
            mpatches.FancyArrow(0, 0, 1, 0, color="green",
                                label="Face Vn  (V_n * n_hat)"),
            mpatches.FancyArrow(0, 0, 1, 0, color="darkorange",
                                label="Face Vt  (V_t * t_hat, double-C)"),
            mpatches.FancyArrow(0, 0, 1, 0, color="dodgerblue",
                                label="Face 2D  (Vn + Vt)"),
        ]
        ax1.legend(handles=legend1, loc="upper left", fontsize=8, framealpha=0.9)
        ax1.set_aspect("equal")
        ax1.set_xlim(_nbr_xlim)
        ax1.set_ylim(_nbr_ylim)
        ax1.set_title(
            f"{self.name} - Cell {cell_idx}  |  Velocity Components\n"
            f"ts={timestep}  wse_interp={wse_interp}  "
            f"vel_weight={vel_weight_method}",
            fontsize=10,
        )
        ax1.set_xlabel("X (model units)")
        ax1.set_ylabel("Y (model units)")

        # -- Right panel (ax_idx): cell & face ID index map ---------------
        # Pure topological reference — no velocity arrows.
        # Every neighbourhood cell is labelled with its cell index;
        # every neighbourhood face with its face index.
        _draw_polys(ax_idx)

        for ci in nbr:
            poly_ci = np.asarray(polys[ci])
            is_focus = ci == cell_idx
            ax_idx.add_patch(MplPolygon(
                poly_ci, closed=True,
                facecolor="steelblue" if is_focus else "lightgray",
                edgecolor="navy"      if is_focus else "gray",
                linewidth=1.5 if is_focus else 0.9,
                alpha=0.35, zorder=2,
            ))
            cxi, cyi = cc_all[ci]
            ax_idx.text(
                cxi, cyi, str(ci),
                ha="center", va="center",
                fontsize=8,
                color="navy" if is_focus else "dimgray",
                fontweight="bold", zorder=8,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.85),
            )

        for fi in nbr_fi:
            fcx_i, fcy_i = fc_all[fi]
            ax_idx.text(
                fcx_i, fcy_i, str(fi),
                ha="center", va="center",
                fontsize=7, color="darkgreen", zorder=9,
                bbox=dict(boxstyle="round,pad=0.10", fc="lightyellow",
                          ec="none", alpha=0.80),
            )

        _nbr_pts = np.vstack([np.asarray(polys[ci]) for ci in nbr])
        _ix_xmin, _ix_ymin = _nbr_pts.min(axis=0)
        _ix_xmax, _ix_ymax = _nbr_pts.max(axis=0)
        _ix_xpad = (_ix_xmax - _ix_xmin) * 0.06 or 1.0
        _ix_ypad = (_ix_ymax - _ix_ymin) * 0.06 or 1.0
        ax_idx.set_xlim(_ix_xmin - _ix_xpad, _ix_xmax + _ix_xpad)
        ax_idx.set_ylim(_ix_ymin - _ix_ypad, _ix_ymax + _ix_ypad)
        ax_idx.set_aspect("equal")
        ax_idx.set_xlabel("X (model units)", fontsize=8)
        ax_idx.tick_params(labelsize=7)
        ax_idx.set_title(
            "Cell & Face Index\n"
            "blue=focus  grey=nbr  green=face #",
            fontsize=8,
        )

        # Leave bottom margin for the speed table, then draw the table
        fig1.tight_layout(rect=[0, 0.18, 1, 1])

        if _arrow_tbl:
            _ncols_tbl = min(3, len(_arrow_tbl))
            _per_col   = max(1, (len(_arrow_tbl) + _ncols_tbl - 1) // _ncols_tbl)
            _tbl_rows  = []
            for _r in range(_per_col):
                _parts = []
                for _c in range(_ncols_tbl):
                    _i = _c * _per_col + _r
                    if _i < len(_arrow_tbl):
                        _n, _desc, _spd = _arrow_tbl[_i]
                        _parts.append(f"{_n:2d}: {_desc:<18s} {_spd:9.4f}")
                    else:
                        _parts.append(" " * 33)
                _tbl_rows.append("   ".join(_parts))
            _tbl_str = "\n".join(_tbl_rows)
            fig1.text(
                0.5, 0.01,
                f"Arrow speeds (model units/s):\n{_tbl_str}",
                fontsize=6.5, family="monospace",
                ha="center", va="bottom",
                transform=fig1.transFigure,
                bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow",
                          ec="gray", lw=0.8, alpha=0.92),
            )

        # --------------------------------------------------------------
        # Figure 2: Scatter Method Comparison
        # --------------------------------------------------------------

        # Source-point definitions per scatter method:
        #   each entry is (points, velocities, colour, legend-label)
        _SOURCES: dict[str, list[tuple]] = {
            "scatter_face": [
                (pts_fc, vel_fc, "dodgerblue", "Face midpoints (2D)"),
            ],
            "scatter_corners": [
                (pts_fp, vel_fp, "green", "Facepoints (double-C avg)"),
            ],
            "scatter_cell_face": [
                (pts_cc, vel_cc, "red",        "Cell centres (WLS)"),
                (pts_fc, vel_fc, "dodgerblue", "Face midpoints (2D)"),
            ],
            "scatter_corners_face": [
                (pts_fp, vel_fp, "green",      "Facepoints (double-C avg)"),
                (pts_fc, vel_fc, "dodgerblue", "Face midpoints (2D)"),
            ],
            "scatter_cell_corners_face": [
                (pts_cc, vel_cc, "red",        "Cell centres (WLS)"),
                (pts_fp, vel_fp, "green",      "Facepoints (double-C avg)"),
                (pts_fc, vel_fc, "dodgerblue", "Face midpoints (2D)"),
            ],
            "scatter_face_normal": [
                (pts_fn, vel_fn, "orange", "Face centroids (vn*n_hat)"),
            ],
        }

        # Global speed scale so arrows are comparable across all panels
        all_sp2 = []
        for v in (vel_cc, vel_fc, vel_fp):
            if len(v):
                all_sp2.extend(np.linalg.norm(v, axis=1).tolist())
        max_sp2 = max(max(all_sp2) if all_sp2 else 0.0, 1e-9)
        sc2 = cell_diam * 0.35 / max_sp2
        _min_len2 = cell_diam * 0.35 * min_arrow_fraction

        n_m   = len(methods)
        n_col = min(3, n_m)
        n_row = (n_m + n_col - 1) // n_col
        fig2, axes2 = plt.subplots(
            n_row, n_col,
            figsize=(6.5 * n_col, 5.5 * n_row),
            squeeze=False,
        )

        for idx, method in enumerate(methods):
            ax = axes2[idx // n_col, idx % n_col]
            _draw_polys(ax)

            for pts, vels, color, label in _SOURCES.get(method, []):
                _quiver(ax, pts, vels, color, label, sc2, min_length=_min_len2)

            ax.set_aspect("equal")
            ax.set_xlim(_nbr_xlim)
            ax.set_ylim(_nbr_ylim)
            ax.set_title(method, fontsize=9, fontweight="bold")
            ax.set_xlabel("X", fontsize=7)
            ax.set_ylabel("Y", fontsize=7)
            ax.tick_params(labelsize=6)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=6, loc="upper right")

        for idx in range(n_m, n_row * n_col):
            axes2[idx // n_col, idx % n_col].set_visible(False)

        fig2.suptitle(
            f"{self.name} - Cell {cell_idx}  |  Scatter Method Comparison\n"
            f"ts={timestep}  vel_weight={vel_weight_method}  "
            f"wse_interp={wse_interp}",
            fontsize=11, fontweight="bold",
        )
        fig2.tight_layout()

        # --------------------------------------------------------------
        # Figure 3: Interpolated Velocity Field (one panel per method)
        # --------------------------------------------------------------
        try:
            from scipy.interpolate import griddata as _griddata
        except ImportError as exc:
            raise ImportError(
                "plot_cell_velocity Figure 3 requires scipy. "
                "Install it with: pip install scipy"
            ) from exc

        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize as MplNorm
        from matplotlib.path import Path as MplPath

        # Source-point lookup (same as Figure 2)
        _SRC3 = {
            "scatter_face":             (pts_fc, vel_fc),
            "scatter_corners":          (pts_fp, vel_fp),
            "scatter_cell_face":        (
                np.vstack([pts_cc, pts_fc]),
                np.vstack([vel_cc, vel_fc]),
            ),
            "scatter_corners_face":     (
                np.vstack([pts_fp, pts_fc]),
                np.vstack([vel_fp, vel_fc]),
            ),
            "scatter_cell_corners_face": (
                np.vstack([pts_cc, pts_fp, pts_fc]),
                np.vstack([vel_cc, vel_fp, vel_fc]),
            ),
            "scatter_face_normal": (pts_fn, vel_fn),
        }

        # Sample grid, masked to neighbourhood cell polygons (computed once).
        # Dry cells (zero WLS speed at the focus timestep) are excluded so
        # interpolated arrows are not drawn over cells with no flow.
        x_min3, y_min3 = _nbr_xmin, _nbr_ymin
        x_max3, y_max3 = _nbr_xmax, _nbr_ymax
        gx = np.linspace(x_min3, x_max3, sample_density)
        gy = np.linspace(y_min3, y_max3, sample_density)
        gxx, gyy = np.meshgrid(gx, gy)
        grid_pts = np.column_stack([gxx.ravel(), gyy.ravel()])
        nbr_wet_cells = [
            ci for ci in nbr
            if np.linalg.norm(cell_vecs[ci]) > 1e-12
        ]
        inside = np.zeros(len(grid_pts), dtype=bool)
        for ci in nbr_wet_cells:
            inside |= MplPath(np.asarray(polys[ci])).contains_points(grid_pts)
        grid_pts_in = grid_pts[inside]

        # Colour scale shared across all panels (use overall source max speed)
        sp_max3 = max(max_sp2, 1e-9)
        norm3   = MplNorm(vmin=0, vmax=sp_max3)
        cmap3   = plt.get_cmap("plasma")
        sc3     = cell_diam * 0.38 / sp_max3  # data-units per unit speed

        n_m3        = len(methods)
        n_col3      = min(3, n_m3)
        n_row3      = (n_m3 + n_col3 - 1) // n_col3
        has_empty   = n_row3 * n_col3 > n_m3
        # Use constrained_layout when the colorbar has no empty slot to
        # occupy - it prevents the bar from overlapping the subplot panels.
        _layout3    = "constrained" if (colorbar and not has_empty) else None
        fig3, axes3 = plt.subplots(
            n_row3, n_col3,
            figsize=(6.5 * n_col3, 5.5 * n_row3),
            squeeze=False,
            layout=_layout3,
        )

        for idx, method in enumerate(methods):
            ax = axes3[idx // n_col3, idx % n_col3]
            _draw_polys(ax)

            src_pts, src_vel = _SRC3[method]

            if len(src_pts) == 0 or len(grid_pts_in) == 0:
                vx_i = np.full(len(grid_pts_in), np.nan)
                vy_i = np.full(len(grid_pts_in), np.nan)
            else:
                vx_i = _griddata(
                    src_pts, src_vel[:, 0], grid_pts_in,
                    method=scatter_interp_method, fill_value=np.nan, rescale=True,
                )
                vy_i = _griddata(
                    src_pts, src_vel[:, 1], grid_pts_in,
                    method=scatter_interp_method, fill_value=np.nan, rescale=True,
                )

            sp_i  = np.sqrt(vx_i**2 + vy_i**2)
            valid = np.isfinite(sp_i) & (sp_i > 0)

            if valid.any():
                u3_raw = vx_i[valid] * sc3
                v3_raw = vy_i[valid] * sc3
                len3   = np.hypot(u3_raw, v3_raw)
                min_len3 = cell_diam * 0.38 * min_arrow_fraction
                boost3 = np.where(
                    (len3 > 1e-12) & (len3 < min_len3),
                    min_len3 / np.maximum(len3, 1e-12),
                    1.0,
                )
                colors_i = cmap3(norm3(sp_i[valid]))
                ax.quiver(
                    grid_pts_in[valid, 0], grid_pts_in[valid, 1],
                    u3_raw * boost3, v3_raw * boost3,
                    color=colors_i,
                    angles="xy", scale_units="xy", scale=1,
                    width=0.004, headwidth=4, headlength=5,
                    zorder=5,
                )

            ax.set_aspect("equal")
            ax.set_xlim(_nbr_xlim)
            ax.set_ylim(_nbr_ylim)
            ax.set_title(method, fontsize=9, fontweight="bold")
            ax.set_xlabel("X", fontsize=7)
            ax.set_ylabel("Y", fontsize=7)
            ax.tick_params(labelsize=6)

        empty_idxs = list(range(n_m3, n_row3 * n_col3))

        # Shared colourbar (skipped when colorbar=False)
        if colorbar:
            sm3 = ScalarMappable(cmap=cmap3, norm=norm3)
            sm3.set_array([])
            if empty_idxs:
                # Turn the first empty slot into a blank host, then place a
                # narrow inset axes centred inside it for the colourbar.
                host_ax = axes3[empty_idxs[0] // n_col3, empty_idxs[0] % n_col3]
                host_ax.set_visible(True)
                host_ax.axis("off")
                for idx in empty_idxs[1:]:
                    axes3[idx // n_col3, idx % n_col3].set_visible(False)
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                cbar_ax = inset_axes(host_ax, width="25%", height="80%",
                                     loc="center")
                fig3.colorbar(sm3, cax=cbar_ax, label="Speed (model units/s)")
                cbar_ax.yaxis.set_label_position("left")
                cbar_ax.yaxis.tick_left()
            else:
                # constrained_layout (set at figure creation) ensures the bar
                # does not overlap the subplot panels.
                fig3.colorbar(sm3, ax=axes3.ravel().tolist(),
                              label="Speed (model units/s)")
        else:
            for idx in empty_idxs:
                axes3[idx // n_col3, idx % n_col3].set_visible(False)

        fig3.suptitle(
            f"{self.name} - Cell {cell_idx}  |  Interpolated Velocity Field\n"
            f"griddata={scatter_interp_method}  ts={timestep}  "
            f"vel_weight={vel_weight_method}  wse_interp={wse_interp}",
            fontsize=11, fontweight="bold",
        )
        if _layout3 != "constrained":
            fig3.tight_layout()
        return fig1, fig2, fig3

    def debug_cell_velocity(
        self,
        cell_idx: int,
        timestep: int,
        wse_interp: Literal["average", "sloped", "max"] = "sloped",
        vel_weight_method: Literal[
            "area_weighted", "length_weighted", "flow_ratio"
        ] = "length_weighted",
        face_velocity_location: Literal[
            "centroid", "normal_intercept"
        ] = "normal_intercept",
    ) -> None:
        """Print a detailed per-face breakdown of velocity reconstruction.

        Displays the WLS input data for each face of the specified cell, then
        shows the reconstructed velocity for all available weight schemes.
        Optionally compares against the HEC-RAS stored ``Cell Velocity``
        scalar when that dataset is present in the HDF file.

        Prints four tables:

        1. **WLS inputs** - per-face normal, length, V_n, face WSE, flow area.
        2. **Double-C stencil face velocities** - full 2D face velocity
           ``vn*n_hat + vt*t_hat`` using the *vel_weight_method* WLS vectors for
           the tangential component.  This is what all intra-cell interpolation
           methods (``triangle_blend``, ``face_idw``, ``face_gradient``,
           ``facepoint_blend``) use at the face midpoints.
        3. **Facepoint velocities** - velocity assigned to each polygon vertex
           by averaging the full 2D velocities of all adjacent wet faces.
           This is what ``facepoint_blend`` interpolates between.
        4. **Face normal velocity** - pure normal component ``vn * n_hat`` for
           each face.  This is the scatter value used by
           ``"scatter_face_normal"`` with no double-C tangential contribution.

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
        face_velocity_location:
            Position of the face normal velocity measurement point used in
            ``wse_interp="sloped"`` and passed to :meth:`cell_velocity_vectors`.
            ``"normal_intercept"`` (default) or ``"centroid"``.
        """
        from ._velocity import (
            _estimate_face_wse_average,
            _estimate_face_wse_max,
            _estimate_face_wse_sloped,
            _interpolate_face_flow_area,
            _wls_velocity,
            average_face_velocities_at_facepoints,
            compute_all_face_velocities,
        )

        if cell_idx < 0 or cell_idx >= self.n_cells:
            raise IndexError(f"cell_idx {cell_idx} is out of range [0, {self.n_cells})")

        n_cells = self.n_cells
        face_vel = np.array(self.face_velocity[timestep, :])
        # Read full water_surface (real + ghost) so boundary faces get
        # ghost-cell WSE rather than NaN when estimating face WSE.
        cell_wse = np.array(self.water_surface[timestep, :])

        cell_face_info, cell_face_values = self.cell_face_info
        start = int(cell_face_info[cell_idx, 0])
        count = int(cell_face_info[cell_idx, 1])
        vals = cell_face_values[start : start + count]
        face_idxs = vals[:, 0].astype(int)
        orientations = vals[:, 1].astype(int)

        normals = self.face_normals[face_idxs, :2]  # (k, 2)
        lengths = self.face_normals[face_idxs, 2]   # (k,)
        vn      = face_vel[face_idxs]               # (k,)

        face_ci          = self.face_cell_indexes    # (n_faces, 2)
        face_centroids_a = (
            self.face_normal_intercept
            if face_velocity_location == "normal_intercept"
            else self.face_centroids
        )
        fp_idx_all       = self.face_facepoint_indexes  # (n_faces, 2)
        fp_coords        = self.facepoint_coordinates   # (n_fp, 2)
        # Stack ghost cell centres so _estimate_face_wse_sloped can index
        # up to n_total (len(cell_wse) includes ghost rows).
        cell_centres = np.vstack([self.cell_centers, self.ghost_cell_centers])

        # -- Face WSE and flow-area weights -----------------------------
        if wse_interp == "sloped":
            face_wse_all = _estimate_face_wse_sloped(
                face_ci, cell_wse, cell_centres, face_centroids_a
            )
        elif wse_interp == "max":
            face_wse_all = _estimate_face_wse_max(face_ci, cell_wse)
        else:
            face_wse_all = _estimate_face_wse_average(face_ci, cell_wse)

        ae_info, ae_values = self.face_area_elevation
        areas = np.array(
            [
                _interpolate_face_flow_area(fi, face_wse_all[fi], ae_info, ae_values)
                for fi in face_idxs
            ]
        )

        # -- Cell-centre WLS vectors (all three methods) -----------------
        vel_aw = _wls_velocity(vn, areas,   normals)
        vel_lw = _wls_velocity(vn, lengths, normals)

        # -- Full cell velocity array for the chosen weight method --------
        # (used by the double-C stencil as V_L / V_R)
        # Returns (n_cells + n_ghost, 2); dry_mask matches.
        all_cell_vecs = self.cell_velocity_vectors(
            timestep,
            method=vel_weight_method,
            wse_interp=wse_interp,
            face_velocity_location=face_velocity_location,
        )
        _vel_mag = np.linalg.norm(all_cell_vecs, axis=1)
        dry_mask = (_vel_mag == 0.0) | ~np.isfinite(_vel_mag)
        n_total = len(dry_mask)

        # -- Face 2D velocities (double-C stencil, vectorised) -----------
        face_vel_2d_all = compute_all_face_velocities(
            face_normals=self.face_normals,
            face_normal_velocity=face_vel,
            face_cell_indexes=face_ci,
            cell_velocity=all_cell_vecs,
            dry_mask=dry_mask,
        )

        # -- Wet-face mask ------------------------------------------------
        l_idx  = face_ci[:, 0]
        r_idx  = face_ci[:, 1]
        l_safe = np.where((l_idx >= 0) & (l_idx < n_total), l_idx, 0)
        r_safe = np.where((r_idx >= 0) & (r_idx < n_total), r_idx, 0)
        wet_face = (
            ((l_idx >= 0) & (l_idx < n_total) & ~dry_mask[l_safe])
            | ((r_idx >= 0) & (r_idx < n_total) & ~dry_mask[r_safe])
        )

        # -- Facepoint velocities (averaged over all adjacent wet faces) --
        fp_vel_all = average_face_velocities_at_facepoints(
            face_facepoint_indexes=fp_idx_all,
            face_vel_2d=face_vel_2d_all,
            wet_face=wet_face,
        )

        # -- Unique facepoints for this cell, ordered by face -------------
        seen_fp: set[int] = set()
        cell_fp_ordered: list[int] = []
        for fi in face_idxs:
            for fp in (int(fp_idx_all[fi, 0]), int(fp_idx_all[fi, 1])):
                if fp not in seen_fp:
                    seen_fp.add(fp)
                    cell_fp_ordered.append(fp)

        # -- Helpers ------------------------------------------------------
        def _angle(v: np.ndarray) -> str:
            spd = np.linalg.norm(v)
            if spd < 1e-10:
                return "     n/a"
            return f"{(90.0 - np.degrees(np.arctan2(v[1], v[0]))) % 360.0:>8.2f}"

        def _vel_line(label: str, v: np.ndarray) -> str:
            spd = np.linalg.norm(v)
            return (
                f"  {label:<18}  Vx={v[0]:+.4f}  Vy={v[1]:+.4f}"
                f"  speed={spd:.4f}  dir={_angle(v)}-"
            )

        SEP = "-" * 72

        # ----------------------------------------------------------------
        # 1. HEADER
        # ----------------------------------------------------------------
        dry_label = "DRY" if dry_mask[cell_idx] else "WET"
        cx, cy   = cell_centres[cell_idx]
        print(f"\n{'-'*72}")
        print(
            f"  debug_cell_velocity  |  area={self.name}"
            f"  cell={cell_idx}  ts={timestep}"
        )
        print(f"{'-'*72}")
        print(f"  Cell centre : ({cx:.3f}, {cy:.3f})")
        print(f"  Cell WSE    : {cell_wse[cell_idx]:.4f}  [{dry_label}]")
        print(f"  wse_interp  : {wse_interp}")
        print(f"  n_faces     : {count}")

        # ----------------------------------------------------------------
        # 2. WLS INPUTS PER FACE
        # ----------------------------------------------------------------
        print(f"\n{SEP}")
        print("  WLS INPUTS PER FACE")
        print(SEP)
        hdr1 = (
            f"  {'Face':>6}  {'Ori':>3}  {'nx':>7}  {'ny':>7}  {'Length':>8}"
            f"  {'V_n':>8}  {'V_n_x':>8}  {'V_n_y':>8}  {'angle':>8}"
            f"  {'L_cell':>7}  {'L_WSE':>8}  {'R_cell':>7}  {'R_WSE':>8}"
            f"  {'face_WSE':>9}  {'A_face':>9}  {'fc_x':>11}  {'fc_y':>11}"
        )
        print(hdr1)
        print(f"  {'-'*68}")
        for fi, ori, (nx, ny), L, v, fwse, a in zip(
            face_idxs, orientations, normals, lengths, vn,
            face_wse_all[face_idxs], areas, strict=False,
        ):
            lc = int(face_ci[fi, 0])
            rc = int(face_ci[fi, 1])
            lc_valid = 0 <= lc < n_cells
            rc_valid = 0 <= rc < n_cells
            lc_str   = (f"{lc:>6}" + ("*" if lc_valid and dry_mask[lc] else " ")) \
                       if lc_valid else "  ghost "
            rc_str   = (f"{rc:>6}" + ("*" if rc_valid and dry_mask[rc] else " ")) \
                       if rc_valid else "  ghost "
            lw_str   = f"{cell_wse[lc]:>8.4f}" if lc_valid else "     n/a"
            rw_str   = f"{cell_wse[rc]:>8.4f}" if rc_valid else "     n/a"
            fcx, fcy = face_centroids_a[fi]
            vnx, vny  = v * nx, v * ny
            ang_str   = _angle(np.array([vnx, vny]))
            print(
                f"  {fi:>6}  {ori:>3}  {nx:>7.4f}  {ny:>7.4f}  {L:>8.2f}"
                f"  {v:>8.4f}  {vnx:>8.4f}  {vny:>8.4f}  {ang_str:>8}"
                f"  {lc_str}  {lw_str}  {rc_str}  {rw_str}"
                f"  {fwse:>9.4f}  {a:>9.4f}  {fcx:>11.3f}  {fcy:>11.3f}"
            )
        print("  (* = dry cell)")

        # ----------------------------------------------------------------
        # 3. CELL-CENTRE VELOCITY (WLS)
        # ----------------------------------------------------------------
        print(f"\n{SEP}")
        print("  CELL-CENTRE VELOCITY (WLS)")
        print(SEP)
        print(_vel_line("area_weighted", vel_aw))
        print(_vel_line("length_weighted", vel_lw))

        if self.face_flow is not None:
            face_flow = np.array(self.face_flow[timestep, :])
            qf         = face_flow[face_idxs]
            weights_fr = np.where(np.abs(vn) > 1e-10, np.abs(qf / vn), 0.0)
            vel_fr     = _wls_velocity(vn, weights_fr, normals)
            print(_vel_line("flow_ratio", vel_fr))
            print(f"    face Q : {face_flow[face_idxs]}")
        else:
            print("  flow_ratio         : (Face Flow not in HDF)")

        if self.cell_velocity is not None:
            ras_spd = float(self.cell_velocity[timestep, cell_idx])
            print(f"  HEC-RAS stored     : speed={ras_spd:.4f}  (scalar only)")
        else:
            print("  HEC-RAS stored     : (Cell Velocity scalar not in HDF)")

        # ----------------------------------------------------------------
        # 4. RECONSTRUCTED FACE VELOCITY (double-C stencil)
        # ----------------------------------------------------------------
        print(f"\n{SEP}")
        print(f"  RECONSTRUCTED FACE VELOCITY  (double-C, wt={vel_weight_method!r})")
        print(SEP)
        hdr2 = (
            f"  {'Face':>6}  {'tx':>7}  {'ty':>7}  {'vt_L':>8}  {'vt_R':>8}"
            f"  {'vt':>8}  {'Vfx':>8}  {'Vfy':>8}  {'|Vf|':>8}  {'dir':>8}  {'side':>6}"
        )
        print(hdr2)
        print(f"  {'-'*68}")
        for fi, (nx, ny), _ in zip(face_idxs, normals, vn, strict=False):
            t_hat = np.array([-ny, nx])
            lc    = int(face_ci[fi, 0])
            rc    = int(face_ci[fi, 1])
            l_ok  = 0 <= lc < n_cells and not dry_mask[lc]
            r_ok  = 0 <= rc < n_cells and not dry_mask[rc]
            vt_L  = float(np.dot(all_cell_vecs[lc], t_hat)) if l_ok else float("nan")
            vt_R  = float(np.dot(all_cell_vecs[rc], t_hat)) if r_ok else float("nan")
            if l_ok and r_ok:
                vt, side = 0.5 * (vt_L + vt_R), "both"
            elif l_ok:
                vt, side = vt_L, "L"
            elif r_ok:
                vt, side = vt_R, "R"
            else:
                vt, side = 0.0, "none"
            vf       = face_vel_2d_all[fi]
            vtL_s    = f"{vt_L:>8.4f}" if l_ok else f"{'dry/g':>8}"
            vtR_s    = f"{vt_R:>8.4f}" if r_ok else f"{'dry/g':>8}"
            print(
                f"  {fi:>6}  {t_hat[0]:>7.4f}  {t_hat[1]:>7.4f}"
                f"  {vtL_s}  {vtR_s}  {vt:>8.4f}"
                f"  {vf[0]:>8.4f}  {vf[1]:>8.4f}  {np.linalg.norm(vf):>8.4f}"
                f"  {_angle(vf)}  {side:>6}"
            )

        # ----------------------------------------------------------------
        # 5. FACEPOINT (CORNER) VELOCITIES
        # ----------------------------------------------------------------
        print(f"\n{SEP}")
        print("  FACEPOINT (CORNER) VELOCITIES")
        print(SEP)
        hdr3 = (
            f"  {'FP':>7}  {'x':>11}  {'y':>11}"
            f"  {'Vfp_x':>8}  {'Vfp_y':>8}  {'|Vfp|':>8}  {'dir':>8}"
            f"  {'n_wet':>5}  wet faces"
        )
        print(hdr3)
        print(f"  {'-'*68}")
        for fp in cell_fp_ordered:
            touching     = np.where(
                (fp_idx_all[:, 0] == fp) | (fp_idx_all[:, 1] == fp)
            )[0]
            wet_touching = touching[wet_face[touching]]
            vfp          = fp_vel_all[fp]
            x, y         = fp_coords[fp]
            faces_str    = " ".join(str(f) for f in wet_touching)
            print(
                f"  {fp:>7}  {x:>11.3f}  {y:>11.3f}"
                f"  {vfp[0]:>8.4f}  {vfp[1]:>8.4f}  {np.linalg.norm(vfp):>8.4f}"
                f"  {_angle(vfp)}  {len(wet_touching):>5}  [{faces_str}]"
            )

        # ----------------------------------------------------------------
        # 6. FACE NORMAL VELOCITY  (scatter_face_normal: vn * n_hat)
        # ----------------------------------------------------------------
        print(f"\n{SEP}")
        print("  FACE NORMAL VELOCITY  (scatter_face_normal: vn \u00d7 n\u0302)")
        print(SEP)
        hdr4 = (
            f"  {'Face':>6}  {'Vfn_x':>8}  {'Vfn_y':>8}  {'|Vfn|':>8}  {'dir':>8}"
        )
        print(hdr4)
        print(f"  {'-'*44}")
        for fi, (nx, ny), v in zip(face_idxs, normals, vn, strict=False):
            vfn = np.array([v * nx, v * ny])
            print(
                f"  {fi:>6}  {vfn[0]:>8.4f}  {vfn[1]:>8.4f}"
                f"  {abs(v):>8.4f}  {_angle(vfn)}"
            )
        print()

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
           - validates every cell against the HEC-RAS rules (convexity, face
           count, duplicate points, cell centre location).
        2. **Triangulation probe** - actually builds the fan-triangulation used
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

            - ``n_triangles_raw`` - triangles before fix.
            - ``n_triangles_after_fix`` - triangles after dedup + zero-area filter.
            - ``n_removed_by_fix`` - triangles removed.
            - ``trifinder_without_fix`` - ``"ok"`` / ``"failed"`` / ``"not_tested"``.
            - ``trifinder_with_fix`` - ``"ok"`` / ``"failed"``.
            - ``fallback_needed`` - ``True`` when even the fixed triangulation
              cannot initialise the trifinder (KDTree fallback will be used).
        """
        import matplotlib.tri as mtri

        from raspy.geo.mesh_validation import print_mesh_report

        # -- Step 1: Mesh geometry validation ------------------------------
        mesh_report = self.check_cells(check_boundary=check_boundary)

        if verbose:
            print(f"=== debug_raster_export: {self.name} ===\n")
            print_mesh_report(mesh_report)

        # -- Step 2: Build dry-cell mask -----------------------------------
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

        # -- Step 3: Build the fan-triangulation ---------------------------
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

        # -- Step 4: Probe trifinder WITHOUT fix ---------------------------
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

        # -- Step 5: Apply fix - track exactly what is removed and why --------
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

        # Duplicate facepoints (global - cause duplicate vertices in all_pts).
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
                print("  fix_triangulation=True is sufficient - no fallback needed")

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
