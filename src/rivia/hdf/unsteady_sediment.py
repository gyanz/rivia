"""UnsteadySediment - read HEC-RAS sediment results from unsteady plan HDF5 files.

Sediment output is written into the same plan HDF file as the hydraulics
results handled by :mod:`rivia.hdf.unsteady_plan`, under
``Results/Unsteady/Output/Output Blocks/Sediment/...``.  It is implemented in
its own module so ``unsteady_plan.py`` -- already large with hydraulics
result classes -- does not grow further.

:class:`UnsteadySediment` does not open its own HDF handle: it borrows the
handle already opened by the parent
:class:`~rivia.hdf.unsteady_plan.UnsteadyPlan` (reached via
``plan.sediment``).  1D cross-section results reuse that plan's existing
cross-section geometry join
(:class:`~rivia.hdf.unsteady_plan.CrossSectionResultsCollection`); 2D flow
area results reuse the plan's flow-area name list
(:attr:`~rivia.hdf.unsteady_plan.UnsteadyPlan.flow_areas`) and read the
``Sediment Bed`` and ``Sediment Transport`` output blocks directly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import pandas as pd

from ._base import _RAS_TS_FMT, _parse_hec_ts_array
from .geometry import _XS_ROOT, _decode
from .unsteady_plan import CrossSectionResultsCollection, _CrossSectionResultsBase

if TYPE_CHECKING:
    import h5py

    from .unsteady_plan import FlowAreaResults, FlowAreaResultsCollection, UnsteadyPlan

# ---------------------------------------------------------------------------
# HDF path constants -- 1D cross sections
# ---------------------------------------------------------------------------
_SED_ROOT = "Results/Unsteady/Output/Output Blocks/Sediment/Sediment Time Series"
_SED_TS_XS = f"{_SED_ROOT}/Cross Sections"
_SED_TIME_STAMP_DS = f"{_SED_ROOT}/Time Date Stamp"
_XS_GEOM_ATTRS = f"{_XS_ROOT}/Attributes"
_GRAIN_CLASS_NAMES = "Sediment/Grain Class Data/Grain Class Names"
_N_GRAIN_CLASSES = 20

Quantity = Literal["mass", "vol"]
_QUANTITY_BASE = {"mass": "Mass", "vol": "Vol"}

# ---------------------------------------------------------------------------
# HDF path constants -- 2D flow areas
# ---------------------------------------------------------------------------
_SED_BED_ROOT = "Results/Unsteady/Output/Output Blocks/Sediment Bed"
_SED_BED_TS_ROOT = f"{_SED_BED_ROOT}/Unsteady Time Series"
_SED_BED_TS_2D = f"{_SED_BED_TS_ROOT}/2D Flow Areas"
_SED_BED_TIME_STAMP_DS = f"{_SED_BED_TS_ROOT}/Time Date Stamp"
_SED_BED_SUM_2D = f"{_SED_BED_ROOT}/Summary Output/2D Flow Areas"

_SED_TRANSPORT_ROOT = "Results/Unsteady/Output/Output Blocks/Sediment Transport"
_SED_TRANSPORT_TS_ROOT = f"{_SED_TRANSPORT_ROOT}/Unsteady Time Series"
_SED_TRANSPORT_TS_2D = f"{_SED_TRANSPORT_TS_ROOT}/2D Flow Areas"
_SED_TRANSPORT_TIME_STAMP_DS = f"{_SED_TRANSPORT_TS_ROOT}/Time Date Stamp"

ShearComponent = Literal["skin", "total"]
_SHEAR_SUFFIX = {"skin": "Skin", "total": "Total"}


def _grain_class_names(hdf: h5py.File) -> list[str]:
    """Return all 20 grain-class names, in HDF fraction-index order.

    Fraction suffix ``n`` (1-based, e.g. ``"Mass In Cum 5"``) names grain
    class ``result[n - 1]``.

    Raises
    ------
    KeyError
        If ``Sediment/Grain Class Data/Grain Class Names`` is absent --
        e.g. this plan has no sediment transport analysis.
    """
    ds = hdf.get(_GRAIN_CLASS_NAMES)
    if ds is None:
        raise KeyError(
            f"'{_GRAIN_CLASS_NAMES}' not found. "
            "Ensure this plan includes a sediment transport analysis."
        )
    return [_decode(v) for v in ds[:]]


class SedimentCrossSectionResults(_CrossSectionResultsBase):
    """Sediment time-series results for one 1-D cross section.

    Returned by ``plan.sediment.cross_sections()[key]``.  Shares its geometry
    and column-index carrier with the hydraulics cross-section results
    (:class:`~rivia.hdf.unsteady_plan._CrossSectionResultsBase`), pointed at
    the sediment result group instead of Base Output.
    """

    @property
    def effective_depth(self) -> pd.Series:
        """Effective depth time series, indexed by :attr:`timestamps`."""
        return self._series("Effective Depth", "Effective Depth")

    @property
    def effective_width(self) -> pd.Series:
        """Effective width time series, indexed by :attr:`timestamps`."""
        return self._series("Effective Width", "Effective Width")

    @property
    def energy_grade(self) -> pd.Series:
        """Energy grade time series, indexed by :attr:`timestamps`."""
        return self._series("Energy Grade", "Energy Grade")

    @property
    def flow(self) -> pd.Series:
        """Flow time series, indexed by :attr:`timestamps`."""
        return self._series("Flow", "Flow")

    @property
    def velocity(self) -> pd.Series:
        """Velocity time series, indexed by :attr:`timestamps`."""
        return self._series("Velocity", "Velocity")

    @property
    def water_surface(self) -> pd.Series:
        """Water surface elevation time series, indexed by :attr:`timestamps`."""
        return self._series("Water Surface", "Water Surface")

    @property
    def invert_elevation(self) -> pd.Series:
        """Channel invert elevation time series, indexed by :attr:`timestamps`."""
        return self._series("Invert Elevation", "Invert Elevation")

    @property
    def invert_change(self) -> pd.Series:
        """Channel invert change time series, indexed by :attr:`timestamps`."""
        return self._series("Invert Change", "Invert Change")

    @property
    def invert_max(self) -> pd.Series:
        """Maximum invert elevation time series, indexed by :attr:`timestamps`."""
        return self._series("Invert Max", "Invert Max")

    @property
    def invert_min(self) -> pd.Series:
        """Minimum invert elevation time series, indexed by :attr:`timestamps`."""
        return self._series("Invert Min", "Invert Min")

    @property
    def mean_effective_invert_change(self) -> pd.Series:
        """Mean effective invert change time series, indexed by :attr:`timestamps`."""
        return self._series(
            "Mean Effective Invert Change", "Mean Effective Invert Change"
        )

    @property
    def mean_effective_invert_elevation(self) -> pd.Series:
        """Mean effective invert elevation, indexed by :attr:`timestamps`."""
        return self._series(
            "Mean Effective Invert Elevation", "Mean Effective Invert Elevation"
        )

    # ------------------------------------------------------------------
    # Mass / Vol consolidated accessors
    # ------------------------------------------------------------------

    def _consolidated(self, quantity: Quantity, base_suffix: str) -> pd.DataFrame:
        """Build a Total + per-grain-class DataFrame for one record family.

        Iterates the full 1-20 grain-fraction range and includes only the
        columns actually present in the HDF file -- the active grain classes
        depend on the user's sediment gradation setup and are never
        hard-coded.

        Raises
        ------
        ValueError
            If *quantity* is not ``"mass"`` or ``"vol"``.
        KeyError
            If the Total record for the requested *quantity* is absent --
            e.g. *quantity* does not match this file's sediment output mode.
        """
        if quantity not in _QUANTITY_BASE:
            raise ValueError(
                f"quantity={quantity!r} is not valid; choose 'mass' or 'vol'."
            )
        base = f"{_QUANTITY_BASE[quantity]} {base_suffix}"
        grain_names = _grain_class_names(self._hdf)

        data: dict[str, np.ndarray] = {"Total": self._load(base)}
        for n in range(1, _N_GRAIN_CLASSES + 1):
            name = f"{base} {n}"
            if f"{self._root}/{name}" in self._hdf:
                data[grain_names[n - 1]] = self._load(name)
        return pd.DataFrame(data, index=self.timestamps)

    def get_cumulative_inflow(self, quantity: Quantity) -> pd.DataFrame:
        """Cumulative sediment inflow, split by grain class.

        Parameters
        ----------
        quantity : {"mass", "vol"}
            Whether to read the ``Mass In Cum`` or ``Vol In Cum`` record
            family.  Required -- a sediment analysis is run in exactly one
            of these two output modes, and only the matching family exists
            in the HDF file.

        Returns
        -------
        pandas.DataFrame
            Indexed by :attr:`timestamps`.  First column ``"Total"``,
            followed by one column per grain class actually present in the
            file (named from ``Sediment/Grain Class Data/Grain Class Names``).

        Raises
        ------
        ValueError
            If *quantity* is not ``"mass"`` or ``"vol"``.
        KeyError
            If the ``{Mass|Vol} In Cum`` total record is absent.
        """
        return self._consolidated(quantity, "In Cum")

    def get_cumulative_outflow(self, quantity: Quantity) -> pd.DataFrame:
        """Cumulative sediment outflow, split by grain class.

        Parameters
        ----------
        quantity : {"mass", "vol"}
            Whether to read the ``Mass Out Cum`` or ``Vol Out Cum`` record
            family.  Required -- see :meth:`get_cumulative_inflow`.

        Returns
        -------
        pandas.DataFrame
            Indexed by :attr:`timestamps`.  First column ``"Total"``,
            followed by one column per grain class actually present in the
            file.

        Raises
        ------
        ValueError
            If *quantity* is not ``"mass"`` or ``"vol"``.
        KeyError
            If the ``{Mass|Vol} Out Cum`` total record is absent.
        """
        return self._consolidated(quantity, "Out Cum")


class SedimentFlowAreaResults:
    """Sediment Bed and Sediment Transport results for one 2-D flow area.

    Returned by ``plan.sediment.flow_areas()[name]``.  ``Sediment Bed`` and
    ``Sediment Transport`` are two independent HEC-RAS output blocks with
    their own timestamp axes; both are exposed here since the two blocks
    describe the same flow area.

    **Naming conventions:** bare-noun properties and mode-only methods
    (``bed_elevation``, ``bed_change``, ``initial_bed_elevation``,
    ``max_bed_elevation``, ``min_bed_elevation``,
    ``transport_bed_shear_stress``) return a raw HDF dataset or a thin
    array/summary over *all* cells or faces -- no cell/face selector is
    accepted or required. ``get_<quantity>(*, cell=/face=, ...)`` methods
    are single-cell/face accessors; the selector is required and raises
    ``ValueError`` if omitted. ``get_bed_elevation``, ``get_bed_change``,
    and ``get_bed_shear_stress`` return a plain ``pandas.Series`` for one cell;
    ``get_fraction_suspended``, ``get_total_load_concentration``, and
    ``get_transport_rate`` additionally split the result by grain class,
    returning a ``pandas.DataFrame``.

    Parameters
    ----------
    hdf :
        Open ``h5py.File`` handle -- kept alive by the parent
        :class:`UnsteadySediment`.
    name :
        Flow area name, matching ``Geometry/2D Flow Areas``.
    bed_timestamps_fn :
        Zero-argument callable returning the Sediment Bed block's
        ``pd.DatetimeIndex``.  Resolved lazily on first access.
    transport_timestamps_fn :
        Zero-argument callable returning the Sediment Transport block's
        ``pd.DatetimeIndex``.  Resolved lazily on first access.
    hydraulics_fn :
        Zero-argument callable returning this flow area's hydraulics
        :class:`~rivia.hdf.unsteady_plan.FlowAreaResults`, used only for
        its mesh geometry
        (:meth:`~rivia.hdf.unsteady_plan.FlowAreaResults.faces_along_line`).
        Resolved lazily on first access.
    """

    def __init__(
        self,
        hdf: h5py.File,
        name: str,
        bed_timestamps_fn: Callable[[], pd.DatetimeIndex],
        transport_timestamps_fn: Callable[[], pd.DatetimeIndex],
        hydraulics_fn: Callable[[], FlowAreaResults],
    ) -> None:
        self.name = name
        self._hdf = hdf
        self._bed_root = f"{_SED_BED_TS_2D}/{name}"
        self._bed_sum_root = f"{_SED_BED_SUM_2D}/{name}"
        self._transport_root = f"{_SED_TRANSPORT_TS_2D}/{name}"
        self._bed_timestamps_fn = bed_timestamps_fn
        self._transport_timestamps_fn = transport_timestamps_fn
        self._hydraulics_fn = hydraulics_fn
        self._bed_timestamps: pd.DatetimeIndex | None = None
        self._transport_timestamps: pd.DatetimeIndex | None = None
        self._hydraulics: FlowAreaResults | None = None

    def __repr__(self) -> str:
        return f"SedimentFlowAreaResults({self.name!r})"

    # ------------------------------------------------------------------
    # Timestamps
    # ------------------------------------------------------------------

    @property
    def bed_timestamps(self) -> pd.DatetimeIndex:
        """Sediment Bed block time stamps as a ``pd.DatetimeIndex``."""
        if self._bed_timestamps is None:
            self._bed_timestamps = self._bed_timestamps_fn()
        return self._bed_timestamps

    @property
    def transport_timestamps(self) -> pd.DatetimeIndex:
        """Sediment Transport block time stamps as a ``pd.DatetimeIndex``."""
        if self._transport_timestamps is None:
            self._transport_timestamps = self._transport_timestamps_fn()
        return self._transport_timestamps

    @property
    def _hydraulics_geometry(self) -> FlowAreaResults:
        """This flow area's hydraulics results, for mesh-geometry access only."""
        if self._hydraulics is None:
            self._hydraulics = self._hydraulics_fn()
        return self._hydraulics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dataset(self, root: str, name: str) -> h5py.Dataset:
        ds = self._hdf.get(f"{root}/{name}")
        if ds is None:
            raise KeyError(f"Dataset '{name}' not found at '{root}'.")
        return ds

    def _column(self, root: str, name: str, index: int) -> np.ndarray:
        return np.array(self._dataset(root, name)[:, index])

    def _consolidated(
        self, root: str, base: str, index: int, timestamps: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Build a Total + per-grain-class DataFrame for one cell/face column.

        Iterates every grain-class name and includes only the columns
        actually present in the HDF file -- the active grain classes depend
        on the user's sediment gradation setup and are never hard-coded.
        """
        total = self._column(root, f"{base} - Total", index)
        data: dict[str, np.ndarray] = {"Total": total}
        for name in _grain_class_names(self._hdf):
            key = f"{base} - {name}"
            if f"{root}/{key}" in self._hdf:
                data[name] = self._column(root, key, index)
        return pd.DataFrame(data, index=timestamps)

    def _consolidated_signed_sum(
        self,
        root: str,
        base: str,
        indexes: list[int],
        signs: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Build a Total + per-grain-class DataFrame summed over several signed columns.

        Same column discovery as :meth:`_consolidated`, but each named record
        is read for every index in *indexes*, multiplied by the matching
        entry of *signs* (``+1``/``-1``), and summed -- used to net a
        multi-face fence rather than read a single cell/face column.
        """

        def signed_sum(name: str) -> np.ndarray:
            cols = np.column_stack([self._column(root, name, i) for i in indexes])
            return np.asarray((cols * signs[np.newaxis, :]).sum(axis=1))

        data: dict[str, np.ndarray] = {"Total": signed_sum(f"{base} - Total")}
        for name in _grain_class_names(self._hdf):
            key = f"{base} - {name}"
            if f"{root}/{key}" in self._hdf:
                data[name] = signed_sum(key)
        return pd.DataFrame(data, index=timestamps)

    # ------------------------------------------------------------------
    # Sediment Bed
    # ------------------------------------------------------------------

    @property
    def bed_elevation(self) -> h5py.Dataset:
        """Cell bed elevation, shape ``(n_t, n_cells)``.

        Slice to control what is loaded, e.g. ``bed_elevation[t]``.
        """
        return self._dataset(self._bed_root, "Cell Bed Elevation")

    @property
    def bed_change(self) -> h5py.Dataset:
        """Cell bed change, shape ``(n_t, n_cells)``.

        Slice to control what is loaded, e.g. ``bed_change[t]``.
        """
        return self._dataset(self._bed_root, "Cell Bed Change")

    @property
    def initial_bed_elevation(self) -> np.ndarray:
        """Cell bed elevation before the simulation starts, shape ``(n_cells,)``."""
        return np.array(self._dataset(self._bed_root, "Cell Initial Bed Elevation")[0])

    @property
    def max_bed_elevation(self) -> np.ndarray:
        """Maximum bed elevation summary, as written by HEC-RAS.

        Returns the ``Maximum Bed Elevation`` dataset unmodified.  Unlike the
        hydraulics ``max_water_surface`` summary (shape ``(2, n_cells)``,
        value/time rows per cell), this dataset's shape has not been
        confirmed to follow that convention across HEC-RAS projects -- it is
        returned as-is rather than parsed into a DataFrame.
        """
        return np.array(self._dataset(self._bed_sum_root, "Maximum Bed Elevation"))

    @property
    def min_bed_elevation(self) -> np.ndarray:
        """Minimum bed elevation summary, as written by HEC-RAS.

        See :attr:`max_bed_elevation` for a caveat on this dataset's shape.
        """
        return np.array(self._dataset(self._bed_sum_root, "Minimum Bed Elevation"))

    def get_bed_elevation(self, *, cell: int | None = None) -> pd.Series:
        """Bed elevation over time for one cell.

        Parameters
        ----------
        cell : int, optional
            0-based cell index.  Required.

        Returns
        -------
        pandas.Series
            Indexed by :attr:`bed_timestamps`, named ``"Bed Elevation"``.

        Raises
        ------
        ValueError
            If *cell* is not specified.
        """
        if cell is None:
            raise ValueError("cell must be specified.")
        values = np.array(self.bed_elevation[:, cell])
        return pd.Series(values, index=self.bed_timestamps, name="Bed Elevation")

    def get_bed_change(self, *, cell: int | None = None) -> pd.Series:
        """Bed change over time for one cell.

        Parameters
        ----------
        cell : int, optional
            0-based cell index.  Required.

        Returns
        -------
        pandas.Series
            Indexed by :attr:`bed_timestamps`, named ``"Bed Change"``.

        Raises
        ------
        ValueError
            If *cell* is not specified.
        """
        if cell is None:
            raise ValueError("cell must be specified.")
        values = np.array(self.bed_change[:, cell])
        return pd.Series(values, index=self.bed_timestamps, name="Bed Change")

    # ------------------------------------------------------------------
    # Sediment Transport
    # ------------------------------------------------------------------

    def transport_bed_shear_stress(self, *, component: ShearComponent) -> h5py.Dataset:
        """Cell bed shear stress, shape ``(n_t, n_cells)``.

        Read from the Sediment Transport output block (see
        :attr:`~UnsteadySediment.transport_timestamps`), despite "bed" in the
        name -- shear stress on the bed is computed as part of the Transport
        block, not the Sediment Bed block.

        Parameters
        ----------
        component : {"skin", "total"}
            Whether to read ``Cell Bed Shear Stress - Skin`` or
            ``... - Total``.  Required -- these are two distinct physical
            quantities with no meaningful default.

        Returns
        -------
        h5py.Dataset
            Slice to control what is loaded, e.g. ``result[t]`` for one
            timestep or ``result[:]`` for the full array.
        """
        return self._dataset(
            self._transport_root, f"Cell Bed Shear Stress - {_SHEAR_SUFFIX[component]}"
        )

    def get_bed_shear_stress(
        self, *, cell: int | None = None, component: ShearComponent
    ) -> pd.Series:
        """Bed shear stress over time for one cell.

        Parameters
        ----------
        cell : int, optional
            0-based cell index.  Required.
        component : {"skin", "total"}
            Whether to read the skin or total bed shear stress.  Required --
            see :meth:`transport_bed_shear_stress`.

        Returns
        -------
        pandas.Series
            Indexed by :attr:`transport_timestamps`, named
            ``"Bed Shear Stress ({component})"``.

        Raises
        ------
        ValueError
            If *cell* is not specified.
        """
        if cell is None:
            raise ValueError("cell must be specified.")
        values = np.array(self.transport_bed_shear_stress(component=component)[:, cell])
        return pd.Series(
            values,
            index=self.transport_timestamps,
            name=f"Bed Shear Stress ({component})",
        )

    def get_fraction_suspended(self, *, cell: int | None = None) -> pd.DataFrame:
        """Suspended-load fraction over time for one cell, split by grain class.

        Parameters
        ----------
        cell : int, optional
            0-based cell index.  Required.

        Returns
        -------
        pandas.DataFrame
            Indexed by :attr:`transport_timestamps`.  First column
            ``"Total"``, followed by one column per grain class actually
            present in the file.

        Raises
        ------
        ValueError
            If *cell* is not specified.
        """
        if cell is None:
            raise ValueError("cell must be specified.")
        return self._consolidated(
            self._transport_root,
            "Cell Fraction Suspended",
            cell,
            self.transport_timestamps,
        )

    def get_total_load_concentration(self, *, cell: int | None = None) -> pd.DataFrame:
        """Total-load concentration over time for one cell, split by grain class.

        Parameters
        ----------
        cell : int, optional
            0-based cell index.  Required.

        Returns
        -------
        pandas.DataFrame
            Indexed by :attr:`transport_timestamps`.  First column
            ``"Total"``, followed by one column per grain class actually
            present in the file.

        Raises
        ------
        ValueError
            If *cell* is not specified.
        """
        if cell is None:
            raise ValueError("cell must be specified.")
        return self._consolidated(
            self._transport_root,
            "Cell Total-load Concentration",
            cell,
            self.transport_timestamps,
        )

    def get_transport_rate(
        self, *, face: int | None = None, capacity: bool = False
    ) -> pd.DataFrame:
        """Total-load transport rate over time for one face, split by grain class.

        Parameters
        ----------
        face : int, optional
            0-based face index.  Required.
        capacity : bool, optional
            When ``True``, read the ``Face Total-load Transport Capacity``
            record instead of ``Face Total-load Transport Rate``.  Unlike
            the rate record, capacity is written as a single flat dataset
            with no per-grain-class breakdown, so the result has only a
            ``"Total"`` column.  Default ``False``.

        Returns
        -------
        pandas.DataFrame
            Indexed by :attr:`transport_timestamps`.  When *capacity* is
            ``False``, first column ``"Total"``, followed by one column per
            grain class actually present in the file.  When *capacity* is
            ``True``, only a ``"Total"`` column.

        Raises
        ------
        ValueError
            If *face* is not specified.
        """
        if face is None:
            raise ValueError("face must be specified.")
        if capacity:
            total = self._column(
                self._transport_root, "Face Total-load Transport Capacity", face
            )
            return pd.DataFrame({"Total": total}, index=self.transport_timestamps)
        return self._consolidated(
            self._transport_root,
            "Face Total-load Transport Rate",
            face,
            self.transport_timestamps,
        )

    def transport_rate_along_line(
        self,
        xy: np.ndarray,
        *,
        method: Literal["walk", "shortest_path"] = "shortest_path",
    ) -> pd.DataFrame:
        """Net total-load transport rate through a user-supplied profile line.

        Identifies the mesh face "fence" that best approximates *xy* via
        this flow area's hydraulics
        :meth:`~rivia.hdf.unsteady_plan.FlowAreaResults.faces_along_line`,
        then sums signed per-face transport rate -- Total and each grain
        class -- across that fence at every timestep.

        The sign convention matches RASMapper and
        :meth:`~rivia.hdf.unsteady_plan.FlowAreaResults.flow_across_line`:
        flow from left bank to right bank (when facing from *xy* start to
        *xy* end) is **positive**.

        Parameters
        ----------
        xy : ndarray, shape ``(n_pts, 2)``
            Profile polyline vertices ``(x, y)`` drawn from left bank to
            right bank.
        method : {"shortest_path", "walk"}, optional
            Face-selection method passed to ``faces_along_line``.  Default
            is ``"shortest_path"``.

        Returns
        -------
        pandas.DataFrame
            Indexed by :attr:`transport_timestamps`.  First column
            ``"Total"``, followed by one column per grain class actually
            present in the file.  Values are the net (signed) transport
            rate across the fence.

        Raises
        ------
        ValueError
            If the polyline does not intersect the mesh or no connected
            face path can be found.
        NotImplementedError
            If ``method="walk"``.

        See Also
        --------
        get_transport_rate : per-face transport rate, no fence summation
        """
        xy = np.asarray(xy, dtype=np.float64)
        faces_df = self._hydraulics_geometry.faces_along_line(xy, method=method)
        face_ids = faces_df["face"].tolist()
        orientations = faces_df["orientation"].to_numpy(dtype=bool)  # True -> negate
        signs = np.where(orientations, -1.0, 1.0)
        return self._consolidated_signed_sum(
            self._transport_root,
            "Face Total-load Transport Rate",
            face_ids,
            signs,
            self.transport_timestamps,
        )


class SedimentFlowAreaResultsCollection(Mapping[str, SedimentFlowAreaResults]):
    """Collection of :class:`SedimentFlowAreaResults`, keyed by flow-area name.

    Returned by :meth:`UnsteadySediment.flow_areas`.  Flow area names are
    taken from the plan's geometry (``Geometry/2D Flow Areas``) rather than
    from the sediment result groups themselves, so a name is available even
    if one of the two sediment blocks is absent for that area; the specific
    accessor raises ``KeyError`` in that case.

    Supports ``[name]`` and 0-based integer index (in geometry order), e.g.
    ``coll[0]``.
    """

    def __init__(
        self,
        hdf: h5py.File,
        names: list[str],
        bed_timestamps_fn: Callable[[], pd.DatetimeIndex],
        transport_timestamps_fn: Callable[[], pd.DatetimeIndex],
        hydraulics_flow_areas: FlowAreaResultsCollection,
    ) -> None:
        self._hdf = hdf
        self._names = names
        self._bed_timestamps_fn = bed_timestamps_fn
        self._transport_timestamps_fn = transport_timestamps_fn
        self._hydraulics_flow_areas = hydraulics_flow_areas
        self._cache: dict[str, SedimentFlowAreaResults] = {}

    @overload
    def __getitem__(self, key: int) -> SedimentFlowAreaResults: ...
    @overload
    def __getitem__(self, key: str) -> SedimentFlowAreaResults: ...

    def __getitem__(self, key: int | str) -> SedimentFlowAreaResults:
        if isinstance(key, int):
            try:
                key = self._names[key]
            except IndexError:
                raise IndexError(
                    f"Index {key} out of range for {len(self._names)} flow areas"
                ) from None
        name = key
        if name not in self._names:
            raise KeyError(f"2D flow area {name!r} not found. Available: {self._names}")
        if name not in self._cache:

            def hydraulics_fn(n: str = name) -> FlowAreaResults:
                return self._hydraulics_flow_areas[n]

            self._cache[name] = SedimentFlowAreaResults(
                self._hdf,
                name,
                self._bed_timestamps_fn,
                self._transport_timestamps_fn,
                hydraulics_fn=hydraulics_fn,
            )
        return self._cache[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)


class UnsteadySediment:
    """Sediment results view for an :class:`~rivia.hdf.unsteady_plan.UnsteadyPlan`.

    Does not own an HDF handle -- borrows ``plan._hdf`` and is invalidated
    once the parent plan is closed.  Reuses the plan's existing cross-section
    geometry join (:class:`~rivia.hdf.unsteady_plan.CrossSectionResultsCollection`)
    pointed at the sediment result group instead of Base Output.

    Parameters
    ----------
    plan :
        The parent :class:`~rivia.hdf.unsteady_plan.UnsteadyPlan`.  Reached
        through its :attr:`~rivia.hdf.unsteady_plan.UnsteadyPlan.sediment`
        property rather than constructed directly.
    """

    def __init__(self, plan: UnsteadyPlan) -> None:
        self._plan = plan
        self._hdf = plan._hdf
        self._cross_sections: CrossSectionResultsCollection | None = None
        self._flow_areas: SedimentFlowAreaResultsCollection | None = None

    @property
    def cross_section_timestamps(self) -> pd.DatetimeIndex:
        """1D cross-section sediment output time stamps as a ``pd.DatetimeIndex``.

        Parsed from ``.../Sediment/Sediment Time Series/Time Date Stamp``.
        This interval is independent of the hydraulics mapping interval
        (:attr:`~rivia.hdf.unsteady_plan.UnsteadyPlan.mapping_timestamps`)
        and of the 2D :attr:`bed_timestamps` / :attr:`transport_timestamps`.

        Raises
        ------
        KeyError
            If the sediment ``Time Date Stamp`` dataset is absent -- e.g.
            this plan has no sediment transport analysis.
        """
        ds = self._hdf.get(_SED_TIME_STAMP_DS)
        if ds is None:
            raise KeyError(
                f"'{_SED_TIME_STAMP_DS}' not found. "
                "Ensure this plan includes a sediment transport analysis."
            )
        raw = np.array(ds).astype(str)
        return _parse_hec_ts_array(raw, _RAS_TS_FMT)

    def cross_sections(self) -> CrossSectionResultsCollection:
        """1-D cross sections with sediment time-series results.

        The sediment result group has no ``Cross Section Attributes`` table
        of its own, so cross sections are joined to result columns using the
        geometry ``Geometry/Cross Sections/Attributes`` table directly.

        Returns
        -------
        CrossSectionResultsCollection
            Collection supporting ``[key]``, integer index, and ``names``,
            with items of type :class:`SedimentCrossSectionResults`.
            Timestamps are available as ``coll.timestamps``
            (:attr:`cross_section_timestamps`).
        """
        if self._cross_sections is None:
            self._cross_sections = CrossSectionResultsCollection(
                self._hdf,
                _SED_TS_XS,
                result_cls=SedimentCrossSectionResults,
                attrs_path=_XS_GEOM_ATTRS,
                timestamps_fn=lambda: self.cross_section_timestamps,
            )
        return self._cross_sections

    # ------------------------------------------------------------------
    # 2D flow areas
    # ------------------------------------------------------------------

    @property
    def bed_timestamps(self) -> pd.DatetimeIndex:
        """Sediment Bed block time stamps as a ``pd.DatetimeIndex``.

        Parsed from ``Sediment Bed/Unsteady Time Series/Time Date Stamp``.
        Independent of :attr:`cross_section_timestamps` (the 1D axis) and of
        :attr:`transport_timestamps`.

        Raises
        ------
        KeyError
            If the Sediment Bed ``Time Date Stamp`` dataset is absent -- e.g.
            this plan has no 2D sediment transport analysis.
        """
        ds = self._hdf.get(_SED_BED_TIME_STAMP_DS)
        if ds is None:
            raise KeyError(
                f"'{_SED_BED_TIME_STAMP_DS}' not found. "
                "Ensure this plan includes a 2D sediment transport analysis."
            )
        raw = np.array(ds).astype(str)
        return _parse_hec_ts_array(raw, _RAS_TS_FMT)

    @property
    def transport_timestamps(self) -> pd.DatetimeIndex:
        """Sediment Transport block time stamps as a ``pd.DatetimeIndex``.

        Parsed from ``Sediment Transport/Unsteady Time Series/Time Date
        Stamp``.  Written at the unsteady-flow mapping interval.

        Raises
        ------
        KeyError
            If the Sediment Transport ``Time Date Stamp`` dataset is absent
            -- e.g. this plan has no 2D sediment transport analysis.
        """
        ds = self._hdf.get(_SED_TRANSPORT_TIME_STAMP_DS)
        if ds is None:
            raise KeyError(
                f"'{_SED_TRANSPORT_TIME_STAMP_DS}' not found. "
                "Ensure this plan includes a 2D sediment transport analysis."
            )
        raw = np.array(ds).astype(str)
        return _parse_hec_ts_array(raw, _RAS_TS_FMT)

    def flow_areas(self) -> SedimentFlowAreaResultsCollection:
        """2D flow areas with sediment bed and transport time-series results.

        Flow area names are taken from ``plan.flow_areas`` (geometry), so
        the collection is available even when a particular area lacks
        sediment output; the specific accessor raises ``KeyError`` in that
        case.

        Returns
        -------
        SedimentFlowAreaResultsCollection
            Mapping keyed by flow-area name, with items of type
            :class:`SedimentFlowAreaResults`.
        """
        if self._flow_areas is None:
            self._flow_areas = SedimentFlowAreaResultsCollection(
                self._hdf,
                self._plan.flow_areas.names,
                bed_timestamps_fn=lambda: self.bed_timestamps,
                transport_timestamps_fn=lambda: self.transport_timestamps,
                hydraulics_flow_areas=self._plan.flow_areas,
            )
        return self._flow_areas
