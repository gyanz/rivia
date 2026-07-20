"""UnsteadySediment - read HEC-RAS sediment results from unsteady plan HDF5 files.

Sediment output is written into the same plan HDF file as the hydraulics
results handled by :mod:`rivia.hdf.unsteady_plan`, under
``Results/Unsteady/Output/Output Blocks/Sediment/...``.  It is implemented in
its own module so ``unsteady_plan.py`` -- already large with hydraulics
result classes -- does not grow further.

:class:`UnsteadySediment` does not open its own HDF handle: it borrows the
handle already opened by the parent
:class:`~rivia.hdf.unsteady_plan.UnsteadyPlan` (reached via
``plan.sediment``) and reuses that plan's cross-section geometry join
(:class:`~rivia.hdf.unsteady_plan.CrossSectionResultsCollection`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from ._base import _RAS_TS_FMT, _parse_hec_ts_array
from .geometry import _XS_ROOT, _decode
from .unsteady_plan import CrossSectionResultsCollection, _CrossSectionResultsBase

if TYPE_CHECKING:
    import h5py

    from .unsteady_plan import UnsteadyPlan

# ---------------------------------------------------------------------------
# HDF path constants
# ---------------------------------------------------------------------------
_SED_ROOT = "Results/Unsteady/Output/Output Blocks/Sediment/Sediment Time Series"
_SED_TS_XS = f"{_SED_ROOT}/Cross Sections"
_SED_TIME_STAMP_DS = f"{_SED_ROOT}/Time Date Stamp"
_XS_GEOM_ATTRS = f"{_XS_ROOT}/Attributes"
_GRAIN_CLASS_NAMES = "Sediment/Grain Class Data/Grain Class Names"
_N_GRAIN_CLASSES = 20

Quantity = Literal["mass", "vol"]
_QUANTITY_BASE = {"mass": "Mass", "vol": "Vol"}


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

    def cumulative_inflow(self, quantity: Quantity) -> pd.DataFrame:
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

    def cumulative_outflow(self, quantity: Quantity) -> pd.DataFrame:
        """Cumulative sediment outflow, split by grain class.

        Parameters
        ----------
        quantity : {"mass", "vol"}
            Whether to read the ``Mass Out Cum`` or ``Vol Out Cum`` record
            family.  Required -- see :meth:`cumulative_inflow`.

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

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        """Sediment output time stamps as a ``pd.DatetimeIndex``.

        Parsed from ``.../Sediment/Sediment Time Series/Time Date Stamp``.
        This interval is independent of the hydraulics mapping interval
        (:attr:`~rivia.hdf.unsteady_plan.UnsteadyPlan.mapping_timestamps`).

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
            Timestamps are available as ``coll.timestamps`` (:attr:`timestamps`).
        """
        if self._cross_sections is None:
            self._cross_sections = CrossSectionResultsCollection(
                self._hdf,
                _SED_TS_XS,
                result_cls=SedimentCrossSectionResults,
                attrs_path=_XS_GEOM_ATTRS,
                timestamps_fn=lambda: self.timestamps,
            )
        return self._cross_sections
