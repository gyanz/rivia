"""QuasiUnsteadyPlan - read HEC-RAS quasi-unsteady flow plan HDF5 files (.p*.hdf).

Quasi-unsteady flow plans are used almost exclusively for 1-D sediment
transport analyses.  Unlike :class:`~rivia.hdf.unsteady_plan.UnsteadyPlan`,
they have no 2-D flow areas or storage areas, and sediment results *are* the
plan's primary (only) result block -- there is no separate hydraulics-only
Base Output to layer sediment on top of, so sediment properties live
directly on the objects returned by :meth:`QuasiUnsteadyPlan.cross_sections`.

Structures (bridges, culverts, inline/lateral structures) may exist in the
geometry -- reachable via the inherited
:attr:`~rivia.hdf.geometry.Geometry.structures` -- but HEC-RAS's
quasi-unsteady sediment engine writes no per-structure time-series results,
so no structure result classes exist here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ._base import _RAS_TS_FMT, _parse_hec_ts_array, _PlanHdf
from .geometry import Geometry, _decode
from .unsteady_plan import CrossSectionResultsCollection, _CrossSectionResultsBase
from .unsteady_sediment import Quantity, _grain_class_names

logger = logging.getLogger("rivia.hdf")

# ---------------------------------------------------------------------------
# HDF path constants
# ---------------------------------------------------------------------------
_SED_ROOT = "Results/Sediment/Output Blocks/Sediment/Sediment Time Series"
_SED_TS_XS = f"{_SED_ROOT}/Cross Sections"
_SED_TIME_STAMP_DS = f"{_SED_ROOT}/Time Date Stamp"
_SED_GEOM_ATTRS = "Results/Sediment/Geometry Info/Cross Section Attributes"

_SED_SE_ROOT = "Results/Sediment/Output Blocks/Sediment SE/Sediment Time Series"
_SED_SE_XS = f"{_SED_SE_ROOT}/Cross Section SE"
_SED_SE_TIME_STAMP_DS = f"{_SED_SE_ROOT}/Time Date Stamp"

_QUANTITY_BASE = {"mass": "Mass", "vol": "Vol"}
_N_GRAIN_CLASSES = 20


# ---------------------------------------------------------------------------
# QuasiUnsteadyCrossSectionResults
# ---------------------------------------------------------------------------


class QuasiUnsteadyCrossSectionResults(_CrossSectionResultsBase):
    """Sediment time-series results for one 1-D cross section.

    Returned by ``plan.cross_sections()[key]``.  Shares its geometry and
    column-index carrier
    (:class:`~rivia.hdf.unsteady_plan._CrossSectionResultsBase`) with the
    unsteady-plan and unsteady-sediment result classes.

    A number of the scalar properties below are intentionally duplicated
    from :class:`~rivia.hdf.unsteady_sediment.SedimentCrossSectionResults`
    rather than shared through a common base -- HEC-RAS's sediment output
    writer produces the same "Cross Sections" schema for both the
    quasi-unsteady and unsteady engines, but this plan type additionally
    writes many variables that don't apply to unsteady runs.

    Which of these properties exist in a given HDF file depends on the
    **sediment output level** chosen when the plan was run (HEC-RAS's
    "Sediment Output Level" setting, 1-6): higher levels write
    progressively more variables (e.g. transport-diagnostic quantities
    like :attr:`reynolds_number`, :attr:`shields_number`, and the
    :attr:`d16_cover`/:attr:`d84_cover` percentiles only appear at level 6).
    Accessing a property whose underlying dataset is absent from this file
    raises ``KeyError`` (via :meth:`_series`/:meth:`_load`), the same as any
    other missing-dataset case in this class.
    """

    # ------------------------------------------------------------------
    # Scalar properties shared with unsteady sediment (duplicated -- see
    # class docstring)
    # ------------------------------------------------------------------

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
    # Quasi-unsteady-specific scalar properties
    #
    # Many of these are only written at higher HEC-RAS "sediment output
    # level" settings -- a lower-level run may be missing some of them, see
    # class docstring. Several also have per-grain-class siblings (suffix
    # " 1".." 20") that are not exposed here -- only the unsuffixed Total
    # record is, consistent with how ``lat_load_mass_in`` and
    # ``longitudinal_cumulative_mass_change`` already treat their own
    # grain-suffixed siblings. The exception is the ``Mass``/``Vol``
    # ``In``/``Out`` family, which is deliberately grain-broken via
    # :meth:`get_cumulative_inflow`/:meth:`get_cumulative_outflow` instead
    # of exposed here as a bare property.
    # ------------------------------------------------------------------

    @property
    def fall_velocity(self) -> pd.Series:
        """Representative grain fall velocity, indexed by :attr:`timestamps`."""
        return self._series("Fall Velocity", "Fall Velocity")

    @property
    def froude_number_channel(self) -> pd.Series:
        """Channel Froude number time series, indexed by :attr:`timestamps`."""
        return self._series("Froude Number Channel", "Froude Number Channel")

    @property
    def fv_ustar_ratio(self) -> pd.Series:
        """Fall-velocity to shear-velocity ratio, indexed by :attr:`timestamps`."""
        return self._series("FV-Ustar Ratio", "FV-Ustar Ratio")

    @property
    def hydraulic_radius(self) -> pd.Series:
        """Hydraulic radius time series, indexed by :attr:`timestamps`."""
        return self._series("Hydraulic Radius", "Hydraulic Radius")

    @property
    def invert_change_max(self) -> pd.Series:
        """Maximum invert change time series, indexed by :attr:`timestamps`."""
        return self._series("Invert Change Max", "Invert Change Max")

    @property
    def invert_change_min(self) -> pd.Series:
        """Minimum invert change time series, indexed by :attr:`timestamps`."""
        return self._series("Invert Change Min", "Invert Change Min")

    @property
    def lat_load_mass_in(self) -> pd.Series:
        """Total lateral sediment load inflow, indexed by :attr:`timestamps`."""
        return self._series("Lat Load Mass In", "Lat Load Mass In")

    @property
    def lat_load_mass_in_cumulative(self) -> pd.Series:
        """Cumulative total lateral sediment load inflow, indexed by timestamps."""
        return self._series("Lat Load Mass In Cum", "Lat Load Mass In Cum")

    @property
    def lat_struc_mass_diverted(self) -> pd.Series:
        """Sediment mass diverted by a lateral structure, indexed by timestamps."""
        return self._series("Lat Struc Mass Div", "Lat Struc Mass Div")

    @property
    def longitudinal_cumulative_mass_change(self) -> pd.Series:
        """Longitudinal cumulative bed mass change, indexed by :attr:`timestamps`."""
        return self._series("Long. Cum Mass Change", "Long. Cum Mass Change")

    @property
    def longitudinal_cumulative_mass_moveable_limit(self) -> pd.Series:
        """Longitudinal cumulative mass at the moveable-bed limit, indexed by ts."""
        return self._series(
            "Long. Cum Mass Moveable Limit", "Long. Cum Mass Moveable Limit"
        )

    @property
    def mannings_n_channel(self) -> pd.Series:
        """Channel Manning's n time series, indexed by :attr:`timestamps`."""
        return self._series("Manning's n Channel", "Manning's n Channel")

    @property
    def mass_bed_change(self) -> pd.Series:
        """Bed mass change for one time step, indexed by :attr:`timestamps`."""
        return self._series("Mass Bed Change", "Mass Bed Change")

    @property
    def mass_bed_change_cumulative(self) -> pd.Series:
        """Cumulative bed mass change, indexed by :attr:`timestamps`."""
        return self._series("Mass Bed Change Cum", "Mass Bed Change Cum")

    @property
    def mass_bed_change_cumulative_max(self) -> pd.Series:
        """Maximum cumulative bed mass change, indexed by :attr:`timestamps`."""
        return self._series("Mass Bed Change Cum Max", "Mass Bed Change Cum Max")

    @property
    def mass_capacity(self) -> pd.Series:
        """Sediment transport mass capacity, indexed by :attr:`timestamps`."""
        return self._series("Mass Capacity", "Mass Capacity")

    @property
    def mass_capacity_cumulative(self) -> pd.Series:
        """Cumulative sediment transport mass capacity, indexed by timestamps."""
        return self._series("Mass Capacity Cum", "Mass Capacity Cum")

    @property
    def mass_cover(self) -> pd.Series:
        """Sediment mass in the cover layer, indexed by :attr:`timestamps`."""
        return self._series("Mass Cover", "Mass Cover")

    @property
    def mass_in(self) -> pd.Series:
        """Sediment mass inflow for one time step, indexed by :attr:`timestamps`.

        See :meth:`get_cumulative_inflow` for the per-grain-class breakdown
        of the cumulative counterpart of this record.
        """
        return self._series("Mass In", "Mass In")

    @property
    def mass_inactive(self) -> pd.Series:
        """Sediment mass in the inactive layer, indexed by :attr:`timestamps`."""
        return self._series("Mass Inactive", "Mass Inactive")

    @property
    def mass_out(self) -> pd.Series:
        """Sediment mass outflow for one time step, indexed by :attr:`timestamps`.

        See :meth:`get_cumulative_outflow` for the per-grain-class breakdown
        of the cumulative counterpart of this record.
        """
        return self._series("Mass Out", "Mass Out")

    @property
    def mass_subsurface(self) -> pd.Series:
        """Sediment mass in the subsurface layer, indexed by :attr:`timestamps`."""
        return self._series("Mass Subsurface", "Mass Subsurface")

    @property
    def moveable_elevation_left(self) -> pd.Series:
        """Left moveable-bed-limit elevation, indexed by :attr:`timestamps`."""
        return self._series("Moveable Elv L", "Moveable Elv L")

    @property
    def moveable_elevation_right(self) -> pd.Series:
        """Right moveable-bed-limit elevation, indexed by :attr:`timestamps`."""
        return self._series("Moveable Elv R", "Moveable Elv R")

    @property
    def moveable_station_left(self) -> pd.Series:
        """Left moveable-bed-limit station, indexed by :attr:`timestamps`."""
        return self._series("Moveable Sta L", "Moveable Sta L")

    @property
    def moveable_station_right(self) -> pd.Series:
        """Right moveable-bed-limit station, indexed by :attr:`timestamps`."""
        return self._series("Moveable Sta R", "Moveable Sta R")

    @property
    def percent_cover(self) -> pd.Series:
        """Percent of bed mass in the cover layer, indexed by :attr:`timestamps`."""
        return self._series("Percentage (Mass) Cover", "Percentage (Mass) Cover")

    @property
    def percent_inactive(self) -> pd.Series:
        """Percent of bed mass in the inactive layer, indexed by timestamps."""
        return self._series(
            "Percentage (Mass) Inactive", "Percentage (Mass) Inactive"
        )

    @property
    def percent_subsurface(self) -> pd.Series:
        """Percent of bed mass in the subsurface layer, indexed by timestamps."""
        return self._series(
            "Percentage (Mass) Subsurface", "Percentage (Mass) Subsurface"
        )

    @property
    def reduce_armor_factor(self) -> pd.Series:
        """Armoring reduction factor, indexed by :attr:`timestamps`."""
        return self._series("Reduce Armor Factor", "Reduce Armor Factor")

    @property
    def relative_roughness(self) -> pd.Series:
        """Relative roughness, indexed by :attr:`timestamps`."""
        return self._series("Relative Roughness", "Relative Roughness")

    @property
    def reynolds_number(self) -> pd.Series:
        """Grain Reynolds number, indexed by :attr:`timestamps`."""
        return self._series("Reynolds", "Reynolds")

    @property
    def rouse_number(self) -> pd.Series:
        """Rouse number, indexed by :attr:`timestamps`."""
        return self._series("Rouse #", "Rouse #")

    @property
    def sediment_concentration(self) -> pd.Series:
        """Sediment concentration time series, indexed by :attr:`timestamps`."""
        return self._series("Sediment Concentration", "Sediment Concentration")

    @property
    def sediment_discharge(self) -> pd.Series:
        """Sediment discharge time series, indexed by :attr:`timestamps`."""
        return self._series("Sediment Discharge", "Sediment Discharge")

    @property
    def shear_stress(self) -> pd.Series:
        """Bed shear stress time series, indexed by :attr:`timestamps`."""
        return self._series("Shear Stress", "Shear Stress")

    @property
    def shear_velocity(self) -> pd.Series:
        """Shear velocity time series, indexed by :attr:`timestamps`."""
        return self._series("Shear Velocity", "Shear Velocity")

    @property
    def shields_number(self) -> pd.Series:
        """Shields number, indexed by :attr:`timestamps`."""
        return self._series("Shields #", "Shields #")

    @property
    def slope(self) -> pd.Series:
        """Energy slope time series, indexed by :attr:`timestamps`."""
        return self._series("Slope", "Slope")

    @property
    def slope_alternative_output(self) -> pd.Series:
        """Alternative-output slope time series, indexed by :attr:`timestamps`."""
        return self._series("Slope Alternative Output", "Slope Alternative Output")

    @property
    def temperature(self) -> pd.Series:
        """Water temperature time series, indexed by :attr:`timestamps`."""
        return self._series("Temperature", "Temperature")

    @property
    def thickness_cover(self) -> pd.Series:
        """Cover-layer thickness, indexed by :attr:`timestamps`."""
        return self._series("Thickness Cover", "Thickness Cover")

    @property
    def thickness_inactive(self) -> pd.Series:
        """Inactive-layer thickness, indexed by :attr:`timestamps`."""
        return self._series("Thickness Inactive", "Thickness Inactive")

    @property
    def thickness_subsurface(self) -> pd.Series:
        """Subsurface-layer thickness, indexed by :attr:`timestamps`."""
        return self._series("Thickness Subsurface", "Thickness Subsurface")

    @property
    def vol_bed_change_cumulative_min(self) -> pd.Series:
        """Minimum cumulative bed volume change, indexed by :attr:`timestamps`."""
        return self._series("Vol Bed Change Cum Min", "Vol Bed Change Cum Min")

    # ------------------------------------------------------------------
    # Grain-size distribution percentiles
    # ------------------------------------------------------------------

    @property
    def d10_cover(self) -> pd.Series:
        """Cover-layer d10 grain size, indexed by :attr:`timestamps`."""
        return self._series("d10 Cover", "d10 Cover")

    @property
    def d16_cover(self) -> pd.Series:
        """Cover-layer d16 grain size, indexed by :attr:`timestamps`."""
        return self._series("d16 Cover", "d16 Cover")

    @property
    def d50_cover(self) -> pd.Series:
        """Cover-layer d50 grain size, indexed by :attr:`timestamps`."""
        return self._series("d50 Cover", "d50 Cover")

    @property
    def d84_cover(self) -> pd.Series:
        """Cover-layer d84 grain size, indexed by :attr:`timestamps`."""
        return self._series("d84 Cover", "d84 Cover")

    @property
    def d90_cover(self) -> pd.Series:
        """Cover-layer d90 grain size, indexed by :attr:`timestamps`."""
        return self._series("d90 Cover", "d90 Cover")

    @property
    def d10_inactive(self) -> pd.Series:
        """Inactive-layer d10 grain size, indexed by :attr:`timestamps`."""
        return self._series("d10 Inactive", "d10 Inactive")

    @property
    def d16_inactive(self) -> pd.Series:
        """Inactive-layer d16 grain size, indexed by :attr:`timestamps`."""
        return self._series("d16 Inactive", "d16 Inactive")

    @property
    def d50_inactive(self) -> pd.Series:
        """Inactive-layer d50 grain size, indexed by :attr:`timestamps`."""
        return self._series("d50 Inactive", "d50 Inactive")

    @property
    def d84_inactive(self) -> pd.Series:
        """Inactive-layer d84 grain size, indexed by :attr:`timestamps`."""
        return self._series("d84 Inactive", "d84 Inactive")

    @property
    def d90_inactive(self) -> pd.Series:
        """Inactive-layer d90 grain size, indexed by :attr:`timestamps`."""
        return self._series("d90 Inactive", "d90 Inactive")

    @property
    def d10_subsurface(self) -> pd.Series:
        """Subsurface d10 grain size, indexed by :attr:`timestamps`."""
        return self._series("d10 Subsurface", "d10 Subsurface")

    @property
    def d16_subsurface(self) -> pd.Series:
        """Subsurface d16 grain size, indexed by :attr:`timestamps`."""
        return self._series("d16 Subsurface", "d16 Subsurface")

    @property
    def d50_subsurface(self) -> pd.Series:
        """Subsurface d50 grain size, indexed by :attr:`timestamps`."""
        return self._series("d50 Subsurface", "d50 Subsurface")

    @property
    def d84_subsurface(self) -> pd.Series:
        """Subsurface d84 grain size, indexed by :attr:`timestamps`."""
        return self._series("d84 Subsurface", "d84 Subsurface")

    @property
    def d90_subsurface(self) -> pd.Series:
        """Subsurface d90 grain size, indexed by :attr:`timestamps`."""
        return self._series("d90 Subsurface", "d90 Subsurface")

    # ------------------------------------------------------------------
    # Mass / Vol consolidated accessors (grain-class breakdown)
    # ------------------------------------------------------------------

    def _consolidated(self, quantity: Quantity, base_suffix: str) -> pd.DataFrame:
        """Build a Total + per-grain-class DataFrame for one record family.

        Iterates the full 1-20 grain-fraction range and includes only the
        columns actually present in the HDF file -- the active grain classes
        depend on the user's sediment gradation setup and are never
        hard-coded.

        The returned DataFrame's ``attrs["units"]`` is set from the ``Total``
        record's HDF ``Units`` attribute, if present -- all grain-class
        columns share the same physical unit. See :meth:`_series` for the
        caveat on ``pandas`` ``.attrs`` durability.

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
        df = pd.DataFrame(data, index=self.timestamps)
        units = self._units(base)
        if units:
            df.attrs["units"] = units
        return df

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

    # ------------------------------------------------------------------
    # Cross-section geometry over time (Sediment SE block)
    # ------------------------------------------------------------------

    def get_xsec(self, timestep: int) -> tuple[np.ndarray, np.ndarray]:
        """Cross-section station/elevation shape at one XS-geometry checkpoint.

        Named with a ``get_`` prefix (unlike the inherited :attr:`station_elevation`
        field, which holds the *static* geometry survey) both to follow the
        ``get_<quantity>(*, timestep=, ...)`` convention used throughout
        :mod:`rivia.hdf.unsteady_plan` for timestep-selected derived
        accessors, and because a same-named method would be silently
        shadowed by the inherited dataclass field of the same name.

        Quasi-unsteady sediment plans write the evolving XS shape (station
        vs. elevation, reflecting bed aggradation/degradation) at its own
        checkpoint interval, independent of the main sediment time series --
        see
        :attr:`~rivia.hdf.quasi_unsteady_plan.QuasiUnsteadyPlan.cross_section_geometry_timestamps`
        for that interval's own timestamps. The checkpoint count is entirely
        file-dependent -- it can be far sparser than the main time series or
        roughly comparable to it, so don't assume either direction.

        Parameters
        ----------
        timestep : int
            0-based index into
            :attr:`~rivia.hdf.quasi_unsteady_plan.QuasiUnsteadyPlan.cross_section_geometry_timestamps`.

        Returns
        -------
        station : ndarray, shape ``(n,)``
        elevation : ndarray, shape ``(n,)``

        Raises
        ------
        KeyError
            If the ``Sediment SE`` output block is absent from this HDF
            file -- e.g. XS geometry output was not enabled for this run.
        IndexError
            If *timestep* is out of range.
        RuntimeError
            If the row order of the ``Sediment SE`` block does not match
            this cross section's column index in the main sediment result
            block (a HEC-RAS write-order assumption this method relies on).
        """
        se_grp = self._hdf.get(_SED_SE_XS)
        if se_grp is None:
            raise KeyError(
                f"'{_SED_SE_XS}' not found. "
                "Ensure this plan has XS geometry output enabled for its "
                "sediment transport analysis."
            )
        ts_ds = self._hdf.get(_SED_SE_TIME_STAMP_DS)
        if ts_ds is None:
            raise KeyError(f"'{_SED_SE_TIME_STAMP_DS}' not found.")
        raw = np.array(ts_ds).astype(str)
        timestamps = _parse_hec_ts_array(raw, _RAS_TS_FMT)
        if not 0 <= timestep < len(timestamps):
            raise IndexError(
                f"timestep {timestep} out of range (n={len(timestamps)})"
            )

        label = _decode(se_grp["River Reach Station"][self._index])
        expected_prefix = f"{self.river} {self.reach} {self.rs}"
        if not label.startswith(expected_prefix):
            raise RuntimeError(
                f"'Sediment SE' row {self._index} ({label!r}) does not match "
                f"expected cross section {expected_prefix!r}. The "
                "'Sediment SE' and 'Cross Section Attributes' row orders "
                "have diverged for this HDF file."
            )

        date_str = timestamps[timestep].strftime(_RAS_TS_FMT).upper()
        info = se_grp[f"Station Elevation ({date_str}) info"]
        values = se_grp[f"Station Elevation ({date_str}) values"]
        start, count = (int(v) for v in info[self._index])
        data = np.array(values[start : start + count])
        return data[:, 0], data[:, 1]


# ---------------------------------------------------------------------------
# QuasiUnsteadyPlan
# ---------------------------------------------------------------------------


class QuasiUnsteadyPlan(_PlanHdf, Geometry):
    """Read HEC-RAS quasi-unsteady flow plan HDF5 output files (``*.p*.hdf``).

    A plan HDF file contains the same ``Geometry/`` data as a geometry HDF
    file, *plus* ``Results/Sediment/...`` time-series and summary output.
    Quasi-unsteady plans have no 2-D flow areas or storage areas -- use
    :attr:`~rivia.hdf.Geometry.cross_sections` (inherited) for geometry-only
    access and :meth:`cross_sections` here for geometry plus sediment
    results.

    Parameters
    ----------
    filename:
        Path to the plan HDF file.  The ``.hdf`` suffix is appended
        automatically if absent.

    Examples
    --------
    ::

        with QuasiUnsteadyPlan("MBex.p04") as plan:
            ts = plan.cross_section_timestamps
            xs = plan.cross_sections["Yang Flume", "Yang Flume", "1000"]

            wse = xs.water_surface          # pd.Series over ts
            inflow = xs.get_cumulative_inflow("mass")   # DataFrame by grain class

            geom_ts = plan.cross_section_geometry_timestamps
            station, elevation = xs.get_xsec(0)
    """

    def __init__(self, filename: str | Path) -> None:
        super().__init__(filename)
        self._geom_view: Geometry | None = None
        self._plan_cross_sections: CrossSectionResultsCollection | None = None

    # ------------------------------------------------------------------
    # Time stamps
    # ------------------------------------------------------------------

    @property
    def cross_section_timestamps(self) -> pd.DatetimeIndex:
        """Sediment time-series output time stamps as a ``pd.DatetimeIndex``.

        Parsed from ``.../Sediment/Sediment Time Series/Time Date Stamp``.

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

    @property
    def cross_section_geometry_timestamps(self) -> pd.DatetimeIndex:
        """XS-geometry ("Sediment SE") checkpoint time stamps.

        This interval is independent of :attr:`cross_section_timestamps` in
        both spacing and count -- the checkpoint count is entirely
        file-dependent (observed anywhere from far sparser than the main
        time series to roughly comparable to it), so don't assume either
        direction. Parsed from ``.../Sediment SE/Sediment Time Series/Time
        Date Stamp``.

        Raises
        ------
        KeyError
            If the ``Sediment SE`` ``Time Date Stamp`` dataset is absent --
            e.g. XS geometry output was not enabled for this run.
        """
        ds = self._hdf.get(_SED_SE_TIME_STAMP_DS)
        if ds is None:
            raise KeyError(
                f"'{_SED_SE_TIME_STAMP_DS}' not found. "
                "Ensure this plan has XS geometry output enabled for its "
                "sediment transport analysis."
            )
        raw = np.array(ds).astype(str)
        return _parse_hec_ts_array(raw, _RAS_TS_FMT)

    # ------------------------------------------------------------------
    # Collections (override Geometry.cross_sections with a results-aware type)
    # ------------------------------------------------------------------

    @property
    def cross_sections(self) -> CrossSectionResultsCollection:
        """1-D cross sections with geometry and sediment time-series results.

        Returns a :class:`CrossSectionResultsCollection` whose items are
        :class:`QuasiUnsteadyCrossSectionResults`. Supports ``[key]`` by
        string, integer, or ``(river, reach, rs)`` tuple. Timestamps are
        available as ``coll.timestamps`` (:attr:`cross_section_timestamps`).
        """
        if self._plan_cross_sections is None:
            self._plan_cross_sections = CrossSectionResultsCollection(
                self._hdf, _SED_TS_XS,
                result_cls=QuasiUnsteadyCrossSectionResults,
                attrs_path=_SED_GEOM_ATTRS,
                timestamps_fn=lambda: self.cross_section_timestamps,
            )
        return self._plan_cross_sections
