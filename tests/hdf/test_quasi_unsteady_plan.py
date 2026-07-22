"""Tests for rivia.hdf.quasi_unsteady_plan (QuasiUnsteadyPlan, 1-D cross sections).

Uses the real 1-D sediment transport example plan (quasi-unsteady / "Mobile
Bed" example) since a faithful synthetic fixture (grain-class table,
per-fraction records, geometry join, XS-geometry-over-time block) would add
more test-fixture code than it saves.  Skipped when the example file is not
present on the machine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rivia.hdf import QuasiUnsteadyPlan
from rivia.hdf.quasi_unsteady_plan import QuasiUnsteadyCrossSectionResults

from .conftest import QUASI_STEADY_SEDIMENT_HDF, skip_if_no_quasi_steady_example

FIRST_XS = ("Yang Flume", "Yang Flume", "1000")


# ---------------------------------------------------------------------------
# Time stamps
# ---------------------------------------------------------------------------


@skip_if_no_quasi_steady_example
class TestCrossSectionTimestamps:
    def test_returns_datetimeindex(self):
        with QuasiUnsteadyPlan(QUASI_STEADY_SEDIMENT_HDF) as plan:
            ts = plan.cross_section_timestamps
        assert isinstance(ts, pd.DatetimeIndex)
        assert ts.is_monotonic_increasing
        assert len(ts) > 0


@skip_if_no_quasi_steady_example
class TestCrossSectionGeometryTimestamps:
    def test_returns_datetimeindex(self):
        with QuasiUnsteadyPlan(QUASI_STEADY_SEDIMENT_HDF) as plan:
            ts = plan.cross_section_geometry_timestamps
        assert isinstance(ts, pd.DatetimeIndex)
        assert ts.is_monotonic_increasing
        assert len(ts) > 0

    def test_sparser_than_main_time_series(self):
        with QuasiUnsteadyPlan(QUASI_STEADY_SEDIMENT_HDF) as plan:
            geom_ts = plan.cross_section_geometry_timestamps
            main_ts = plan.cross_section_timestamps
        assert len(geom_ts) <= len(main_ts)


# ---------------------------------------------------------------------------
# Cross-section collection / geometry join
# ---------------------------------------------------------------------------


@skip_if_no_quasi_steady_example
class TestCrossSectionCollection:
    def test_collection_cached(self):
        with QuasiUnsteadyPlan(QUASI_STEADY_SEDIMENT_HDF) as plan:
            assert plan.cross_sections is plan.cross_sections

    def test_nonempty(self):
        with QuasiUnsteadyPlan(QUASI_STEADY_SEDIMENT_HDF) as plan:
            coll = plan.cross_sections
            assert len(coll) > 0

    def test_lookup_by_location_tuple(self):
        with QuasiUnsteadyPlan(QUASI_STEADY_SEDIMENT_HDF) as plan:
            xs = plan.cross_sections[FIRST_XS]
        assert isinstance(xs, QuasiUnsteadyCrossSectionResults)
        assert xs.river == FIRST_XS[0]
        assert xs.reach == FIRST_XS[1]
        assert xs.rs == FIRST_XS[2]

    def test_lookup_by_int_index(self):
        with QuasiUnsteadyPlan(QUASI_STEADY_SEDIMENT_HDF) as plan:
            coll = plan.cross_sections
            by_index = coll[0]
            by_tuple = coll[FIRST_XS]
        assert by_index.location == by_tuple.location


# ---------------------------------------------------------------------------
# Scalar time-series properties
# ---------------------------------------------------------------------------


@skip_if_no_quasi_steady_example
class TestScalarProperties:
    @pytest.mark.parametrize(
        "attr",
        [
            "effective_depth",
            "effective_width",
            "energy_grade",
            "flow",
            "velocity",
            "water_surface",
            "invert_elevation",
            "invert_change",
            "invert_max",
            "invert_min",
            "mean_effective_invert_change",
            "mean_effective_invert_elevation",
            "froude_number_channel",
            "invert_change_max",
            "invert_change_min",
            "lat_load_mass_in",
            "lat_load_mass_in_cumulative",
            "lat_struc_mass_diverted",
            "longitudinal_cumulative_mass_change",
            "mannings_n_channel",
            "mass_bed_change_cumulative",
            "sediment_concentration",
            "shear_stress",
            "slope",
            "slope_alternative_output",
            "d10_cover",
            "d50_cover",
            "d90_cover",
            "d10_inactive",
            "d50_inactive",
            "d90_inactive",
            "d10_subsurface",
            "d50_subsurface",
            "d90_subsurface",
            # Level-6-only additions (MBex.p04.hdf re-run at sediment output
            # level 6; see QuasiUnsteadyCrossSectionResults class docstring).
            "fall_velocity",
            "fv_ustar_ratio",
            "hydraulic_radius",
            "longitudinal_cumulative_mass_moveable_limit",
            "mass_bed_change",
            "mass_bed_change_cumulative_max",
            "mass_capacity",
            "mass_capacity_cumulative",
            "mass_cover",
            "mass_in",
            "mass_inactive",
            "mass_out",
            "mass_subsurface",
            "moveable_elevation_left",
            "moveable_elevation_right",
            "moveable_station_left",
            "moveable_station_right",
            "percent_cover",
            "percent_inactive",
            "percent_subsurface",
            "reduce_armor_factor",
            "relative_roughness",
            "reynolds_number",
            "rouse_number",
            "sediment_discharge",
            "shear_velocity",
            "shields_number",
            "temperature",
            "thickness_cover",
            "thickness_inactive",
            "thickness_subsurface",
            "vol_bed_change_cumulative_min",
            "d16_cover",
            "d84_cover",
            "d16_inactive",
            "d84_inactive",
            "d16_subsurface",
            "d84_subsurface",
        ],
    )
    def test_returns_series_indexed_by_timestamps(self, attr):
        with QuasiUnsteadyPlan(QUASI_STEADY_SEDIMENT_HDF) as plan:
            xs = plan.cross_sections[FIRST_XS]
            series = getattr(xs, attr)
            ts = plan.cross_section_timestamps
        assert isinstance(series, pd.Series)
        assert len(series) == len(ts)
        assert (series.index == ts).all()

    def test_effective_depth_units_is_feet(self):
        with QuasiUnsteadyPlan(QUASI_STEADY_SEDIMENT_HDF) as plan:
            xs = plan.cross_sections[FIRST_XS]
            series = xs.effective_depth
        assert series.attrs["units"] == "ft"


# ---------------------------------------------------------------------------
# Mass consolidated accessors
# ---------------------------------------------------------------------------


@skip_if_no_quasi_steady_example
class TestConsolidatedMass:
    def test_cumulative_inflow_total_first_column(self):
        with QuasiUnsteadyPlan(QUASI_STEADY_SEDIMENT_HDF) as plan:
            xs = plan.cross_sections[FIRST_XS]
            df = xs.get_cumulative_inflow("mass")
        assert isinstance(df, pd.DataFrame)
        assert df.columns[0] == "Total"
        assert len(df.columns) > 1  # at least one active grain class present

    def test_cumulative_inflow_units_present(self):
        with QuasiUnsteadyPlan(QUASI_STEADY_SEDIMENT_HDF) as plan:
            xs = plan.cross_sections[FIRST_XS]
            df = xs.get_cumulative_inflow("mass")
        assert "units" in df.attrs

    def test_cumulative_inflow_invalid_quantity_raises(self):
        with QuasiUnsteadyPlan(QUASI_STEADY_SEDIMENT_HDF) as plan:
            xs = plan.cross_sections[FIRST_XS]
            with pytest.raises(ValueError):
                xs.get_cumulative_inflow("bogus")  # type: ignore[arg-type]

    def test_cumulative_outflow_total_first_column(self):
        # Present at sediment output level 6 (absent at lower levels, where
        # this used to raise KeyError -- see get_cumulative_outflow docstring).
        with QuasiUnsteadyPlan(QUASI_STEADY_SEDIMENT_HDF) as plan:
            xs = plan.cross_sections[FIRST_XS]
            df = xs.get_cumulative_outflow("mass")
        assert isinstance(df, pd.DataFrame)
        assert df.columns[0] == "Total"

    def test_cumulative_outflow_wrong_quantity_raises_keyerror(self):
        # This example model was run in Mass mode, not Volume mode.
        with QuasiUnsteadyPlan(QUASI_STEADY_SEDIMENT_HDF) as plan:
            xs = plan.cross_sections[FIRST_XS]
            with pytest.raises(KeyError):
                xs.get_cumulative_outflow("vol")


# ---------------------------------------------------------------------------
# Cross-section geometry over time (Sediment SE block)
# ---------------------------------------------------------------------------


@skip_if_no_quasi_steady_example
class TestStationElevation:
    def test_returns_station_elevation_arrays(self):
        with QuasiUnsteadyPlan(QUASI_STEADY_SEDIMENT_HDF) as plan:
            xs = plan.cross_sections[FIRST_XS]
            station, elevation = xs.get_xsec(0)
        assert isinstance(station, np.ndarray)
        assert isinstance(elevation, np.ndarray)
        assert station.ndim == 1
        assert elevation.ndim == 1
        assert station.shape == elevation.shape
        assert station.shape[0] > 0

    def test_matches_geometry_station_elevation_at_time_zero(self):
        # At the first checkpoint the bed has not yet moved, so the XS shape
        # should equal the static geometry survey.
        with QuasiUnsteadyPlan(QUASI_STEADY_SEDIMENT_HDF) as plan:
            xs = plan.cross_sections[FIRST_XS]
            station0, elevation0 = xs.get_xsec(0)
            geom_xs = plan.geometry.cross_sections[FIRST_XS]
        np.testing.assert_allclose(station0, geom_xs.station_elevation[:, 0])
        np.testing.assert_allclose(elevation0, geom_xs.station_elevation[:, 1])

    def test_timestep_out_of_range_raises(self):
        with QuasiUnsteadyPlan(QUASI_STEADY_SEDIMENT_HDF) as plan:
            xs = plan.cross_sections[FIRST_XS]
            n = len(plan.cross_section_geometry_timestamps)
            with pytest.raises(IndexError):
                xs.get_xsec(n)
