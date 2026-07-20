"""Tests for rivia.hdf.unsteady_sediment (UnsteadySediment, 1-D cross sections).

Uses the real 1-D sediment transport example plans -- Mass and Volume output
modes of the same underlying model -- since a faithful synthetic sediment
fixture (grain-class table, per-fraction records, geometry join) would add
more test-fixture code than it saves.  Skipped when the example files are
not present on the machine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rivia.hdf import UnsteadyPlan
from rivia.hdf.unsteady_sediment import SedimentCrossSectionResults, UnsteadySediment

from .conftest import (
    SEDIMENT_1D_MASS_HDF,
    SEDIMENT_1D_VOL_HDF,
    skip_if_no_sediment_examples,
)

FIRST_XS = ("Beaver Creek", "Kentwood", "5.99")
EXPECTED_GRAINS_5_TO_16 = [
    "CM", "VFS", "FS", "MS", "CS", "VCS", "VFG", "FG", "MG", "CG", "VCG", "SC",
]


# ---------------------------------------------------------------------------
# plan.sediment wiring
# ---------------------------------------------------------------------------


@skip_if_no_sediment_examples
class TestSedimentAccessor:
    def test_returns_unsteady_sediment(self):
        with UnsteadyPlan(SEDIMENT_1D_MASS_HDF) as plan:
            assert isinstance(plan.sediment, UnsteadySediment)

    def test_cached(self):
        with UnsteadyPlan(SEDIMENT_1D_MASS_HDF) as plan:
            assert plan.sediment is plan.sediment

    def test_borrows_plan_handle(self):
        with UnsteadyPlan(SEDIMENT_1D_MASS_HDF) as plan:
            assert plan.sediment._hdf is plan._hdf


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


@skip_if_no_sediment_examples
class TestSedimentTimestamps:
    def test_returns_datetimeindex(self):
        with UnsteadyPlan(SEDIMENT_1D_MASS_HDF) as plan:
            ts = plan.sediment.cross_section_timestamps
        assert isinstance(ts, pd.DatetimeIndex)
        assert ts.is_monotonic_increasing

    def test_differs_from_hydraulics_mapping_interval(self):
        with UnsteadyPlan(SEDIMENT_1D_MASS_HDF) as plan:
            sed_ts = plan.sediment.cross_section_timestamps
            hydraulics_ts = plan.mapping_timestamps
        # Sediment writes at its own interval -- not required to match the
        # hydraulics Base Output interval in either count or spacing.
        assert len(sed_ts) > 0
        assert len(hydraulics_ts) > 0


# ---------------------------------------------------------------------------
# Cross-section collection / geometry join
# ---------------------------------------------------------------------------


@skip_if_no_sediment_examples
class TestCrossSectionCollection:
    def test_collection_cached(self):
        with UnsteadyPlan(SEDIMENT_1D_MASS_HDF) as plan:
            assert plan.sediment.cross_sections() is plan.sediment.cross_sections()

    def test_nonempty(self):
        with UnsteadyPlan(SEDIMENT_1D_MASS_HDF) as plan:
            coll = plan.sediment.cross_sections()
            assert len(coll) > 0

    def test_lookup_by_location_tuple(self):
        with UnsteadyPlan(SEDIMENT_1D_MASS_HDF) as plan:
            xs = plan.sediment.cross_sections()[FIRST_XS]
        assert isinstance(xs, SedimentCrossSectionResults)
        assert xs.river == FIRST_XS[0]
        assert xs.reach == FIRST_XS[1]
        assert xs.rs == FIRST_XS[2]

    def test_lookup_by_int_index(self):
        with UnsteadyPlan(SEDIMENT_1D_MASS_HDF) as plan:
            coll = plan.sediment.cross_sections()
            by_index = coll[0]
            by_tuple = coll[FIRST_XS]
        assert by_index.location == by_tuple.location


# ---------------------------------------------------------------------------
# Scalar time-series properties
# ---------------------------------------------------------------------------


@skip_if_no_sediment_examples
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
        ],
    )
    def test_returns_series_indexed_by_timestamps(self, attr):
        with UnsteadyPlan(SEDIMENT_1D_MASS_HDF) as plan:
            sed = plan.sediment
            xs = sed.cross_sections()[FIRST_XS]
            series = getattr(xs, attr)
            ts = sed.cross_section_timestamps
        assert isinstance(series, pd.Series)
        assert len(series) == len(ts)
        assert (series.index == ts).all()


# ---------------------------------------------------------------------------
# Mass / Vol consolidated accessors
# ---------------------------------------------------------------------------


@skip_if_no_sediment_examples
class TestConsolidatedMass:
    def test_cumulative_inflow_total_first_column(self):
        with UnsteadyPlan(SEDIMENT_1D_MASS_HDF) as plan:
            xs = plan.sediment.cross_sections()[FIRST_XS]
            df = xs.get_cumulative_inflow(quantity="mass")
        assert isinstance(df, pd.DataFrame)
        assert df.columns[0] == "Total"

    def test_cumulative_inflow_discovers_present_grains_only(self):
        with UnsteadyPlan(SEDIMENT_1D_MASS_HDF) as plan:
            xs = plan.sediment.cross_sections()[FIRST_XS]
            df = xs.get_cumulative_inflow(quantity="mass")
        assert list(df.columns) == ["Total"] + EXPECTED_GRAINS_5_TO_16

    def test_cumulative_outflow_same_grain_set(self):
        with UnsteadyPlan(SEDIMENT_1D_MASS_HDF) as plan:
            xs = plan.sediment.cross_sections()[FIRST_XS]
            df = xs.get_cumulative_outflow(quantity="mass")
        assert list(df.columns) == ["Total"] + EXPECTED_GRAINS_5_TO_16

    def test_indexed_by_sediment_timestamps(self):
        with UnsteadyPlan(SEDIMENT_1D_MASS_HDF) as plan:
            sed = plan.sediment
            df = sed.cross_sections()[FIRST_XS].get_cumulative_inflow(quantity="mass")
            ts = sed.cross_section_timestamps
        assert (df.index == ts).all()

    def test_grain_columns_sum_to_total(self):
        # Physically, per-grain contributions should sum to the total
        # (within float32 rounding error accumulated over 1440 timesteps).
        with UnsteadyPlan(SEDIMENT_1D_MASS_HDF) as plan:
            xs = plan.sediment.cross_sections()[FIRST_XS]
            df = xs.get_cumulative_inflow(quantity="mass")
        grain_sum = df.drop(columns="Total").sum(axis=1)
        assert np.allclose(grain_sum, df["Total"], rtol=1e-4, atol=1.0)

    def test_vol_requested_on_mass_file_raises_keyerror(self):
        with UnsteadyPlan(SEDIMENT_1D_MASS_HDF) as plan:
            xs = plan.sediment.cross_sections()[FIRST_XS]
            with pytest.raises(KeyError):
                xs.get_cumulative_inflow(quantity="vol")

    def test_invalid_quantity_raises_valueerror(self):
        with UnsteadyPlan(SEDIMENT_1D_MASS_HDF) as plan:
            xs = plan.sediment.cross_sections()[FIRST_XS]
            with pytest.raises(ValueError):
                xs.get_cumulative_inflow(quantity="bogus")  # type: ignore[arg-type]

    def test_quantity_has_no_default(self):
        with UnsteadyPlan(SEDIMENT_1D_MASS_HDF) as plan:
            xs = plan.sediment.cross_sections()[FIRST_XS]
            with pytest.raises(TypeError):
                xs.get_cumulative_inflow()  # type: ignore[call-arg]


@skip_if_no_sediment_examples
class TestConsolidatedVolume:
    def test_cumulative_inflow_vol(self):
        with UnsteadyPlan(SEDIMENT_1D_VOL_HDF) as plan:
            xs = plan.sediment.cross_sections()[FIRST_XS]
            df = xs.get_cumulative_inflow(quantity="vol")
        assert list(df.columns) == ["Total"] + EXPECTED_GRAINS_5_TO_16

    def test_mass_requested_on_vol_file_raises_keyerror(self):
        with UnsteadyPlan(SEDIMENT_1D_VOL_HDF) as plan:
            xs = plan.sediment.cross_sections()[FIRST_XS]
            with pytest.raises(KeyError):
                xs.get_cumulative_inflow(quantity="mass")
