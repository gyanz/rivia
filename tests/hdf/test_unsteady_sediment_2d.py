"""Tests for rivia.hdf.unsteady_sediment (UnsteadySediment, 2D flow areas).

Uses the real 2D sediment transport example plan (Chippewa_2D). Skipped
when the example file is not present on the machine.
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest

from rivia.hdf import UnsteadyPlan
from rivia.hdf.unsteady_sediment import (
    SedimentFlowAreaResults,
    SedimentFlowAreaResultsCollection,
)

from .conftest import SEDIMENT_2D_HDF, skip_if_no_2d_sediment_example

AREA = "Perimeter 1"
EXPECTED_TRANSPORT_GRAINS = [
    "FS", "MS", "CS", "VCS", "VFG", "FG", "MG", "CG", "VCG", "SC",
]
CELL_METHODS = ["fraction_suspended", "total_load_concentration"]


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


@skip_if_no_2d_sediment_example
class TestFlowAreaTimestamps:
    def test_bed_timestamps_returns_datetimeindex(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            ts = plan.sediment.bed_timestamps
        assert isinstance(ts, pd.DatetimeIndex)
        assert ts.is_monotonic_increasing

    def test_transport_timestamps_returns_datetimeindex(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            ts = plan.sediment.transport_timestamps
        assert isinstance(ts, pd.DatetimeIndex)
        assert ts.is_monotonic_increasing

    def test_bed_and_transport_timestamps_same_length(self):
        # Not guaranteed by HEC-RAS in general, but true for this fixture --
        # both blocks are written at the mapping interval here.
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            bed_len = len(plan.sediment.bed_timestamps)
            transport_len = len(plan.sediment.transport_timestamps)
        assert bed_len == transport_len


# ---------------------------------------------------------------------------
# Flow area collection
# ---------------------------------------------------------------------------


@skip_if_no_2d_sediment_example
class TestFlowAreaCollection:
    def test_returns_collection_type(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            coll = plan.sediment.flow_areas()
        assert isinstance(coll, SedimentFlowAreaResultsCollection)

    def test_cached(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            assert plan.sediment.flow_areas() is plan.sediment.flow_areas()

    def test_contains_expected_area(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            coll = plan.sediment.flow_areas()
            assert AREA in coll
            assert list(coll) == [AREA]

    def test_getitem_cached(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            coll = plan.sediment.flow_areas()
            assert coll[AREA] is coll[AREA]

    def test_getitem_returns_result_type(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
        assert isinstance(fa, SedimentFlowAreaResults)
        assert fa.name == AREA

    def test_unknown_area_raises_keyerror(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            coll = plan.sediment.flow_areas()
            with pytest.raises(KeyError):
                coll["Bogus Area"]


# ---------------------------------------------------------------------------
# Sediment Bed
# ---------------------------------------------------------------------------


@skip_if_no_2d_sediment_example
class TestSedimentBed:
    def test_bed_elevation_is_dataset(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            ds = fa.bed_elevation
            assert isinstance(ds, h5py.Dataset)
            n_t = len(plan.sediment.bed_timestamps)
            assert ds.shape[0] == n_t

    def test_bed_change_is_dataset(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            ds = fa.bed_change
            assert isinstance(ds, h5py.Dataset)
            assert ds.shape == fa.bed_elevation.shape

    def test_initial_bed_elevation_is_1d(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            initial = fa.initial_bed_elevation
            n_cells = fa.bed_elevation.shape[1]
        assert initial.shape == (n_cells,)

    def test_initial_bed_elevation_matches_first_timestep(self):
        # At t=0 the bed has not changed yet, so elevation should equal
        # the initial condition (within float32 rounding).
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            initial = fa.initial_bed_elevation
            first_step = fa.bed_elevation[0]
        assert np.allclose(initial, first_step, rtol=1e-4, atol=1e-3)

    def test_max_bed_elevation_returns_ndarray(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            arr = fa.max_bed_elevation()
        assert arr.ndim == 2

    def test_min_bed_elevation_returns_ndarray(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            arr = fa.min_bed_elevation()
        assert arr.ndim == 2


# ---------------------------------------------------------------------------
# Sediment Transport -- bed shear stress
# ---------------------------------------------------------------------------


@skip_if_no_2d_sediment_example
class TestBedShearStress:
    def test_skin_and_total_are_datasets(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            skin = fa.bed_shear_stress(component="skin")
            total = fa.bed_shear_stress(component="total")
            assert isinstance(skin, h5py.Dataset)
            assert isinstance(total, h5py.Dataset)
            assert skin.shape == total.shape

    def test_component_has_no_default(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            with pytest.raises(TypeError):
                fa.bed_shear_stress()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Sediment Transport -- consolidated grain accessors
# ---------------------------------------------------------------------------


@skip_if_no_2d_sediment_example
class TestConsolidatedCellAccessors:
    @pytest.mark.parametrize("method", CELL_METHODS)
    def test_total_first_column(self, method):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            df = getattr(fa, method)(cell=100)
        assert isinstance(df, pd.DataFrame)
        assert df.columns[0] == "Total"

    @pytest.mark.parametrize("method", CELL_METHODS)
    def test_discovers_present_grains_only(self, method):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            df = getattr(fa, method)(cell=100)
        assert list(df.columns) == ["Total"] + EXPECTED_TRANSPORT_GRAINS

    @pytest.mark.parametrize("method", CELL_METHODS)
    def test_indexed_by_transport_timestamps(self, method):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            sed = plan.sediment
            fa = sed.flow_areas()[AREA]
            df = getattr(fa, method)(cell=100)
            ts = sed.transport_timestamps
        assert (df.index == ts).all()

    def test_cell_required_raises_valueerror(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            with pytest.raises(ValueError):
                fa.fraction_suspended()
            with pytest.raises(ValueError):
                fa.total_load_concentration()


@skip_if_no_2d_sediment_example
class TestTransportRate:
    def test_total_first_column(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            df = fa.transport_rate(face=200)
        assert df.columns[0] == "Total"

    def test_discovers_present_grains_only(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            df = fa.transport_rate(face=200)
        assert list(df.columns) == ["Total"] + EXPECTED_TRANSPORT_GRAINS

    def test_face_required_raises_valueerror(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            with pytest.raises(ValueError):
                fa.transport_rate()
