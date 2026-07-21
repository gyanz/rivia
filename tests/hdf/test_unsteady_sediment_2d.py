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
CELL_METHODS = ["get_fraction_suspended", "get_total_load_concentration"]


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
            arr = fa.max_bed_elevation
        assert arr.ndim == 2

    def test_min_bed_elevation_returns_ndarray(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            arr = fa.min_bed_elevation
        assert arr.ndim == 2

    def test_get_bed_elevation_matches_column(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            series = fa.get_bed_elevation(cell=100)
            column = np.array(fa.bed_elevation[:, 100])
        assert isinstance(series, pd.Series)
        assert series.name == "Bed Elevation"
        assert (series.index == fa.bed_timestamps).all()
        assert np.allclose(series.to_numpy(), column)

    def test_get_bed_elevation_cell_required(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            with pytest.raises(ValueError):
                fa.get_bed_elevation()

    def test_get_bed_change_matches_column(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            series = fa.get_bed_change(cell=100)
            column = np.array(fa.bed_change[:, 100])
        assert isinstance(series, pd.Series)
        assert series.name == "Bed Change"
        assert (series.index == fa.bed_timestamps).all()
        assert np.allclose(series.to_numpy(), column)

    def test_get_bed_change_cell_required(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            with pytest.raises(ValueError):
                fa.get_bed_change()


# ---------------------------------------------------------------------------
# Sediment Transport -- bed shear stress
# ---------------------------------------------------------------------------


@skip_if_no_2d_sediment_example
class TestBedShearStress:
    def test_skin_and_total_are_datasets(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            skin = fa.transport_bed_shear_stress(component="skin")
            total = fa.transport_bed_shear_stress(component="total")
            assert isinstance(skin, h5py.Dataset)
            assert isinstance(total, h5py.Dataset)
            assert skin.shape == total.shape

    def test_component_has_no_default(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            with pytest.raises(TypeError):
                fa.transport_bed_shear_stress()  # type: ignore[call-arg]

    def test_get_bed_shear_stress_matches_column(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            series = fa.get_bed_shear_stress(cell=100, component="skin")
            column = np.array(fa.transport_bed_shear_stress(component="skin")[:, 100])
        assert isinstance(series, pd.Series)
        assert series.name == "Bed Shear Stress (skin)"
        assert (series.index == fa.transport_timestamps).all()
        assert np.allclose(series.to_numpy(), column)

    def test_get_bed_shear_stress_cell_required(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            with pytest.raises(ValueError):
                fa.get_bed_shear_stress(component="skin")

    def test_get_bed_shear_stress_component_has_no_default(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            with pytest.raises(TypeError):
                fa.get_bed_shear_stress(cell=100)  # type: ignore[call-arg]


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
                fa.get_fraction_suspended()
            with pytest.raises(ValueError):
                fa.get_total_load_concentration()


@skip_if_no_2d_sediment_example
class TestTransportRate:
    def test_total_first_column(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            df = fa.get_transport_rate(face=200)
        assert df.columns[0] == "Total"

    def test_discovers_present_grains_only(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            df = fa.get_transport_rate(face=200)
        assert list(df.columns) == ["Total"] + EXPECTED_TRANSPORT_GRAINS

    def test_face_required_raises_valueerror(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            with pytest.raises(ValueError):
                fa.get_transport_rate()

    def test_capacity_reads_capacity_record(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            capacity = fa.get_transport_rate(face=200, capacity=True)
        assert isinstance(capacity, pd.DataFrame)
        assert list(capacity.columns) == ["Total"]

    def test_capacity_defaults_to_false(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            default = fa.get_transport_rate(face=200)
            explicit_rate = fa.get_transport_rate(face=200, capacity=False)
        assert default["Total"].equals(explicit_rate["Total"])


# ---------------------------------------------------------------------------
# Sediment Transport -- transport rate along a profile line
# ---------------------------------------------------------------------------


def _left_to_right_line(hydraulics_fa) -> np.ndarray:
    """Build a profile line spanning the flow area from its cell centers."""
    centers = hydraulics_fa.cell_centers
    return np.array(
        [centers[np.argmin(centers[:, 0])], centers[np.argmax(centers[:, 0])]]
    )


@skip_if_no_2d_sediment_example
class TestTransportRateAlongLine:
    def test_returns_dataframe_with_grain_columns(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            xy = _left_to_right_line(plan.flow_areas[AREA])
            df = fa.transport_rate_along_line(xy)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["Total"] + EXPECTED_TRANSPORT_GRAINS

    def test_indexed_by_transport_timestamps(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            sed = plan.sediment
            fa = sed.flow_areas()[AREA]
            xy = _left_to_right_line(plan.flow_areas[AREA])
            df = fa.transport_rate_along_line(xy)
            ts = sed.transport_timestamps
        assert (df.index == ts).all()

    def test_matches_manual_signed_sum_of_faces(self):
        # Cross-check the fence summation against the already-verified
        # single-face get_transport_rate accessor.
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            hydraulics_fa = plan.flow_areas[AREA]
            xy = _left_to_right_line(hydraulics_fa)
            df = fa.transport_rate_along_line(xy)
            faces_df = hydraulics_fa.faces_along_line(xy)
            signs = np.where(faces_df["orientation"].to_numpy(dtype=bool), -1.0, 1.0)
            expected_total = np.zeros(len(df))
            for face, sign in zip(faces_df["face"], signs, strict=True):
                expected_total += sign * fa.get_transport_rate(face=int(face))[
                    "Total"
                ].to_numpy()
        assert np.allclose(df["Total"].to_numpy(), expected_total)

    def test_walk_method_raises_notimplementederror(self):
        with UnsteadyPlan(SEDIMENT_2D_HDF) as plan:
            fa = plan.sediment.flow_areas()[AREA]
            xy = _left_to_right_line(plan.flow_areas[AREA])
            with pytest.raises(NotImplementedError):
                fa.transport_rate_along_line(xy, method="walk")
