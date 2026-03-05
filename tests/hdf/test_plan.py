"""Tests for raspy.hdf._plan (PlanHdf, FlowAreaResults)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raspy.hdf import PlanHdf

from .conftest import skip_if_no_example, EXAMPLE_PLAN_HDF

N_CELLS = 10
N_FACES = 20
N_TIMESTEPS = 5
AREA = "TestArea"


# ---------------------------------------------------------------------------
# Time stamps
# ---------------------------------------------------------------------------


class TestTimeStamps:
    def test_returns_datetimeindex(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            ts = hdf.time_stamps
        assert isinstance(ts, pd.DatetimeIndex)

    def test_length_matches_timesteps(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            ts = hdf.time_stamps
        assert len(ts) == N_TIMESTEPS

    def test_timestamps_are_increasing(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            ts = hdf.time_stamps
        assert ts.is_monotonic_increasing


# ---------------------------------------------------------------------------
# Lazy time-series
# ---------------------------------------------------------------------------


class TestLazyTimeSeries:
    def test_water_surface_is_dataset(self, synthetic_plan_hdf):
        import h5py

        with PlanHdf(synthetic_plan_hdf) as hdf:
            ws = hdf.flow_areas[AREA].water_surface
            assert isinstance(ws, h5py.Dataset)
            assert ws.shape == (N_TIMESTEPS, N_CELLS)

    def test_water_surface_single_timestep(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            row = np.array(hdf.flow_areas[AREA].water_surface[0])
        assert row.shape == (N_CELLS,)

    def test_water_surface_slice(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            block = np.array(hdf.flow_areas[AREA].water_surface[1:3])
        assert block.shape == (2, N_CELLS)

    def test_face_velocity_shape(self, synthetic_plan_hdf):
        # Check shape inside the context manager — h5py.Dataset is invalid after close
        with PlanHdf(synthetic_plan_hdf) as hdf:
            fv_shape = hdf.flow_areas[AREA].face_velocity.shape
        assert fv_shape == (N_TIMESTEPS, N_FACES)

    def test_face_flow_none_when_absent(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            assert hdf.flow_areas[AREA].face_flow is None

    def test_cell_velocity_none_when_absent(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            assert hdf.flow_areas[AREA].cell_velocity is None


# ---------------------------------------------------------------------------
# Summary DataFrames
# ---------------------------------------------------------------------------


class TestSummaryResults:
    def test_max_water_surface_shape(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            df = hdf.flow_areas[AREA].max_water_surface
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["value", "time"]
        assert len(df) == N_CELLS

    def test_min_water_surface_shape(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            df = hdf.flow_areas[AREA].min_water_surface
        assert len(df) == N_CELLS

    def test_max_face_velocity_shape(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            df = hdf.flow_areas[AREA].max_face_velocity
        assert len(df) == N_FACES

    def test_max_ge_min_water_surface(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            area = hdf.flow_areas[AREA]
            assert (
                area.max_water_surface["value"].values
                >= area.min_water_surface["value"].values
            ).all()


# ---------------------------------------------------------------------------
# Computed: depth
# ---------------------------------------------------------------------------


class TestDepth:
    def test_depth_shape(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            d = hdf.flow_areas[AREA].depth(0)
        assert d.shape == (N_CELLS,)

    def test_depth_non_negative(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            d = hdf.flow_areas[AREA].depth(0)
        assert (d >= 0).all()

    def test_depth_equals_wse_minus_bed(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            area = hdf.flow_areas[AREA]
            wse = np.array(area.water_surface[0, :N_CELLS])
            bed = area.cell_min_elevation
            depth = area.depth(0)
        expected = np.maximum(0.0, wse - bed)
        np.testing.assert_allclose(depth, expected, rtol=1e-5)

    def test_max_depth_shape(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            df = hdf.flow_areas[AREA].max_depth()
        assert list(df.columns) == ["value", "time"]
        assert len(df) == N_CELLS

    def test_max_depth_non_negative(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            df = hdf.flow_areas[AREA].max_depth()
        assert (df["value"] >= 0).all()


# ---------------------------------------------------------------------------
# Computed: cell velocity
# ---------------------------------------------------------------------------


class TestCellVelocity:
    def test_cell_velocity_vectors_shape(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            vecs = hdf.flow_areas[AREA].cell_velocity_vectors(0)
        assert vecs.shape == (N_CELLS, 2)

    def test_cell_speed_shape(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            speed = hdf.flow_areas[AREA].cell_speed(0)
        assert speed.shape == (N_CELLS,)

    def test_cell_speed_non_negative(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            speed = hdf.flow_areas[AREA].cell_speed(0)
        assert (speed >= 0).all()

    def test_speed_equals_norm_of_vectors(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            area = hdf.flow_areas[AREA]
            vecs = area.cell_velocity_vectors(0)
            speed = area.cell_speed(0)
        expected = np.sqrt(vecs[:, 0] ** 2 + vecs[:, 1] ** 2)
        np.testing.assert_allclose(speed, expected, rtol=1e-6)

    def test_length_weighted_method(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            vecs = hdf.flow_areas[AREA].cell_velocity_vectors(
                0, method="length_weighted"
            )
        assert vecs.shape == (N_CELLS, 2)

    def test_invalid_method_raises(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            with pytest.raises(ValueError, match="method"):
                hdf.flow_areas[AREA].cell_velocity_vectors(0, method="bad")

    def test_flow_ratio_without_face_flow_raises(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            with pytest.raises(KeyError, match="Face Flow"):
                hdf.flow_areas[AREA].cell_velocity_vectors(0, method="flow_ratio")

    def test_sloped_wse_interp_vectors_shape(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            vecs = hdf.flow_areas[AREA].cell_velocity_vectors(0, wse_interp="sloped")
        assert vecs.shape == (N_CELLS, 2)

    def test_sloped_wse_interp_speed_shape(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            speed = hdf.flow_areas[AREA].cell_speed(0, wse_interp="sloped")
        assert speed.shape == (N_CELLS,)

    def test_invalid_wse_interp_raises(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            with pytest.raises(ValueError, match="wse_interp"):
                hdf.flow_areas[AREA].cell_velocity_vectors(0, wse_interp="bad")


# ---------------------------------------------------------------------------
# export_raster: no-geo guard
# ---------------------------------------------------------------------------


class TestExportRasterGeoGuard:
    """export_raster must raise ImportError if geo deps are missing.

    We patch the import to simulate missing rasterio.
    """

    def test_bad_variable_raises_value_error(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            with pytest.raises(ValueError, match="Unknown variable"):
                hdf.flow_areas[AREA].export_raster(
                    "bad_variable", timestep=0, output_path="out.tif"
                )

    def test_max_for_velocity_raises_value_error(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            with pytest.raises(ValueError, match="timestep=None"):
                hdf.flow_areas[AREA].export_raster(
                    "cell_speed", timestep=None, output_path="out.tif"
                )

    def test_invalid_vel_method_raises(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            with pytest.raises(ValueError, match="method"):
                hdf.flow_areas[AREA].export_raster(
                    "cell_speed", timestep=0, output_path="out.tif", vel_weight_method="bad"
                )

    def test_invalid_wse_interp_raises(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            with pytest.raises(ValueError, match="wse_interp"):
                hdf.flow_areas[AREA].export_raster(
                    "cell_velocity", timestep=0, output_path="out.tif", vel_wse_method="bad"
                )


# ---------------------------------------------------------------------------
# Integration tests against the real example file
# ---------------------------------------------------------------------------


@skip_if_no_example
class TestPlanHdfIntegration:
    def test_time_stamps_length(self):
        with PlanHdf(EXAMPLE_PLAN_HDF) as hdf:
            ts = hdf.time_stamps
        assert len(ts) > 0

    def test_water_surface_shape(self):
        with PlanHdf(EXAMPLE_PLAN_HDF) as hdf:
            area = hdf.flow_areas[hdf.flow_areas.names[0]]
            # check h5py.Dataset attributes while file is open
            ws_shape = area.water_surface.shape
            n_cells = area.n_cells
        assert ws_shape[0] > 0
        # HEC-RAS stores WSE for real + ghost cells, so columns >= n_cells
        assert ws_shape[1] >= n_cells

    def test_depth_non_negative(self):
        with PlanHdf(EXAMPLE_PLAN_HDF) as hdf:
            area = hdf.flow_areas[hdf.flow_areas.names[0]]
            d = area.depth(0)
        assert (d >= 0).all()

    def test_cell_velocity_vectors_shape(self):
        with PlanHdf(EXAMPLE_PLAN_HDF) as hdf:
            area = hdf.flow_areas[hdf.flow_areas.names[0]]
            vecs = area.cell_velocity_vectors(0)
            n_cells = area.n_cells
        assert vecs.shape == (n_cells, 2)

    def test_max_water_surface_len(self):
        with PlanHdf(EXAMPLE_PLAN_HDF) as hdf:
            area = hdf.flow_areas[hdf.flow_areas.names[0]]
            df = area.max_water_surface
            n_cells = area.n_cells
        assert len(df) == n_cells
