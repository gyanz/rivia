"""Tests for rivia.hdf.unsteady_plan (UnsteadyPlan, FlowAreaResults)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rivia.hdf import UnsteadyPlan

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
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            ts = hdf.mapping_timestamps
        assert isinstance(ts, pd.DatetimeIndex)

    def test_length_matches_timesteps(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            ts = hdf.mapping_timestamps
        assert len(ts) == N_TIMESTEPS

    def test_timestamps_are_increasing(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            ts = hdf.mapping_timestamps
        assert ts.is_monotonic_increasing


# ---------------------------------------------------------------------------
# Lazy time-series
# ---------------------------------------------------------------------------


class TestLazyTimeSeries:
    def test_water_surface_is_dataset(self, synthetic_plan_hdf):
        import h5py

        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            ws = hdf.flow_areas[AREA].water_surface
            assert isinstance(ws, h5py.Dataset)
            assert ws.shape == (N_TIMESTEPS, N_CELLS)

    def test_water_surface_single_timestep(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            row = np.array(hdf.flow_areas[AREA].water_surface[0])
        assert row.shape == (N_CELLS,)

    def test_water_surface_slice(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            block = np.array(hdf.flow_areas[AREA].water_surface[1:3])
        assert block.shape == (2, N_CELLS)

    def test_face_velocity_shape(self, synthetic_plan_hdf):
        # Check shape inside the context manager — h5py.Dataset is invalid after close
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            fv_shape = hdf.flow_areas[AREA].face_velocity.shape
        assert fv_shape == (N_TIMESTEPS, N_FACES)

    def test_face_flow_none_when_absent(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            assert hdf.flow_areas[AREA].face_flow is None

    def test_cell_velocity_none_when_absent(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            assert hdf.flow_areas[AREA].cell_velocity is None


# ---------------------------------------------------------------------------
# Summary DataFrames
# ---------------------------------------------------------------------------


class TestSummaryResults:
    def test_max_water_surface_shape(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            df = hdf.flow_areas[AREA].max_water_surface
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["value", "time"]
        assert len(df) == N_CELLS

    def test_min_water_surface_shape(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            df = hdf.flow_areas[AREA].min_water_surface
        assert len(df) == N_CELLS

    def test_max_face_velocity_shape(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            df = hdf.flow_areas[AREA].max_face_velocity
        assert len(df) == N_FACES

    def test_max_ge_min_water_surface(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
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
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            d = hdf.flow_areas[AREA].get_depth(timestep=0)
        assert d.shape == (N_CELLS,)

    def test_depth_non_negative(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            d = hdf.flow_areas[AREA].get_depth(timestep=0)
        assert (d >= 0).all()

    def test_depth_equals_wse_minus_bed(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            area = hdf.flow_areas[AREA]
            wse = np.array(area.water_surface[0, :N_CELLS])
            bed = area.cell_min_elevation
            depth = area.get_depth(timestep=0)
        expected = np.maximum(0.0, wse - bed)
        np.testing.assert_allclose(depth, expected, rtol=1e-5)

    def test_max_depth_shape(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            df = hdf.flow_areas[AREA].get_max_depth()
        assert list(df.columns) == ["value", "time"]
        assert len(df) == N_CELLS

    def test_max_depth_non_negative(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            df = hdf.flow_areas[AREA].get_max_depth()
        assert (df["value"] >= 0).all()


# ---------------------------------------------------------------------------
# Computed: cell velocity
# ---------------------------------------------------------------------------


class TestCellVelocity:
    def test_cell_velocity_vectors_shape(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            area = hdf.flow_areas[AREA]
            vecs = area.get_cell_velocity(0)
        assert vecs.shape == (N_CELLS, 2)

    def test_cell_speed_shape(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            speed = hdf.flow_areas[AREA].get_cell_velocity(0, component="speed")
        assert speed.shape == (N_CELLS,)

    def test_cell_speed_non_negative(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            speed = hdf.flow_areas[AREA].get_cell_velocity(0, component="speed")
        assert (speed >= 0).all()

    def test_speed_equals_norm_of_vectors(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            area = hdf.flow_areas[AREA]
            vecs = area.get_cell_velocity(0)
            speed = area.get_cell_velocity(0, component="speed")
        expected = np.sqrt(vecs[:, 0] ** 2 + vecs[:, 1] ** 2)
        np.testing.assert_allclose(speed, expected, rtol=1e-6)

    def test_length_weighted_method(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            vecs = hdf.flow_areas[AREA].get_cell_velocity(
                0, method="length_weighted"
            )
        assert vecs.shape == (N_CELLS, 2)

    def test_invalid_method_raises(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            with pytest.raises(ValueError, match="method"):
                hdf.flow_areas[AREA].get_cell_velocity(0, method="bad")

    def test_flow_ratio_without_face_flow_raises(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            with pytest.raises(KeyError, match="Face Flow"):
                hdf.flow_areas[AREA].get_cell_velocity(0, method="flow_ratio")

    def test_sloped_wse_interp_vectors_shape(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            vecs = hdf.flow_areas[AREA].get_cell_velocity(0, wse_interp="sloped")
        assert vecs.shape == (N_CELLS, 2)

    def test_sloped_wse_interp_speed_shape(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            speed = hdf.flow_areas[AREA].get_cell_velocity(
                0, component="speed", wse_interp="sloped"
            )
        assert speed.shape == (N_CELLS,)

    def test_invalid_wse_interp_raises(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            with pytest.raises(ValueError, match="wse_interp"):
                hdf.flow_areas[AREA].get_cell_velocity(0, wse_interp="bad")


# ---------------------------------------------------------------------------
# Computed: face / facepoint velocity
# ---------------------------------------------------------------------------


class TestFaceVelocity:
    def test_face_velocity_vectors_shape(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            vecs = hdf.flow_areas[AREA].get_face_velocity(timestep=0, component="vector")
        assert vecs.shape == (N_FACES, 2)

    def test_face_velocity_vectors_finite(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            vecs = hdf.flow_areas[AREA].get_face_velocity(timestep=0, component="vector")
        assert np.isfinite(vecs).all()

    def test_facepoint_velocity_vectors_shape(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            vecs = hdf.flow_areas[AREA].get_facepoint_velocity_field(0)
        # synthetic fixture maps all facepoints to index 0 → 1 unique facepoint
        assert vecs.ndim == 2
        assert vecs.shape[1] == 2

    def test_facepoint_velocity_vectors_finite(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            vecs = hdf.flow_areas[AREA].get_facepoint_velocity_field(0)
        assert np.isfinite(vecs).all()


# ---------------------------------------------------------------------------
# export_raster guard tests (no rasterio needed for ValueError path)
# ---------------------------------------------------------------------------


class TestExportRasterGuards:
    """export_raster error guards — no geo dependencies needed."""

    def test_velocity_requires_timestep(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            with pytest.raises(ValueError, match="timestep=None"):
                hdf.flow_areas[AREA].export_raster("velocity", timestep=None)


pytest.importorskip("rasterio", reason="rasterio not installed")
pytest.importorskip("shapely", reason="shapely not installed")


class TestExportRaster:
    """export_raster functional tests (require rasterio + shapely)."""

    def test_wse_flat_returns_dataset(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            area = hdf.flow_areas[AREA]
            ds = area.export_raster(
                "water_surface", timestep=0, render_mode="horizontal",
                cell_size=5.0, tight_extent=False,
            )
        assert ds is not None
        ds.close()

    def test_wse_sloping_returns_dataset(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            area = hdf.flow_areas[AREA]
            ds = area.export_raster(
                "water_surface", timestep=0, render_mode="sloping",
                cell_size=5.0, tight_extent=False,
            )
        assert ds is not None
        ds.close()

    def test_wse_max_timestep(self, synthetic_plan_hdf):
        """timestep=None uses max_water_surface."""
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            area = hdf.flow_areas[AREA]
            ds = area.export_raster(
                "water_surface", timestep=None, render_mode="horizontal",
                cell_size=5.0, tight_extent=False,
            )
        assert ds is not None
        ds.close()

    def test_velocity_flat_returns_dataset(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            area = hdf.flow_areas[AREA]
            ds = area.export_raster(
                "velocity", timestep=0, render_mode="horizontal",
                cell_size=5.0, tight_extent=False,
            )
        assert ds is not None
        ds.close()

    def test_output_path_writes_file(self, synthetic_plan_hdf, tmp_path):
        out = tmp_path / "wse.tif"
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            result = hdf.flow_areas[AREA].export_raster(
                "water_surface", timestep=0, output_path=str(out),
                cell_size=5.0, tight_extent=False,
            )
        assert out.exists()
        assert str(result) == str(out)

    def test_default_cell_size_used_when_none_given(self, synthetic_plan_hdf):
        """Neither reference_raster nor cell_size → auto cell_size from face lengths."""
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            area = hdf.flow_areas[AREA]
            ds = area.export_raster(
                "water_surface", timestep=0, render_mode="horizontal",
                tight_extent=False,
            )
        assert ds is not None
        ds.close()


# ---------------------------------------------------------------------------
# Integration tests against the real example file
# ---------------------------------------------------------------------------


@skip_if_no_example
class TestUnsteadyPlanIntegration:
    def test_time_stamps_length(self):
        with UnsteadyPlan(EXAMPLE_PLAN_HDF) as hdf:
            ts = hdf.mapping_timestamps
        assert len(ts) > 0

    def test_water_surface_shape(self):
        with UnsteadyPlan(EXAMPLE_PLAN_HDF) as hdf:
            area = hdf.flow_areas[hdf.flow_areas.names[0]]
            # check h5py.Dataset attributes while file is open
            ws_shape = area.water_surface.shape
            n_cells = area.n_cells
        assert ws_shape[0] > 0
        # HEC-RAS stores WSE for real + ghost cells, so columns >= n_cells
        assert ws_shape[1] >= n_cells

    def test_depth_non_negative(self):
        with UnsteadyPlan(EXAMPLE_PLAN_HDF) as hdf:
            area = hdf.flow_areas[hdf.flow_areas.names[0]]
            d = area.get_depth(timestep=0)
        assert (d >= 0).all()

    def test_cell_velocity_vectors_shape(self):
        with UnsteadyPlan(EXAMPLE_PLAN_HDF) as hdf:
            area = hdf.flow_areas[hdf.flow_areas.names[0]]
            vecs = area.get_cell_velocity(0)
            n_cells = area.n_cells
        assert vecs.shape == (n_cells, 2)

    def test_max_water_surface_len(self):
        with UnsteadyPlan(EXAMPLE_PLAN_HDF) as hdf:
            area = hdf.flow_areas[hdf.flow_areas.names[0]]
            df = area.max_water_surface
            n_cells = area.n_cells
        assert len(df) == n_cells


# ---------------------------------------------------------------------------
# collection methods (storage_areas / structures / sa2d_connections)
# ---------------------------------------------------------------------------


class TestCollectionMethods:
    """storage_areas / structures / sa2d_connections are methods, not properties."""

    def test_storage_areas_is_callable(self, synthetic_plan_hdf):
        from rivia.hdf.unsteady_plan import StorageAreaResultsCollection
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            result = hdf.storage_areas()
        assert isinstance(result, StorageAreaResultsCollection)

    def test_storage_areas_default_output_is_mapping(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            coll = hdf.storage_areas()
        assert coll._output == "mapping"

    def test_storage_areas_accepts_output_kwarg(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            coll = hdf.storage_areas(output="output")
        assert coll._output == "output"

    def test_structures_is_callable(self, synthetic_plan_hdf):
        from rivia.hdf.unsteady_plan import StructureResultsCollection
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            result = hdf.structures()
        assert isinstance(result, StructureResultsCollection)

    def test_structures_default_output_is_output(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            coll = hdf.structures()
        assert coll._output == "output"

    def test_sa2d_connections_is_callable(self, synthetic_plan_hdf):
        from rivia.hdf.geometry import StructureIndex
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            result = hdf.sa2d_connections()
        assert isinstance(result, StructureIndex)

    def test_cross_sections_is_callable(self, synthetic_plan_hdf):
        from rivia.hdf.unsteady_plan import CrossSectionResultsCollection
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            result = hdf.cross_sections()
        assert isinstance(result, CrossSectionResultsCollection)

    def test_cross_sections_default_uses_mapping_result_cls(self, synthetic_plan_hdf):
        from rivia.hdf.unsteady_plan import CrossSectionMappingResults
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            coll = hdf.cross_sections()
        assert coll._result_cls is CrossSectionMappingResults


# ---------------------------------------------------------------------------
# detailed_timestamps (DSS Profile block absent → KeyError)
# ---------------------------------------------------------------------------


class TestDetailedTimestamps:
    def test_n_detailed_timestamps_returns_none_when_absent(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            assert hdf.n_detailed_timestamps is None

    def test_detailed_interval_returns_none_when_absent(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            assert hdf.detailed_interval is None

    def test_detailed_timestamps_raises_when_absent(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            with pytest.raises(KeyError, match="DSS Profile"):
                _ = hdf.detailed_timestamps

    def test_structures_timestamps_matches_mapping(self, synthetic_plan_hdf):
        """StructureResultsCollection.timestamps reads mapping block timestamps."""
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            ts_plan = hdf.mapping_timestamps
            # Use output="mapping" — synthetic fixture has no DSS Hydrograph block
            ts_coll = hdf.structures(output="mapping").timestamps
        pd.testing.assert_index_equal(ts_plan, ts_coll)

    def test_storage_areas_timestamps_matches_mapping(self, synthetic_plan_hdf):
        """StorageAreaResultsCollection.timestamps reads mapping block timestamps."""
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            ts_plan = hdf.mapping_timestamps
            ts_coll = hdf.storage_areas().timestamps
        pd.testing.assert_index_equal(ts_plan, ts_coll)


# ---------------------------------------------------------------------------
# StorageAreaResults — pd.Series return types
# ---------------------------------------------------------------------------


class TestStorageAreaResults:
    def test_wse_returns_series(self, plan_with_storage_areas_hdf):
        with UnsteadyPlan(plan_with_storage_areas_hdf) as hdf:
            sa = hdf.storage_areas()["Pond"]
            wse = sa.wse
        assert isinstance(wse, pd.Series)

    def test_flow_returns_series(self, plan_with_storage_areas_hdf):
        with UnsteadyPlan(plan_with_storage_areas_hdf) as hdf:
            sa = hdf.storage_areas()["Pond"]
            flow = sa.flow
        assert isinstance(flow, pd.Series)

    def test_wse_index_is_datetimeindex(self, plan_with_storage_areas_hdf):
        with UnsteadyPlan(plan_with_storage_areas_hdf) as hdf:
            sa = hdf.storage_areas()["Pond"]
            wse = sa.wse
        assert isinstance(wse.index, pd.DatetimeIndex)

    def test_wse_length_matches_timestamps(self, plan_with_storage_areas_hdf):
        with UnsteadyPlan(plan_with_storage_areas_hdf) as hdf:
            coll = hdf.storage_areas()
            sa = coll["Pond"]
            wse = sa.wse
            n_ts = len(coll.timestamps)
        assert len(wse) == n_ts

    def test_wse_index_matches_collection_timestamps(self, plan_with_storage_areas_hdf):
        with UnsteadyPlan(plan_with_storage_areas_hdf) as hdf:
            coll = hdf.storage_areas()
            sa = coll["Pond"]
            pd.testing.assert_index_equal(sa.wse.index, coll.timestamps)

    def test_inflow_net_returns_series(self, plan_with_storage_areas_hdf):
        with UnsteadyPlan(plan_with_storage_areas_hdf) as hdf:
            sa = hdf.storage_areas()["Pond"]
            result = sa.inflow_net
        assert isinstance(result, pd.Series)

    def test_max_wse_returns_dataframe(self, plan_with_storage_areas_hdf):
        with UnsteadyPlan(plan_with_storage_areas_hdf) as hdf:
            sa = hdf.storage_areas()["Pond"]
            df = sa.max_wse
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["value", "time"]
