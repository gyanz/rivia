"""Tests for rivia.hdf._base."""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from rivia.hdf._base import PlanInformation, _resolve_hdf_path


class TestResolveHdfPath:
    def test_no_suffix_appended(self, tmp_path):
        p = _resolve_hdf_path(tmp_path / "model.p01")
        assert p.name == "model.p01.hdf"

    def test_existing_hdf_suffix_unchanged(self, tmp_path):
        p = _resolve_hdf_path(tmp_path / "model.p01.hdf")
        assert p.name == "model.p01.hdf"

    def test_uppercase_hdf_suffix_unchanged(self, tmp_path):
        p = _resolve_hdf_path(tmp_path / "model.p01.HDF")
        assert p.name == "model.p01.HDF"

    def test_string_input_accepted(self):
        p = _resolve_hdf_path("path/to/model.g01")
        assert p.name == "model.g01.hdf"

    def test_returns_path_object(self):
        from pathlib import Path

        p = _resolve_hdf_path("model.p01")
        assert isinstance(p, Path)


class TestHdfFileLifecycle:
    def test_file_not_found_raises(self, tmp_path):
        from rivia.hdf._base import _HdfFile

        with pytest.raises(FileNotFoundError):
            _HdfFile(tmp_path / "nonexistent.p01.hdf")

    def test_context_manager_closes_file(self, synthetic_plan_hdf):
        from rivia.hdf import UnsteadyPlan

        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            assert hdf._hdf.id.valid
        assert not hdf._hdf.id.valid

    def test_explicit_close(self, synthetic_plan_hdf):
        from rivia.hdf import UnsteadyPlan

        hdf = UnsteadyPlan(synthetic_plan_hdf)
        assert hdf._hdf.id.valid
        hdf.close()
        assert not hdf._hdf.id.valid

    def test_suffix_auto_appended(self, synthetic_plan_hdf):
        from rivia.hdf import UnsteadyPlan

        # Pass path without .hdf — it points to a file that doesn't exist,
        # but we just want to confirm the suffix logic; use the full path.
        no_suffix = str(synthetic_plan_hdf).removesuffix(".hdf")
        with UnsteadyPlan(no_suffix) as hdf:
            assert hdf.filename == synthetic_plan_hdf


# ---------------------------------------------------------------------------
# PlanInformation
# ---------------------------------------------------------------------------

N_CELLS = 10
AREA = "TestArea"


class TestPlanInformation:
    def test_returns_plan_information_instance(self, plan_with_geometry_hdfs):
        from rivia.hdf import UnsteadyPlan

        plan_path, _ = plan_with_geometry_hdfs
        with UnsteadyPlan(plan_path) as plan:
            info = plan.plan_information
        assert isinstance(info, PlanInformation)

    def test_geometry_filename_decoded(self, plan_with_geometry_hdfs):
        from rivia.hdf import UnsteadyPlan

        plan_path, _ = plan_with_geometry_hdfs
        with UnsteadyPlan(plan_path) as plan:
            assert plan.plan_information.geometry_filename == "MyModel.g01"

    def test_raw_contains_all_attrs(self, plan_with_geometry_hdfs):
        from rivia.hdf import UnsteadyPlan

        plan_path, _ = plan_with_geometry_hdfs
        with UnsteadyPlan(plan_path) as plan:
            raw = plan.plan_information.raw
        assert "Geometry Filename" in raw
        assert "Plan Name" in raw

    def test_raw_values_are_strings(self, plan_with_geometry_hdfs):
        from rivia.hdf import UnsteadyPlan

        plan_path, _ = plan_with_geometry_hdfs
        with UnsteadyPlan(plan_path) as plan:
            raw = plan.plan_information.raw
        assert all(isinstance(v, str) for v in raw.values())

    def test_missing_group_raises_key_error(self, synthetic_plan_hdf):
        from rivia.hdf import UnsteadyPlan

        # synthetic_plan_hdf has no Plan Data/Plan Information group
        with UnsteadyPlan(synthetic_plan_hdf) as plan:
            with pytest.raises(KeyError, match="Plan Data/Plan Information"):
                plan.plan_information


# ---------------------------------------------------------------------------
# geometry_hdf_path
# ---------------------------------------------------------------------------


class TestGeometryHdfPath:
    def test_returns_path_object(self, plan_with_geometry_hdfs):
        from rivia.hdf import UnsteadyPlan

        plan_path, _ = plan_with_geometry_hdfs
        with UnsteadyPlan(plan_path) as plan:
            assert isinstance(plan.geometry_hdf_path, Path)

    def test_same_folder_as_plan(self, plan_with_geometry_hdfs):
        from rivia.hdf import UnsteadyPlan

        plan_path, _ = plan_with_geometry_hdfs
        with UnsteadyPlan(plan_path) as plan:
            assert plan.geometry_hdf_path.parent == plan_path.parent

    def test_correct_filename(self, plan_with_geometry_hdfs):
        from rivia.hdf import UnsteadyPlan

        plan_path, geom_path = plan_with_geometry_hdfs
        with UnsteadyPlan(plan_path) as plan:
            assert plan.geometry_hdf_path == geom_path

    def test_missing_geometry_file_raises(self, tmp_path):
        from rivia.hdf import UnsteadyPlan

        plan_path = tmp_path / "model.p01.hdf"
        with h5py.File(plan_path, "w") as f:
            grp = f.create_group("Plan Data/Plan Information")
            grp.attrs["Geometry Filename"] = "nonexistent.g01"
        with UnsteadyPlan(plan_path) as plan:
            with pytest.raises(FileNotFoundError, match="nonexistent.g01"):
                plan.geometry_hdf_path


# ---------------------------------------------------------------------------
# geometry property
# ---------------------------------------------------------------------------


class TestGeometryProperty:
    def test_returns_geometry_instance(self, plan_with_geometry_hdfs):
        from rivia.hdf import Geometry, UnsteadyPlan

        plan_path, _ = plan_with_geometry_hdfs
        with UnsteadyPlan(plan_path) as plan:
            assert isinstance(plan.geometry, Geometry)

    def test_flow_areas_accessible(self, plan_with_geometry_hdfs):
        from rivia.hdf import UnsteadyPlan

        plan_path, _ = plan_with_geometry_hdfs
        with UnsteadyPlan(plan_path) as plan:
            assert AREA in plan.geometry.flow_areas

    def test_cell_centers_shape(self, plan_with_geometry_hdfs):
        from rivia.hdf import UnsteadyPlan

        plan_path, _ = plan_with_geometry_hdfs
        with UnsteadyPlan(plan_path) as plan:
            centers = plan.geometry.flow_areas[AREA].cell_centers
        assert centers.shape == (N_CELLS, 2)

    def test_center_coordinates_2d_cell(self, plan_with_geometry_hdfs):
        from rivia.hdf import UnsteadyPlan

        plan_path, _ = plan_with_geometry_hdfs
        with UnsteadyPlan(plan_path) as plan:
            xy = plan.geometry.center_coordinates((AREA, "", "0"))
        assert xy.shape == (2,)

    def test_geometry_is_cached(self, plan_with_geometry_hdfs):
        from rivia.hdf import UnsteadyPlan

        plan_path, _ = plan_with_geometry_hdfs
        with UnsteadyPlan(plan_path) as plan:
            g1 = plan.geometry
            g2 = plan.geometry
        assert g1 is g2

    def test_geometry_hdf_open_while_plan_open(self, plan_with_geometry_hdfs):
        from rivia.hdf import UnsteadyPlan

        plan_path, _ = plan_with_geometry_hdfs
        with UnsteadyPlan(plan_path) as plan:
            geom = plan.geometry
            assert geom._hdf.id.valid

    def test_geometry_hdf_closed_when_plan_closed(self, plan_with_geometry_hdfs):
        from rivia.hdf import UnsteadyPlan

        plan_path, _ = plan_with_geometry_hdfs
        plan = UnsteadyPlan(plan_path)
        geom = plan.geometry  # open and cache
        assert geom._hdf.id.valid
        plan.close()
        assert not geom._hdf.id.valid

    def test_plan_hdf_also_closed(self, plan_with_geometry_hdfs):
        from rivia.hdf import UnsteadyPlan

        plan_path, _ = plan_with_geometry_hdfs
        plan = UnsteadyPlan(plan_path)
        _ = plan.geometry
        plan.close()
        assert not plan._hdf.id.valid
