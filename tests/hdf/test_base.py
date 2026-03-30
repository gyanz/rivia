"""Tests for rivia.hdf._base."""

from __future__ import annotations

import pytest

from rivia.hdf._base import _resolve_hdf_path


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
        from rivia.hdf import PlanHdf

        with PlanHdf(synthetic_plan_hdf) as hdf:
            assert hdf._hdf.id.valid
        assert not hdf._hdf.id.valid

    def test_explicit_close(self, synthetic_plan_hdf):
        from rivia.hdf import PlanHdf

        hdf = PlanHdf(synthetic_plan_hdf)
        assert hdf._hdf.id.valid
        hdf.close()
        assert not hdf._hdf.id.valid

    def test_suffix_auto_appended(self, synthetic_plan_hdf):
        from rivia.hdf import PlanHdf

        # Pass path without .hdf — it points to a file that doesn't exist,
        # but we just want to confirm the suffix logic; use the full path.
        no_suffix = str(synthetic_plan_hdf).removesuffix(".hdf")
        with PlanHdf(no_suffix) as hdf:
            assert hdf.filename == synthetic_plan_hdf
