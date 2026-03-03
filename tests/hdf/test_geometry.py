"""Tests for raspy.hdf._geometry (GeometryHdf, FlowAreaCollection, FlowArea)."""
from __future__ import annotations

import numpy as np
import pytest

from raspy.hdf import GeometryHdf, PlanHdf

from .conftest import skip_if_no_example, EXAMPLE_PLAN_HDF

N_CELLS = 10
N_FACES = 20
AREA    = "TestArea"


# ---------------------------------------------------------------------------
# FlowAreaCollection
# ---------------------------------------------------------------------------

class TestFlowAreaCollection:
    def test_names_returns_list(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            names = hdf.flow_areas.names
        assert isinstance(names, list)
        assert AREA in names

    def test_summary_has_name_column(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            df = hdf.flow_areas.summary
        assert "name" in df.columns
        assert AREA in df["name"].values

    def test_contains(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            assert AREA in hdf.flow_areas

    def test_len(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            assert len(hdf.flow_areas) == 1

    def test_missing_area_raises_key_error(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            with pytest.raises(KeyError):
                _ = hdf.flow_areas["DoesNotExist"]


# ---------------------------------------------------------------------------
# FlowArea geometry
# ---------------------------------------------------------------------------

class TestFlowArea:
    def test_n_cells(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            area = hdf.flow_areas[AREA]
            assert area.n_cells == N_CELLS

    def test_n_faces(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            area = hdf.flow_areas[AREA]
            assert area.n_faces == N_FACES

    def test_cell_centers_shape(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            cc = hdf.flow_areas[AREA].cell_centers
        assert cc.shape == (N_CELLS, 2)

    def test_cell_centers_excludes_ghost_cells(self, synthetic_plan_hdf):
        """Cell centers array should have exactly n_cells rows, not n_cells+2."""
        with PlanHdf(synthetic_plan_hdf) as hdf:
            area = hdf.flow_areas[AREA]
            assert area.cell_centers.shape[0] == area.n_cells

    def test_cell_min_elevation_shape(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            elev = hdf.flow_areas[AREA].cell_min_elevation
        assert elev.shape == (N_CELLS,)

    def test_cell_mannings_n_shape(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            n_val = hdf.flow_areas[AREA].cell_mannings_n
        assert n_val.shape == (N_CELLS,)

    def test_cell_surface_area_shape(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            sa = hdf.flow_areas[AREA].cell_surface_area
        assert sa.shape == (N_CELLS,)

    def test_face_normals_shape(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            fn = hdf.flow_areas[AREA].face_normals
        assert fn.shape == (N_FACES, 3)

    def test_face_cell_indexes_shape(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            fci = hdf.flow_areas[AREA].face_cell_indexes
        assert fci.shape == (N_FACES, 2)

    def test_face_area_elevation_info_shape(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            info, values = hdf.flow_areas[AREA].face_area_elevation
        assert info.shape == (N_FACES, 2)
        assert values.shape[1] == 4

    def test_cell_face_info_shape(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            info, values = hdf.flow_areas[AREA].cell_face_info
        assert info.shape[1] == 2
        assert values.shape[1] == 2

    def test_perimeter_is_2d(self, synthetic_plan_hdf):
        with PlanHdf(synthetic_plan_hdf) as hdf:
            p = hdf.flow_areas[AREA].perimeter
        assert p.ndim == 2 and p.shape[1] == 2

    def test_geometry_cached(self, synthetic_plan_hdf):
        """Repeated access returns equal values (backing array is cached in memory)."""
        with PlanHdf(synthetic_plan_hdf) as hdf:
            area = hdf.flow_areas[AREA]
            a = area.cell_centers
            b = area.cell_centers
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Integration tests against the real example file
# ---------------------------------------------------------------------------

@skip_if_no_example
class TestGeometryHdfIntegration:
    def test_open_plan_as_geometry(self):
        """PlanHdf should expose geometry just like GeometryHdf."""
        with PlanHdf(EXAMPLE_PLAN_HDF) as hdf:
            names = hdf.flow_areas.names
            assert len(names) >= 1
            area = hdf.flow_areas[names[0]]
            assert area.n_cells > 0
            assert area.n_faces > 0
            assert area.cell_centers.shape == (area.n_cells, 2)
            assert area.face_normals.shape == (area.n_faces, 3)
