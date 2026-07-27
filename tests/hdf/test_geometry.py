"""Tests for rivia.hdf._geometry (Geometry, FlowAreaCollection, FlowArea)."""

from __future__ import annotations

import numpy as np
import pytest

from rivia.hdf import FlowArea, Geometry, UnsteadyPlan

from .conftest import skip_if_no_example, EXAMPLE_PLAN_HDF

N_CELLS = 10
N_FACES = 20
AREA = "TestArea"


def _make_grid_flow_area() -> FlowArea:
    """4-cell 2x2 unit-square grid ``FlowArea`` for polygon-query tests.

    A plain ``dict`` stands in for the ``h5py.Group`` -- ``FlowArea`` only
    ever does ``self._g[key]`` / ``key in self._g``, both of which a dict
    supports, so no real HDF file is needed for geometry-only unit tests.

    Facepoints (index: coordinate)::

        6:(0,2) 7:(1,2) 8:(2,2)
        3:(0,1) 4:(1,1) 5:(2,1)
        0:(0,0) 1:(1,0) 2:(2,0)

    Cells (CCW corner facepoints, centre)::

        0: fp 0,1,4,3  centre (0.5, 0.5)
        1: fp 1,2,5,4  centre (1.5, 0.5)
        2: fp 3,4,7,6  centre (0.5, 1.5)
        3: fp 4,5,8,7  centre (1.5, 1.5)
    """
    fp_coords = np.array([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
        [0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
    ])
    cell_fp = np.full((4, 8), -1, dtype=np.int32)
    cell_fp[0, :4] = [0, 1, 4, 3]
    cell_fp[1, :4] = [1, 2, 5, 4]
    cell_fp[2, :4] = [3, 4, 7, 6]
    cell_fp[3, :4] = [4, 5, 8, 7]

    centers = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]])

    n_faces = 4  # dummy count -- unused since no face is marked curved
    fake_group = {
        "Cells Center Coordinate": centers,
        "Cells FacePoint Indexes": cell_fp,
        "FacePoints Coordinate": fp_coords,
        "Faces FacePoint Indexes": np.zeros((n_faces, 2), dtype=np.int32),
        "Faces Perimeter Info": np.zeros((n_faces, 2), dtype=np.int32),
        "Faces Perimeter Values": np.zeros((0, 2), dtype=np.float64),
        "Cells Face and Orientation Info": np.zeros((4, 2), dtype=np.int32),
        "Cells Face and Orientation Values": np.zeros((1, 2), dtype=np.int32),
    }
    return FlowArea(fake_group, "Grid", n_cells=4)


# ---------------------------------------------------------------------------
# FlowAreaCollection
# ---------------------------------------------------------------------------


class TestFlowAreaCollection:
    def test_names_returns_list(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            names = hdf.flow_areas.names
        assert isinstance(names, list)
        assert AREA in names

    def test_summary_has_name_column(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            df = hdf.flow_areas.summary
        assert "name" in df.columns
        assert AREA in df["name"].values

    def test_contains(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            assert AREA in hdf.flow_areas

    def test_len(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            assert len(hdf.flow_areas) == 1

    def test_missing_area_raises_key_error(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            with pytest.raises(KeyError):
                _ = hdf.flow_areas["DoesNotExist"]


# ---------------------------------------------------------------------------
# FlowArea geometry
# ---------------------------------------------------------------------------


class TestFlowArea:
    def test_n_cells(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            area = hdf.flow_areas[AREA]
            assert area.n_cells == N_CELLS

    def test_n_faces(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            area = hdf.flow_areas[AREA]
            assert area.n_faces == N_FACES

    def test_cell_centers_shape(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            cc = hdf.flow_areas[AREA].cell_centers
        assert cc.shape == (N_CELLS, 2)

    def test_cell_centers_excludes_ghost_cells(self, synthetic_plan_hdf):
        """Cell centers array should have exactly n_cells rows, not n_cells+2."""
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            area = hdf.flow_areas[AREA]
            assert area.cell_centers.shape[0] == area.n_cells

    def test_cell_min_elevation_shape(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            elev = hdf.flow_areas[AREA].cell_min_elevation
        assert elev.shape == (N_CELLS,)

    def test_cell_mannings_n_shape(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            n_val = hdf.flow_areas[AREA].cell_mannings_n
        assert n_val.shape == (N_CELLS,)

    def test_cell_surface_area_shape(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            sa = hdf.flow_areas[AREA].cell_surface_area
        assert sa.shape == (N_CELLS,)

    def test_face_normals_shape(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            fn = hdf.flow_areas[AREA].face_normals
        assert fn.shape == (N_FACES, 3)

    def test_face_cell_indexes_shape(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            fci = hdf.flow_areas[AREA].face_cell_indexes
        assert fci.shape == (N_FACES, 2)

    def test_face_area_elevation_info_shape(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            info, values = hdf.flow_areas[AREA].face_area_elevation
        assert info.shape == (N_FACES, 2)
        assert values.shape[1] == 4

    def test_cell_face_info_shape(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            info, values = hdf.flow_areas[AREA].cell_face_info
        assert info.shape[1] == 2
        assert values.shape[1] == 2

    def test_perimeter_is_2d(self, synthetic_plan_hdf):
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            p = hdf.flow_areas[AREA].perimeter
        assert p.ndim == 2 and p.shape[1] == 2

    def test_geometry_cached(self, synthetic_plan_hdf):
        """Repeated access returns equal values (backing array is cached in memory)."""
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            area = hdf.flow_areas[AREA]
            a = area.cell_centers
            b = area.cell_centers
        np.testing.assert_array_equal(a, b)

    def test_facepoint_face_orientation_shapes(self, synthetic_plan_hdf):
        """fp_face_info is (n_fp, 2) and fp_face_values is (total, 2)."""
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            area = hdf.flow_areas[AREA]
            info, vals = area.facepoint_face_orientation
        n_fp = len(area.facepoint_coordinates)
        assert info.shape == (n_fp, 2)
        assert vals.ndim == 2 and vals.shape[1] == 2

    def test_facepoint_face_orientation_dtypes(self, synthetic_plan_hdf):
        """Both arrays must be int32."""
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            info, vals = hdf.flow_areas[AREA].facepoint_face_orientation
        assert info.dtype == np.int32
        assert vals.dtype == np.int32

    def test_facepoint_face_orientation_counts_sum(self, synthetic_plan_hdf):
        """Sum of counts in fp_face_info equals len(fp_face_values)."""
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            info, vals = hdf.flow_areas[AREA].facepoint_face_orientation
        assert int(info[:, 1].sum()) == len(vals)

    def test_facepoint_face_orientation_total_entries(self, synthetic_plan_hdf):
        """Total entries == 2 * n_faces (each face contributes fpA and fpB)."""
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            area = hdf.flow_areas[AREA]
            info, vals = area.facepoint_face_orientation
        assert len(vals) == 2 * N_FACES

    def test_facepoint_face_orientation_valid_face_indices(self, synthetic_plan_hdf):
        """All face indices in fp_face_values are within [0, n_faces)."""
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            area = hdf.flow_areas[AREA]
            _, vals = area.facepoint_face_orientation
        assert (vals[:, 0] >= 0).all()
        assert (vals[:, 0] < N_FACES).all()

    def test_facepoint_face_orientation_valid_orientations(self, synthetic_plan_hdf):
        """Orientation flags are -1 or +1 only (fpA=-1, fpB=+1)."""
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            _, vals = hdf.flow_areas[AREA].facepoint_face_orientation
        assert set(vals[:, 1].tolist()).issubset({-1, 1})

    def test_facepoint_face_orientation_cached(self, synthetic_plan_hdf):
        """Second call returns the same array objects (cached)."""
        with UnsteadyPlan(synthetic_plan_hdf) as hdf:
            area = hdf.flow_areas[AREA]
            info1, vals1 = area.facepoint_face_orientation
            info2, vals2 = area.facepoint_face_orientation
        assert info1 is info2
        assert vals1 is vals2


# ---------------------------------------------------------------------------
# cells_within_polygon
# ---------------------------------------------------------------------------


class TestCellsWithinPolygon:
    def test_polygon_covers_whole_mesh(self):
        area = _make_grid_flow_area()
        polygon = np.array([[-1.0, -1.0], [3.0, -1.0], [3.0, 3.0], [-1.0, 3.0]])
        centroid = area.cells_within_polygon(polygon, mode="centroid")
        full = area.cells_within_polygon(polygon, mode="full")
        np.testing.assert_array_equal(centroid, [0, 1, 2, 3])
        np.testing.assert_array_equal(full, [0, 1, 2, 3])

    def test_polygon_outside_mesh_bbox_is_empty(self):
        area = _make_grid_flow_area()
        polygon = np.array([[10.0, 10.0], [11.0, 10.0], [11.0, 11.0], [10.0, 11.0]])
        centroid = area.cells_within_polygon(polygon, mode="centroid")
        full = area.cells_within_polygon(polygon, mode="full")
        assert centroid.shape == (0,)
        assert full.shape == (0,)
        assert centroid.dtype == np.int64
        assert full.dtype == np.int64

    def test_centroid_mode_matches_where_full_mode_does_not(self):
        """A vertical strip that spans every cell's centroid but not its corners:

        centroid mode should match all 4 cells; full mode should match none,
        since every cell has a corner at x=0 or x=2 which lies outside the
        strip's x range of [0.2, 1.8].
        """
        area = _make_grid_flow_area()
        strip = np.array([[0.2, -0.5], [1.8, -0.5], [1.8, 2.5], [0.2, 2.5]])
        centroid = area.cells_within_polygon(strip, mode="centroid")
        full = area.cells_within_polygon(strip, mode="full")
        np.testing.assert_array_equal(centroid, [0, 1, 2, 3])
        assert full.shape == (0,)

    def test_concave_polygon_excludes_notch_cells(self):
        """An L-shaped (concave) polygon covering only the bottom row and the
        left column should include cells 0, 1, 2 but exclude cell 3, whose
        centre (1.5, 1.5) sits in the notch cut out of the top-right.
        """
        area = _make_grid_flow_area()
        l_shape = np.array([
            [-0.5, -0.5], [2.5, -0.5], [2.5, 1.0],
            [1.0, 1.0], [1.0, 2.5], [-0.5, 2.5],
        ])
        centroid = area.cells_within_polygon(l_shape, mode="centroid")
        np.testing.assert_array_equal(centroid, [0, 1, 2])

    def test_malformed_cell_polygon_excluded_only_under_full_mode(self):
        """A cell with < 3 valid corner facepoints (broken adjacency) has no
        usable footprint for "full" mode, but its centre is still valid, so
        it can still match under "centroid" mode.
        """
        fp_coords = np.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
        ])
        cell_fp = np.full((2, 8), -1, dtype=np.int32)
        cell_fp[0, :4] = [0, 1, 2, 3]      # normal quad
        cell_fp[1, :2] = [0, 1]            # malformed: only 2 valid corners
        centers = np.array([[0.5, 0.5], [1.5, 0.5]])
        n_faces = 2
        fake_group = {
            "Cells Center Coordinate": centers,
            "Cells FacePoint Indexes": cell_fp,
            "FacePoints Coordinate": fp_coords,
            "Faces FacePoint Indexes": np.zeros((n_faces, 2), dtype=np.int32),
            "Faces Perimeter Info": np.zeros((n_faces, 2), dtype=np.int32),
            "Faces Perimeter Values": np.zeros((0, 2), dtype=np.float64),
            "Cells Face and Orientation Info": np.zeros((2, 2), dtype=np.int32),
            "Cells Face and Orientation Values": np.zeros((1, 2), dtype=np.int32),
        }
        area = FlowArea(fake_group, "Malformed", n_cells=2)

        polygon = np.array([[-1.0, -1.0], [3.0, -1.0], [3.0, 3.0], [-1.0, 3.0]])
        centroid = area.cells_within_polygon(polygon, mode="centroid")
        full = area.cells_within_polygon(polygon, mode="full")
        np.testing.assert_array_equal(centroid, [0, 1])
        np.testing.assert_array_equal(full, [0])

    def test_invalid_polygon_shape_raises(self):
        area = _make_grid_flow_area()
        with pytest.raises(ValueError):
            area.cells_within_polygon(np.array([[0.0, 0.0], [1.0, 1.0]]))

    def test_invalid_polygon_ndim_raises(self):
        area = _make_grid_flow_area()
        with pytest.raises(ValueError):
            area.cells_within_polygon(np.array([0.0, 1.0, 2.0]))

    def test_invalid_mode_raises(self):
        area = _make_grid_flow_area()
        polygon = np.array([[-1.0, -1.0], [3.0, -1.0], [3.0, 3.0], [-1.0, 3.0]])
        with pytest.raises(ValueError):
            area.cells_within_polygon(polygon, mode="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Integration tests against the real example file
# ---------------------------------------------------------------------------


@skip_if_no_example
class TestGeometryIntegration:
    def test_open_plan_as_geometry(self):
        """UnsteadyPlan should expose geometry just like Geometry."""
        with UnsteadyPlan(EXAMPLE_PLAN_HDF) as hdf:
            names = hdf.flow_areas.names
            assert len(names) >= 1
            area = hdf.flow_areas[names[0]]
            assert area.n_cells > 0
            assert area.n_faces > 0
            assert area.cell_centers.shape == (area.n_cells, 2)
            assert area.face_normals.shape == (area.n_faces, 3)
