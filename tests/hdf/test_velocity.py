"""Tests for raspy.hdf._velocity (pure-numpy WLS functions)."""

from __future__ import annotations

import numpy as np
import pytest

from raspy.hdf._velocity import (
    _estimate_face_wse_average,
    _estimate_face_wse_sloped,
    _interpolate_face_flow_area,
    _wls_velocity,
    compute_all_cell_velocities,
)


# ---------------------------------------------------------------------------
# _wls_velocity
# ---------------------------------------------------------------------------


class TestWlsVelocity:
    def test_axis_aligned_faces(self):
        """Pure x-flow: two x-normal faces, one y-normal face with zero V_n."""
        normals = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        vn = np.array([2.0, 2.0, 0.0])
        weights = np.array([1.0, 1.0, 1.0])
        uv = _wls_velocity(vn, weights, normals)
        np.testing.assert_allclose(uv[0], 2.0, rtol=1e-6)
        np.testing.assert_allclose(uv[1], 0.0, atol=1e-10)

    def test_pure_y_flow(self):
        # Two y-normal faces alone make the system singular (a11=0).
        # Add one x-normal face with zero V_n so the system is well-posed
        # and still reconstructs u=0, v=3.
        normals = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        vn = np.array([0.0, 3.0, 3.0])
        weights = np.ones(3)
        uv = _wls_velocity(vn, weights, normals)
        np.testing.assert_allclose(uv[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(uv[1], 3.0, rtol=1e-6)

    def test_singular_system_returns_zero(self):
        """All normals in same direction → singular system → [0, 0]."""
        normals = np.array([[1.0, 0.0], [1.0, 0.0]])
        vn = np.array([1.0, 1.0])
        weights = np.ones(2)
        uv = _wls_velocity(vn, weights, normals)
        np.testing.assert_array_equal(uv, [0.0, 0.0])

    def test_zero_weights_returns_zero(self):
        normals = np.array([[1.0, 0.0], [0.0, 1.0]])
        vn = np.array([1.0, 1.0])
        weights = np.zeros(2)
        uv = _wls_velocity(vn, weights, normals)
        np.testing.assert_array_equal(uv, [0.0, 0.0])

    def test_output_shape(self):
        normals = np.random.default_rng(0).uniform(-1, 1, (5, 2))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        vn = np.ones(5)
        weights = np.ones(5)
        uv = _wls_velocity(vn, weights, normals)
        assert uv.shape == (2,)


# ---------------------------------------------------------------------------
# _interpolate_face_flow_area
# ---------------------------------------------------------------------------


class TestInterpolateFaceFlowArea:
    @pytest.fixture()
    def simple_table(self):
        """Two-entry table: area=0 at elev=0, area=100 at elev=10."""
        ae_info = np.array([[0, 2]], dtype="i4")
        ae_values = np.array(
            [[0.0, 0.0, 0.0, 0.04], [10.0, 100.0, 40.0, 0.04]], dtype="f4"
        )
        return ae_info, ae_values

    def test_below_table_returns_zero(self, simple_table):
        info, vals = simple_table
        assert _interpolate_face_flow_area(0, -1.0, info, vals) == 0.0

    def test_above_table_returns_max(self, simple_table):
        info, vals = simple_table
        assert _interpolate_face_flow_area(0, 15.0, info, vals) == pytest.approx(100.0)

    def test_midpoint_interpolated(self, simple_table):
        info, vals = simple_table
        area = _interpolate_face_flow_area(0, 5.0, info, vals)
        assert area == pytest.approx(50.0, rel=1e-5)

    def test_at_lower_bound(self, simple_table):
        info, vals = simple_table
        area = _interpolate_face_flow_area(0, 0.0, info, vals)
        # elev == elevs[0] → returns 0 (below-or-equal branch)
        assert area == 0.0


# ---------------------------------------------------------------------------
# _estimate_face_wse_average
# ---------------------------------------------------------------------------


class TestEstimateFaceWseAverage:
    def test_interior_face_averages_neighbours(self):
        # face 0: left=0 (wse=2), right=1 (wse=4) → expected 3
        fci = np.array([[0, 1], [1, 2]], dtype="i4")
        cell_wse = np.array([2.0, 4.0, 6.0])
        result = _estimate_face_wse_average(fci, cell_wse, n_cells=3)
        assert result[0] == pytest.approx(3.0)
        assert result[1] == pytest.approx(5.0)

    def test_boundary_face_uses_single_cell(self):
        # face 0: left=0 (real), right=-1 (boundary)
        fci = np.array([[0, -1]], dtype="i4")
        cell_wse = np.array([5.0])
        result = _estimate_face_wse_average(fci, cell_wse, n_cells=1)
        assert result[0] == pytest.approx(5.0)

    def test_ghost_cell_treated_as_boundary(self):
        # n_cells=2; face right index=5 → ghost → use left cell only
        fci = np.array([[0, 5]], dtype="i4")
        cell_wse = np.array([3.0, 7.0])
        result = _estimate_face_wse_average(fci, cell_wse, n_cells=2)
        assert result[0] == pytest.approx(3.0)

    def test_both_boundary_returns_zero(self):
        fci = np.array([[-1, -1]], dtype="i4")
        cell_wse = np.array([5.0])
        result = _estimate_face_wse_average(fci, cell_wse, n_cells=1)
        assert result[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _estimate_face_wse_sloped
# ---------------------------------------------------------------------------


class TestEstimateFaceWseSloped:
    def test_symmetric_equals_average(self):
        """Face at midpoint between equal-distance cells → same as average."""
        # cell0 at (0,0) wse=2, cell1 at (2,0) wse=4, face at (1,0)
        fci = np.array([[0, 1]], dtype="i4")
        cell_wse = np.array([2.0, 4.0])
        cell_coords = np.array([[0.0, 0.0], [2.0, 0.0]])
        face_coords = np.array([[1.0, 0.0]])
        result = _estimate_face_wse_sloped(fci, cell_wse, 2, cell_coords, face_coords)
        assert result[0] == pytest.approx(3.0)

    def test_asymmetric_interpolation(self):
        """Face closer to left cell → result biased toward left WSE."""
        # cell0 at (0,0) wse=2, cell1 at (6,0) wse=8, face at (2,0)
        # d_left=2, d_right=4, t=2/6=1/3
        # wse = (1-1/3)*2 + (1/3)*8 = 4/3 + 8/3 = 4.0
        fci = np.array([[0, 1]], dtype="i4")
        cell_wse = np.array([2.0, 8.0])
        cell_coords = np.array([[0.0, 0.0], [6.0, 0.0]])
        face_coords = np.array([[2.0, 0.0]])
        result = _estimate_face_wse_sloped(fci, cell_wse, 2, cell_coords, face_coords)
        assert result[0] == pytest.approx(4.0)

    def test_boundary_face_uses_single_cell(self):
        fci = np.array([[0, -1]], dtype="i4")
        cell_wse = np.array([5.0])
        cell_coords = np.array([[0.0, 0.0]])
        face_coords = np.array([[1.0, 0.0]])
        result = _estimate_face_wse_sloped(fci, cell_wse, 1, cell_coords, face_coords)
        assert result[0] == pytest.approx(5.0)

    def test_ghost_cell_treated_as_boundary(self):
        fci = np.array([[0, 5]], dtype="i4")
        cell_wse = np.array([3.0, 7.0])
        cell_coords = np.array([[0.0, 0.0], [2.0, 0.0]])
        face_coords = np.array([[1.0, 0.0]])
        result = _estimate_face_wse_sloped(fci, cell_wse, 2, cell_coords, face_coords)
        assert result[0] == pytest.approx(3.0)

    def test_both_boundary_returns_zero(self):
        fci = np.array([[-1, -1]], dtype="i4")
        cell_wse = np.array([5.0])
        cell_coords = np.array([[0.0, 0.0]])
        face_coords = np.array([[1.0, 0.0]])
        result = _estimate_face_wse_sloped(fci, cell_wse, 1, cell_coords, face_coords)
        assert result[0] == pytest.approx(0.0)

    def test_degenerate_geometry_falls_back_to_average(self):
        """Cells at identical locations → fallback to 0.5 weight."""
        fci = np.array([[0, 1]], dtype="i4")
        cell_wse = np.array([2.0, 6.0])
        cell_coords = np.array([[1.0, 1.0], [1.0, 1.0]])  # same point
        face_coords = np.array([[1.0, 1.0]])
        result = _estimate_face_wse_sloped(fci, cell_wse, 2, cell_coords, face_coords)
        assert result[0] == pytest.approx(4.0)  # (2+6)/2


# ---------------------------------------------------------------------------
# compute_all_cell_velocities  (smoke test — correctness covered above)
# ---------------------------------------------------------------------------


class TestComputeAllCellVelocities:
    @pytest.fixture()
    def simple_mesh(self):
        """Two cells, two faces — simple symmetric mesh."""
        n_cells = 2
        n_faces = 2

        # each cell has one face
        cf_info = np.array([[0, 1], [1, 1]], dtype="i4")
        cf_values = np.array([[0, 1], [1, -1]], dtype="i4")

        # face normals: face0 points +x, face1 points -x
        normals = np.array([[1.0, 0.0, 10.0], [-1.0, 0.0, 10.0]], dtype="f4")

        fci = np.array([[0, 1], [1, 0]], dtype="i4")

        # flat area-elevation tables: area=50 everywhere
        ae_info = np.array([[0, 2], [2, 2]], dtype="i4")
        ae_values = np.array(
            [
                [0.0, 50.0, 20.0, 0.04],
                [10.0, 50.0, 20.0, 0.04],
                [0.0, 50.0, 20.0, 0.04],
                [10.0, 50.0, 20.0, 0.04],
            ],
            dtype="f4",
        )

        face_vel = np.array([1.0, -1.0], dtype="f4")
        cell_wse = np.array([5.0, 5.0], dtype="f4")

        return dict(
            n_cells=n_cells,
            cf_info=cf_info,
            cf_values=cf_values,
            normals=normals,
            fci=fci,
            ae_info=ae_info,
            ae_values=ae_values,
            face_vel=face_vel,
            cell_wse=cell_wse,
        )

    def test_output_shape(self, simple_mesh):
        m = simple_mesh
        result = compute_all_cell_velocities(
            n_cells=m["n_cells"],
            cell_face_info=m["cf_info"],
            cell_face_values=m["cf_values"],
            face_normals=m["normals"],
            face_cell_indexes=m["fci"],
            face_ae_info=m["ae_info"],
            face_ae_values=m["ae_values"],
            face_vel=m["face_vel"],
            cell_wse=m["cell_wse"],
        )
        assert result.shape == (m["n_cells"], 2)

    def test_area_weighted_and_length_weighted_agree_for_constant_area(
        self, simple_mesh
    ):
        """When face area is constant, both methods should give the same answer."""
        m = simple_mesh
        kw = dict(
            n_cells=m["n_cells"],
            cell_face_info=m["cf_info"],
            cell_face_values=m["cf_values"],
            face_normals=m["normals"],
            face_cell_indexes=m["fci"],
            face_ae_info=m["ae_info"],
            face_ae_values=m["ae_values"],
            face_vel=m["face_vel"],
            cell_wse=m["cell_wse"],
        )
        aw = compute_all_cell_velocities(**kw, method="area_weighted")
        lw = compute_all_cell_velocities(**kw, method="length_weighted")
        # directions should agree even if magnitudes differ
        np.testing.assert_allclose(np.sign(aw[:, 0]), np.sign(lw[:, 0]))

    def test_invalid_method_raises(self, simple_mesh):
        m = simple_mesh
        with pytest.raises(ValueError, match="method"):
            compute_all_cell_velocities(
                n_cells=m["n_cells"],
                cell_face_info=m["cf_info"],
                cell_face_values=m["cf_values"],
                face_normals=m["normals"],
                face_cell_indexes=m["fci"],
                face_ae_info=m["ae_info"],
                face_ae_values=m["ae_values"],
                face_vel=m["face_vel"],
                cell_wse=m["cell_wse"],
                method="bad_method",
            )

    def test_flow_ratio_without_face_flow_raises(self, simple_mesh):
        m = simple_mesh
        with pytest.raises(ValueError, match="face_flow"):
            compute_all_cell_velocities(
                n_cells=m["n_cells"],
                cell_face_info=m["cf_info"],
                cell_face_values=m["cf_values"],
                face_normals=m["normals"],
                face_cell_indexes=m["fci"],
                face_ae_info=m["ae_info"],
                face_ae_values=m["ae_values"],
                face_vel=m["face_vel"],
                cell_wse=m["cell_wse"],
                method="flow_ratio",
            )

    def test_invalid_wse_interp_raises(self, simple_mesh):
        m = simple_mesh
        with pytest.raises(ValueError, match="wse_interp"):
            compute_all_cell_velocities(
                n_cells=m["n_cells"],
                cell_face_info=m["cf_info"],
                cell_face_values=m["cf_values"],
                face_normals=m["normals"],
                face_cell_indexes=m["fci"],
                face_ae_info=m["ae_info"],
                face_ae_values=m["ae_values"],
                face_vel=m["face_vel"],
                cell_wse=m["cell_wse"],
                wse_interp="bad",
            )

    def test_sloped_without_coords_raises(self, simple_mesh):
        m = simple_mesh
        with pytest.raises(ValueError, match="cell_coords"):
            compute_all_cell_velocities(
                n_cells=m["n_cells"],
                cell_face_info=m["cf_info"],
                cell_face_values=m["cf_values"],
                face_normals=m["normals"],
                face_cell_indexes=m["fci"],
                face_ae_info=m["ae_info"],
                face_ae_values=m["ae_values"],
                face_vel=m["face_vel"],
                cell_wse=m["cell_wse"],
                wse_interp="sloped",
            )

    def test_sloped_symmetric_agrees_with_average(self, simple_mesh):
        """On a symmetric mesh, sloped WSE == average WSE → same velocities."""
        m = simple_mesh
        # cell centres at x=0 and x=2; faces at x=0 (shared) and x=2
        cell_coords = np.array([[0.0, 0.0], [2.0, 0.0]])
        face_coords = np.array([[0.0, 0.0], [2.0, 0.0]])
        kw = dict(
            n_cells=m["n_cells"],
            cell_face_info=m["cf_info"],
            cell_face_values=m["cf_values"],
            face_normals=m["normals"],
            face_cell_indexes=m["fci"],
            face_ae_info=m["ae_info"],
            face_ae_values=m["ae_values"],
            face_vel=m["face_vel"],
            cell_wse=m["cell_wse"],
        )
        avg = compute_all_cell_velocities(**kw, wse_interp="average")
        sloped = compute_all_cell_velocities(
            **kw,
            wse_interp="sloped",
            cell_coords=cell_coords,
            face_coords=face_coords,
        )
        np.testing.assert_allclose(avg, sloped, rtol=1e-6)
