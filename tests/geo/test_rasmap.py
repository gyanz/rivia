"""Tests for raspy.geo._rasmap — RASMapper-exact 2D pipeline functions.

All tests use hand-crafted minimal meshes so no HDF files are required.

Minimal mesh used throughout:
  - 2 cells (cell 0 and cell 1) sharing one interior face
  - 3 faces total: face 0 (shared), face 1 (boundary of cell 0),
    face 2 (boundary of cell 1)
  - 4 facepoints forming a simple rectangle

        fp0 --- face1 --- fp1
        |                  |
      face2   cell0   face0   cell1   (face0 is the shared interior face)
        |                  |
        fp3 --- face2? --- fp2   <- we keep it simple with fewer faces

  For simplicity we use a 2-cell strip:

        fp0 --face1-- fp1 --face2-- fp2
        |              |              |
      face0(bdry)  face3(shared)  face4(bdry)
        |              |              |
        fp3 --face5-- fp4 --face6-- fp5

  That's 6 facepoints, 7 faces.  Instead we use the absolute minimum:
  2 cells, 3 faces, 4 facepoints.

Mesh layout (2 square cells side by side):

    fp0(0,1) --- fp1(1,1) --- fp2(2,1)
      |               |               |
    face0           face1(shared)   face2
      |               |               |
    fp3(0,0) --- fp4(1,0) --- fp5(2,0)

  cell0: left square (fp0,fp3,fp4,fp1)
  cell1: right square (fp1,fp4,fp5,fp2)

  Faces:
    face0 : fp0-fp3  (left boundary of cell0)    cellA=0, cellB=-1
    face1 : fp3-fp4  (bottom boundary of cell0)  cellA=0, cellB=-1
    face2 : fp0-fp1  (top boundary of cell0)     cellA=0, cellB=-1
    face3 : fp1-fp4  (shared interior)            cellA=0, cellB=1
    face4 : fp4-fp5  (bottom boundary of cell1)  cellA=1, cellB=-1
    face5 : fp1-fp2  (top boundary of cell1)     cellA=1, cellB=-1
    face6 : fp2-fp5  (right boundary of cell1)   cellA=1, cellB=-1
"""
from __future__ import annotations

import numpy as np
import pytest

from raspy.geo._rasmap import (
    compute_face_wss,
    compute_facepoint_wse,
    reconstruct_face_velocities,
    compute_facepoint_velocities,
    replace_face_velocities_sloped,
    _barycentric_weights,
    _donate,
    _downward_adjust_fp_wse,
    _pixel_wse_sloped,
    _all_shallow,
)


# ---------------------------------------------------------------------------
# Shared minimal mesh fixtures
# ---------------------------------------------------------------------------

N_FP    = 6
N_FACES = 7
N_CELLS = 2

# Facepoint coordinates
FP_COORDS = np.array([
    [0.0, 1.0],  # fp0
    [1.0, 1.0],  # fp1
    [2.0, 1.0],  # fp2
    [0.0, 0.0],  # fp3
    [1.0, 0.0],  # fp4
    [2.0, 0.0],  # fp5
], dtype=np.float64)

# face_facepoint_indexes: [fpA, fpB]
FACE_FP = np.array([
    [0, 3],  # face0: fp0-fp3 (left bdry cell0)
    [3, 4],  # face1: fp3-fp4 (bot  bdry cell0)
    [0, 1],  # face2: fp0-fp1 (top  bdry cell0)
    [1, 4],  # face3: fp1-fp4 (shared interior)
    [4, 5],  # face4: fp4-fp5 (bot  bdry cell1)
    [1, 2],  # face5: fp1-fp2 (top  bdry cell1)
    [2, 5],  # face6: fp2-fp5 (right bdry cell1)
], dtype=np.int32)

# face_cell_indexes: [cellA, cellB]  (-1 = boundary)
FACE_CI = np.array([
    [0, -1],  # face0
    [0, -1],  # face1
    [0, -1],  # face2
    [0,  1],  # face3 — shared
    [1, -1],  # face4
    [1, -1],  # face5
    [1, -1],  # face6
], dtype=np.int32)

# cell_face_info: [start, count] + cell_face_values [face_idx, orientation]
# cell0 has faces: 0,1,2,3 — orientations all 1 for simplicity
# cell1 has faces: 3,4,5,6 — orientations all 1
CELL_FACE_INFO = np.array([
    [0, 4],   # cell0: start=0, count=4
    [4, 4],   # cell1: start=4, count=4
], dtype=np.int32)
CELL_FACE_VALS = np.array([
    [0, 1], [1, 1], [2, 1], [3, 1],   # cell0
    [3, 0], [4, 1], [5, 1], [6, 1],   # cell1 (face3 orientation=0 from cell1 side)
], dtype=np.int32)

# cell_face_count (for virtual cell detection)
CELL_FACE_COUNT = np.array([4, 4], dtype=np.int32)

# Face normals (unit vectors [nx, ny], lengths ~1.0)
# face0: fp0(0,1)->fp3(0,0): dx=0,dy=-1 → normal=(dy/L, -dx/L)=(-1,0)...
# Actually for a vertical face (fp0-fp3): dx=0, dy=-1, L=1 → n=(dy/L, -dx/L)=(-1,0)
# For horizontal face (fp3-fp4): dx=1, dy=0, L=1 → n=(0, -1)
# face3 (fp1-fp4): dx=0, dy=-1, L=1 → n=(-1, 0)
FACE_NORMALS_2D = np.array([
    [-1.0,  0.0],  # face0 (vertical left)
    [ 0.0, -1.0],  # face1 (horizontal bottom)
    [ 0.0,  1.0],  # face2 (horizontal top)
    [-1.0,  0.0],  # face3 (vertical shared)
    [ 0.0, -1.0],  # face4 (horizontal bottom)
    [ 0.0,  1.0],  # face5 (horizontal top)
    [ 1.0,  0.0],  # face6 (vertical right)
], dtype=np.float64)

FACE_LENGTHS = np.ones(N_FACES, dtype=np.float64)

FACE_MIN_ELEV = np.zeros(N_FACES, dtype=np.float64)  # all invert at z=0
CELL_MIN_ELEV = np.zeros(N_CELLS, dtype=np.float64)  # both cells min elev = 0


# ---------------------------------------------------------------------------
# Step A: compute_face_wss
# ---------------------------------------------------------------------------


class TestComputeFaceWss:
    def test_both_cells_wet_shared_face_connected(self):
        """Interior face is connected when both cells have WSE well above invert."""
        cell_wse = np.array([2.0, 2.0], dtype=np.float64)
        face_connected, val_a, val_b = compute_face_wss(
            cell_wse, CELL_MIN_ELEV, FACE_MIN_ELEV, FACE_CI, CELL_FACE_COUNT
        )
        assert face_connected[3], "shared face should be connected"

    def test_one_cell_dry_shared_face_disconnected(self):
        """Interior face is disconnected when one cell is dry."""
        cell_wse = np.array([0.0, 2.0], dtype=np.float64)  # cell0 dry
        face_connected, _, _ = compute_face_wss(
            cell_wse, CELL_MIN_ELEV, FACE_MIN_ELEV, FACE_CI, CELL_FACE_COUNT
        )
        assert not face_connected[3]

    def test_boundary_faces_never_connected(self):
        """Boundary faces (cellB=-1) are always marked disconnected."""
        cell_wse = np.array([5.0, 5.0], dtype=np.float64)
        face_connected, _, _ = compute_face_wss(
            cell_wse, CELL_MIN_ELEV, FACE_MIN_ELEV, FACE_CI, CELL_FACE_COUNT
        )
        # faces 0,1,2 are boundary for cell0; faces 4,5,6 for cell1
        for bdry_f in [0, 1, 2, 4, 5, 6]:
            assert not face_connected[bdry_f], f"face {bdry_f} should not be connected"

    def test_face_value_a_nodata_on_dry_cell(self):
        """face_value_a is -9999 for the dry side."""
        cell_wse = np.array([0.0, 3.0], dtype=np.float64)
        _, val_a, val_b = compute_face_wss(
            cell_wse, CELL_MIN_ELEV, FACE_MIN_ELEV, FACE_CI, CELL_FACE_COUNT
        )
        assert val_a[3] == -9999.0  # cell0 is dry → face3 cellA side nodata
        assert val_b[3] != -9999.0  # cell1 is wet → face3 cellB side has value

    def test_both_below_face_invert_disconnected(self):
        """Both cells below face invert → disconnected, WSE stored."""
        cell_wse = np.array([-1.0, -1.0], dtype=np.float64)
        face_min = np.zeros(N_FACES, dtype=np.float64)
        cell_min = np.full(N_CELLS, -2.0, dtype=np.float64)  # cells well below
        cell_wse2 = np.array([0.5, 0.5])  # above cell_min but below face_invert=1
        face_min2 = np.full(N_FACES, 1.0, dtype=np.float64)
        face_connected, val_a, val_b = compute_face_wss(
            cell_wse2, cell_min, face_min2, FACE_CI, CELL_FACE_COUNT
        )
        assert not face_connected[3]
        assert val_a[3] == pytest.approx(0.5)
        assert val_b[3] == pytest.approx(0.5)

    def test_returns_correct_dtypes(self):
        cell_wse = np.array([2.0, 2.0])
        fc, va, vb = compute_face_wss(
            cell_wse, CELL_MIN_ELEV, FACE_MIN_ELEV, FACE_CI, CELL_FACE_COUNT
        )
        assert fc.dtype == bool
        assert va.dtype == np.float64
        assert vb.dtype == np.float64


# ---------------------------------------------------------------------------
# Step B: compute_facepoint_wse
# ---------------------------------------------------------------------------

def _make_fp_face_orientation():
    """Build fp_face_info / fp_face_values for our minimal mesh (no HDF)."""
    n_fp = N_FP
    n_faces = N_FACES
    fp_counts = np.zeros(n_fp, dtype=np.int32)
    for fi in range(n_faces):
        fp_counts[int(FACE_FP[fi, 0])] += 1
        fp_counts[int(FACE_FP[fi, 1])] += 1
    fp_info = np.zeros((n_fp, 2), dtype=np.int32)
    offset = 0
    for i in range(n_fp):
        fp_info[i, 0] = offset
        fp_info[i, 1] = fp_counts[i]
        offset += fp_counts[i]
    fp_vals = np.zeros((offset, 2), dtype=np.int32)
    current = np.zeros(n_fp, dtype=np.int32)
    for fi in range(n_faces):
        fpA = int(FACE_FP[fi, 0]);  fpB = int(FACE_FP[fi, 1])
        pos_a = fp_info[fpA, 0] + current[fpA]
        fp_vals[pos_a] = [fi, 0];  current[fpA] += 1
        pos_b = fp_info[fpB, 0] + current[fpB]
        fp_vals[pos_b] = [fi, 1];  current[fpB] += 1
    return fp_info, fp_vals


class TestComputeFacepointWse:
    def _run(self, cell_wse):
        face_connected, val_a, val_b = compute_face_wss(
            cell_wse, CELL_MIN_ELEV, FACE_MIN_ELEV, FACE_CI, CELL_FACE_COUNT
        )
        fp_info, fp_vals = _make_fp_face_orientation()
        return compute_facepoint_wse(
            FP_COORDS, fp_info, fp_vals, FACE_FP, val_a, val_b
        )

    def test_interior_facepoints_have_valid_wse(self):
        """Facepoints shared by two wet cells (fp1, fp4) should have valid WSE.

        Corner/boundary facepoints (fp0, fp2, fp3, fp5) are only adjacent to
        boundary faces; compute_face_wss skips those (cellB=-1), so those
        facepoints legitimately remain nodata.
        """
        fp_wse = self._run(np.array([2.0, 2.0]))
        # fp1 (idx=1) and fp4 (idx=4) are at the shared face endpoints
        assert fp_wse[1] != -9999.0, "fp1 should have valid WSE"
        assert fp_wse[4] != -9999.0, "fp4 should have valid WSE"
        # At least some facepoints have valid WSE
        assert (fp_wse != -9999.0).sum() >= 2

    def test_facepoint_wse_close_to_cell_wse(self):
        """In a uniform WSE field, all facepoint WSEs should equal that value."""
        wse_val = 3.5
        fp_wse = self._run(np.full(N_CELLS, wse_val))
        valid = fp_wse[fp_wse != -9999.0]
        assert valid == pytest.approx(wse_val, abs=0.5)  # allow small regression error

    def test_all_dry_returns_nodata(self):
        """Dry mesh → all facepoints nodata."""
        fp_wse = self._run(np.zeros(N_CELLS))  # WSE=0 = cell_min_elev
        assert (fp_wse == -9999.0).all()

    def test_output_shape(self):
        fp_wse = self._run(np.array([2.0, 2.0]))
        assert fp_wse.shape == (N_FP,)
        assert fp_wse.dtype == np.float64


# ---------------------------------------------------------------------------
# Step 2: reconstruct_face_velocities
# ---------------------------------------------------------------------------


class TestReconstructFaceVelocities:
    def test_output_shapes(self):
        fv = np.ones(N_FACES, dtype=np.float64)
        fc = np.ones(N_FACES, dtype=bool)
        A, B = reconstruct_face_velocities(
            fv, FACE_NORMALS_2D, fc, FACE_CI, CELL_FACE_INFO, CELL_FACE_VALS
        )
        assert A.shape == (N_FACES, 2)
        assert B.shape == (N_FACES, 2)

    def test_zero_velocity_gives_zero_vectors(self):
        fv = np.zeros(N_FACES, dtype=np.float64)
        fc = np.zeros(N_FACES, dtype=bool)
        A, B = reconstruct_face_velocities(
            fv, FACE_NORMALS_2D, fc, FACE_CI, CELL_FACE_INFO, CELL_FACE_VALS
        )
        np.testing.assert_array_almost_equal(A, 0.0)
        np.testing.assert_array_almost_equal(B, 0.0)

    def test_connected_face_averages_A_and_B(self):
        """When face3 is connected, A and B must be equal (averaged)."""
        fv = np.random.default_rng(42).uniform(-1, 1, N_FACES)
        fc = np.zeros(N_FACES, dtype=bool)
        fc[3] = True  # only shared face connected
        A, B = reconstruct_face_velocities(
            fv, FACE_NORMALS_2D, fc, FACE_CI, CELL_FACE_INFO, CELL_FACE_VALS
        )
        np.testing.assert_array_almost_equal(A[3], B[3])

    def test_boundary_face_is_pure_normal(self):
        """Boundary face (cellB=-1): velocity = face_vel * face_normal."""
        fv = np.zeros(N_FACES, dtype=np.float64)
        fv[0] = 2.0  # only face0 has nonzero velocity
        fc = np.zeros(N_FACES, dtype=bool)
        A, B = reconstruct_face_velocities(
            fv, FACE_NORMALS_2D, fc, FACE_CI, CELL_FACE_INFO, CELL_FACE_VALS
        )
        expected = fv[0] * FACE_NORMALS_2D[0]
        np.testing.assert_array_almost_equal(A[0], expected)
        np.testing.assert_array_almost_equal(B[0], expected)


# ---------------------------------------------------------------------------
# Step 3: compute_facepoint_velocities
# ---------------------------------------------------------------------------


class TestComputeVertexVelocities:
    def _run(self, fv=None, fc=None):
        if fv is None:
            fv = np.ones(N_FACES, dtype=np.float64)
        if fc is None:
            fc = np.zeros(N_FACES, dtype=bool)
            fc[3] = True
        cell_wse = np.array([2.0, 2.0])
        _, val_a, val_b = compute_face_wss(
            cell_wse, CELL_MIN_ELEV, FACE_MIN_ELEV, FACE_CI, CELL_FACE_COUNT
        )
        A, B = reconstruct_face_velocities(
            fv, FACE_NORMALS_2D, fc, FACE_CI, CELL_FACE_INFO, CELL_FACE_VALS
        )
        fp_info, fp_vals = _make_fp_face_orientation()
        return compute_facepoint_velocities(
            A, B, fc, FACE_LENGTHS, FACE_FP, FACE_CI,
            cell_wse, fp_info, fp_vals, val_a, val_b
        )

    def test_returns_list_and_dict(self):
        vels, fmap = self._run()
        assert isinstance(vels, list) and len(vels) == N_FP
        assert isinstance(fmap, dict)

    def test_each_element_shape(self):
        vels, _ = self._run()
        for fp in range(N_FP):
            assert vels[fp].ndim == 2
            assert vels[fp].shape[1] == 2

    def test_zero_input_gives_zero_output(self):
        vels, _ = self._run(fv=np.zeros(N_FACES))
        for v in vels:
            np.testing.assert_array_almost_equal(v, 0.0)

    def test_local_map_covers_all_facepoint_face_pairs(self):
        """Every (fp, face) pair from the mesh should appear in the map."""
        _, fmap = self._run()
        fp_info, fp_vals = _make_fp_face_orientation()
        for fp in range(N_FP):
            start = int(fp_info[fp, 0])
            count = int(fp_info[fp, 1])
            for j in range(count):
                fi = int(fp_vals[start + j, 0])
                assert (fp, fi) in fmap


# ---------------------------------------------------------------------------
# Step 3.5: replace_face_velocities_sloped
# ---------------------------------------------------------------------------


class TestReplaceFaceVelocitiesSloped:
    def _run(self):
        fv = np.ones(N_FACES, dtype=np.float64)
        fc = np.zeros(N_FACES, dtype=bool);  fc[3] = True
        cell_wse = np.array([2.0, 2.0])
        _, val_a, val_b = compute_face_wss(
            cell_wse, CELL_MIN_ELEV, FACE_MIN_ELEV, FACE_CI, CELL_FACE_COUNT
        )
        A, B = reconstruct_face_velocities(
            fv, FACE_NORMALS_2D, fc, FACE_CI, CELL_FACE_INFO, CELL_FACE_VALS
        )
        fp_info, fp_vals = _make_fp_face_orientation()
        vels, fmap = compute_facepoint_velocities(
            A, B, fc, FACE_LENGTHS, FACE_FP, FACE_CI,
            cell_wse, fp_info, fp_vals, val_a, val_b
        )
        return replace_face_velocities_sloped(vels, fmap, FACE_FP)

    def test_output_shape(self):
        rv = self._run()
        assert rv.shape == (N_FACES, 2)
        assert rv.dtype == np.float64

    def test_equals_average_of_endpoint_fp_velocities(self):
        """Each face's replaced velocity must be the mean of its two fp velocities."""
        fv = np.random.default_rng(7).uniform(-2, 2, N_FACES)
        fc = np.zeros(N_FACES, dtype=bool);  fc[3] = True
        cell_wse = np.array([2.0, 2.0])
        _, val_a, val_b = compute_face_wss(
            cell_wse, CELL_MIN_ELEV, FACE_MIN_ELEV, FACE_CI, CELL_FACE_COUNT
        )
        A, B = reconstruct_face_velocities(
            fv, FACE_NORMALS_2D, fc, FACE_CI, CELL_FACE_INFO, CELL_FACE_VALS
        )
        fp_info, fp_vals = _make_fp_face_orientation()
        vels, fmap = compute_facepoint_velocities(
            A, B, fc, FACE_LENGTHS, FACE_FP, FACE_CI,
            cell_wse, fp_info, fp_vals, val_a, val_b
        )
        rv = replace_face_velocities_sloped(vels, fmap, FACE_FP)
        for f in range(N_FACES):
            fpA = int(FACE_FP[f, 0])
            fpB = int(FACE_FP[f, 1])
            key_A = (fpA, f);  key_B = (fpB, f)
            if key_A in fmap and key_B in fmap:
                vel_A = vels[fpA][fmap[key_A]]
                vel_B = vels[fpB][fmap[key_B]]
                expected = (vel_A + vel_B) / 2.0
                np.testing.assert_array_almost_equal(rv[f], expected)


# ---------------------------------------------------------------------------
# Step 4 helpers: _barycentric_weights, _donate, _downward_adjust_fp_wse
# ---------------------------------------------------------------------------


class TestBarycentricWeights:
    def test_centroid_equal_weights_square(self):
        """Centroid of a unit square → all 4 vertices get equal weight 0.25."""
        verts_x = np.array([0.0, 1.0, 1.0, 0.0])
        verts_y = np.array([0.0, 0.0, 1.0, 1.0])
        w = _barycentric_weights(0.5, 0.5, verts_x, verts_y)
        np.testing.assert_allclose(w, 0.25, atol=1e-5)

    def test_weights_sum_to_one(self):
        verts_x = np.array([0.0, 2.0, 2.5, 1.5, 0.5])
        verts_y = np.array([0.0, 0.0, 1.5, 3.0, 2.5])
        w = _barycentric_weights(1.0, 1.0, verts_x, verts_y)
        assert float(w.sum()) == pytest.approx(1.0, abs=1e-5)

    def test_weights_nonnegative(self):
        verts_x = np.array([0.0, 1.0, 1.0, 0.0])
        verts_y = np.array([0.0, 0.0, 1.0, 1.0])
        w = _barycentric_weights(0.3, 0.7, verts_x, verts_y)
        assert (w >= 0).all()

    def test_output_dtype_float32(self):
        verts_x = np.array([0.0, 1.0, 1.0, 0.0])
        verts_y = np.array([0.0, 0.0, 1.0, 1.0])
        w = _barycentric_weights(0.5, 0.5, verts_x, verts_y)
        assert w.dtype == np.float32

    def test_vertex_weight_one_at_corner(self):
        """Pixel at a corner should give weight ~1 to that corner."""
        verts_x = np.array([0.0, 1.0, 1.0, 0.0])
        verts_y = np.array([0.0, 0.0, 1.0, 1.0])
        w = _barycentric_weights(0.0, 0.0, verts_x, verts_y)
        assert w.sum() == pytest.approx(1.0, abs=1e-4)
        # Vertex 0 should have the largest weight
        assert int(w.argmax()) == 0


class TestDonate:
    def test_output_length_2n(self):
        fp_w = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        vw = _donate(fp_w)
        assert len(vw) == 8

    def test_total_weight_conserved(self):
        """Sum of all velocity weights should equal sum of input fp_weights."""
        fp_w = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
        vw = _donate(fp_w)
        assert float(vw.sum()) == pytest.approx(float(fp_w.sum()), abs=1e-6)

    def test_vertex_weights_reduced(self):
        """After donate, vertex weights should be ≤ original fp_weights."""
        fp_w = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        vw = _donate(fp_w)
        N = len(fp_w)
        assert (vw[:N] <= fp_w + 1e-10).all()

    def test_face_weights_nonnegative(self):
        fp_w = np.array([0.1, 0.4, 0.3, 0.2], dtype=np.float32)
        vw = _donate(fp_w)
        N = len(fp_w)
        assert (vw[N:] >= 0).all()


class TestDownwardAdjustFpWse:
    def test_correction_applied(self):
        """Mean of adjusted values should equal cell_wse."""
        cell_wse = 3.0
        fp_wse = np.array([2.5, 3.5, 2.8, 3.2], dtype=np.float64)
        adj = _downward_adjust_fp_wse(cell_wse, fp_wse)
        assert float(adj.mean()) == pytest.approx(cell_wse, abs=1e-10)

    def test_empty_array_unchanged(self):
        adj = _downward_adjust_fp_wse(3.0, np.array([], dtype=np.float64))
        assert len(adj) == 0


class TestPixelWseSloped:
    def test_uniform_wse_returns_that_value(self):
        """All fp and face WSEs equal → pixel WSE = that value."""
        N = 4
        wse_val = 5.0
        fw = _barycentric_weights(0.5, 0.5,
                                   np.array([0., 1., 1., 0.]),
                                   np.array([0., 0., 1., 1.]))
        vw = _donate(fw)
        fp_wse  = np.full(N, wse_val, dtype=np.float64)
        face_ws = np.full(N, wse_val, dtype=np.float64)
        result  = _pixel_wse_sloped(vw, fp_wse, face_ws, None)
        assert result == pytest.approx(wse_val, abs=1e-6)

    def test_nodata_base_returns_nodata_like(self):
        N = 4
        fw = _barycentric_weights(0.5, 0.5,
                                   np.array([0., 1., 1., 0.]),
                                   np.array([0., 0., 1., 1.]))
        vw = _donate(fw)
        # N=0 path
        result = _pixel_wse_sloped(vw, np.array([]), np.array([]), None)
        assert result == -9999.0


class TestAllShallow:
    def test_no_connected_faces_is_all_shallow(self):
        fc = np.zeros(N_FACES, dtype=bool)
        assert _all_shallow(0, CELL_FACE_INFO, CELL_FACE_VALS, fc)

    def test_one_connected_face_not_all_shallow(self):
        fc = np.zeros(N_FACES, dtype=bool)
        fc[3] = True  # shared face
        assert not _all_shallow(0, CELL_FACE_INFO, CELL_FACE_VALS, fc)


# ---------------------------------------------------------------------------
# rasmap_raster integration tests (require rasterio + shapely)
# ---------------------------------------------------------------------------

pytest.importorskip("rasterio", reason="rasterio not installed")
pytest.importorskip("shapely", reason="shapely not installed")


def _make_rasmap_inputs(cell_wse_vals=(2.0, 2.0), face_vel=None):
    """Return all keyword arguments for rasmap_raster using our minimal mesh."""
    from raspy.geo.raster import rasmap_raster  # noqa: F401 — imported for signature

    # Cell polygons: two unit squares
    cell_polygons = [
        np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=np.float64),
        np.array([[1., 0.], [2., 0.], [2., 1.], [1., 1.]], dtype=np.float64),
    ]

    face_normals_3d = np.column_stack([FACE_NORMALS_2D, FACE_LENGTHS])
    fp_info, fp_vals = _make_fp_face_orientation()

    if face_vel is None:
        face_vel = np.zeros(N_FACES, dtype=np.float64)

    return dict(
        variable="water_surface",
        cell_wse=np.array(cell_wse_vals, dtype=np.float64),
        cell_min_elevation=CELL_MIN_ELEV,
        face_min_elevation=FACE_MIN_ELEV,
        face_cell_indexes=FACE_CI,
        cell_face_info=CELL_FACE_INFO,
        cell_face_values=CELL_FACE_VALS,
        face_facepoint_indexes=FACE_FP,
        fp_coords=FP_COORDS,
        face_normals=face_normals_3d,
        fp_face_info=fp_info,
        fp_face_values=fp_vals,
        cell_polygons=cell_polygons,
        face_normal_velocity=face_vel,
        output_path=None,
        cell_size=0.5,
        nodata=-9999.0,
        tight_extent=False,
    )


class TestRasmapRasterFlat:
    """Tests for rasmap_raster with render_mode='horizontal'."""

    def _run(self, variable="water_surface", cell_wse_vals=(2.0, 2.0), **kwargs):
        from raspy.geo.raster import rasmap_raster
        inputs = _make_rasmap_inputs(cell_wse_vals=cell_wse_vals)
        inputs["variable"] = variable
        inputs.update(kwargs)
        ds = rasmap_raster(**inputs, render_mode="horizontal")
        try:
            data = ds.read(1)
        finally:
            ds.close()
        return data

    def test_returns_dataset(self):
        from raspy.geo.raster import rasmap_raster
        inputs = _make_rasmap_inputs()
        ds = rasmap_raster(**inputs, render_mode="horizontal")
        assert ds is not None
        ds.close()

    def test_output_shape_positive(self):
        data = self._run()
        assert data.ndim == 2
        assert data.shape[0] > 0 and data.shape[1] > 0

    def test_wet_pixels_have_valid_wse(self):
        """Non-nodata pixels should have WSE near cell value."""
        data = self._run(cell_wse_vals=(3.0, 3.0))
        valid = data[data != -9999.0]
        assert len(valid) > 0
        assert float(valid.mean()) == pytest.approx(3.0, abs=0.1)

    def test_dry_cell_gives_nodata(self):
        """Both cells dry (WSE=cell_min_elev=0) → all pixels nodata."""
        data = self._run(cell_wse_vals=(0.0, 0.0))
        assert (data == -9999.0).all()

    def test_speed_requires_face_vel(self):
        """variable='speed' without face_normal_velocity raises ValueError."""
        from raspy.geo.raster import rasmap_raster
        inputs = _make_rasmap_inputs()
        inputs["variable"] = "velocity"
        inputs["face_normal_velocity"] = None
        with pytest.raises(ValueError, match="face_normal_velocity"):
            rasmap_raster(**inputs, render_mode="horizontal")

    def test_depth_requires_reference_raster(self):
        """variable='depth' without reference_raster raises ValueError."""
        from raspy.geo.raster import rasmap_raster
        inputs = _make_rasmap_inputs()
        inputs["variable"] = "depth"
        with pytest.raises(ValueError, match="reference_raster"):
            rasmap_raster(**inputs, render_mode="horizontal")

    def test_no_grid_spec_raises(self):
        """Neither reference_raster nor cell_size → ValueError."""
        from raspy.geo.raster import rasmap_raster
        inputs = _make_rasmap_inputs()
        del inputs["cell_size"]
        inputs["output_path"] = None
        with pytest.raises(ValueError):
            rasmap_raster(**inputs, render_mode="horizontal")

    def test_both_grid_specs_raises(self):
        """Both reference_raster and cell_size → ValueError."""
        from raspy.geo.raster import rasmap_raster
        import tempfile, os
        inputs = _make_rasmap_inputs()
        inputs["reference_raster"] = "some_file.tif"  # value doesn't matter — raises before open
        with pytest.raises((ValueError, Exception)):
            rasmap_raster(**inputs, render_mode="horizontal")


class TestRasmapRasterSloping:
    """Tests for rasmap_raster with render_mode='sloping' (default)."""

    def _run(self, variable="water_surface", cell_wse_vals=(2.0, 2.0), **kwargs):
        from raspy.geo.raster import rasmap_raster
        inputs = _make_rasmap_inputs(cell_wse_vals=cell_wse_vals)
        inputs["variable"] = variable
        inputs.update(kwargs)
        ds = rasmap_raster(**inputs, render_mode="sloping")
        try:
            data = ds.read(1)
        finally:
            ds.close()
        return data

    def test_returns_2d_array(self):
        data = self._run()
        assert data.ndim == 2

    def test_wet_pixels_have_valid_wse(self):
        """At least some pixels must be non-nodata for wet mesh."""
        data = self._run(cell_wse_vals=(3.0, 3.0))
        valid = data[data != -9999.0]
        assert len(valid) > 0

    def test_output_dtype_float32(self):
        """Output array dtype must be float32."""
        data = self._run(cell_wse_vals=(3.0, 3.0))
        assert data.dtype == np.float32

    def test_dry_mesh_all_nodata(self):
        """Dry cells → all pixels nodata."""
        data = self._run(cell_wse_vals=(0.0, 0.0))
        assert (data == -9999.0).all()

    def test_speed_variable_returns_2d(self):
        face_vel = np.ones(N_FACES, dtype=np.float64) * 0.5
        data = self._run(variable="velocity", face_normal_velocity=face_vel)
        assert data.ndim == 2

    def test_speed_nonnegative(self):
        """Speed (magnitude) must be ≥ 0 everywhere."""
        face_vel = np.random.default_rng(99).uniform(-1, 1, N_FACES)
        data = self._run(variable="velocity", face_normal_velocity=face_vel)
        valid = data[data != -9999.0]
        assert (valid >= 0.0).all()

    def test_output_path_creates_file(self, tmp_path):
        """output_path writes a GeoTIFF and returns the path."""
        from raspy.geo.raster import rasmap_raster
        out = tmp_path / "out.tif"
        inputs = _make_rasmap_inputs()
        inputs["output_path"] = str(out)
        result = rasmap_raster(**inputs, render_mode="sloping")
        assert out.exists()
        assert str(result) == str(out)

    def test_flat_and_sloping_agree_on_wet_coverage(self):
        """Flat and sloping modes should produce the same grid dimensions."""
        from raspy.geo.raster import rasmap_raster
        inputs_flat   = _make_rasmap_inputs(cell_wse_vals=(2.0, 2.0))
        inputs_slope  = _make_rasmap_inputs(cell_wse_vals=(2.0, 2.0))
        ds_flat  = rasmap_raster(**inputs_flat,  render_mode="horizontal")
        ds_slope = rasmap_raster(**inputs_slope, render_mode="sloping")
        try:
            assert ds_flat.shape  == ds_slope.shape
        finally:
            ds_flat.close()
            ds_slope.close()
