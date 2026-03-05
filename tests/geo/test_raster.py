"""Tests for raspy.geo.raster.points_to_raster."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scattered_scalar(n: int = 50, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return random (points, values) for a scalar field."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(0, 100, (n, 2))
    values = rng.uniform(0, 10, n)
    return points, values


def _scattered_vector(n: int = 50, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return random (points, values) for a vector field (Vx, Vy)."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(0, 100, (n, 2))
    values = rng.uniform(-2, 2, (n, 2))
    return points, values


# ---------------------------------------------------------------------------
# Import guard (skip all tests if rasterio/scipy not installed)
# ---------------------------------------------------------------------------

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")
scipy = pytest.importorskip("scipy", reason="scipy not installed")


# ---------------------------------------------------------------------------
# Scalar raster
# ---------------------------------------------------------------------------


class TestScalarRaster:
    def test_creates_file(self, tmp_path):
        from raspy.geo.raster import points_to_raster

        pts, vals = _scattered_scalar()
        out = points_to_raster(pts, vals, tmp_path / "out.tif", cell_size=5.0)
        assert out.exists()

    def test_returns_path_object(self, tmp_path):
        from raspy.geo.raster import points_to_raster

        pts, vals = _scattered_scalar()
        result = points_to_raster(pts, vals, tmp_path / "out.tif", cell_size=5.0)
        assert isinstance(result, Path)

    def test_single_band(self, tmp_path):
        from raspy.geo.raster import points_to_raster

        pts, vals = _scattered_scalar()
        out = points_to_raster(pts, vals, tmp_path / "out.tif", cell_size=5.0)
        with rasterio.open(out) as src:
            assert src.count == 1

    def test_cell_size_respected(self, tmp_path):
        from raspy.geo.raster import points_to_raster

        pts, vals = _scattered_scalar()
        cell_size = 10.0
        out = points_to_raster(pts, vals, tmp_path / "out.tif", cell_size=cell_size)
        with rasterio.open(out) as src:
            assert src.transform.a == pytest.approx(cell_size)
            assert abs(src.transform.e) == pytest.approx(cell_size)

    def test_nodata_written(self, tmp_path):
        from raspy.geo.raster import points_to_raster

        pts, vals = _scattered_scalar()
        nodata = -999.0
        out = points_to_raster(
            pts, vals, tmp_path / "out.tif", cell_size=5.0, nodata=nodata
        )
        with rasterio.open(out) as src:
            assert src.nodata == nodata

    def test_crs_set(self, tmp_path):
        from raspy.geo.raster import points_to_raster

        pts, vals = _scattered_scalar()
        out = points_to_raster(
            pts, vals, tmp_path / "out.tif", cell_size=5.0, crs="EPSG:4326"
        )
        with rasterio.open(out) as src:
            assert src.crs is not None
            assert src.crs.to_epsg() == 4326

    def test_no_crs_writes_without_error(self, tmp_path):
        from raspy.geo.raster import points_to_raster

        pts, vals = _scattered_scalar()
        out = points_to_raster(pts, vals, tmp_path / "out.tif", cell_size=5.0)
        with rasterio.open(out) as src:
            assert src.crs is None

    def test_parent_dir_created(self, tmp_path):
        from raspy.geo.raster import points_to_raster

        pts, vals = _scattered_scalar()
        out_path = tmp_path / "subdir" / "nested" / "out.tif"
        assert not out_path.parent.exists()
        out = points_to_raster(pts, vals, out_path, cell_size=5.0)
        assert out.exists()


# ---------------------------------------------------------------------------
# Vector raster
# ---------------------------------------------------------------------------


class TestVectorRaster:
    def test_four_bands(self, tmp_path):
        from raspy.geo.raster import points_to_raster

        pts, vals = _scattered_vector()
        out = points_to_raster(pts, vals, tmp_path / "vel.tif", cell_size=5.0)
        with rasterio.open(out) as src:
            assert src.count == 4

    def test_band_names_stored(self, tmp_path):
        from raspy.geo.raster import points_to_raster

        pts, vals = _scattered_vector()
        out = points_to_raster(pts, vals, tmp_path / "vel.tif", cell_size=5.0)
        with rasterio.open(out) as src:
            tags_1 = src.tags(1)
            assert "name" in tags_1
            assert tags_1["name"] == "Vx"


# ---------------------------------------------------------------------------
# Grid alignment — transform snapping
# ---------------------------------------------------------------------------


class TestGridAlignment:
    def test_transform_snaps_origin(self, tmp_path):
        from rasterio.transform import from_origin
        from raspy.geo.raster import points_to_raster

        ref_transform = from_origin(0.0, 200.0, 10.0, 10.0)
        pts, vals = _scattered_scalar()  # x in [0,100], y in [0,100]
        out = points_to_raster(
            pts, vals, tmp_path / "snapped.tif", reference_transform=ref_transform
        )
        with rasterio.open(out) as src:
            # pixel size must equal the reference
            assert src.transform.a == pytest.approx(10.0)
            assert abs(src.transform.e) == pytest.approx(10.0)
            # origin must be an exact multiple of 10
            assert src.transform.c % 10.0 == pytest.approx(0.0)

    def test_reference_raster_inherits_crs(self, tmp_path):
        from rasterio.transform import from_origin
        from raspy.geo.raster import points_to_raster

        # Write a reference raster with a known CRS
        ref_path = tmp_path / "ref.tif"
        ref_transform = from_origin(0.0, 200.0, 10.0, 10.0)
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": 20,
            "height": 20,
            "count": 1,
            "crs": rasterio.crs.CRS.from_epsg(32610),
            "transform": ref_transform,
        }
        with rasterio.open(ref_path, "w", **profile) as dst:
            dst.write(np.ones((20, 20), dtype="f4"), 1)

        pts, vals = _scattered_scalar()
        out = points_to_raster(
            pts, vals, tmp_path / "aligned.tif", reference_raster=ref_path
        )
        with rasterio.open(out) as src:
            assert src.crs.to_epsg() == 32610
            assert src.transform.a == pytest.approx(10.0)

    def test_reference_raster_crs_overridden(self, tmp_path):
        from rasterio.transform import from_origin
        from raspy.geo.raster import points_to_raster

        ref_path = tmp_path / "ref.tif"
        ref_transform = from_origin(0.0, 200.0, 10.0, 10.0)
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": 20,
            "height": 20,
            "count": 1,
            "crs": rasterio.crs.CRS.from_epsg(32610),
            "transform": ref_transform,
        }
        with rasterio.open(ref_path, "w", **profile) as dst:
            dst.write(np.ones((20, 20), dtype="f4"), 1)

        pts, vals = _scattered_scalar()
        out = points_to_raster(
            pts,
            vals,
            tmp_path / "override.tif",
            reference_raster=ref_path,
            crs="EPSG:4326",
        )
        with rasterio.open(out) as src:
            assert src.crs.to_epsg() == 4326

    def test_transform_and_reference_raster_mutual_exclusion(self, tmp_path):
        from rasterio.transform import from_origin
        from raspy.geo.raster import points_to_raster

        pts, vals = _scattered_scalar()
        with pytest.raises(ValueError, match="reference_raster or reference_transform"):
            points_to_raster(
                pts,
                vals,
                tmp_path / "out.tif",
                reference_transform=from_origin(0, 100, 5, 5),
                reference_raster=tmp_path / "nonexistent.tif",
            )


# ---------------------------------------------------------------------------
# min_value masking
# ---------------------------------------------------------------------------


class TestMinValueMask:
    def test_raises_when_too_few_points_remain(self, tmp_path):
        from raspy.geo.raster import points_to_raster

        pts = np.array([[0.0, 0.0], [10.0, 0.0]])  # only 2 points
        vals = np.array([0.0, 0.0])
        with pytest.raises(ValueError, match="Fewer than 3"):
            points_to_raster(
                pts, vals, tmp_path / "out.tif", cell_size=1.0, min_value=1.0
            )

    def test_dry_cells_excluded(self, tmp_path):
        from raspy.geo.raster import points_to_raster

        # 50 points, half with value < 1.0
        rng = np.random.default_rng(1)
        pts = rng.uniform(0, 100, (50, 2))
        vals = np.concatenate([rng.uniform(0, 0.5, 25), rng.uniform(2.0, 5.0, 25)])
        # Should not raise; dry cells are simply excluded
        out = points_to_raster(
            pts, vals, tmp_path / "masked.tif", cell_size=10.0, min_value=1.0
        )
        assert out.exists()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_invalid_cell_size_raises(self, tmp_path):
        from raspy.geo.raster import points_to_raster

        pts, vals = _scattered_scalar()
        with pytest.raises(ValueError, match="cell_size"):
            points_to_raster(pts, vals, tmp_path / "out.tif", cell_size=0)

    def test_no_grid_spec_and_no_cell_size_raises(self, tmp_path):
        from raspy.geo.raster import points_to_raster

        pts, vals = _scattered_scalar()
        # No cell_size, transform, or reference_raster
        # cell_size=None is passed explicitly to trigger the error path
        with pytest.raises(ValueError, match="cell_size"):
            points_to_raster(pts, vals, tmp_path / "out.tif", cell_size=None)


# ---------------------------------------------------------------------------
# mesh_to_raster
# ---------------------------------------------------------------------------

def _minimal_mesh():
    """Return a tiny 2-cell mesh suitable for mesh_to_raster tests.

    Layout (not to scale):
        cell 0 centred at (5, 5),  cell 1 centred at (15, 5)
        Three facepoints: (0,0), (10,0), (10,10), (0,10), (20,0), (20,10)
        Two cells share the face between fp1=(10,0) and fp2=(10,10).
    """
    cell_centers = np.array([[5.0, 5.0], [15.0, 5.0]])
    facepoint_coords = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 10.0],
        [0.0, 10.0],
        [20.0, 0.0],
        [20.0, 10.0],
    ])
    # Each face: [fp_start, fp_end]
    face_facepoint_indexes = np.array([
        [0, 1],  # face 0: bottom of cell 0
        [1, 2],  # face 1: shared between cell 0 and cell 1
        [2, 3],  # face 2: top of cell 0
        [3, 0],  # face 3: left of cell 0
        [1, 4],  # face 4: bottom of cell 1
        [4, 5],  # face 5: right of cell 1
        [5, 2],  # face 6: top of cell 1
    ])
    # [left_cell, right_cell]; -1 = boundary
    face_cell_indexes = np.array([
        [0, -1],
        [0, 1],
        [0, -1],
        [0, -1],
        [1, -1],
        [1, -1],
        [1, -1],
    ])
    # cell_face_info: [start, count] into cell_face_values
    cell_face_info = np.array([
        [0, 4],  # cell 0 uses entries 0..3
        [4, 3],  # cell 1 uses entries 4..6
    ])
    # cell_face_values: [face_idx, orientation]
    cell_face_values = np.array([
        [0, 1], [1, 1], [2, 1], [3, 1],   # cell 0
        [1, -1], [4, 1], [5, 1], [6, 1],  # cell 1 (face 1 is inward)
    ])
    return dict(
        cell_centers=cell_centers,
        facepoint_coordinates=facepoint_coords,
        face_facepoint_indexes=face_facepoint_indexes,
        face_cell_indexes=face_cell_indexes,
        cell_face_info=cell_face_info,
        cell_face_values=cell_face_values,
    )


class TestMeshToRaster:
    def test_rejects_vector_cell_values(self, tmp_path):
        """mesh_to_raster must raise when given a (n, 2) vector array."""
        from raspy.geo.raster import mesh_to_raster

        mesh = _minimal_mesh()
        vector_values = np.ones((2, 2))  # (n_cells, 2) — invalid
        with pytest.raises((ValueError, IndexError)):
            mesh_to_raster(
                **mesh,
                cell_values=vector_values,
                output_path=tmp_path / "out.tif",
                cell_size=2.0,
            )

    def test_scalar_output_one_band(self, tmp_path):
        """Scalar cell_values produce a single-band raster."""
        from raspy.geo.raster import mesh_to_raster

        mesh = _minimal_mesh()
        cell_values = np.array([10.0, 12.0])
        out = mesh_to_raster(
            **mesh,
            cell_values=cell_values,
            output_path=tmp_path / "wse.tif",
            cell_size=2.0,
        )
        with rasterio.open(out) as src:
            assert src.count == 1

    def test_min_above_ref_masks_shallow_pixels(self, tmp_path):
        """min_above_ref with a reference DEM masks pixels shallower than the threshold."""
        from rasterio.transform import from_origin
        from raspy.geo.raster import mesh_to_raster

        mesh = _minimal_mesh()
        # DEM elevation = 8.0 everywhere; WSE = 9.0 → depth = 1.0
        dem_path = tmp_path / "dem.tif"
        dem_transform = from_origin(0.0, 10.0, 2.0, 2.0)
        dem_profile = {
            "driver": "GTiff", "dtype": "float32",
            "width": 10, "height": 5, "count": 1,
            "transform": dem_transform, "nodata": None,
        }
        dem_data = np.full((5, 10), 8.0, dtype="f4")
        with rasterio.open(dem_path, "w", **dem_profile) as dst:
            dst.write(dem_data, 1)

        cell_wse = np.array([9.0, 9.0])  # depth = 1.0 everywhere

        # min_above_ref=0.5: depth 1.0 >= 0.5 → pixels should remain
        ds_keep = mesh_to_raster(
            **mesh,
            cell_values=cell_wse,
            output_path=None,
            reference_raster=dem_path,
            snap_to_reference_extent=True,
            min_above_ref=0.5,
        )
        data_keep = ds_keep.read(1)
        ds_keep.close()
        wet_keep = np.isfinite(data_keep) & (data_keep != -9999.0)
        assert wet_keep.any(), "Expected wet pixels when depth > min_above_ref"

        # min_above_ref=2.0: depth 1.0 < 2.0 → all pixels should be masked
        ds_mask = mesh_to_raster(
            **mesh,
            cell_values=cell_wse,
            output_path=None,
            reference_raster=dem_path,
            snap_to_reference_extent=True,
            min_above_ref=2.0,
        )
        data_mask = ds_mask.read(1)
        ds_mask.close()
        wet_mask = np.isfinite(data_mask) & (data_mask != -9999.0)
        assert not wet_mask.any(), "Expected all pixels masked when depth < min_above_ref"

    def test_below_ground_masked_by_default(self, tmp_path):
        """Without min_above_ref, pixels where WSE <= DEM are masked."""
        from rasterio.transform import from_origin
        from raspy.geo.raster import mesh_to_raster

        mesh = _minimal_mesh()
        dem_path = tmp_path / "dem.tif"
        dem_transform = from_origin(0.0, 10.0, 2.0, 2.0)
        dem_profile = {
            "driver": "GTiff", "dtype": "float32",
            "width": 10, "height": 5, "count": 1,
            "transform": dem_transform, "nodata": None,
        }
        # DEM = 10.0; WSE = 9.0 → WSE below ground → should be masked
        dem_data = np.full((5, 10), 10.0, dtype="f4")
        with rasterio.open(dem_path, "w", **dem_profile) as dst:
            dst.write(dem_data, 1)

        cell_wse = np.array([9.0, 9.0])
        ds = mesh_to_raster(
            **mesh,
            cell_values=cell_wse,
            output_path=None,
            reference_raster=dem_path,
            snap_to_reference_extent=True,
        )
        data = ds.read(1)
        ds.close()
        wet = np.isfinite(data) & (data != -9999.0)
        assert not wet.any(), "Expected all pixels masked when WSE <= DEM"


# ---------------------------------------------------------------------------
# Helpers for velocity-interp tests
# ---------------------------------------------------------------------------


def _minimal_vel_mesh() -> dict:
    """Extend _minimal_mesh() with uniform-flow velocity fields.

    Uniform flow u=(1, 0) m/s so face normal velocities are u·n̂ and
    both cells have WLS velocity [1.0, 0.0].
    """
    mesh = _minimal_mesh()
    face_normals = np.array([
        [0.0, -1.0, 10.0],   # face 0: south boundary of cell 0
        [1.0,  0.0, 10.0],   # face 1: shared face, normal east
        [0.0,  1.0, 10.0],   # face 2: north boundary of cell 0
        [-1.0, 0.0, 10.0],   # face 3: west boundary of cell 0
        [0.0, -1.0, 10.0],   # face 4: south boundary of cell 1
        [1.0,  0.0, 10.0],   # face 5: east boundary of cell 1
        [0.0,  1.0, 10.0],   # face 6: north boundary of cell 1
    ])
    # vn = u · n̂  for u=(1,0)
    face_vel = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0])
    cell_wse = np.array([5.0, 5.0])
    cell_velocity = np.array([[1.0, 0.0], [1.0, 0.0]])
    return dict(**mesh, face_normals=face_normals, face_vel=face_vel,
                cell_wse=cell_wse, cell_velocity=cell_velocity)


# ---------------------------------------------------------------------------
# _compute_face_midpoints
# ---------------------------------------------------------------------------


class TestComputeFaceMidpoints:
    def test_basic(self):
        from raspy.geo.raster import _compute_face_midpoints

        fp_coords = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
        fp_idx = np.array([[0, 1], [1, 2], [2, 0]])
        midpoints = _compute_face_midpoints(fp_coords, fp_idx)
        expected = np.array([[5.0, 0.0], [10.0, 5.0], [5.0, 5.0]])
        np.testing.assert_allclose(midpoints, expected)

    def test_shape(self):
        from raspy.geo.raster import _compute_face_midpoints

        mesh = _minimal_mesh()
        mp = _compute_face_midpoints(
            mesh["facepoint_coordinates"],
            mesh["face_facepoint_indexes"],
        )
        assert mp.shape == (len(mesh["face_facepoint_indexes"]), 2)


# ---------------------------------------------------------------------------
# _interp_face_idw
# ---------------------------------------------------------------------------


class TestInterpFaceIdw:
    def _face_midpoints_and_vel2d(self):
        """Pre-compute face midpoints and 2D face velocities for uniform flow."""
        from raspy.geo.raster import _compute_face_midpoints, _compute_face_velocity_2d

        vm = _minimal_vel_mesh()
        face_midpoints = _compute_face_midpoints(
            vm["facepoint_coordinates"], vm["face_facepoint_indexes"]
        )
        dry_mask = np.isnan(vm["cell_wse"])
        face_vel_2d = _compute_face_velocity_2d(
            face_normals=np.asarray(vm["face_normals"]),
            face_vel=np.asarray(vm["face_vel"]),
            face_cell_indexes=np.asarray(vm["face_cell_indexes"]),
            cell_velocity=np.asarray(vm["cell_velocity"]),
            dry_mask=dry_mask,
            n_cells=2,
        )
        return vm, face_midpoints, face_vel_2d, dry_mask

    def test_output_shape(self):
        from raspy.geo.raster import _interp_face_idw

        vm, face_midpoints, face_vel_2d, dry_mask = self._face_midpoints_and_vel2d()
        px = np.array([5.0, 15.0, 3.0])
        py = np.array([5.0, 5.0, 7.0])
        cell_idx_hit = np.array([0, 1, 0])
        result = _interp_face_idw(
            px, py, cell_idx_hit, 2,
            vm["cell_face_info"], vm["cell_face_values"],
            face_midpoints, face_vel_2d, dry_mask,
        )
        assert result.shape == (3, 2)

    def test_uniform_flow_returns_unit_x(self):
        """For uniform u=(1,0) flow, IDW must return [1, 0] at every pixel."""
        from raspy.geo.raster import _interp_face_idw

        vm, face_midpoints, face_vel_2d, dry_mask = self._face_midpoints_and_vel2d()
        px = np.array([5.0, 5.0, 15.0, 15.0])
        py = np.array([2.0, 8.0, 2.0, 8.0])
        cell_idx_hit = np.array([0, 0, 1, 1])
        result = _interp_face_idw(
            px, py, cell_idx_hit, 2,
            vm["cell_face_info"], vm["cell_face_values"],
            face_midpoints, face_vel_2d, dry_mask,
        )
        np.testing.assert_allclose(result[:, 0], 1.0, atol=1e-12)
        np.testing.assert_allclose(result[:, 1], 0.0, atol=1e-12)

    def test_dry_cell_returns_zeros(self):
        from raspy.geo.raster import _interp_face_idw

        vm, face_midpoints, face_vel_2d, _ = self._face_midpoints_and_vel2d()
        dry_mask = np.array([True, False])   # cell 0 is dry
        px = np.array([5.0])
        py = np.array([5.0])
        cell_idx_hit = np.array([0])
        result = _interp_face_idw(
            px, py, cell_idx_hit, 2,
            vm["cell_face_info"], vm["cell_face_values"],
            face_midpoints, face_vel_2d, dry_mask,
        )
        np.testing.assert_array_equal(result, [[0.0, 0.0]])


# ---------------------------------------------------------------------------
# _interp_face_gradient
# ---------------------------------------------------------------------------


class TestInterpFaceGradient:
    def _face_midpoints_and_vel2d(self):
        from raspy.geo.raster import _compute_face_midpoints, _compute_face_velocity_2d

        vm = _minimal_vel_mesh()
        face_midpoints = _compute_face_midpoints(
            vm["facepoint_coordinates"], vm["face_facepoint_indexes"]
        )
        dry_mask = np.isnan(vm["cell_wse"])
        face_vel_2d = _compute_face_velocity_2d(
            face_normals=np.asarray(vm["face_normals"]),
            face_vel=np.asarray(vm["face_vel"]),
            face_cell_indexes=np.asarray(vm["face_cell_indexes"]),
            cell_velocity=np.asarray(vm["cell_velocity"]),
            dry_mask=dry_mask,
            n_cells=2,
        )
        return vm, face_midpoints, face_vel_2d, dry_mask

    def test_output_shape(self):
        from raspy.geo.raster import _interp_face_gradient

        vm, face_midpoints, face_vel_2d, dry_mask = self._face_midpoints_and_vel2d()
        px = np.array([5.0, 15.0])
        py = np.array([5.0, 5.0])
        cell_idx_hit = np.array([0, 1])
        result = _interp_face_gradient(
            px, py, cell_idx_hit, 2,
            vm["cell_centers"], vm["cell_face_info"], vm["cell_face_values"],
            face_midpoints, face_vel_2d, dry_mask, vm["cell_velocity"],
        )
        assert result.shape == (2, 2)

    def test_uniform_flow_exact(self):
        """Constant velocity field must be recovered exactly by the gradient fit."""
        from raspy.geo.raster import _interp_face_gradient

        vm, face_midpoints, face_vel_2d, dry_mask = self._face_midpoints_and_vel2d()
        px = np.array([3.0, 7.0, 5.0])
        py = np.array([3.0, 7.0, 5.0])
        cell_idx_hit = np.array([0, 0, 0])
        result = _interp_face_gradient(
            px, py, cell_idx_hit, 2,
            vm["cell_centers"], vm["cell_face_info"], vm["cell_face_values"],
            face_midpoints, face_vel_2d, dry_mask, vm["cell_velocity"],
        )
        # Constant field [1, 0] must be recovered with machine precision.
        np.testing.assert_allclose(result[:, 0], 1.0, atol=1e-12)
        np.testing.assert_allclose(result[:, 1], 0.0, atol=1e-12)

    def test_dry_cell_returns_zeros(self):
        from raspy.geo.raster import _interp_face_gradient

        vm, face_midpoints, face_vel_2d, _ = self._face_midpoints_and_vel2d()
        dry_mask = np.array([True, False])
        px = np.array([5.0])
        py = np.array([5.0])
        cell_idx_hit = np.array([0])
        result = _interp_face_gradient(
            px, py, cell_idx_hit, 2,
            vm["cell_centers"], vm["cell_face_info"], vm["cell_face_values"],
            face_midpoints, face_vel_2d, dry_mask, vm["cell_velocity"],
        )
        np.testing.assert_array_equal(result, [[0.0, 0.0]])


# ---------------------------------------------------------------------------
# _barycentric_weights
# ---------------------------------------------------------------------------


class TestBarycentricWeights:
    def test_centroid_gives_equal_weights(self):
        from raspy.geo.raster import _barycentric_weights

        # Centroid of triangle (0,0)-(6,0)-(0,6) is (2,2).
        w0, w1, w2 = _barycentric_weights(
            np.array([2.0]), np.array([2.0]),
            np.array([0.0]), np.array([0.0]),
            np.array([6.0]), np.array([0.0]),
            np.array([0.0]), np.array([6.0]),
        )
        np.testing.assert_allclose(w0, [1/3], atol=1e-12)
        np.testing.assert_allclose(w1, [1/3], atol=1e-12)
        np.testing.assert_allclose(w2, [1/3], atol=1e-12)

    def test_at_vertex_a_weight_is_one(self):
        from raspy.geo.raster import _barycentric_weights

        w0, w1, w2 = _barycentric_weights(
            np.array([0.0]), np.array([0.0]),
            np.array([0.0]), np.array([0.0]),
            np.array([6.0]), np.array([0.0]),
            np.array([0.0]), np.array([6.0]),
        )
        np.testing.assert_allclose(w0, [1.0], atol=1e-12)
        np.testing.assert_allclose(w1, [0.0], atol=1e-12)
        np.testing.assert_allclose(w2, [0.0], atol=1e-12)

    def test_weights_sum_to_one(self):
        from raspy.geo.raster import _barycentric_weights

        rng = np.random.default_rng(42)
        # Random points inside the unit triangle (0,0)-(1,0)-(0,1).
        u = rng.uniform(0, 1, 20)
        v = rng.uniform(0, 1 - u, 20)
        w0, w1, w2 = _barycentric_weights(
            u, v,
            np.zeros(20), np.zeros(20),
            np.ones(20),  np.zeros(20),
            np.zeros(20), np.ones(20),
        )
        np.testing.assert_allclose(w0 + w1 + w2, np.ones(20), atol=1e-12)

    def test_degenerate_returns_vertex_a(self):
        from raspy.geo.raster import _barycentric_weights

        # fp0 == fp1 → degenerate triangle.
        w0, w1, w2 = _barycentric_weights(
            np.array([1.0]), np.array([1.0]),
            np.array([0.0]), np.array([0.0]),
            np.array([2.0]), np.array([2.0]),
            np.array([2.0]), np.array([2.0]),  # B == C
        )
        np.testing.assert_array_equal(w0, [1.0])
        np.testing.assert_array_equal(w1, [0.0])
        np.testing.assert_array_equal(w2, [0.0])


# ---------------------------------------------------------------------------
# _compute_facepoint_velocities
# ---------------------------------------------------------------------------


class TestComputeFacepointVelocities:
    def test_uniform_flow_all_facepoints_unit_x(self):
        """For uniform u=(1,0), every facepoint should get velocity [1, 0]."""
        from raspy.geo.raster import _compute_facepoint_velocities, _compute_face_velocity_2d

        vm = _minimal_vel_mesh()
        dry_mask = np.zeros(2, dtype=bool)
        face_vel_2d = _compute_face_velocity_2d(
            face_normals=np.asarray(vm["face_normals"]),
            face_vel=np.asarray(vm["face_vel"]),
            face_cell_indexes=np.asarray(vm["face_cell_indexes"]),
            cell_velocity=np.asarray(vm["cell_velocity"]),
            dry_mask=dry_mask,
            n_cells=2,
        )
        n_fp = len(vm["facepoint_coordinates"])
        fp_vel = _compute_facepoint_velocities(
            np.asarray(vm["face_facepoint_indexes"]),
            face_vel_2d, n_fp,
            np.asarray(vm["face_cell_indexes"]),
            dry_mask, 2,
        )
        assert fp_vel.shape == (n_fp, 2)
        np.testing.assert_allclose(fp_vel[:, 0], 1.0, atol=1e-12)
        np.testing.assert_allclose(fp_vel[:, 1], 0.0, atol=1e-12)

    def test_all_dry_gives_zeros(self):
        from raspy.geo.raster import _compute_facepoint_velocities

        vm = _minimal_vel_mesh()
        n_fp = len(vm["facepoint_coordinates"])
        n_faces = len(vm["face_facepoint_indexes"])
        dry_mask = np.ones(2, dtype=bool)
        fp_vel = _compute_facepoint_velocities(
            np.asarray(vm["face_facepoint_indexes"]),
            np.ones((n_faces, 2)),
            n_fp,
            np.asarray(vm["face_cell_indexes"]),
            dry_mask, 2,
        )
        np.testing.assert_array_equal(fp_vel, np.zeros((n_fp, 2)))


# ---------------------------------------------------------------------------
# mesh_to_velocity_raster_interp — method parameter
# ---------------------------------------------------------------------------


matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")


class TestMeshToVelocityRasterInterpMethod:
    def test_invalid_method_raises(self):
        from raspy.geo.raster import mesh_to_velocity_raster_interp

        vm = _minimal_vel_mesh()
        with pytest.raises(ValueError, match="method must be"):
            mesh_to_velocity_raster_interp(
                **vm,
                output_path=None,
                cell_size=2.0,
                method="bad_method",
            )

    @pytest.mark.parametrize(
        "method",
        ["triangle_blend", "face_idw", "face_gradient", "facepoint_blend", "scatter_interp", "scatter_interp2"],
    )
    def test_returns_four_bands(self, method):
        from raspy.geo.raster import mesh_to_velocity_raster_interp

        vm = _minimal_vel_mesh()
        ds = mesh_to_velocity_raster_interp(
            **vm,
            output_path=None,
            cell_size=2.0,
            method=method,
        )
        assert ds.count == 4
        ds.close()

    @pytest.mark.parametrize(
        "method",
        ["triangle_blend", "face_idw", "face_gradient", "facepoint_blend", "scatter_interp", "scatter_interp2"],
    )
    def test_has_wet_pixels(self, method):
        from raspy.geo.raster import mesh_to_velocity_raster_interp

        vm = _minimal_vel_mesh()
        ds = mesh_to_velocity_raster_interp(
            **vm,
            output_path=None,
            cell_size=2.0,
            method=method,
        )
        speed = ds.read(3).astype(np.float64)
        ds.close()
        nodata = -9999.0
        wet = speed != nodata
        assert wet.any(), f"Expected wet pixels for method={method!r}"

    @pytest.mark.parametrize(
        "method", ["face_idw", "face_gradient", "facepoint_blend", "scatter_interp", "scatter_interp2"]
    )
    def test_uniform_flow_speed_approx_one(self, method):
        """Uniform u=(1,0) flow should give speed ≈ 1 at all wet pixels."""
        from raspy.geo.raster import mesh_to_velocity_raster_interp

        vm = _minimal_vel_mesh()
        ds = mesh_to_velocity_raster_interp(
            **vm,
            output_path=None,
            cell_size=2.0,
            method=method,
        )
        speed = ds.read(3).astype(np.float64)
        ds.close()
        wet = speed != -9999.0
        np.testing.assert_allclose(speed[wet], 1.0, atol=1e-6)

    def test_scatter_interp_vx_vy_direction(self):
        """scatter_interp: uniform u=(1,0) → Vx≈1, Vy≈0 at wet pixels."""
        from raspy.geo.raster import mesh_to_velocity_raster_interp

        vm = _minimal_vel_mesh()
        ds = mesh_to_velocity_raster_interp(
            **vm,
            output_path=None,
            cell_size=2.0,
            method="scatter_interp",
        )
        vx = ds.read(1).astype(np.float64)
        vy = ds.read(2).astype(np.float64)
        ds.close()
        nodata = -9999.0
        wet = vx != nodata
        assert wet.any()
        np.testing.assert_allclose(vx[wet], 1.0, atol=1e-6)
        np.testing.assert_allclose(vy[wet], 0.0, atol=1e-6)

    def test_scatter_interp2_vx_vy_direction(self):
        """scatter_interp2 (face midpoints only): uniform u=(1,0) → Vx≈1, Vy≈0."""
        from raspy.geo.raster import mesh_to_velocity_raster_interp

        vm = _minimal_vel_mesh()
        ds = mesh_to_velocity_raster_interp(
            **vm,
            output_path=None,
            cell_size=2.0,
            method="scatter_interp2",
        )
        vx = ds.read(1).astype(np.float64)
        vy = ds.read(2).astype(np.float64)
        ds.close()
        nodata = -9999.0
        wet = vx != nodata
        assert wet.any()
        np.testing.assert_allclose(vx[wet], 1.0, atol=1e-6)
        np.testing.assert_allclose(vy[wet], 0.0, atol=1e-6)
