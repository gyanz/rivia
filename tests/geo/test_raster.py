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
            pts, vals, tmp_path / "snapped.tif", transform=ref_transform
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
        with pytest.raises(ValueError, match="reference_raster or transform"):
            points_to_raster(
                pts,
                vals,
                tmp_path / "out.tif",
                transform=from_origin(0, 100, 5, 5),
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
