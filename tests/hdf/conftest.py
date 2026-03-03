"""Shared fixtures for raspy.hdf tests.

The real HEC-RAS example HDF is used for integration tests.
Synthetic in-memory HDF fixtures are used for unit tests so they run
without the example files on disk.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Path to the real example file (skip integration tests if absent)
# ---------------------------------------------------------------------------

EXAMPLE_PLAN_HDF = Path(
    r"d:/Dropbox/repositories/Z/HEC-RAS Examples"
    r"/Tulloch HEC-RAS (WEST)/Tullochspillway.p01.hdf"
)

skip_if_no_example = pytest.mark.skipif(
    not EXAMPLE_PLAN_HDF.exists(),
    reason="Example HDF file not found on this machine",
)


# ---------------------------------------------------------------------------
# Synthetic HDF fixture helpers
# ---------------------------------------------------------------------------


def _make_synthetic_hdf(
    path: Path,
    n_cells: int = 10,
    n_faces: int = 20,
    n_timesteps: int = 5,
    area_name: str = "TestArea",
) -> None:
    """Write a minimal but structurally correct HEC-RAS plan HDF to *path*."""
    rng = np.random.default_rng(0)

    with h5py.File(path, "w") as f:
        # ── Geometry ──────────────────────────────────────────────────
        geom_root = f.create_group("Geometry/2D Flow Areas")

        # Attributes structured array (one row per area)
        attrs_dtype = np.dtype(
            [
                ("Name", "S16"),
                ("Cell Count", "<i4"),
            ]
        )
        attrs_data = np.array(
            [(area_name.encode(), n_cells)],
            dtype=attrs_dtype,
        )
        geom_root.create_dataset("Attributes", data=attrs_data)

        g = geom_root.create_group(area_name)

        # Include ghost cells (n_cells + 2) in coordinate array
        all_cells = n_cells + 2
        g.create_dataset(
            "Cells Center Coordinate",
            data=rng.uniform(0, 100, (all_cells, 2)).astype("f8"),
        )
        g.create_dataset(
            "Cells Minimum Elevation", data=rng.uniform(0, 5, (all_cells,)).astype("f4")
        )
        g.create_dataset(
            "Cells Center Manning's n", data=np.full(all_cells, 0.04, "f4")
        )
        g.create_dataset(
            "Cells Surface Area", data=rng.uniform(10, 100, (all_cells,)).astype("f4")
        )

        # Volume-elevation tables (one entry per cell)
        vol_info = np.zeros((all_cells, 2), dtype="i4")
        vol_values = np.zeros((all_cells * 2, 2), dtype="f4")
        for i in range(all_cells):
            vol_info[i] = [i * 2, 2]
            vol_values[i * 2] = [0.0, 0.0]
            vol_values[i * 2 + 1] = [10.0, 500.0]
        g.create_dataset("Cells Volume Elevation Info", data=vol_info)
        g.create_dataset("Cells Volume Elevation Values", data=vol_values)

        # Cell-face connectivity (each cell has 3 faces for simplicity)
        faces_per_cell = 3
        total_cf = all_cells * faces_per_cell
        cf_info = np.array(
            [[i * faces_per_cell, faces_per_cell] for i in range(all_cells)],
            dtype="i4",
        )
        cf_values = np.zeros((total_cf, 2), dtype="i4")
        for i in range(all_cells):
            for j in range(faces_per_cell):
                cf_values[i * faces_per_cell + j] = [
                    (i * faces_per_cell + j) % n_faces,
                    1,
                ]
        g.create_dataset("Cells Face and Orientation Info", data=cf_info)
        g.create_dataset("Cells Face and Orientation Values", data=cf_values)
        # FacePoint indexes (not used in computation but present in real files)
        g.create_dataset(
            "Cells FacePoint Indexes", data=np.zeros((all_cells, 8), dtype="i4")
        )

        # Face normals: random unit vectors + length
        angles = rng.uniform(0, 2 * np.pi, n_faces)
        normals = np.column_stack(
            [
                np.cos(angles),
                np.sin(angles),
                rng.uniform(5, 20, n_faces),
            ]
        ).astype("f4")
        g.create_dataset("Faces NormalUnitVector and Length", data=normals)

        # Face-cell indexes: left and right cell
        fci = np.column_stack(
            [
                np.arange(n_faces) % n_cells,
                (np.arange(n_faces) + 1) % n_cells,
            ]
        ).astype("i4")
        g.create_dataset("Faces Cell Indexes", data=fci)

        g.create_dataset(
            "Faces Minimum Elevation", data=rng.uniform(0, 3, n_faces).astype("f4")
        )
        g.create_dataset(
            "Faces FacePoint Indexes", data=np.zeros((n_faces, 2), dtype="i4")
        )
        g.create_dataset(
            "FacePoints Coordinate",
            data=rng.uniform(0, 100, (n_faces * 2, 2)).astype("f8"),
        )

        # Area-elevation tables (two entries per face)
        ae_info = np.array([[i * 2, 2] for i in range(n_faces)], dtype="i4")
        ae_values = np.zeros((n_faces * 2, 4), dtype="f4")
        for i in range(n_faces):
            ae_values[i * 2] = [0.0, 0.0, 0.0, 0.04]
            ae_values[i * 2 + 1] = [10.0, 50.0, 20.0, 0.04]
        g.create_dataset("Faces Area Elevation Info", data=ae_info)
        g.create_dataset("Faces Area Elevation Values", data=ae_values)

        # Face perimeter info/values
        g.create_dataset(
            "Faces Perimeter Info", data=np.zeros((n_faces, 2), dtype="i4")
        )
        g.create_dataset("Faces Perimeter Values", data=np.zeros((4, 2), dtype="f8"))
        g.create_dataset("Perimeter", data=rng.uniform(0, 100, (8, 2)).astype("f8"))

        # ── Results ───────────────────────────────────────────────────
        ts_path = (
            "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series"
        )
        ts_root = f.create_group(ts_path)

        # Time stamps (DD Mon YYYY HH:MM:SS)
        stamps = [
            b"01Jan2000 00:00:00",
            b"01Jan2000 01:00:00",
            b"01Jan2000 02:00:00",
            b"01Jan2000 03:00:00",
            b"01Jan2000 04:00:00",
        ]
        ts_root.create_dataset("Time Date Stamp", data=np.array(stamps, dtype="S19"))
        ts_root.create_dataset("Time", data=np.linspace(0, 4, n_timesteps))
        ts_root.create_dataset("Time Step", data=np.ones(n_timesteps, dtype="f4"))

        area_ts = ts_root.create_group(f"2D Flow Areas/{area_name}")
        bed = np.array(f[f"Geometry/2D Flow Areas/{area_name}/Cells Minimum Elevation"])
        wse_data = rng.uniform(3, 8, (n_timesteps, n_cells)).astype("f4")
        # ensure WSE > bed for all real cells
        wse_data = np.maximum(wse_data, bed[:n_cells][np.newaxis, :] + 0.5)
        area_ts.create_dataset("Water Surface", data=wse_data)
        area_ts.create_dataset(
            "Face Velocity",
            data=rng.uniform(-2, 2, (n_timesteps, n_faces)).astype("f4"),
        )

        sum_path = "Results/Unsteady/Output/Output Blocks/Base Output/Summary Output"
        area_sum = f.create_group(f"{sum_path}/2D Flow Areas/{area_name}")
        max_wse = np.vstack([wse_data.max(axis=0), np.ones(n_cells) * 2.0])
        min_wse = np.vstack([wse_data.min(axis=0), np.zeros(n_cells)])
        area_sum.create_dataset("Maximum Water Surface", data=max_wse.astype("f4"))
        area_sum.create_dataset("Minimum Water Surface", data=min_wse.astype("f4"))
        max_fv = rng.uniform(0, 3, n_faces).astype("f4")
        area_sum.create_dataset(
            "Maximum Face Velocity",
            data=np.vstack([max_fv, np.ones(n_faces)]).astype("f4"),
        )


@pytest.fixture(scope="session")
def synthetic_plan_hdf(tmp_path_factory) -> Path:
    """Session-scoped path to a synthetic plan HDF file."""
    path = tmp_path_factory.mktemp("hdf") / "synthetic.p01.hdf"
    _make_synthetic_hdf(path)
    return path
