"""Base class for HEC-RAS HDF5 file access.

Manages file lifecycle: open on construction, close explicitly or via context
manager.  Appends '.hdf' to the path when the suffix is missing.
"""
from __future__ import annotations

from pathlib import Path

import h5py


def _resolve_hdf_path(filename: str | Path) -> Path:
    """Return *filename* as a Path, appending '.hdf' if the suffix is absent."""
    path = Path(filename)
    if path.suffix.lower() != ".hdf":
        path = path.with_name(path.name + ".hdf")
    return path


class _HdfFile:
    """Open an HEC-RAS HDF5 file and keep the handle alive until closed.

    Subclasses receive ``self._hdf`` as a ready-to-use ``h5py.File`` object
    after ``__init__`` completes.

    Usage
    -----
    Direct::

        obj = SomeHdfSubclass("model.p01")
        data = obj._hdf["Results/..."][:]
        obj.close()

    Context manager (preferred)::

        with SomeHdfSubclass("model.p01") as obj:
            data = obj._hdf["Results/..."][:]
    """

    def __init__(self, filename: str | Path) -> None:
        self._filename = _resolve_hdf_path(filename)
        if not self._filename.is_file():
            raise FileNotFoundError(f"HEC-RAS HDF file not found: {self._filename}")
        self._hdf: h5py.File = h5py.File(self._filename, "r")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def filename(self) -> Path:
        """Resolved path to the HDF file."""
        return self._filename

    def close(self) -> None:
        """Close the underlying HDF5 file handle."""
        if self._hdf.id.valid:
            self._hdf.close()

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "_HdfFile":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False  # do not suppress exceptions

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
