"""Base classes for HEC-RAS HDF5 file access.

:class:`_HdfFile` manages file lifecycle: open on construction, close
explicitly or via context manager.  Appends ``.hdf`` to the path when
the suffix is missing.

:class:`_PlanHdf` is a mixin that adds :meth:`~_PlanHdf.runtime_log` to any
plan HDF class.  It has no base class of its own; it accesses ``self._hdf``
and ``self._filename`` via the MRO of the concrete plan class.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

from rivia.utils import parse_hec_datetime

if TYPE_CHECKING:
    import pandas as pd

# ---------------------------------------------------------------------------
# HEC-RAS datetime format constants — single authoritative definitions.
# All hdf/ modules import from here; never redefine locally.
# ---------------------------------------------------------------------------

# HDF attribute timestamps and runtime-log datestamps: "11APR2026 11:53:38"
_RAS_TS_FMT = "%d%b%Y %H:%M:%S"

# Post-process profile dates (instantaneous output): "01JAN2026 0002"
_POSTPROC_TS_FMT = "%d%b%Y %H%M"


def _parse_hec_ts_array(raw: np.ndarray, fmt: str) -> pd.DatetimeIndex:
    """Convert an array of HEC-RAS timestamp strings to a :class:`pandas.DatetimeIndex`.

    Delegates each element to :func:`~rivia.utils.parse_hec_datetime` so that
    ``HH=24`` midnight values are handled correctly.

    Parameters
    ----------
    raw : numpy.ndarray
        1-D array of HEC-RAS timestamp strings (any dtype castable to ``str``).
    fmt : str
        ``strptime`` format string passed through to
        :func:`~rivia.utils.parse_hec_datetime`.  Use :data:`_RAS_TS_FMT` for
        standard HDF timestamps or :data:`_POSTPROC_TS_FMT` for post-process
        profile dates.

    Returns
    -------
    pandas.DatetimeIndex
    """
    import pandas as pd  # local import — pandas is optional at module level

    return pd.DatetimeIndex([parse_hec_datetime(str(s), fmt=fmt) for s in raw])


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


# ---------------------------------------------------------------------------
# _PlanHdf — mixin for plan HDF files
# ---------------------------------------------------------------------------

# HDF path constants (plan files only)
_SUMMARY_ROOT = "Results/Summary"
_MSG_TEXT = f"{_SUMMARY_ROOT}/Compute Messages (text)"
_MSG_RTF = f"{_SUMMARY_ROOT}/Compute Messages (rtf)"
_PROCESSES = f"{_SUMMARY_ROOT}/Compute Processes"


class _PlanHdf:
    """Mixin that adds :meth:`runtime_log` to HEC-RAS plan HDF classes.

    Has no base class of its own.  Accesses ``self._hdf`` and
    ``self._filename`` through the MRO of the concrete plan class, which
    must include :class:`_HdfFile` (directly or via :class:`Geometry`).

    Usage::

        class SteadyPlan(_PlanHdf, Geometry): ...
        class UnsteadyPlan(_PlanHdf, Geometry): ...

    The MRO for both becomes ``Plan → _PlanHdf → Geometry → _HdfFile``,
    so ``self._hdf`` is always available when :meth:`runtime_log` runs.
    """

    def _runtime_log_raw(self) -> tuple[bytes, bytes, np.ndarray]:
        """Read raw runtime-log data from ``Results/Summary/``.

        Returns
        -------
        tuple[bytes, bytes, numpy.ndarray]
            ``(text_bytes, rtf_bytes, processes)`` — the two message byte
            strings and the structured ``Compute Processes`` array.

        Raises
        ------
        KeyError
            If ``Results/Summary`` is absent (e.g. a geometry HDF file).
        """
        hdf = self._hdf  # type: ignore[attr-defined]
        if _SUMMARY_ROOT not in hdf:
            raise KeyError(
                f"'Results/Summary' group not found in "
                f"{self._filename!r}. "  # type: ignore[attr-defined]
                "This may be a geometry HDF file, not a plan HDF file."
            )
        text_bytes: bytes = hdf[_MSG_TEXT][0]
        rtf_bytes: bytes = hdf[_MSG_RTF][0]
        processes: np.ndarray = hdf[_PROCESSES][:]
        return text_bytes, rtf_bytes, processes

    def runtime_log(self):  # return type annotated in subclasses
        """Read the runtime compute log from ``Results/Summary/``.

        Returns
        -------
        RuntimeLog
            Container with the full text log, RTF log, and compute-process
            table.  Concrete plan classes override this to return the
            appropriate :class:`~rivia.hdf.RuntimeLog` subclass.

        Raises
        ------
        KeyError
            If ``Results/Summary`` is absent from the HDF file.

        Examples
        --------
        ::

            with SteadyPlan("model.p01") as hdf:
                log = hdf.runtime_log()
                print(log.plan_name)
                print(log.simulation_start)
                for proc in log.compute_processes:
                    print(proc.process, proc.compute_time)
        """
        from .log import RuntimeLog  # local import avoids circular dependency

        return RuntimeLog(*self._runtime_log_raw())
