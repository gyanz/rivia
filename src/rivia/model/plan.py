"""Read/write HEC-RAS plan files (.p**)."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from ..utils.helpers import check_sim_date as _check_sim_date
from ..utils.helpers import check_sim_time as _check_sim_time

logger = logging.getLogger("rivia.model")

_MONTHS = {"JAN", "FEB", "MAR", "APR", "MAY", "JUN",
           "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"}

_MONTH_ABBR = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
               "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def _to_sim_date_str(bound: str | tuple[str, str] | datetime) -> str:
    """Normalise a simulation window bound to ``"DDMONYYYY,HHMM"`` format."""
    if isinstance(bound, datetime):
        return f"{bound.day:02d}{_MONTH_ABBR[bound.month - 1]}{bound.year},{bound.hour:02d}{bound.minute:02d}"
    if isinstance(bound, tuple):
        _check_sim_date(bound[0])
        _check_sim_time(bound[1])
        return f"{bound[0]},{bound[1]}"
    parts = bound.split(",")
    _check_sim_date(parts[0])
    _check_sim_time(parts[1])
    return bound


class Plan:
    """Parser and editor for a HEC-RAS plan file.

    Reads the file into memory once, exposes typed properties for commonly
    changed fields, and writes back to the source path via ``save()``.

    Unknown lines and lines without an ``=`` (e.g. ``Subcritical Flow``) are
    preserved verbatim so round-trips are faithful.

    Derived from: ``archive/ras_tools/planParser.py``
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.is_file():
            raise FileNotFoundError(f"Plan file not found: {self._path}")
        with open(self._path, encoding="utf-8", errors="replace") as fh:
            self._lines: list[str] = fh.readlines()
        self._modified: bool = False

    def __repr__(self) -> str:
        return f"Plan({self._path.name!r}, title={self.title!r})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, key: str) -> str | None:
        """Return the stripped value for *key*, or ``None`` if absent/empty."""
        prefix = key + "="
        for line in self._lines:
            if line.startswith(prefix):
                value = line[len(prefix) :].strip()
                return value if value else None
        return None

    def _set(self, key: str, raw_value: str) -> None:
        """Replace the value for the *first* occurrence of *key*.

        Raises ``KeyError`` if the key is not present in the file.
        """
        prefix = key + "="
        for i, line in enumerate(self._lines):
            if line.startswith(prefix):
                self._lines[i] = f"{prefix}{raw_value}\n"
                self._modified = True
                return
        raise KeyError(f"Key not found in plan file: {key!r}")

    @staticmethod
    def _to_bool(raw: str | None) -> bool:
        """Convert a raw plan-file flag value to bool.

        HEC-RAS uses several conventions:
        - ``" 1 "`` or ``"-1"`` -> True
        - ``" 0 "``             -> False
        """
        if raw is None:
            return False
        return raw.strip() != "0"

    @staticmethod
    def _from_bool(value: bool) -> str:
        """Convert a bool to the ``" 1 "`` / ``" 0 "`` style used by run flags."""
        return " 1 " if value else " 0 "

    # ------------------------------------------------------------------
    # Modification state
    # ------------------------------------------------------------------

    @property
    def is_modified(self) -> bool:
        """``True`` if any value has been changed since the last :meth:`save`."""
        return self._modified

    # ------------------------------------------------------------------
    # Generic escape hatch
    # ------------------------------------------------------------------

    def get(self, key: str) -> str | None:
        """Return the raw stripped value for *key*, or ``None`` if absent/empty.

        Use this for fields not exposed as typed properties.
        """
        return self._get(key)

    def set(self, key: str, value: str) -> None:
        """Set *key* to *value* verbatim.

        Raises ``KeyError`` if the key does not already exist in the file.
        Use this for fields not exposed as typed properties.
        """
        self._set(key, value)

    # ------------------------------------------------------------------
    # Identity / metadata
    # ------------------------------------------------------------------

    @property
    def title(self) -> str | None:
        """Full plan title (``Plan Title=``)."""
        return self._get("Plan Title")

    @title.setter
    def title(self, value: str) -> None:
        self._set("Plan Title", value)

    @property
    def short_id(self) -> str | None:
        """Short identifier, stripped of padding (``Short Identifier=``)."""
        return self._get("Short Identifier")

    @short_id.setter
    def short_id(self, value: str) -> None:
        self._set("Short Identifier", value)

    @property
    def program_version(self) -> str | None:
        """HEC-RAS version that wrote this plan (``Program Version=``).

        Treat as read-only; HEC-RAS manages this field.
        """
        return self._get("Program Version")

    # ------------------------------------------------------------------
    # File references
    # ------------------------------------------------------------------

    @property
    def geom_file(self) -> str | None:
        """Geometry file extension reference, e.g. ``g01`` (``Geom File=``)."""
        return self._get("Geom File")

    @geom_file.setter
    def geom_file(self, value: str) -> None:
        self._set("Geom File", value)

    @property
    def flow_file(self) -> str | None:
        """Flow file extension reference, e.g. ``u01`` or ``f01`` (``Flow File=``)."""
        return self._get("Flow File")

    @flow_file.setter
    def flow_file(self, value: str) -> None:
        self._set("Flow File", value)

    @property
    def sediment_file(self) -> str | None:
        """Sediment file extension reference, e.g. ``s04`` (``Sediment File=``)."""
        return self._get("Sediment File")

    @sediment_file.setter
    def sediment_file(self, value: str) -> None:
        self._set("Sediment File", value)

    @property
    def water_quality_file(self) -> str | None:
        """Water quality file extension reference, e.g. ``w01``
        (``Water Quality File=``).
        """
        return self._get("Water Quality File")

    @water_quality_file.setter
    def water_quality_file(self, value: str) -> None:
        self._set("Water Quality File", value)

    @property
    def is_steady(self) -> bool:
        """True if this is a steady flow plan.

        Determined by ``Flow File=`` extension starting with ``f``.
        """
        ref = self.flow_file
        return ref is not None and ref.strip().lower().startswith("f")

    @property
    def is_unsteady(self) -> bool:
        """True if this is an unsteady flow plan.

        Determined by ``Flow File=`` extension starting with ``u``.
        """
        ref = self.flow_file
        return ref is not None and ref.strip().lower().startswith("u")

    @property
    def is_quasi_steady(self) -> bool:
        """True if this is a quasi-steady flow plan.

        Determined by ``Flow File=`` extension starting with ``q``.
        """
        ref = self.flow_file
        return ref is not None and ref.strip().lower().startswith("q")

    @property
    def is_sediment(self) -> bool:
        """True if this plan includes a sediment file.

        Determined by ``Sediment File=`` being present.
        """
        return self.sediment_file is not None

    @property
    def is_water_quality(self) -> bool:
        """True if this plan includes a water quality file.

        Determined by ``Water Quality File=`` being present.
        """
        return self.water_quality_file is not None

    # ------------------------------------------------------------------
    # Simulation window
    # ------------------------------------------------------------------

    @property
    def simulation_window(self) -> tuple[tuple[str, str], tuple[str, str]] | None:
        """Simulation start and end as ``((date, time), (date, time))``.

        Each date is ``"DDMONYYYY"`` and each time is ``"HHMM"``
        (e.g. ``(("18FEB1999", "0000"), ("20FEB1999", "2400"))``).
        Returns ``None`` if the key is absent.
        """
        raw = self._get("Simulation Date")
        if raw is None:
            return None
        parts = raw.split(",")
        if len(parts) < 4:
            raise ValueError(
                f"Unexpected Simulation Date format: {raw!r}. "
                "Expected 'DDMONYYYY,HHMM,DDMONYYYY,HHMM'."
            )
        return (parts[0], parts[1]), (parts[2], parts[3])

    @simulation_window.setter
    def simulation_window(
        self,
        value: (
            tuple[str, str]
            | tuple[tuple[str, str], tuple[str, str]]
            | tuple[datetime, datetime]
        ),
    ) -> None:
        """Set simulation date.

        Each bound can be supplied in one of three forms:

        - Flat string: ``"DDMONYYYY,HHMM"`` (e.g. ``"01JAN2020,0000"``)
        - Nested tuple: ``("DDMONYYYY", "HHMM")``
        - :class:`~datetime.datetime` object
        """
        start, end = value
        start = _to_sim_date_str(start)
        end = _to_sim_date_str(end)
        self._set("Simulation Date", f"{start},{end}")

    # ------------------------------------------------------------------
    # Computation / output intervals
    # ------------------------------------------------------------------

    @property
    def computation_interval(self) -> str | None:
        """Computation time step, e.g. ``"2MIN"``, ``"30SEC"``.

        Key: ``Computation Interval=``
        """
        return self._get("Computation Interval")

    @computation_interval.setter
    def computation_interval(self, value: str) -> None:
        self._set("Computation Interval", value)

    @property
    def output_interval(self) -> str | None:
        """Output write interval, e.g. ``"1HOUR"`` (``Output Interval=``)."""
        return self._get("Output Interval")

    @output_interval.setter
    def output_interval(self, value: str) -> None:
        self._set("Output Interval", value)

    @property
    def instantaneous_interval(self) -> str | None:
        """Instantaneous output interval (``Instantaneous Interval=``).

        Returns ``None`` if not present in the plan file.
        """
        return self._get("Instantaneous Interval")

    @instantaneous_interval.setter
    def instantaneous_interval(self, value: str) -> None:
        self._set("Instantaneous Interval", value)

    @property
    def mapping_interval(self) -> str | None:
        """Mapping output interval (``Mapping Interval=``).

        Returns ``None`` if not present in the plan file.
        """
        return self._get("Mapping Interval")

    @mapping_interval.setter
    def mapping_interval(self, value: str) -> None:
        self._set("Mapping Interval", value)

    # ------------------------------------------------------------------
    # Run flags  (HEC-RAS stores these as " 1 " / " 0 " or "-1" / " 0 ")
    # ------------------------------------------------------------------

    @property
    def run_hydraulic_tables(self) -> bool:
        """Whether to run hydraulic tables (``Run HTab=``)."""
        return self._to_bool(self._get("Run HTab"))

    @run_hydraulic_tables.setter
    def run_hydraulic_tables(self, value: bool) -> None:
        self._set("Run HTab", self._from_bool(value))

    @property
    def run_unsteady(self) -> bool:
        """Whether to run the unsteady-flow engine (``Run UNet=``)."""
        return self._to_bool(self._get("Run UNet"))

    @run_unsteady.setter
    def run_unsteady(self, value: bool) -> None:
        self._set("Run UNet", self._from_bool(value))

    @property
    def run_sediment(self) -> bool:
        """Whether to run sediment transport (``Run Sediment=``)."""
        return self._to_bool(self._get("Run Sediment"))

    @run_sediment.setter
    def run_sediment(self, value: bool) -> None:
        self._set("Run Sediment", self._from_bool(value))

    @property
    def run_post_process(self) -> bool:
        """Whether to run post-processing (``Run PostProcess=``)."""
        return self._to_bool(self._get("Run PostProcess"))

    @run_post_process.setter
    def run_post_process(self, value: bool) -> None:
        self._set("Run PostProcess", self._from_bool(value))

    @property
    def run_water_quality(self) -> bool:
        """Whether to run water quality (``Run WQNet=``)."""
        return self._to_bool(self._get("Run WQNet"))

    @run_water_quality.setter
    def run_water_quality(self, value: bool) -> None:
        self._set("Run WQNet", self._from_bool(value))

    @property
    def run_rasmapper(self) -> bool:
        """Whether to run RAS Mapper post-processing (``Run RASMapper=``)."""
        return self._to_bool(self._get("Run RASMapper"))

    @run_rasmapper.setter
    def run_rasmapper(self, value: bool) -> None:
        self._set("Run RASMapper", self._from_bool(value))

    # ------------------------------------------------------------------
    # 1-D hydraulics
    # ------------------------------------------------------------------

    @property
    def theta(self) -> float | None:
        """1-D implicit weighting factor (``UNET Theta=``)."""
        raw = self._get("UNET Theta")
        return float(raw) if raw is not None else None

    @theta.setter
    def theta(self, value: float) -> None:
        self._set("UNET Theta", str(value))

    @property
    def theta_warmup(self) -> float | None:
        """1-D implicit weighting factor during warmup (``UNET Theta Warmup=``)."""
        raw = self._get("UNET Theta Warmup")
        return float(raw) if raw is not None else None

    @theta_warmup.setter
    def theta_warmup(self, value: float) -> None:
        self._set("UNET Theta Warmup", str(value))

    @property
    def z_tolerance(self) -> float | None:
        """Water surface convergence tolerance (``UNET ZTol=``)."""
        raw = self._get("UNET ZTol")
        return float(raw) if raw is not None else None

    @z_tolerance.setter
    def z_tolerance(self, value: float) -> None:
        self._set("UNET ZTol", str(value))

    @property
    def max_iterations(self) -> int | None:
        """Maximum iterations per time step (``UNET MxIter=``)."""
        raw = self._get("UNET MxIter")
        return int(raw) if raw is not None else None

    @max_iterations.setter
    def max_iterations(self, value: int) -> None:
        self._set("UNET MxIter", str(value))

    # ------------------------------------------------------------------
    # Initial conditions output
    # ------------------------------------------------------------------

    @property
    def write_ic_file(self) -> bool:
        """Whether to write an initial conditions file (``Write IC File=``)."""
        return self._to_bool(self._get("Write IC File"))

    @write_ic_file.setter
    def write_ic_file(self, value: bool) -> None:
        self._set("Write IC File", self._from_bool(value))

    @property
    def write_ic_at_end(self) -> bool:
        """Whether to write IC file at simulation end (``Write IC File at Sim End=``)."""
        return self._to_bool(self._get("Write IC File at Sim End"))

    @write_ic_at_end.setter
    def write_ic_at_end(self, value: bool) -> None:
        self._set("Write IC File at Sim End", self._from_bool(value))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write all in-memory lines back to the source plan file."""
        with open(self._path, "w", encoding="utf-8") as fh:
            fh.writelines(self._lines)
        self._modified = False
