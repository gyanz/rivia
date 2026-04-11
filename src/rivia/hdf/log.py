"""Runtime log classes for HEC-RAS plan HDF5 files.

Reads ``Results/Summary/Compute Messages (text|rtf)`` and
``Results/Summary/Compute Processes`` from any plan HDF file (steady,
unsteady, sediment, quasi-steady, water quality).

Exposes:

* :class:`ComputeProcess` — one row of the Compute Processes table.
* :class:`RuntimeLog` — common log container, valid for all plan types.
* :class:`SteadyRuntimeLog` — steady-specific subclass (stub; methods TBD).
* :class:`UnsteadyRuntimeLog` — unsteady-specific parsed methods.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger("rivia.hdf")

# ---------------------------------------------------------------------------
# Regex helpers used in UnsteadyRuntimeLog
# ---------------------------------------------------------------------------
# Matches any line that starts with a HEC-RAS datestamp:
#   01OCT2024 00:00:02
_RE_DATESTAMP = re.compile(
    r"^(?P<date>\d{2}[A-Z]{3}\d{4})\s+(?P<time>\d{2}:\d{2}:\d{2})"
)
# Matches adaptive-timestep lines:
#   01OCT2024 00:00:02       timestep =   1  (sec)
_RE_TIMESTEP = re.compile(
    r"^(?P<date>\d{2}[A-Z]{3}\d{4})\s+(?P<time>\d{2}:\d{2}:\d{2})"
    r"\s+timestep\s*=\s*(?P<val>\S+)\s*\(sec\)",
    re.IGNORECASE,
)
# Matches the Computation Speed section lines:
#   Unsteady Flow Computations<tab>9.10x
_RE_SPEED = re.compile(r"^(?P<task>.+?)\t(?P<speed>\S+x)\s*$")

_HEC_DATE_FMT = "%d%b%Y %H:%M:%S"  # e.g. "01OCT2024 00:00:02"


def _parse_hec_datetime(date_str: str, time_str: str) -> dt.datetime:
    """Parse a HEC-RAS date + time token pair into a :class:`datetime.datetime`.

    Parameters
    ----------
    date_str:
        Date token in ``DD<MON>YYYY`` format, e.g. ``"01OCT2024"``.
    time_str:
        Time token in ``HH:MM:SS`` format, e.g. ``"00:00:02"``.

    Returns
    -------
    datetime.datetime
    """
    return dt.datetime.strptime(f"{date_str} {time_str}", _HEC_DATE_FMT)


# ---------------------------------------------------------------------------
# ComputeProcess
# ---------------------------------------------------------------------------


@dataclass
class ComputeProcess:
    """One row from ``Results/Summary/Compute Processes``.

    Parameters
    ----------
    process:
        Short process label, e.g. ``"Ras.exe"``,
        ``"Unsteady Flow Computations"``.
    filename:
        Executable or DLL filename (may be blank).
    file_date:
        Build date of the executable (may be blank).
    file_size:
        File size in bytes (0 when not recorded).
    file_version:
        Version string, e.g. ``"6.6"``.
    arguments:
        Command-line arguments used (may be blank).
    compute_time:
        Human-readable elapsed time string, e.g. ``"00:00:01.000"``,
        ``"<1"``, or ``""`` when not applicable.
    compute_time_ms:
        Elapsed time in milliseconds (0 when not recorded).
    """

    process: str
    filename: str
    file_date: str
    file_size: int
    file_version: str
    arguments: str
    compute_time: str
    compute_time_ms: int


# ---------------------------------------------------------------------------
# RuntimeLog — common to all plan types
# ---------------------------------------------------------------------------


class RuntimeLog:
    """Runtime log for any HEC-RAS plan type.

    Wraps the ``Results/Summary/Compute Messages`` datasets and the
    ``Compute Processes`` structured array.  This class is the return type
    of :meth:`_PlanHdf.runtime_log` and is subclassed by
    :class:`SteadyRuntimeLog` and :class:`UnsteadyRuntimeLog` to add
    plan-specific parsing methods.

    Parameters
    ----------
    text_bytes:
        Raw bytes from ``Compute Messages (text)``.
    rtf_bytes:
        Raw bytes from ``Compute Messages (rtf)``.
    processes:
        Numpy structured array from ``Compute Processes``.

    Notes
    -----
    Both message datasets store a single-element byte-string array whose
    element holds the entire log as a ``\\r\\n``-delimited string.
    """

    def __init__(
        self,
        text_bytes: bytes,
        rtf_bytes: bytes,
        processes: np.ndarray,
    ) -> None:
        self._text = text_bytes.decode("utf-8", errors="replace")
        self._rtf = rtf_bytes.decode("utf-8", errors="replace")
        self._processes_raw = processes

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def text(self) -> str:
        """Full plain-text compute log."""
        return self._text

    @property
    def rtf(self) -> str:
        """Full RTF compute log (preserves color/bold markup from HEC-RAS)."""
        return self._rtf

    @property
    def lines(self) -> list[str]:
        """Plain-text log split into individual lines."""
        return self._text.splitlines()

    @property
    def compute_processes(self) -> list[ComputeProcess]:
        """Rows of the ``Compute Processes`` table as :class:`ComputeProcess` objects.

        Returns
        -------
        list[ComputeProcess]
            One entry per process/phase recorded by HEC-RAS.

        Examples
        --------
        ::

            log = plan.runtime_log()
            for proc in log.compute_processes:
                print(proc.process, proc.compute_time)
        """
        result: list[ComputeProcess] = []
        def _s(field: bytes | str) -> str:
            if isinstance(field, (bytes, np.bytes_)):
                return field.decode("utf-8", errors="replace").strip()
            return str(field).strip()

        for row in self._processes_raw:
            result.append(
                ComputeProcess(
                    process=_s(row["Process"]),
                    filename=_s(row["Filename"]),
                    file_date=_s(row["File Date"]),
                    file_size=int(row["File Size"]),
                    file_version=_s(row["File Version"]),
                    arguments=_s(row["Arguments"]),
                    compute_time=_s(row["Compute Time"]),
                    compute_time_ms=int(row["Compute Time (ms)"]),
                )
            )
        return result

    @property
    def plan_name(self) -> str | None:
        """Plan name parsed from the first log line.

        The first line has the form ``Plan: '<name>' (file.p01)``.
        Returns ``None`` if the line is absent or does not match.
        """
        lines = self.lines
        if not lines:
            return None
        m = re.match(r"^Plan:\s*'(.+?)'\s*\(", lines[0])
        return m.group(1) if m else None

    @property
    def simulation_start(self) -> dt.datetime | None:
        """Simulation start time parsed from ``"Simulation started at: ..."`` line.

        Returns
        -------
        datetime.datetime or None
            ``None`` when the line is absent or cannot be parsed.
        """
        for line in self.lines:
            m = re.match(r"^Simulation started at:\s*(.+)$", line)
            if m:
                raw = m.group(1).strip()
                for fmt in ("%d%b%Y %I:%M:%S %p", "%d%b%Y %H:%M:%S"):
                    try:
                        return dt.datetime.strptime(raw, fmt)
                    except ValueError:
                        continue
        return None

    def computation_time(self) -> float | None:
        """Computation time in seconds for the flow-computations process.

        Searches :attr:`compute_processes` for the entry whose ``process``
        attribute ends with ``"Flow Computations"`` (case-insensitive,
        ignoring leading/trailing whitespace) and returns its recorded
        elapsed time converted to seconds.

        Returns
        -------
        float or None
            Elapsed time in seconds, or ``None`` when no matching process
            entry is found.

        Examples
        --------
        ::

            log = plan.runtime_log()
            secs = log.computation_time()
            if secs is not None:
                print(f"Flow computations took {secs:.1f} s")
        """
        for proc in self.compute_processes:
            if proc.process.strip().lower().endswith("flow computations"):
                secs = proc.compute_time_ms / 1000.0
                logger.info("Flow computation time: %.3f seconds", secs)
                return secs
        return None


# ---------------------------------------------------------------------------
# SteadyRuntimeLog — steady-specific (stub)
# ---------------------------------------------------------------------------


class SteadyRuntimeLog(RuntimeLog):
    """Runtime log for a HEC-RAS steady-flow plan.

    Inherits all :class:`RuntimeLog` accessors.  Steady-specific parsing
    methods (profile convergence, critical depth warnings, etc.) will be
    added here as the library evolves.
    """


# ---------------------------------------------------------------------------
# UnsteadyRuntimeLog — unsteady-specific parsing
# ---------------------------------------------------------------------------


class UnsteadyRuntimeLog(RuntimeLog):
    """Runtime log for a HEC-RAS unsteady-flow plan.

    Inherits all :class:`RuntimeLog` accessors and adds parsing methods
    specific to unsteady (and quasi-unsteady) compute logs.

    The text log records one iteration-summary line per compute timestep in
    the form::

        DD<MON>YYYY HH:MM:SS  <location>  <rs_or_cell>  <wsel>  <error>  <iters>

    where fields are tab-separated and location is space-padded.  Adaptive
    timestep changes appear as::

        DD<MON>YYYY HH:MM:SS       timestep = <val>  (sec)
    """

    # ------------------------------------------------------------------
    # max_iterations
    # ------------------------------------------------------------------

    def max_iterations(self) -> pd.DataFrame:
        """Per-timestep maximum-iteration summary.

        Parses datestamped iteration lines.  HEC-RAS writes two layouts
        depending on whether the controlling element is 1-D or 2-D:

        * **1-D** (6 tab-delimited fields after the datestamp)::

              02JAN1900 00:00:01  White  \\tMuncie  \\t15696.24\\t952.69\\t0.235\\t20
              ^-- location (river)  ^-- reach ^-- RS     ^-- wsel ^-- err ^-- iters

        * **2-D** (5 fields after the datestamp)::

              01OCT2024 00:00:02  Kagman\\t     35337\\t  228.75\\t   0.012\\t20
              ^-- location (area)        ^-- cell     ^-- wsel    ^-- error ^-- iters

        * **Storage area** (4 fields after the datestamp)::

              01OCT2024 00:00:02  Pond1\\t  228.75\\t   0.012\\t20
              ^-- storage area name  ^-- wsel    ^-- error ^-- iters

          ``rs_or_cell`` is set to an empty string for storage-area rows.

        Mixed 1-D/2-D/storage-area plans produce multiple layouts in the same log.

        Returns
        -------
        pandas.DataFrame
            Columns:

            * ``datetime`` — :class:`datetime.datetime`
            * ``location_type`` — str, one of ``"1D"``, ``"FlowArea"``,
              or ``"StorageArea"``
            * ``location`` — str, river name (1-D), 2-D flow area name,
              or storage area name
            * ``reach`` — str, reach name for 1-D rows; empty string otherwise
            * ``rs_or_cell`` — str, river station (1-D) or cell identifier
              (2-D); empty string for storage-area rows
            * ``wsel`` — float, water-surface elevation (ft or m)
            * ``error`` — float, iteration error
            * ``iterations`` — int, number of iterations at this step
            * ``timestep`` — float, active adaptive timestep in seconds at
              the time of this row; ``NaN`` when no adaptive-timestep change
              has been logged yet or the run uses a fixed timestep

            Rows are in log order (chronological).  Returns an empty
            DataFrame if no iteration lines are found.

        Notes
        -----
        Adaptive-timestep change lines share the same datestamp prefix but
        contain the word ``timestep``; they are not returned as rows but are
        used to populate the ``timestep`` column of subsequent iteration rows.
        """
        _COLS = [
            "datetime", "location_type", "location", "reach",
            "rs_or_cell", "wsel", "error", "iterations", "timestep",
        ]
        records: list[dict] = []
        current_timestep: float = float("nan")
        for line in self.lines:
            m = _RE_DATESTAMP.match(line)
            if m is None:
                continue
            rest = line[m.end():].strip()
            # Capture adaptive-timestep lines and keep scanning
            ts_m = re.match(r"timestep\s*=\s*(\S+)\s*\(sec\)", rest, re.IGNORECASE)
            if ts_m:
                with contextlib.suppress(ValueError):
                    current_timestep = float(ts_m.group(1))
                continue
            parts = rest.split("\t")
            try:
                if len(parts) == 6:
                    # 1-D layout: river \t reach \t rs \t wsel \t error \t iters
                    location_type = "1D"
                    location = parts[0].strip()
                    reach = parts[1].strip()
                    rs_or_cell = parts[2].strip()
                    wsel = float(parts[3])
                    error = float(parts[4])
                    iterations = int(parts[5].strip())
                elif len(parts) == 5:
                    # 2-D layout: area \t cell \t wsel \t error \t iters
                    location_type = "FlowArea"
                    location = parts[0].strip()
                    reach = ""
                    rs_or_cell = parts[1].strip()
                    wsel = float(parts[2])
                    error = float(parts[3])
                    iterations = int(parts[4].strip())
                elif len(parts) == 4:
                    # Storage area layout: name \t wsel \t error \t iters
                    location_type = "StorageArea"
                    location = parts[0].strip()
                    reach = ""
                    rs_or_cell = ""
                    wsel = float(parts[1])
                    error = float(parts[2])
                    iterations = int(parts[3].strip())
                else:
                    continue
            except (ValueError, IndexError):
                continue
            records.append(
                {
                    "datetime": _parse_hec_datetime(m.group("date"), m.group("time")),
                    "location_type": location_type,
                    "location": location,
                    "reach": reach,
                    "rs_or_cell": rs_or_cell,
                    "wsel": wsel,
                    "error": error,
                    "iterations": iterations,
                    "timestep": current_timestep,
                }
            )
        if not records:
            return pd.DataFrame(columns=_COLS)
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # max_1d2d_iterations
    # ------------------------------------------------------------------

    def max_1d2d_iterations(self) -> pd.DataFrame:
        """Per-timestep maximum 1D/2D connection iteration summary.

        Parses lines from the ``Maximum 1D/2D iterations`` section of the
        compute log.  These lines record timesteps at which the 1D/2D
        interface flow equation failed to converge within the iteration
        limit.  They are distinct from the per-element rows returned by
        :meth:`max_iterations` and are absent from plans that have no
        1D/2D connections.

        Each data line has the form::

            DD<MON>YYYY HH:MM:SS    1D/2D Flow error\\t<flow_error>\\t
                <river>   <reach>   <rs>

        where the third tab field holds river name, reach name, and river
        station separated by two or more spaces.

        Returns
        -------
        pandas.DataFrame
            Columns:

            * ``datetime`` — :class:`datetime.datetime`
            * ``flow_error`` — float, flow error at the 1D/2D interface
              (cfs or cms; can be negative)
            * ``river`` — str, river name
            * ``reach`` — str, reach name
            * ``rs`` — str, river station

            Rows are in log order (chronological).  Returns an empty
            DataFrame if no 1D/2D iteration lines are found.
        """
        _COLS = ["datetime", "flow_error", "river", "reach", "rs"]
        records: list[dict] = []
        for line in self.lines:
            m = _RE_DATESTAMP.match(line)
            if m is None:
                continue
            rest = line[m.end():].strip()
            parts = rest.split("\t")
            if len(parts) != 3 or parts[0].strip() != "1D/2D Flow error":
                continue
            try:
                flow_error = float(parts[1])
                connection = parts[2].strip()
                tokens = re.split(r"\s{2,}", connection)
                if len(tokens) == 3:
                    river, reach, rs = tokens[0], tokens[1], tokens[2]
                else:
                    river, reach, rs = connection, "", ""
            except (ValueError, IndexError):
                continue
            records.append(
                {
                    "datetime": _parse_hec_datetime(m.group("date"), m.group("time")),
                    "flow_error": flow_error,
                    "river": river,
                    "reach": reach,
                    "rs": rs,
                }
            )
        if not records:
            return pd.DataFrame(columns=_COLS)
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # adaptive_timesteps
    # ------------------------------------------------------------------

    def adaptive_timesteps(self) -> pd.DataFrame:
        """Adaptive timestep change events recorded in the compute log.

        Parses lines of the form::

            01OCT2024 00:00:02       timestep =   1   (sec)

        These lines are only present when adaptive time stepping is active
        and a timestep reduction or increase occurs.

        Returns
        -------
        pandas.DataFrame
            Columns:

            * ``datetime`` — :class:`datetime.datetime`
            * ``timestep_sec`` — float, new timestep in seconds

            Returns an empty DataFrame when no adaptive-timestep lines are
            present (fixed-timestep runs).
        """
        records: list[dict] = []
        for line in self.lines:
            m = _RE_TIMESTEP.match(line)
            if m is None:
                continue
            try:
                ts = float(m.group("val"))
            except ValueError:
                continue
            records.append(
                {
                    "datetime": _parse_hec_datetime(m.group("date"), m.group("time")),
                    "timestep_sec": ts,
                }
            )
        if not records:
            return pd.DataFrame(columns=["datetime", "timestep_sec"])
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # computation_speed
    # ------------------------------------------------------------------

    def computation_speed(self) -> dict[str, str]:
        """Simulation-to-runtime speed ratios from the Computation Speed section.

        Returns
        -------
        dict[str, str]
            Mapping of task name to speed string, e.g.
            ``{"Unsteady Flow Computations": "9.10x",
            "Complete Process": "8.99x"}``.

            Returns an empty dict when the section is absent.
        """
        lines = self.lines
        in_section = False
        result: dict[str, str] = {}
        for line in lines:
            if line.strip() == "Computation Speed\tSimulation/Runtime":
                in_section = True
                continue
            if in_section:
                if not line.strip():
                    continue
                m = _RE_SPEED.match(line)
                if m:
                    result[m.group("task").strip()] = m.group("speed")
                else:
                    # Any non-matching, non-empty line ends the section
                    break
        return result

    # ------------------------------------------------------------------
    # input_summary
    # ------------------------------------------------------------------

    def input_summary(self) -> str:
        """Raw text of the ``Unsteady Input Summary:`` block.

        Returns
        -------
        str
            All lines from ``"Unsteady Input Summary:"`` up to (but not
            including) the next blank-then-non-blank transition that marks
            the start of iteration output.  Returns an empty string when
            the section is absent.
        """
        lines = self.lines
        start: int | None = None
        for i, line in enumerate(lines):
            if line.strip() == "Unsteady Input Summary:":
                start = i
                break
        if start is None:
            return ""
        # Collect until the "Maximum iteration location" header (or similar)
        block: list[str] = []
        for line in lines[start:]:
            if re.match(r"^Maximum\s+iteration\s+location", line, re.IGNORECASE):
                break
            if re.match(r"^Pipe\s+Network\s+Iter", line, re.IGNORECASE):
                break
            block.append(line)
        return "\n".join(block).strip()

