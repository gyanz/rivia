"""Plan staleness diagnostics for HEC-RAS HDF5 files.

Provides :func:`check_plan_staleness` which inspects a plan file and its
associated HDF files to report:

* Whether geometry input layers (terrain, land cover, infiltration, etc.)
  have been modified on disk since the geometry was last preprocessed.
* Whether the geometry HDF has been re-preprocessed since the plan was run,
  meaning the plan results were produced with an older geometry.
* Whether the plan's simulation results appear complete or were interrupted
  (user stop, crash, power outage).

Usage::

    from rivia.hdf.staleness import check_plan_staleness

    report = check_plan_staleness("KagmanWatershed.p01")
    print(report)
    if not report.run_appears_complete:
        print("Results may be incomplete or from a prior run!")
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rivia.utils import parse_hec_datetime

from ._base import _RAS_TS_FMT, _SUMMARY_ROOT, _MSG_TEXT, _MSG_RTF, _PROCESSES
from .geometry import Geometry
from .log import RunCompletion, RuntimeLog

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_RUN_WINDOW_PATHS = (
    "Results/Unsteady/Summary",
    "Results/Steady/Summary",
)


def _os_mtime(p: Path) -> dt.datetime | None:
    """Return the OS modification time of *p*, or ``None`` on any error."""
    try:
        return dt.datetime.fromtimestamp(p.stat().st_mtime)
    except OSError:
        return None


def _plan_geom_ext(plan_file: Path) -> str | None:
    """Scan *plan_file* for ``Geom File=`` and return the extension token."""
    try:
        with open(plan_file, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("Geom File="):
                    val = line.split("=", 1)[1].strip()
                    return val if val else None
    except OSError:
        pass
    return None


def _read_run_window(hdf) -> str | None:  # hdf: h5py.File
    """Read the ``Run Time Window`` attribute from whichever summary group exists."""
    for path in _RUN_WINDOW_PATHS:
        grp = hdf.get(path)
        if grp is not None:
            raw = grp.attrs.get("Run Time Window")
            if raw is not None:
                return raw.decode() if isinstance(raw, (bytes, np.bytes_)) else str(raw)
    return None


def _parse_run_window(
    raw: str,
) -> tuple[dt.datetime, dt.datetime] | None:
    """Parse ``"DDMONYYYY HH:MM:SS to DDMONYYYY HH:MM:SS"`` into a pair."""
    parts = raw.split(" to ", maxsplit=1)
    if len(parts) != 2:
        return None
    try:
        start = parse_hec_datetime(parts[0].strip(), fmt=_RAS_TS_FMT)
        end = parse_hec_datetime(parts[1].strip(), fmt=_RAS_TS_FMT)
        return start, end
    except ValueError:
        return None


def _runtime_log_from_hdf(hdf) -> RuntimeLog | None:  # hdf: h5py.File
    """Read raw log bytes from plan HDF and return a :class:`RuntimeLog`."""
    if _SUMMARY_ROOT not in hdf:
        return None
    try:
        text_bytes: bytes = hdf[_MSG_TEXT][0]
        rtf_bytes: bytes = hdf[_MSG_RTF][0]
        processes: np.ndarray = hdf[_PROCESSES][:]
        return RuntimeLog(text_bytes, rtf_bytes, processes)
    except (KeyError, IndexError):
        return None


def _read_compute_scalars(
    hdf,  # h5py.File
) -> tuple[bool | None, float | None, float | None]:
    """Read ``(compute_ok, volume_error_pct, time_unstable)`` from the plan HDF.

    Tries ``Results/Unsteady/Summary`` first, then ``Results/Steady/Summary``.
    Returns ``(None, None, None)`` when neither group is present or readable.

    For unsteady plans:
    * ``compute_ok`` — ``True`` when ``"Time Solution Went Unstable"`` is NaN
      (i.e. the solution did not go unstable).
    * ``volume_error_pct`` — overall volume balance error % from
      ``Volume Accounting/Error Percent``.
    * ``time_unstable`` — elapsed sim time (days) at instability, or ``None``.

    For steady plans:
    * ``compute_ok`` — ``True`` when ``"Solution"`` attribute contains
      ``"successfully"`` (case-insensitive).
    * ``volume_error_pct`` — ``None`` (steady runs have no volume accounting).
    * ``time_unstable`` — ``None`` (not applicable).
    """
    def _decode(v: object) -> str:
        return v.decode("utf-8", errors="replace") if isinstance(v, (bytes, np.bytes_)) else str(v)  # type: ignore[union-attr]

    # --- unsteady ---
    grp = hdf.get("Results/Unsteady/Summary")
    if grp is not None:
        try:
            t_raw = float(grp.attrs["Time Solution Went Unstable"])
            time_unstable: float | None = None if np.isnan(t_raw) else t_raw
            compute_ok: bool | None = time_unstable is None
            volume_error_pct: float | None = None
            va = grp.get("Volume Accounting")
            if va is not None:
                volume_error_pct = float(va.attrs["Error Percent"])
            return compute_ok, volume_error_pct, time_unstable
        except (KeyError, TypeError, ValueError):
            pass

    # --- steady ---
    grp = hdf.get("Results/Steady/Summary")
    if grp is not None:
        try:
            solution = _decode(grp.attrs["Solution"])
            compute_ok = "successfully" in solution.lower()
            return compute_ok, None, None
        except (KeyError, TypeError):
            pass

    return None, None, None


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LayerStaleness:
    """Staleness information for one geometry input layer.

    Attributes
    ----------
    name:
        Layer role: ``"terrain"``, ``"land_cover"``, ``"infiltration"``,
        or ``"pct_impervious"``.
    layer_file:
        Resolved path to the layer HDF file (from :attr:`~rivia.hdf.LayerRef.filename`).
    layer_file_exists:
        Whether the layer file is present on disk.
    layer_file_disk_mtime:
        Current OS modification time of the layer file, or ``None`` when
        the file does not exist.
    recorded_file_date:
        Timestamp HEC-RAS recorded at geometry preprocessing time
        (:attr:`~rivia.hdf.LayerRef.file_date`).
    recorded_modified_date:
        For terrain: the terrain HDF mtime recorded per 2D flow area at
        preprocessing, resolved to the most recent across all areas
        (:attr:`~rivia.hdf.LayerRef.date_modified`).  ``None`` for other
        layer types or when not stored.
    delta_seconds:
        ``(layer_file_disk_mtime − recorded_file_date).total_seconds()``.
        Positive means the file has been modified since preprocessing.
        ``None`` when the file does not exist.
    is_stale:
        ``True`` when ``layer_file_disk_mtime > recorded_file_date``,
        i.e. the layer file has changed since the geometry was preprocessed.
    """

    name: str
    layer_file: Path
    layer_file_exists: bool
    layer_file_disk_mtime: dt.datetime | None
    recorded_file_date: dt.datetime
    recorded_modified_date: dt.datetime | None
    delta_seconds: float | None
    is_stale: bool

    def __repr__(self) -> str:
        def _dt(d: dt.datetime | None) -> str:
            return d.strftime("%Y-%m-%d %H:%M:%S") if d else "—"

        stale_tag = "  [STALE]" if self.is_stale else ""
        rows: list[tuple[str, str]] = [
            ("layer file",         str(self.layer_file)),
            ("file exists",        "yes" if self.layer_file_exists else "no"),
            ("recorded file date", _dt(self.recorded_file_date)),
            ("disk mtime",         _dt(self.layer_file_disk_mtime)),
        ]
        if self.delta_seconds is not None:
            sign = "+" if self.delta_seconds >= 0 else ""
            rows.append(("delta (s)", f"{sign}{self.delta_seconds:.0f}"))
        if self.recorded_modified_date is not None:
            rows.append(("recorded mod date", _dt(self.recorded_modified_date)))
        w = max(len(k) for k, _ in rows)
        lines = [f"LayerStaleness  {self.name}{stale_tag}"] + [
            f"  {k:<{w}} : {v}" for k, v in rows
        ]
        return "\n".join(lines)


@dataclass
class GeomVersionStaleness:
    """Comparison of geometry preprocessing timestamps between the plan HDF
    and the geometry HDF on disk.

    If the geometry HDF was re-preprocessed after the plan was last run,
    the plan results were produced with an older geometry.

    Attributes
    ----------
    geom_hdf_path:
        Path to the geometry HDF file (``*.g**.hdf``).
    geom_hdf_exists:
        Whether the geometry HDF file is present on disk.
    geom_hdf_disk_mtime:
        Current OS modification time of the geometry HDF, or ``None``.
    plan_hdf_geom_preprocessed_at:
        ``GeometrySummary.preprocessed_at`` read from the **plan** HDF —
        i.e. when the geometry was preprocessed at the time the plan was run.
        ``None`` when the plan HDF has no ``Geometry`` group (1D-only plans).
    geom_hdf_geom_preprocessed_at:
        ``GeometrySummary.preprocessed_at`` read from the **geometry** HDF.
        ``None`` when the geometry HDF is absent or unreadable.
    geom_complete_in_plan_hdf:
        ``GeometrySummary.complete`` from the plan HDF.  ``None`` when the
        plan HDF has no ``Geometry`` group.
    geom_complete_in_geom_hdf:
        ``GeometrySummary.complete`` from the geometry HDF.  ``None`` when
        the geometry HDF is absent.
    geom_reprocessed_since_run:
        ``True`` when the geometry HDF shows a newer preprocessing time than
        the plan HDF records — meaning the plan was run against an older
        geometry.  ``None`` when either timestamp is unavailable.
    preprocessed_at_delta_seconds:
        ``(geom_hdf_preprocessed_at − plan_hdf_preprocessed_at).total_seconds()``.
        Positive means the geometry HDF is newer.  ``None`` when either
        timestamp is unavailable.
    """

    geom_hdf_path: Path | None
    geom_hdf_exists: bool
    geom_hdf_disk_mtime: dt.datetime | None
    plan_hdf_geom_preprocessed_at: dt.datetime | None
    geom_hdf_geom_preprocessed_at: dt.datetime | None
    geom_complete_in_plan_hdf: bool | None
    geom_complete_in_geom_hdf: bool | None
    geom_reprocessed_since_run: bool | None
    preprocessed_at_delta_seconds: float | None

    def __repr__(self) -> str:
        def _dt(d: dt.datetime | None) -> str:
            return d.strftime("%Y-%m-%d %H:%M:%S") if d else "—"

        def _bool(v: bool | None) -> str:
            return "—" if v is None else ("yes" if v else "no")

        stale_tag = ""
        if self.geom_reprocessed_since_run is True:
            stale_tag = "  [STALE]"
        elif self.geom_reprocessed_since_run is None:
            stale_tag = "  [unknown]"

        rows: list[tuple[str, str]] = [
            ("geom HDF",              str(self.geom_hdf_path) if self.geom_hdf_path else "—"),
            ("geom HDF exists",       _bool(self.geom_hdf_exists)),
            ("geom HDF mtime",        _dt(self.geom_hdf_disk_mtime)),
            ("preprocessed (plan HDF)", _dt(self.plan_hdf_geom_preprocessed_at)),
            ("preprocessed (geom HDF)", _dt(self.geom_hdf_geom_preprocessed_at)),
        ]
        if self.preprocessed_at_delta_seconds is not None:
            sign = "+" if self.preprocessed_at_delta_seconds >= 0 else ""
            rows.append(("delta (s)", f"{sign}{self.preprocessed_at_delta_seconds:.0f}"))
        rows += [
            ("complete (plan HDF)", _bool(self.geom_complete_in_plan_hdf)),
            ("complete (geom HDF)", _bool(self.geom_complete_in_geom_hdf)),
            ("reprocessed since run", _bool(self.geom_reprocessed_since_run)),
        ]
        w = max(len(k) for k, _ in rows)
        lines = [f"GeomVersionStaleness{stale_tag}"] + [
            f"  {k:<{w}} : {v}" for k, v in rows
        ]
        return "\n".join(lines)


@dataclass
class ResultStaleness:
    """File-level and runtime-log staleness indicators for plan results.

    Attributes
    ----------
    plan_hdf_disk_mtime:
        Current OS modification time of the plan HDF file, or ``None``
        when the file is absent.
    plan_text_disk_mtime:
        Current OS modification time of the plan text file (``.p**``).
    plan_text_newer_than_hdf:
        ``True`` when the plan text file is newer than the plan HDF,
        indicating plan settings may have changed since the last run.
        ``None`` when the plan HDF is absent.

    Plan HDF-sourced
    ----------------
    run_window:
        Raw ``"Run Time Window"`` attribute string from the plan HDF
        results group, e.g. ``"01OCT2024 00:00:00 to 02OCT2024 00:00:00"``.
        ``None`` when absent or unreadable.
    run_window_start:
        Parsed start of :attr:`run_window`, or ``None``.
    run_window_end:
        Parsed end of :attr:`run_window`, or ``None``.
    compute_ok:
        ``True`` when the solution finished without instability — for
        unsteady plans ``"Time Solution Went Unstable"`` is NaN; for
        steady plans ``"Solution"`` contains ``"successfully"``.  ``None``
        when the summary group is absent or unreadable.
    volume_error_pct:
        Overall volume balance error as a percentage of total inflow
        (unsteady plans only).  ``None`` for steady plans or when the
        ``Volume Accounting`` group is absent.
    time_unstable:
        Elapsed simulation time in days when the solution went unstable
        (unsteady plans only).  ``None`` when the run was stable or for
        steady plans.

    Runtime log-sourced
    -------------------
    temp_hdf_exists:
        ``True`` when a temporary results HDF (``*.p**.tmp.hdf``) is
        present — indicating a run is currently in progress.
    simulation_start:
        Wall-clock start time parsed from ``"Simulation started at:"`` in
        the runtime log.  ``None`` when absent.
    last_simulation_time:
        Last simulation-time datestamp from iteration lines in the runtime
        log (unsteady / sediment only).  ``None`` for steady plans or when
        no datestamped lines are found.
    run_completion:
        Parsed run outcome from the runtime log text.  ``None`` when the
        plan HDF is absent or has no ``Results/Summary`` group.
    """

    plan_hdf_disk_mtime: dt.datetime | None
    plan_text_disk_mtime: dt.datetime
    plan_text_newer_than_hdf: bool | None
    # Plan HDF-sourced
    run_window: str | None
    run_window_start: dt.datetime | None
    run_window_end: dt.datetime | None
    compute_ok: bool | None
    volume_error_pct: float | None
    time_unstable: float | None
    # Runtime log-sourced
    temp_hdf_exists: bool
    simulation_start: dt.datetime | None
    last_simulation_time: dt.datetime | None
    run_completion: RunCompletion | None

    def is_stale(self) -> tuple[str, ...]:
        """Return the reasons the results are stale, or an empty tuple when complete.

        Staleness is defined as: results are present in the plan HDF **and**
        the runtime log indicates the run did not finish.  Returns an empty
        tuple when the run completed normally, when the plan HDF is absent,
        or when no runtime log is available (indeterminate).

        Returns
        -------
        tuple[str, ...]
            Empty when results are not stale (or indeterminate).  One or more
            of the following when stale:

            * ``"user stopped"`` — user halted the computation.
            * ``"process error: <msg>"`` — ``"Error with program:"`` was found
              in the runtime log.
            * ``"run did not complete"`` — run ended abnormally with no
              specific cause identified.
        """
        if self.plan_hdf_disk_mtime is None:
            return ()
        if self.run_completion is None:
            return ()
        if self.run_completion.finished:
            return ()
        rc = self.run_completion
        reasons: list[str] = []
        if rc.user_stopped:
            reasons.append("user stopped")
        if rc.process_error:
            msg = rc.error_message or "unknown error"
            reasons.append(f"process error: {msg}")
        if not reasons:
            reasons.append("run did not complete")
        reasons.insert(0,"yes")
        return tuple(reasons)

    def __repr__(self) -> str:
        def _dt(d: dt.datetime | None) -> str:
            return d.strftime("%Y-%m-%d %H:%M:%S") if d else "—"

        def _bool(v: bool | None) -> str:
            return "—" if v is None else ("yes" if v else "no")

        rc = self.run_completion
        finished = rc.finished if rc is not None else None

        if self.run_window_start is not None and self.run_window_end is not None:
            duration_h = (
                (self.run_window_end - self.run_window_start).total_seconds() / 3600
            )
            rw_val = (
                f"{_dt(self.run_window_start)} to {_dt(self.run_window_end)}"
                f" ({duration_h:.2f} hr)"
            )
        elif self.run_window is not None:
            rw_val = self.run_window
        else:
            rw_val = "—"

        Row = tuple[str, str]
        top_rows: list[Row] = [
            ("plan HDF mtime",      _dt(self.plan_hdf_disk_mtime)),
            ("plan text mtime",     _dt(self.plan_text_disk_mtime)),
            ("text newer than HDF", _bool(self.plan_text_newer_than_hdf)),
        ]
        hdf_rows: list[Row] = [
            ("run window",          rw_val),
            ("compute ok",          _bool(self.compute_ok)),
            ("volume error %",      f"{self.volume_error_pct:.4f}" if self.volume_error_pct is not None else "—"),
            ("time unstable (d)",   f"{self.time_unstable}" if self.time_unstable is not None else "—"),
        ]
        log_rows: list[Row] = [
            ("run in progress",   _bool(self.temp_hdf_exists)),
            ("simulation start",  _dt(self.simulation_start)),
            ("last sim timestep", _dt(self.last_simulation_time)),
            ("run complete",      _bool(finished)),
        ]
        if rc is not None:
            log_rows.append(("finish message", rc.finish_message or "—"))
            if rc.user_stopped:
                log_rows.append(("user stopped", "yes"))
            if rc.process_error:
                log_rows.append(("process error", rc.error_message or "yes"))

        w = max(len(k) for k, _ in top_rows + hdf_rows + log_rows + [("stale", "")])

        def _row(k: str, v: str) -> str:
            return f"  {k:<{w}} : {v}"

        def _header(label: str) -> str:
            return f"  ({label})"

        stale_reasons = self.is_stale()
        stale_val = ", ".join(stale_reasons) if stale_reasons else "no"

        lines = ["ResultStaleness"]
        lines += [_row(k, v) for k, v in top_rows]
        lines.append(_row("stale", stale_val))
        lines.append(_header("Plan HDF Results"))
        lines += [_row(k, v) for k, v in hdf_rows]
        lines.append(_header("Runtime Log Information"))
        lines += [_row(k, v) for k, v in log_rows]
        return "\n".join(lines)


@dataclass
class PlanStalenessReport:
    """Full staleness diagnostic report for one HEC-RAS plan.

    Returned by :func:`check_plan_staleness` and
    :meth:`~rivia.model.Project.plan_staleness`.

    Attributes
    ----------
    plan_path:
        Path to the plan text file (``.p**``).
    plan_hdf_path:
        Path to the plan HDF file (``.p**.hdf``).
    plan_hdf_exists:
        Whether the plan HDF file is present on disk.
    geometry_layers:
        Per-layer staleness for terrain, land cover, infiltration, and
        percent-impervious.  Only layers present in the plan HDF
        ``GeometrySummary`` are included.  Empty when the plan HDF is
        absent or has no ``Geometry`` group.
    geom_version:
        Geometry preprocessing version comparison between plan HDF and
        geometry HDF.
    results:
        File-level and runtime-log staleness indicators.
    any_layer_stale:
        ``True`` when at least one layer in :attr:`geometry_layers` is
        stale (layer file modified since preprocessing).
    geom_stale:
        ``True`` when :attr:`GeomVersionStaleness.geom_reprocessed_since_run`
        is ``True`` — the geometry HDF has been re-preprocessed since the
        plan was last run.
    run_appears_complete:
        ``True`` when :attr:`ResultStaleness.run_completion` reports a
        finished run.  ``None`` when the plan HDF is absent.
    run_in_progress:
        ``True`` when a temporary results HDF is present (run currently
        in progress).
    """

    plan_path: Path
    plan_hdf_path: Path
    plan_hdf_exists: bool
    geometry_layers: list[LayerStaleness]
    geom_version: GeomVersionStaleness
    results: ResultStaleness
    any_layer_stale: bool
    geom_stale: bool
    run_appears_complete: bool | None
    run_in_progress: bool

    def __repr__(self) -> str:
        def _bool(v: bool | None) -> str:
            return "—" if v is None else ("yes" if v else "no")

        def _indent(text: str, prefix: str = "  ") -> str:
            return "\n".join(prefix + line for line in text.splitlines())

        lines = [f"PlanStalenessReport  {self.plan_path.name}"]
        lines.append(f"  plan HDF exists         : {_bool(self.plan_hdf_exists)}")
        lines.append(f"  run complete            : {_bool(self.run_appears_complete)}")
        lines.append(f"  run in progress         : {_bool(self.run_in_progress)}")
        lines.append(f"  geom stale              : {_bool(self.geom_stale)}")
        lines.append(f"  any layer stale         : {_bool(self.any_layer_stale)}")
        lines.append(f"  plan text newer than HDF: {_bool(self.results.plan_text_newer_than_hdf)}")

        lines.append("")
        lines.append(_indent(repr(self.results)))

        lines.append("")
        lines.append(_indent(repr(self.geom_version)))

        if self.geometry_layers:
            lines.append("")
            lines.append("  Geometry layers")
            for ls in self.geometry_layers:
                lines.append(_indent(repr(ls), prefix="    "))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_plan_staleness(plan_file: str | Path) -> PlanStalenessReport:
    """Return a full staleness diagnostic report for one HEC-RAS plan.

    Opens the plan text file, plan HDF, and geometry HDF (when present) and
    assembles a :class:`PlanStalenessReport` with timestamps and staleness
    flags for geometry layers, geometry version, and simulation results.

    No COM connection is required.  The function is safe to call on plans
    that have never been run (missing HDF files are handled gracefully).

    Parameters
    ----------
    plan_file:
        Path to the plan text file, e.g. ``"KagmanWatershed.p01"``.

    Returns
    -------
    PlanStalenessReport

    Raises
    ------
    FileNotFoundError
        If *plan_file* does not exist.

    Examples
    --------
    ::

        from rivia.hdf.staleness import check_plan_staleness

        report = check_plan_staleness("KagmanWatershed.p01")
        print(report)
        if not report.run_appears_complete:
            print("Results are incomplete or from a prior run.")
    """
    plan_path = Path(plan_file)
    if not plan_path.is_file():
        raise FileNotFoundError(f"Plan file not found: {plan_path}")

    # --- derive paths -------------------------------------------------------
    plan_hdf_path = Path(str(plan_path) + ".hdf")
    temp_hdf_path = Path(str(plan_path) + ".tmp.hdf")

    geom_ext = _plan_geom_ext(plan_path)
    if geom_ext is not None:
        geom_path = plan_path.with_suffix(f".{geom_ext}")
        geom_hdf_path: Path | None = Path(str(geom_path) + ".hdf")
    else:
        geom_hdf_path = None

    # --- disk mtimes --------------------------------------------------------
    plan_hdf_exists = plan_hdf_path.is_file()
    plan_hdf_mtime = _os_mtime(plan_hdf_path)
    plan_text_mtime = _os_mtime(plan_path)  # plan file always exists (checked above)
    geom_hdf_exists = geom_hdf_path is not None and geom_hdf_path.is_file()
    geom_hdf_mtime = _os_mtime(geom_hdf_path) if geom_hdf_path is not None else None
    temp_hdf_exists = temp_hdf_path.is_file()

    # --- plan text staleness vs HDF -----------------------------------------
    if plan_hdf_mtime is not None and plan_text_mtime is not None:
        plan_text_newer = plan_text_mtime > plan_hdf_mtime
    else:
        plan_text_newer = None

    # --- read plan HDF ------------------------------------------------------
    geometry_layers: list[LayerStaleness] = []
    plan_geom_preprocessed_at: dt.datetime | None = None
    geom_complete_in_plan: bool | None = None
    run_window_raw: str | None = None
    run_completion: RunCompletion | None = None
    _simulation_start: dt.datetime | None = None
    _compute_ok: bool | None = None
    _volume_error_pct: float | None = None
    _time_unstable: float | None = None

    if plan_hdf_exists:
        with Geometry(plan_hdf_path) as g:
            # geometry summary (Geometry group may be absent on unrun or 1D plans)
            try:
                gs = g.geometry_summary()
                plan_geom_preprocessed_at = gs.preprocessed_at
                geom_complete_in_plan = gs.complete
                for name, ref in [
                    ("terrain",       gs.terrain),
                    ("land_cover",    gs.land_cover),
                    ("infiltration",  gs.infiltration),
                    ("pct_impervious", gs.pct_impervious),
                ]:
                    if ref is None:
                        continue
                    lf = ref.filename
                    lf_mtime = _os_mtime(lf)
                    if lf_mtime is not None:
                        delta: float | None = (lf_mtime - ref.file_date).total_seconds()
                        is_stale = lf_mtime > ref.file_date
                    else:
                        delta = None
                        is_stale = False
                    geometry_layers.append(LayerStaleness(
                        name=name,
                        layer_file=lf,
                        layer_file_exists=lf.is_file(),
                        layer_file_disk_mtime=lf_mtime,
                        recorded_file_date=ref.file_date,
                        recorded_modified_date=ref.date_modified,
                        delta_seconds=delta,
                        is_stale=is_stale,
                    ))
            except KeyError:
                pass  # no Geometry group — 1D-only plan or never preprocessed

            # run window, runtime log, and compute-summary scalars via h5py handle
            hdf = g._hdf
            run_window_raw = _read_run_window(hdf)
            log = _runtime_log_from_hdf(hdf)
            if log is not None:
                run_completion = log.run_completion()
                _simulation_start = log.simulation_start
            _compute_ok, _volume_error_pct, _time_unstable = _read_compute_scalars(hdf)

    # --- read geometry HDF --------------------------------------------------
    geom_hdf_preprocessed_at: dt.datetime | None = None
    geom_complete_in_geom: bool | None = None

    if geom_hdf_exists and geom_hdf_path is not None:
        try:
            with Geometry(geom_hdf_path) as g:
                gs2 = g.geometry_summary()
            geom_hdf_preprocessed_at = gs2.preprocessed_at
            geom_complete_in_geom = gs2.complete
        except (KeyError, FileNotFoundError):
            pass

    # --- parse run window ---------------------------------------------------
    run_window_start: dt.datetime | None = None
    run_window_end: dt.datetime | None = None
    if run_window_raw is not None:
        parsed = _parse_run_window(run_window_raw)
        if parsed is not None:
            run_window_start, run_window_end = parsed

    # --- geometry version staleness -----------------------------------------
    if (
        plan_geom_preprocessed_at is not None
        and geom_hdf_preprocessed_at is not None
    ):
        delta_pp = (
            geom_hdf_preprocessed_at - plan_geom_preprocessed_at
        ).total_seconds()
        geom_reprocessed = geom_hdf_preprocessed_at > plan_geom_preprocessed_at
    else:
        delta_pp = None
        geom_reprocessed = None

    geom_version = GeomVersionStaleness(
        geom_hdf_path=geom_hdf_path,
        geom_hdf_exists=geom_hdf_exists,
        geom_hdf_disk_mtime=geom_hdf_mtime,
        plan_hdf_geom_preprocessed_at=plan_geom_preprocessed_at,
        geom_hdf_geom_preprocessed_at=geom_hdf_preprocessed_at,
        geom_complete_in_plan_hdf=geom_complete_in_plan,
        geom_complete_in_geom_hdf=geom_complete_in_geom,
        geom_reprocessed_since_run=geom_reprocessed,
        preprocessed_at_delta_seconds=delta_pp,
    )

    results = ResultStaleness(
        plan_hdf_disk_mtime=plan_hdf_mtime,
        plan_text_disk_mtime=plan_text_mtime,  # type: ignore[arg-type]
        plan_text_newer_than_hdf=plan_text_newer,
        run_window=run_window_raw,
        run_window_start=run_window_start,
        run_window_end=run_window_end,
        compute_ok=_compute_ok,
        volume_error_pct=_volume_error_pct,
        time_unstable=_time_unstable,
        temp_hdf_exists=temp_hdf_exists,
        simulation_start=_simulation_start,
        last_simulation_time=(
            run_completion.last_simulation_time if run_completion is not None else None
        ),
        run_completion=run_completion,
    )

    # --- synthesised flags --------------------------------------------------
    any_layer_stale = any(ls.is_stale for ls in geometry_layers)
    geom_stale = geom_reprocessed is True
    run_appears_complete: bool | None = (
        run_completion.finished if run_completion is not None else None
    )

    return PlanStalenessReport(
        plan_path=plan_path,
        plan_hdf_path=plan_hdf_path,
        plan_hdf_exists=plan_hdf_exists,
        geometry_layers=geometry_layers,
        geom_version=geom_version,
        results=results,
        any_layer_stale=any_layer_stale,
        geom_stale=geom_stale,
        run_appears_complete=run_appears_complete,
        run_in_progress=temp_hdf_exists,
    )
