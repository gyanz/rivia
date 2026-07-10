"""Read/write HEC-RAS unsteady flow files (.u**).

:class:`UnsteadyFlow` — structured editor.  Boundary conditions are
parsed into typed dataclass objects and may be sorted by river station.
``save()`` reconstructs the boundary section from the objects; trailing
meteorological / non-Newtonian lines are still written verbatim.

Convention
----------
``get_*`` methods return ``None`` when the requested item is not found.
``set_*`` methods raise :exc:`KeyError` when the target does not exist.
"""

from __future__ import annotations

import copy
import datetime as dt
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Literal

import pandas as pd

from rivia.utils import (
    format_interval_strict,
    parse_hec_datetime,
    parse_interval,
    parse_interval_strict,
)

logger = logging.getLogger("rivia.model")


def _parse_window(
    use_fixed_start: bool,
    fixed_start: str,
    interval: str,
    n_values: int,
) -> tuple[dt.datetime, dt.datetime] | None:
    """Return ``(start, end)`` datetimes for a boundary with a fixed start.

    Returns ``None`` when *use_fixed_start* is ``False``.

    Parameters
    ----------
    use_fixed_start:
        Mirror of the boundary's ``use_fixed_start`` flag.
    fixed_start:
        ``"DDMONYYYY,HHMM"`` or ``"DDMONYYYY,HHMMSS"`` string stored on the
        boundary (e.g. ``"01JAN2020,0600"``).
    interval:
        HEC-RAS interval string (e.g. ``"5MIN"``); parsed by
        :func:`rivia.utils.parse_interval`.
    n_values:
        Number of time-series values; determines the end date.
    """
    if not use_fixed_start:
        return None
    start = parse_hec_datetime(fixed_start)
    if n_values == 0:
        return start, start
    end = start + (n_values - 1) * parse_interval(interval)
    return start, end

# Scalar or sequence accepted by all set_* methods.
# A bare float/int is broadcast to fill the current time-series length.
_Values = list[float | int] | float | int


def _coerce_values(values: _Values, count: int) -> list[float]:
    """Return *values* as a list of floats of length *count*.

    If *values* is a scalar it is broadcast to *count* elements.  If *values*
    is already a sequence its length is used as-is (``count`` is ignored).
    """
    if isinstance(values, (int, float)):
        return [float(values)] * count
    return [float(v) for v in values]


def _resolve_interval(value: str | float | int | dt.timedelta) -> str:
    """Return *value* as a canonical, HEC-RAS-dropdown-valid interval string.

    Accepts a HEC-RAS interval string (validated and re-canonicalized via
    :func:`rivia.utils.parse_interval_strict`), a :class:`datetime.timedelta`,
    or a bare ``int``/``float`` (seconds) — all resolved via
    :func:`rivia.utils.format_interval_strict`.

    Raises
    ------
    ValueError
        *value* is a string that does not match a HEC-RAS dropdown interval,
        or a duration that cannot be expressed as one in any unit.
    """
    if isinstance(value, str):
        value = parse_interval_strict(value)
    return format_interval_strict(value)


def _expand_steps(
    steps: dict[float, float], n_periods: int, interval_minutes: float
) -> list[float]:
    """Return *n_periods* per-timestep values by holding each step until the next.

    Used by :meth:`_TimeSeriesBoundary.set_time_series_window` to expand a
    ``{elapsed_minutes: value}`` mapping into a full timestep-by-timestep
    series: the value at a given timestep is whichever key is the largest
    one not greater than that timestep's elapsed minutes.

    Parameters
    ----------
    steps:
        Mapping of elapsed minutes since the window start to the value held
        from that point until the next key. Must contain a ``0`` key.
    n_periods:
        Number of evenly spaced timesteps to generate (inclusive of both
        window endpoints).
    interval_minutes:
        Spacing between timesteps, in minutes.

    Raises
    ------
    ValueError
        *steps* is empty, has a negative key, or has no ``0`` key.
    """
    if not steps:
        raise ValueError("steps dict must not be empty.")
    if any(k < 0 for k in steps):
        raise ValueError("steps dict keys (elapsed minutes) must be non-negative.")
    if 0 not in steps:
        raise ValueError(
            "steps dict must define a value at elapsed minute 0 (the start "
            "of the window)."
        )
    breakpoints = sorted(steps)
    values = []
    current = breakpoints[0]
    bp_iter = iter(breakpoints)
    next_bp = next(bp_iter)
    for i in range(n_periods):
        elapsed = i * interval_minutes
        while next_bp is not None and next_bp <= elapsed:
            current = next_bp
            next_bp = next(bp_iter, None)
        values.append(float(steps[current]))
    return values


def _format_fixed_start(d: dt.datetime) -> str:
    """Return *d* formatted as a HEC-RAS ``fixed_start`` string (``"DDMONYYYY,HHMM"``).

    Distinct from :func:`rivia.utils.format_hec_datetime`, which uses a
    space/colon format for a different HEC-RAS context (HDF attributes,
    runtime log timestamps).
    """
    return f"{d.strftime('%d%b%Y').upper()},{d.strftime('%H%M')}"


def _infer_interval(index: pd.DatetimeIndex) -> str:
    """Return the HEC-RAS interval string for an evenly spaced *index*.

    Raises
    ------
    ValueError
        *index* has fewer than 2 points, is not evenly spaced, or the
        spacing does not correspond to a HEC-RAS dropdown interval (see
        :func:`rivia.utils.format_interval_strict`).
    """
    if len(index) < 2:
        raise ValueError(
            "Cannot infer interval from a series with fewer than 2 points; "
            "pass interval= explicitly."
        )
    diffs = index.to_series().diff().dropna().unique()
    if len(diffs) != 1:
        raise ValueError(
            "data.index is not evenly spaced; cannot infer a single interval. "
            "Resample first or pass interval= explicitly."
        )
    return format_interval_strict(pd.Timedelta(diffs[0]).to_pytimedelta())


# ---------------------------------------------------------------------------
# Formatting helpers (shared)
# ---------------------------------------------------------------------------

_COL_WIDTH = 8
_COLS_PER_ROW = 10


def _fit_width(value: float, width: int = _COL_WIDTH) -> str:
    """Right-justify *value* inside *width* characters.

    Tries integer, then progressively fewer decimal places, then scientific
    notation.  Truncates as last resort.
    """
    # Integer shortcut
    if isinstance(value, int) or (
        isinstance(value, float)
        and value == int(value)
        and len(str(int(value))) <= width
    ):
        s = str(int(value))
        if len(s) <= width:
            return s.rjust(width)

    s = repr(value)
    if len(s) <= width:
        return s.rjust(width)

    fv = float(value)
    for decimals in range(6, -1, -1):
        s = f"{fv:.{decimals}f}"
        if len(s) <= width:
            return s.rjust(width)

    for decimals in range(2, -1, -1):
        s = f"{fv:.{decimals}E}"
        if len(s) <= width:
            return s.rjust(width)

    return repr(value)[:width]


def _format_data_block(
    values: list[float], cols: int = _COLS_PER_ROW, width: int = _COL_WIDTH
) -> list[str]:
    """Return a list of fixed-width data lines (no trailing newline)."""
    lines: list[str] = []
    for i in range(0, len(values), cols):
        chunk = values[i : i + cols]
        lines.append("".join(_fit_width(v, width) for v in chunk))
    return lines


def _parse_data_block(
    lines: list[str], count: int, width: int = _COL_WIDTH
) -> list[float]:
    """Parse *count* fixed-width values from *lines*."""
    values: list[float] = []
    for line in lines:
        pos = 0
        while pos < len(line) and len(values) < count:
            token = line[pos : pos + width].strip()
            if token:
                try:
                    values.append(float(token))
                except ValueError:
                    values.append(0.0)
            pos += width
    return values[:count]


def _data_line_count(n: int, cols: int = _COLS_PER_ROW) -> int:
    """Number of data lines needed for *n* values at *cols* per line."""
    return ceil(n / cols) if n > 0 else 0


# ---------------------------------------------------------------------------
# Shared boundary dataclasses
# ---------------------------------------------------------------------------


@dataclass
class InitialFlowLoc:
    """Initial flow at a river / reach / station."""

    river: str
    reach: str
    river_station: str
    flow: float

    @classmethod
    def _from_raw(cls, raw: str) -> "InitialFlowLoc":
        parts = raw.split(",")
        return cls(
            river=parts[0].strip() if len(parts) > 0 else "",
            reach=parts[1].strip() if len(parts) > 1 else "",
            river_station=parts[2].strip() if len(parts) > 2 else "",
            flow=float(parts[3].strip()) if len(parts) > 3 else 0.0,
        )

    def _to_raw(self) -> str:
        return f"{self.river:16},{self.reach:16},{self.river_station:8},{self.flow}"


@dataclass
class InitialStorageElev:
    """Initial water surface elevation for a storage area."""

    name: str
    elevation: float

    @classmethod
    def _from_raw(cls, raw: str) -> "InitialStorageElev":
        parts = raw.split(",")
        return cls(
            name=parts[0].strip() if parts else "",
            elevation=float(parts[1].strip()) if len(parts) > 1 else 0.0,
        )

    def _to_raw(self) -> str:
        return f"{self.name},{self.elevation}"


@dataclass
class InitialRainfallRunoffElev:
    """Initial water surface elevation for a reservoir / RRR."""

    river: str
    reach: str
    river_station: str
    elevation: float

    @classmethod
    def _from_raw(cls, raw: str) -> "InitialRainfallRunoffElev":
        parts = raw.split(",")
        return cls(
            river=parts[0].strip() if len(parts) > 0 else "",
            reach=parts[1].strip() if len(parts) > 1 else "",
            river_station=parts[2].strip() if len(parts) > 2 else "",
            elevation=float(parts[3].strip()) if len(parts) > 3 else 0.0,
        )

    def _to_raw(self) -> str:
        return (
            f"{self.river:16},{self.reach:16},{self.river_station:8},{self.elevation}"
        )


# ---- boundary base ---------------------------------------------------------


@dataclass
class _Boundary:
    """Base class for all boundary condition types."""

    river: str
    reach: str
    river_station: str
    # Raw comma-separated tail of the Boundary Location= line after the first
    # three fields (preserved verbatim for roundtrip fidelity).
    _location_tail: str = field(default="", repr=False)

    def _location_line(self) -> str:
        return (
            f"Boundary Location={self.river:16},{self.reach:16},"
            f"{self.river_station:8},{self._location_tail}"
        )

    def location(
        self, *, rs_float: bool = False
    ) -> tuple[str, str, str] | tuple[str, str, float]:
        """Return ``(river, reach, river_station)``.

        Parameters
        ----------
        rs_float:
            If ``True``, river_station is returned as ``float``
            (strips trailing ``'*'``); otherwise as ``str`` (default).
        """
        if rs_float:
            return (self.river, self.reach, self._rs_float())
        return (self.river, self.reach, self.river_station)

    def _rs_float(self) -> float:
        """River station as float for sorting (strips trailing '*')."""
        try:
            return float(self.river_station.rstrip("*").strip())
        except ValueError:
            return float("-inf")


@dataclass
class _TimeSeriesBoundary(_Boundary):
    """Base for boundary types carrying an inline value time series.

    Shared by :class:`FlowHydrograph`, :class:`LateralInflow`, and
    :class:`StageHydrograph`.  :class:`RatingCurve` is *not* a subclass of
    this: it stores a static stage/flow lookup table (``pairs``), not a
    time-indexed signal.
    """

    interval: str = "1HOUR"
    values: list[float] = field(default_factory=list)
    use_dss: bool = False
    use_fixed_start: bool = False
    fixed_start: str = ","
    # Extra lines between the standard fields and next Boundary Location
    # that we don't model explicitly (e.g. CWMS InputPosition).
    _extra_lines: list[str] = field(default_factory=list, repr=False)

    @property
    def window(self) -> tuple[dt.datetime, dt.datetime] | None:
        """Return ``(start, end)`` as Python datetimes, or ``None``.

        ``None`` is returned when ``use_fixed_start`` is ``False``.
        The end date is ``start + len(values) * parse_interval(interval)``.
        """
        return _parse_window(
            self.use_fixed_start, self.fixed_start, self.interval, len(self.values)
        )

    def time_series(
        self, start_datetime: dt.datetime | None = None, raw_values: bool = True
    ) -> pd.Series:
        """Return :attr:`values` as a :class:`pandas.Series` indexed by datetime.

        Parameters
        ----------
        start_datetime:
            Start of the time series.  Required when :attr:`use_fixed_start`
            is ``False``.  Ignored (with a warning) when
            :attr:`use_fixed_start` is ``True``, since :attr:`fixed_start`
            is authoritative in that case.
        raw_values:
            If ``True`` (default), return :attr:`values` exactly as stored.
            If ``False``, apply ``QMin``/``QMult`` to each value as
            ``max(value * q_mult, q_min)`` (a missing/``None`` ``q_mult`` is
            treated as ``1.0``; a missing/``None`` ``q_min`` skips the floor).
            Has no effect on boundary types without ``q_min``/``q_mult``
            fields (e.g. :class:`StageHydrograph`), which have nothing to
            apply and always return raw values.

        Returns
        -------
        pd.Series
            Values indexed by a :class:`pandas.DatetimeIndex` built from the
            resolved start datetime and :attr:`interval`.  Empty if
            :attr:`values` is empty.

        Raises
        ------
        ValueError
            *start_datetime* was not provided and :attr:`use_fixed_start`
            is ``False``.
        NotImplementedError
            :attr:`use_dss` is ``True`` — values live in an external DSS
            file and are not available inline.
        """
        if self.use_dss:
            raise NotImplementedError(
                "values are stored in an external DSS file (use_dss=True); "
                "time_series() only supports inline values."
            )
        if self.use_fixed_start:
            if start_datetime is not None:
                logger.warning(
                    "start_datetime is ignored because use_fixed_start is "
                    "True; using fixed_start=%r instead.",
                    self.fixed_start,
                )
            start = parse_hec_datetime(self.fixed_start)
        else:
            if start_datetime is None:
                raise ValueError(
                    "start_datetime is required when use_fixed_start is False."
                )
            start = start_datetime
        values = self.values
        if not raw_values:
            q_mult = getattr(self, "q_mult", None)
            q_min = getattr(self, "q_min", None)
            mult = q_mult if q_mult is not None else 1.0
            values = [v * mult for v in values]
            if q_min is not None:
                values = [max(v, q_min) for v in values]
        if not values:
            return pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
        index = pd.date_range(
            start=start, periods=len(values), freq=parse_interval(self.interval)
        )
        return pd.Series(values, index=index)

    def set_time_series(
        self,
        data: pd.Series | Sequence[float | int] | float | int,
        *,
        interval: str | float | int | dt.timedelta | None = None,
        start_datetime: dt.datetime | None = None,
        use_fixed_start: bool | None = None,
        q_min: float = 0.0,
        q_mult: float = 1.0,
    ) -> None:
        """Replace this boundary's time series, in place.

        More flexible than ``UnsteadyFlow.set_flow_hydrograph``/
        ``set_lateral_inflow`` (and their ``_at`` variants), which only ever
        replace :attr:`values`.  This method can also change :attr:`interval`,
        switch between a fixed start date and the plan's simulation start,
        and accept a :class:`pandas.Series` directly.

        Parameters
        ----------
        data:
            New time series. One of:

            * :class:`pandas.Series` with a :class:`pandas.DatetimeIndex` —
              :attr:`values`, :attr:`interval`, and the fixed start are all
              inferred from the series (unless overridden below). The index
              must be evenly spaced.
            * a sequence of numbers — used as-is; pass *interval* and/or
              *start_datetime* explicitly to change those fields, otherwise
              the existing values are kept.
            * a scalar — broadcast to the current length of :attr:`values`
              (same convention as the ``UnsteadyFlow.set_*`` methods).
        interval:
            New interval. One of a HEC-RAS interval string (e.g.
            ``"15MIN"``), a :class:`datetime.timedelta`, or a bare
            ``int``/``float`` interpreted as seconds. In every case the
            resolved duration is strictly validated and re-canonicalized via
            :func:`rivia.utils.parse_interval_strict` /
            :func:`rivia.utils.format_interval_strict` — only values
            HEC-RAS's own interval dropdown offers are accepted (e.g.
            ``"7HOUR"`` or ``dt.timedelta(hours=5)`` raise ``ValueError``).
            Mutually exclusive with passing a :class:`pandas.Series` for
            *data* — the series' own index spacing is always used in that
            case; pass a plain sequence instead if you want to combine
            external values with an explicit interval.
        start_datetime:
            New fixed start. Implies ``use_fixed_start=True`` unless
            *use_fixed_start* is also passed. Mutually exclusive with
            passing a :class:`pandas.Series` for *data* — pass a plain
            sequence instead if you want to combine external values with an
            explicit start.
        use_fixed_start:
            Explicitly set :attr:`use_fixed_start`. Defaults to ``True``
            automatically whenever a start is resolved (from *data* or
            *start_datetime*); pass ``False`` to keep the new values/interval
            but fall back to the plan's simulation start instead.
        q_min:
            New ``QMin`` value. Always applied, overwriting any previously
            set value (same convention as ``UnsteadyFlow.set_flow_hydrograph``
            / ``set_lateral_inflow``). Has no effect on boundary types
            without a ``q_min`` field (e.g. :class:`StageHydrograph`).
        q_mult:
            New ``QMult`` value. Always applied, overwriting any previously
            set value. Has no effect on boundary types without a ``q_mult``
            field (e.g. :class:`StageHydrograph`).

        Notes
        -----
        Always clears :attr:`use_dss` to ``False``: a DSS-linked boundary
        ignores inline values entirely, so supplying new inline data means
        the caller wants them used.

        Raises
        ------
        ValueError
            *data* is a :class:`pandas.Series` with a non-uniform index,
            *data* is a Series and *start_datetime* and/or *interval* are
            also given, or *interval* does not resolve to a HEC-RAS
            dropdown interval.
        TypeError
            *data* is a :class:`pandas.Series` without a
            :class:`pandas.DatetimeIndex`.
        """
        resolved_start: dt.datetime | None = None
        resolved_interval: str | None = (
            _resolve_interval(interval) if interval is not None else None
        )

        if isinstance(data, pd.Series):
            if start_datetime is not None:
                raise ValueError(
                    "start_datetime cannot be combined with a pandas.Series "
                    "for data; the series' own DatetimeIndex is "
                    "authoritative. Pass a plain sequence for data if you "
                    "want to combine external values with an explicit "
                    "start_datetime."
                )
            if resolved_interval is not None:
                raise ValueError(
                    "interval cannot be combined with a pandas.Series for "
                    "data; the series' own index spacing is always used. "
                    "Pass a plain sequence for data if you want to combine "
                    "external values with an explicit interval."
                )
            if not isinstance(data.index, pd.DatetimeIndex):
                raise TypeError("data.index must be a pandas.DatetimeIndex.")
            values = [float(v) for v in data.to_numpy()]
            resolved_interval = _infer_interval(data.index)
            resolved_start = data.index[0].to_pydatetime()
        elif isinstance(data, (int, float)):
            values = _coerce_values(data, len(self.values))
        else:
            values = [float(v) for v in data]

        if start_datetime is not None:
            resolved_start = start_datetime

        # Resolve everything that can still raise (formatting, float casts)
        # before mutating self, so a failure here leaves the instance
        # untouched instead of partially updated.
        resolved_fixed_start = (
            _format_fixed_start(resolved_start) if resolved_start is not None else None
        )
        if resolved_fixed_start is not None and use_fixed_start is None:
            use_fixed_start = True
        resolved_q_min = float(q_min) if hasattr(self, "q_min") else None
        resolved_q_mult = float(q_mult) if hasattr(self, "q_mult") else None

        # Nothing below this point can raise.
        self.values = values
        self.use_dss = False
        if resolved_interval is not None:
            self.interval = resolved_interval
        if resolved_fixed_start is not None:
            self.fixed_start = resolved_fixed_start
        if use_fixed_start is not None:
            self.use_fixed_start = use_fixed_start
        if resolved_q_min is not None:
            self.q_min = resolved_q_min
        if resolved_q_mult is not None:
            self.q_mult = resolved_q_mult

    def set_time_series_window(
        self,
        window: tuple[dt.datetime, dt.datetime],
        interval: str | float | int | dt.timedelta,
        data: float | int | Sequence[float | int] | dict[float, float],
        *,
        q_min: float = 0.0,
        q_mult: float = 1.0,
    ) -> None:
        """Replace this boundary's time series over an explicit ``(start, end)`` window.

        Convenience wrapper around :meth:`set_time_series` for specifying a
        fixed start and end time directly, instead of letting the end be
        inferred from the length of *data*. Also accepts a ``dict`` of step
        values keyed by elapsed minutes, for building a step (piecewise
        constant) hydrograph without precomputing every timestep by hand.

        Parameters
        ----------
        window:
            ``(start, end)`` datetimes, inclusive of both endpoints. *end*
            must be reachable from *start* by a whole number of *interval*
            steps.
        interval:
            Spacing between timesteps. One of a HEC-RAS interval string
            (e.g. ``"15MIN"``), a :class:`datetime.timedelta`, or a bare
            ``int``/``float`` interpreted as seconds — resolved the same way
            as :meth:`set_time_series`'s *interval* (validated and
            re-canonicalized via :func:`rivia.utils.parse_interval_strict` /
            :func:`rivia.utils.format_interval_strict`).
        data:
            Values to fill the window with. One of:

            * a scalar — broadcast to every timestep in the window.
            * a sequence of numbers — used as-is; its length must exactly
              match the number of timesteps implied by *window* and
              *interval*.
            * a ``dict[float, float]`` mapping elapsed minutes since *start*
              to a step value, e.g. ``{0: 20, 60: 50}`` holds ``20`` from
              the start of the window until minute 60, then ``50`` for the
              rest of the window. Must contain a ``0`` key.
        q_min:
            New ``QMin`` value. Always applied, overwriting any previously
            set value. Has no effect on boundary types without a ``q_min``
            field (e.g. :class:`StageHydrograph`).
        q_mult:
            New ``QMult`` value. Always applied, overwriting any previously
            set value. Has no effect on boundary types without a ``q_mult``
            field (e.g. :class:`StageHydrograph`).

        Raises
        ------
        ValueError
            *end* is not after *start*; the window duration is not evenly
            divisible by *interval*; *data* is a sequence whose length
            doesn't match the number of timesteps implied by *window* and
            *interval*; *data* is a dict that is empty, has a negative key,
            or has no ``0`` key; or *interval* does not resolve to a
            HEC-RAS dropdown interval.

        Examples
        --------
        >>> fh.set_time_series_window(
        ...     (dt.datetime(2021, 1, 1), dt.datetime(2021, 1, 1, 2)),
        ...     "15MIN",
        ...     {0: 20, 60: 50},
        ... )
        """
        start, end = window
        if end <= start:
            raise ValueError(
                f"window end ({end}) must be after window start ({start})."
            )

        resolved_interval = _resolve_interval(interval)
        interval_td = parse_interval_strict(resolved_interval)

        duration = end - start
        if duration % interval_td != dt.timedelta(0):
            raise ValueError(
                f"window duration {duration} is not evenly divisible by "
                f"interval {resolved_interval!r}."
            )
        n_periods = duration // interval_td + 1

        if isinstance(data, dict):
            interval_minutes = interval_td.total_seconds() / 60
            values = _expand_steps(data, n_periods, interval_minutes)
        elif isinstance(data, (int, float)):
            values = _coerce_values(data, n_periods)
        else:
            values = [float(v) for v in data]
            if len(values) != n_periods:
                raise ValueError(
                    f"data has {len(values)} values but window/interval "
                    f"implies {n_periods} timesteps."
                )

        self.set_time_series(
            values,
            interval=resolved_interval,
            start_datetime=start,
            use_fixed_start=True,
            q_min=q_min,
            q_mult=q_mult,
        )

    def reset_values(
        self,
        data: float | int | Sequence[float | int] | dict[float, float],
        *,
        q_min: float = 0.0,
        q_mult: float = 1.0,
    ) -> None:
        """Replace this boundary's values in place, keeping window and interval fixed.

        Unlike :meth:`set_time_series_window`, this never changes
        :attr:`interval`, :attr:`fixed_start`, or :attr:`use_fixed_start` —
        only :attr:`values` (and, where present, ``q_min``/``q_mult``) are
        replaced, using the existing number of timesteps.

        Parameters
        ----------
        data:
            New values. One of:

            * a scalar — broadcast to the existing number of timesteps.
            * a sequence of numbers — used as-is; its length must exactly
              match the existing number of timesteps (since the window is
              not changing).
            * a ``dict[float, float]`` mapping elapsed minutes since the
              start of the existing window to a step value, e.g.
              ``{0: 20, 60: 50}`` (same step/hold semantics as
              :meth:`set_time_series_window`). Must contain a ``0`` key.
        q_min:
            New ``QMin`` value. Always applied, overwriting any previously
            set value. Has no effect on boundary types without a ``q_min``
            field (e.g. :class:`StageHydrograph`).
        q_mult:
            New ``QMult`` value. Always applied, overwriting any previously
            set value. Has no effect on boundary types without a ``q_mult``
            field (e.g. :class:`StageHydrograph`).

        Raises
        ------
        ValueError
            :attr:`values` is currently empty (there is no existing window
            length to reset into); *data* is a sequence whose length
            doesn't match the existing number of timesteps; or *data* is a
            dict that is empty, has a negative key, or has no ``0`` key.

        Examples
        --------
        >>> fh.values
        [1.0, 2.0, 3.0]
        >>> fh.reset_values(0.0)
        >>> fh.values
        [0.0, 0.0, 0.0]
        """
        n = len(self.values)
        if n == 0:
            raise ValueError(
                "Cannot reset an empty time series; there is no existing "
                "window length to reset into. Use set_time_series_window() "
                "to establish one."
            )
        if isinstance(data, dict):
            interval_minutes = parse_interval(self.interval).total_seconds() / 60
            values = _expand_steps(data, n, interval_minutes)
        elif isinstance(data, (int, float)):
            values = _coerce_values(data, n)
        else:
            values = [float(v) for v in data]
            if len(values) != n:
                raise ValueError(
                    f"data has {len(values)} values but this boundary has "
                    f"{n} existing timesteps; length must match since "
                    "window/interval are not being changed."
                )
        self.set_time_series(values, q_min=q_min, q_mult=q_mult)

    def resize_window(
        self,
        window: tuple[dt.datetime | None, dt.datetime | None],
        *,
        start_datetime: dt.datetime | None = None,
    ) -> None:
        """Clip and/or extend this boundary's time series to a new ``(start, end)``.

        Unlike :meth:`set_time_series_window`, this does not replace the
        data — it reshapes the *existing* :attr:`values` to cover a new
        window: timesteps inside both the old and new window are kept
        as-is, timesteps dropped from either end are clipped, and timesteps
        added beyond either end are filled by repeating the existing edge
        value (:attr:`values`'s first element for the front, last element
        for the back). :attr:`interval` and (where present) ``q_min``/
        ``q_mult`` are left unchanged.

        Parameters
        ----------
        window:
            ``(new_start, new_end)``. Either side may be ``None`` to leave
            that side untouched. Both, when given, must land exactly on the
            existing interval grid (i.e. be reachable from the current
            start by a whole number of :attr:`interval` steps).
        start_datetime:
            The time series' *current* start — needed only to interpret
            *window* when :attr:`use_fixed_start` is ``False`` (a non-fixed
            boundary has no start of its own; it implicitly starts at the
            plan's simulation start). Ignored (with a warning) when
            :attr:`use_fixed_start` is ``True``, since :attr:`fixed_start`
            is authoritative in that case — same convention as
            :meth:`time_series`.

        Raises
        ------
        ValueError
            *start_datetime* was not provided and :attr:`use_fixed_start`
            is ``False``; :attr:`values` is empty; the effective new end is
            before the effective new start; or either window bound is not
            reachable from the current start by a whole number of
            :attr:`interval` steps.
        NotImplementedError
            :attr:`use_dss` is ``True`` — values live in an external DSS
            file and there is nothing inline to reshape.

        Examples
        --------
        >>> fh.values
        [10.0, 20.0, 30.0]
        >>> fh.resize_window((dt.datetime(2020, 12, 31, 23), None))
        >>> fh.values
        [10.0, 10.0, 20.0, 30.0]
        """
        if self.use_dss:
            raise NotImplementedError(
                "values are stored in an external DSS file (use_dss=True); "
                "resize_window() only supports inline values."
            )
        if not self.values:
            raise ValueError(
                "Cannot resize an empty time series; there are no edge "
                "values to clip or extend from."
            )
        if self.use_fixed_start:
            if start_datetime is not None:
                logger.warning(
                    "start_datetime is ignored because use_fixed_start is "
                    "True; using fixed_start=%r instead.",
                    self.fixed_start,
                )
            current_start = parse_hec_datetime(self.fixed_start)
        else:
            if start_datetime is None:
                raise ValueError(
                    "start_datetime is required when use_fixed_start is False."
                )
            current_start = start_datetime

        interval_td = parse_interval(self.interval)
        n = len(self.values)
        current_end = current_start + (n - 1) * interval_td

        new_start, new_end = window
        effective_start = new_start if new_start is not None else current_start
        effective_end = new_end if new_end is not None else current_end
        if effective_end < effective_start:
            raise ValueError(
                f"effective window end ({effective_end}) is before the "
                f"effective window start ({effective_start})."
            )
        if (effective_start - current_start) % interval_td != dt.timedelta(0):
            raise ValueError(
                f"window start {effective_start} is not reachable from the "
                f"current start {current_start} by a whole number of "
                f"{self.interval!r} steps."
            )
        if (effective_end - current_start) % interval_td != dt.timedelta(0):
            raise ValueError(
                f"window end {effective_end} is not reachable from the "
                f"current start {current_start} by a whole number of "
                f"{self.interval!r} steps."
            )
        i_start = (effective_start - current_start) // interval_td
        i_end = (effective_end - current_start) // interval_td

        def _value_at(i: int) -> float:
            if i < 0:
                return self.values[0]
            if i > n - 1:
                return self.values[-1]
            return self.values[i]

        new_values = [_value_at(i) for i in range(i_start, i_end + 1)]

        # Resolve everything that can still raise before mutating self.
        resolved_fixed_start = (
            _format_fixed_start(effective_start)
            if effective_start != current_start
            else None
        )

        # Nothing below this point can raise.
        self.values = new_values
        if resolved_fixed_start is not None:
            self.fixed_start = resolved_fixed_start
            self.use_fixed_start = True


@dataclass
class FlowHydrograph(_TimeSeriesBoundary):
    """Upstream / internal flow hydrograph boundary."""

    flow_hydrograph_slope: str | None = None
    stage_tw_check: int = 0
    q_min: float | None = None
    q_mult: float | None = None
    dss_file: str = ""
    dss_path: str = ""
    is_critical: bool = False
    critical_boundary_flow: str = ""


@dataclass
class LateralInflow(_TimeSeriesBoundary):
    """Lateral or uniform lateral inflow hydrograph."""

    is_uniform: bool = False
    q_min: float | None = None
    q_mult: float | None = None
    dss_file: str = ""
    dss_path: str = ""
    is_critical: bool = False
    critical_boundary_flow: str = ""


@dataclass
class StageHydrograph(_TimeSeriesBoundary):
    """Stage (water-surface) hydrograph boundary."""

    dss_path: str = ""


@dataclass
class RatingCurve(_Boundary):
    """Rating-curve downstream boundary."""

    pairs: list[tuple[float, float]] = field(default_factory=list)
    dss_path: str = ""
    use_dss: bool = False
    use_fixed_start: bool = False
    fixed_start: str = ","
    is_critical: bool = False
    critical_boundary_flow: str = ""
    _extra_lines: list[str] = field(default_factory=list, repr=False)


@dataclass
class FrictionSlope(_Boundary):
    """Normal-depth (friction slope) downstream boundary."""

    slope: float = 0.0
    value2: float = 0.0


@dataclass
class NormalDepth(_Boundary):
    """Normal-depth boundary specified as a single slope value."""

    slope: float = 0.0


@dataclass
class GateOpening:
    """Time series of openings for one gate."""

    gate_name: str = ""
    dss_path: str = ""
    use_dss: bool = False
    time_interval: str = "1HOUR"
    use_fixed_start: bool = False
    fixed_start: str = ","
    values: list[float] = field(default_factory=list)


@dataclass
class GateBoundary(_Boundary):
    """Inline structure with one or more gated openings."""

    gates: list[GateOpening] = field(default_factory=list)


# Type alias for the flat boundary list
BoundaryType = (
    FlowHydrograph
    | LateralInflow
    | StageHydrograph
    | RatingCurve
    | FrictionSlope
    | NormalDepth
    | GateBoundary
)

# Type alias for a class (not an instance) accepted by the boundary_types=
# parameter of UnsteadyFlow's bulk time-series methods.
_TimeSeriesBoundaryClass = (
    type[FlowHydrograph] | type[LateralInflow] | type[StageHydrograph]
)


# ---------------------------------------------------------------------------
# Boundary parser (shared by both classes)
# ---------------------------------------------------------------------------


def _parse_boundary_blocks(lines: list[str]) -> list[BoundaryType]:
    """Parse all boundary condition blocks from *lines*.

    *lines* should be the slice of the file that contains boundary blocks
    (i.e. from the first ``Boundary Location=`` line to the start of the
    trailing Met / Non-Newtonian section).
    """
    boundaries: list[BoundaryType] = []
    n = len(lines)
    i = 0

    def _next_key(idx: int) -> tuple[str, str]:
        """Return (key, value) for lines[idx], splitting on first '='."""
        raw = lines[idx]
        eq = raw.find("=")
        if eq == -1:
            return raw.rstrip("\n"), ""
        return raw[:eq].rstrip(), raw[eq + 1 :].rstrip("\n")

    while i < n:
        key, val = _next_key(i)
        if key != "Boundary Location":
            i += 1
            continue

        # Parse location fields
        parts = val.split(",", 3)
        river = parts[0].strip() if len(parts) > 0 else ""
        reach = parts[1].strip() if len(parts) > 1 else ""
        rs = parts[2].strip() if len(parts) > 2 else ""
        tail = parts[3] if len(parts) > 3 else ""

        i += 1

        # Peek ahead to determine BC type
        bc: BoundaryType | None = None

        while i < n:
            key2, val2 = _next_key(i)

            # ---- stop conditions (next Boundary Location or trailing section)
            if key2 == "Boundary Location":
                break
            if _is_trailing_key(key2):
                break

            # ---- Flow hydrograph
            if key2 == "Interval":
                interval = val2.strip()
                i += 1
                if i >= n:
                    break
                key3, val3 = _next_key(i)

                if key3 in ("Flow Hydrograph",):
                    count = int(val3.strip())
                    i += 1
                    nlines = _data_line_count(count)
                    data_lines = lines[i : i + nlines]
                    i += nlines
                    values = _parse_data_block(data_lines, count)
                    bc = FlowHydrograph(
                        river=river,
                        reach=reach,
                        river_station=rs,
                        _location_tail=tail,
                        interval=interval,
                        values=values,
                    )
                    # Consume metadata
                    while i < n:
                        k, v = _next_key(i)
                        if k in ("Boundary Location",) or _is_trailing_key(k):
                            break
                        if k == "Flow Hydrograph Slope":
                            bc.flow_hydrograph_slope = v.strip()
                        elif k == "Stage Hydrograph TW Check":
                            bc.stage_tw_check = int(v.strip())
                        elif k == "Flow Hydrograph QMin":
                            bc.q_min = float(v.strip())
                        elif k == "Flow Hydrograph QMult":
                            bc.q_mult = float(v.strip())
                        elif k == "DSS File":
                            bc.dss_file = v.strip()
                        elif k == "DSS Path":
                            bc.dss_path = v.strip()
                        elif k == "Use DSS":
                            bc.use_dss = v.strip().lower() == "true"
                        elif k == "Use Fixed Start Time":
                            bc.use_fixed_start = v.strip().lower() == "true"
                        elif k == "Fixed Start Date/Time":
                            bc.fixed_start = v.strip()
                        elif k == "Is Critical Boundary":
                            bc.is_critical = v.strip().lower() == "true"
                        elif k == "Critical Boundary Flow":
                            bc.critical_boundary_flow = v.strip()
                            i += 1
                            break
                        else:
                            bc._extra_lines.append(lines[i])
                        i += 1
                    break

                elif key3 in (
                    "Lateral Inflow Hydrograph",
                    "Uniform Lateral Inflow Hydrograph",
                ):
                    count = int(val3.strip())
                    i += 1
                    nlines = _data_line_count(count)
                    data_lines = lines[i : i + nlines]
                    i += nlines
                    values = _parse_data_block(data_lines, count)
                    bc = LateralInflow(
                        river=river,
                        reach=reach,
                        river_station=rs,
                        _location_tail=tail,
                        interval=interval,
                        values=values,
                        is_uniform=(key3 == "Uniform Lateral Inflow Hydrograph"),
                    )
                    while i < n:
                        k, v = _next_key(i)
                        if k in ("Boundary Location",) or _is_trailing_key(k):
                            break
                        if k == "Flow Hydrograph QMin":
                            bc.q_min = float(v.strip())
                        elif k == "Flow Hydrograph QMult":
                            bc.q_mult = float(v.strip())
                        elif k == "DSS File":
                            bc.dss_file = v.strip()
                        elif k == "DSS Path":
                            bc.dss_path = v.strip()
                        elif k == "Use DSS":
                            bc.use_dss = v.strip().lower() == "true"
                        elif k == "Use Fixed Start Time":
                            bc.use_fixed_start = v.strip().lower() == "true"
                        elif k == "Fixed Start Date/Time":
                            bc.fixed_start = v.strip()
                        elif k == "Is Critical Boundary":
                            bc.is_critical = v.strip().lower() == "true"
                        elif k == "Critical Boundary Flow":
                            bc.critical_boundary_flow = v.strip()
                            i += 1
                            break
                        else:
                            bc._extra_lines.append(lines[i])
                        i += 1
                    break

                elif key3 == "Stage Hydrograph":
                    count = int(val3.strip())
                    i += 1
                    nlines = _data_line_count(count)
                    data_lines = lines[i : i + nlines]
                    i += nlines
                    values = _parse_data_block(data_lines, count)
                    bc = StageHydrograph(
                        river=river,
                        reach=reach,
                        river_station=rs,
                        _location_tail=tail,
                        interval=interval,
                        values=values,
                    )
                    while i < n:
                        k, v = _next_key(i)
                        if k in ("Boundary Location",) or _is_trailing_key(k):
                            break
                        if k == "DSS Path":
                            bc.dss_path = v.strip()
                        elif k == "Use DSS":
                            bc.use_dss = v.strip().lower() == "true"
                        elif k == "Use Fixed Start Time":
                            bc.use_fixed_start = v.strip().lower() == "true"
                        elif k == "Fixed Start Date/Time":
                            bc.fixed_start = v.strip()
                        else:
                            bc._extra_lines.append(lines[i])
                        i += 1
                    break

                else:
                    # Unknown type after Interval= — skip
                    i += 1
                    continue

            # ---- Rating curve
            elif key2 == "Rating Curve":
                count = int(val2.strip())
                i += 1
                # Rating curve data: pairs of (elev, flow), 10 values per row
                # so ceil(count*2 / 10) lines
                nlines = _data_line_count(count * 2)
                data_lines = lines[i : i + nlines]
                i += nlines
                flat = _parse_data_block(data_lines, count * 2)
                pairs = [(flat[j], flat[j + 1]) for j in range(0, len(flat), 2)]
                bc = RatingCurve(
                    river=river,
                    reach=reach,
                    river_station=rs,
                    _location_tail=tail,
                    pairs=pairs,
                )
                while i < n:
                    k, v = _next_key(i)
                    if k in ("Boundary Location",) or _is_trailing_key(k):
                        break
                    if k == "DSS Path":
                        bc.dss_path = v.strip()
                    elif k == "Use DSS":
                        bc.use_dss = v.strip().lower() == "true"
                    elif k == "Use Fixed Start Time":
                        bc.use_fixed_start = v.strip().lower() == "true"
                    elif k == "Fixed Start Date/Time":
                        bc.fixed_start = v.strip()
                    elif k == "Is Critical Boundary":
                        bc.is_critical = v.strip().lower() == "true"
                    elif k == "Critical Boundary Flow":
                        bc.critical_boundary_flow = v.strip()
                        i += 1
                        break
                    else:
                        bc._extra_lines.append(lines[i])
                    i += 1
                break

            # ---- Friction slope
            elif key2 == "Friction Slope":
                parts2 = val2.split(",")
                slope = float(parts2[0].strip()) if parts2 else 0.0
                v2 = float(parts2[1].strip()) if len(parts2) > 1 else 0.0
                bc = FrictionSlope(
                    river=river,
                    reach=reach,
                    river_station=rs,
                    _location_tail=tail,
                    slope=slope,
                    value2=v2,
                )
                i += 1
                break

            # ---- Normal depth
            elif key2 == "Normal Depth":
                bc = NormalDepth(
                    river=river,
                    reach=reach,
                    river_station=rs,
                    _location_tail=tail,
                    slope=float(val2.strip()),
                )
                i += 1
                break

            # ---- Gate boundary (inline structure)
            elif key2 == "Gate Name":
                bc = GateBoundary(
                    river=river, reach=reach, river_station=rs, _location_tail=tail
                )
                while i < n:
                    k, v = _next_key(i)
                    if k in ("Boundary Location",) or _is_trailing_key(k):
                        break
                    if k == "Gate Name":
                        bc.gates.append(GateOpening(gate_name=v.strip()))
                        i += 1
                    elif k == "Gate DSS Path":
                        if bc.gates:
                            bc.gates[-1].dss_path = v.strip()
                        i += 1
                    elif k == "Gate Use DSS":
                        if bc.gates:
                            bc.gates[-1].use_dss = v.strip().lower() == "true"
                        i += 1
                    elif k == "Gate Time Interval":
                        if bc.gates:
                            bc.gates[-1].time_interval = v.strip()
                        i += 1
                    elif k == "Gate Use Fixed Start Time":
                        if bc.gates:
                            bc.gates[-1].use_fixed_start = v.strip().lower() == "true"
                        i += 1
                    elif k == "Gate Fixed Start Date/Time":
                        if bc.gates:
                            bc.gates[-1].fixed_start = v.strip()
                        i += 1
                    elif k == "Gate Openings":
                        count = int(v.strip())
                        i += 1
                        nlines = _data_line_count(count)
                        data_lines = lines[i : i + nlines]
                        i += nlines
                        if bc.gates:
                            bc.gates[-1].values = _parse_data_block(data_lines, count)
                    else:
                        i += 1
                break

            else:
                # Unknown line within a boundary block — skip
                i += 1

        if bc is not None:
            boundaries.append(bc)

    return boundaries


def _is_trailing_key(key: str) -> bool:
    """Return True for keys that mark the start of the trailing section."""
    return (
        key.startswith("Met ")
        or key.startswith("Met BC")
        or key
        in (
            "Met Point Raster Parameters",
            "Non-Newtonian Method",
            "Non-Newtonian Constant Vol Conc",
            "Precipitation Mode",
            "Wind Mode",
            "Air Density Mode",
            "Lava Activation",
        )
    )



# ---------------------------------------------------------------------------
# UnsteadyFlow — structured, sortable editor
# ---------------------------------------------------------------------------


class UnsteadyFlow:
    """Structured editor for HEC-RAS unsteady flow files (.u**).

    Boundary conditions are parsed into typed dataclass objects stored in
    :attr:`boundaries`.  Boundaries may be sorted by river station (useful
    for workflows that address gates or lateral inflows by index).

    ``save()`` reconstructs the boundary section from the objects; the
    header, initial conditions, and trailing meteorological / Non-Newtonian
    lines are preserved verbatim.

    .. note::
        ``save()`` is **not** byte-identical to the original when boundaries
        are reordered or values are changed, because the file is reconstructed
        from parsed objects — the file is reconstructed from parsed data.

    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.is_file():
            raise FileNotFoundError(f"Unsteady flow file not found: {self._path}")
        with open(self._path, encoding="utf-8", errors="replace") as fh:
            self._all_lines: list[str] = fh.readlines()
        self._parse()
        self._modified: bool = False

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self) -> None:
        lines = self._all_lines

        # Split file into:
        #   1. header lines (before first Initial/Boundary)
        #   2. initial condition lines
        #   3. boundary lines
        #   4. trailing lines (Met BC, Non-Newtonian, etc.)
        self._header_lines: list[str] = []
        self._initial_lines: list[str] = []
        self._trailing_lines: list[str] = []
        boundary_lines: list[str] = []

        section: Literal["header", "initial", "boundary", "trailing"] = "header"

        for line in lines:
            key = line.split("=", 1)[0].rstrip() if "=" in line else line.rstrip("\n")

            if section == "header":
                if key in (
                    "Initial Flow Loc",
                    "Initial Storage Elev",
                    "Initial RRR Elev",
                ):
                    section = "initial"
                    self._initial_lines.append(line)
                elif key == "Boundary Location":
                    section = "boundary"
                    boundary_lines.append(line)
                else:
                    self._header_lines.append(line)

            elif section == "initial":
                if key == "Boundary Location":
                    section = "boundary"
                    boundary_lines.append(line)
                elif key not in (
                    "Initial Flow Loc",
                    "Initial Storage Elev",
                    "Initial RRR Elev",
                ):
                    # Non-initial, non-boundary line — stays in initial block
                    # (e.g. blank lines or unknown keys between initial conds)
                    self._initial_lines.append(line)
                else:
                    self._initial_lines.append(line)

            elif section == "boundary":
                if _is_trailing_key(key):
                    section = "trailing"
                    self._trailing_lines.append(line)
                else:
                    boundary_lines.append(line)

            else:  # trailing
                self._trailing_lines.append(line)

        # Parse initial conditions into typed objects
        self.initial_flow_locs: list[InitialFlowLoc] = []
        self.initial_storage_elevs: list[InitialStorageElev] = []
        self.initial_rainfall_runoff_elevs: list[InitialRainfallRunoffElev] = []
        for line in self._initial_lines:
            if line.startswith("Initial Flow Loc="):
                raw = line[len("Initial Flow Loc=") :].strip()
                self.initial_flow_locs.append(InitialFlowLoc._from_raw(raw))
            elif line.startswith("Initial Storage Elev="):
                raw = line[len("Initial Storage Elev=") :].strip()
                self.initial_storage_elevs.append(InitialStorageElev._from_raw(raw))
            elif line.startswith("Initial RRR Elev="):
                raw = line[len("Initial RRR Elev=") :].strip()
                self.initial_rainfall_runoff_elevs.append(InitialRainfallRunoffElev._from_raw(raw))

        # Parse boundaries
        boundary_lines_stripped = [l.rstrip("\n") for l in boundary_lines]
        self.boundaries: list[BoundaryType] = _parse_boundary_blocks(
            boundary_lines_stripped
        )

    # ------------------------------------------------------------------
    # Modification state
    # ------------------------------------------------------------------

    @property
    def is_modified(self) -> bool:
        """``True`` if any value has been changed since the last :meth:`save`."""
        return self._modified

    # ------------------------------------------------------------------
    # Scalar properties (read from header_lines, write back in-place)
    # ------------------------------------------------------------------

    def _header_get(self, key: str) -> str | None:
        prefix = key + "="
        for line in self._header_lines:
            if line.startswith(prefix):
                val = line[len(prefix) :].strip()
                return val if val else None
        return None

    def _header_set(self, key: str, raw_value: str) -> None:
        prefix = key + "="
        for i, line in enumerate(self._header_lines):
            if line.startswith(prefix):
                self._header_lines[i] = f"{prefix}{raw_value}\n"
                self._modified = True
                return
        raise KeyError(f"Key not found in header: {key!r}")

    @property
    def flow_title(self) -> str | None:
        """Flow file title (``Flow Title=``), or ``None`` if absent."""
        return self._header_get("Flow Title")

    @flow_title.setter
    def flow_title(self, value: str) -> None:
        self._header_set("Flow Title", value)

    @property
    def program_version(self) -> str | None:
        """HEC-RAS version string (``Program Version=``), or ``None`` if absent."""
        return self._header_get("Program Version")

    @property
    def write_ic_file(self) -> bool | None:
        """Whether to write an initial conditions file at the end of simulation.

        Returns ``True`` if ``Write IC File at Sim End=-1``, ``False`` if
        ``0``, or ``None`` if the key is absent (older files).
        """
        raw = self._header_get("Write IC File at Sim End")
        if raw is None:
            return None
        return int(raw) == -1

    @write_ic_file.setter
    def write_ic_file(self, value: bool) -> None:
        self._header_set("Write IC File at Sim End", "-1" if value else "0")

    def _header_set_restart_filename(self, filename: str) -> None:
        prefix = "Restart Filename="
        for i, line in enumerate(self._header_lines):
            if line.startswith(prefix):
                self._header_lines[i] = f"{prefix}{filename}\n"
                self._modified = True
                return
        ur_prefix = "Use Restart="
        for i, line in enumerate(self._header_lines):
            if line.startswith(ur_prefix):
                self._header_lines.insert(i + 1, f"{prefix}{filename}\n")
                self._modified = True
                return
        raise KeyError(
            "'Use Restart' not found in header; cannot insert 'Restart Filename'"
        )

    @property
    def restart(self) -> tuple[int, str | None]:
        """Return ``(flag, filename)`` for the restart configuration.

        *flag* is the ``Use Restart`` value (``0`` = disabled, ``1`` = enabled).
        *filename* is the ``Restart Filename`` value, or ``None`` if absent.
        """
        raw = self._header_get("Use Restart")
        flag = int(raw.strip()) if raw is not None else 0
        filename = self._header_get("Restart Filename")
        return (flag, filename)

    @restart.setter
    def restart(self, value: int | bool | str | None) -> None:
        if value is None or value is False:
            self._header_set("Use Restart", " 0 ")
        elif value is True:
            self._header_set("Use Restart", " 1 ")
        elif isinstance(value, str):
            self._header_set_restart_filename(value)
            self._header_set("Use Restart", " 1 ")
        else:
            self._header_set("Use Restart", " 1 " if value else " 0 ")

    # ------------------------------------------------------------------
    # Typed boundary views
    # ------------------------------------------------------------------

    @property
    def flow_hydrographs(self) -> list[FlowHydrograph]:
        """All :class:`FlowHydrograph` boundaries, in file order."""
        return [b for b in self.boundaries if isinstance(b, FlowHydrograph)]

    @property
    def lateral_inflows(self) -> list[LateralInflow]:
        """All :class:`LateralInflow` boundaries, in file order."""
        return [b for b in self.boundaries if isinstance(b, LateralInflow)]

    @property
    def stage_hydrographs(self) -> list[StageHydrograph]:
        """All :class:`StageHydrograph` boundaries, in file order."""
        return [b for b in self.boundaries if isinstance(b, StageHydrograph)]

    @property
    def gate_boundaries(self) -> list[GateBoundary]:
        """All :class:`GateBoundary` boundaries, in file order."""
        return [b for b in self.boundaries if isinstance(b, GateBoundary)]

    @property
    def friction_slopes(self) -> list[FrictionSlope]:
        """All :class:`FrictionSlope` boundaries, in file order."""
        return [b for b in self.boundaries if isinstance(b, FrictionSlope)]

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------

    def _sort_type(self, bc_type: type, *, descending: bool) -> None:
        """Sort boundaries of *bc_type* by river station within each (river, reach)
        group, preserving both group order and the positions of all other types."""
        targets = [
            (i, b) for i, b in enumerate(self.boundaries) if isinstance(b, bc_type)
        ]
        # Group in first-appearance order (dict preserves insertion order, Python 3.7+)
        groups: dict[tuple[str, str], list[tuple[int, object]]] = {}
        for item in targets:
            key = (item[1].river, item[1].reach)
            if key not in groups:
                groups[key] = []
            groups[key].append(item)
        # Sort by RS within each group, then flatten preserving group order
        sorted_targets = []
        for group in groups.values():
            sorted_targets.extend(
                sorted(group, key=lambda t: t[1]._rs_float(), reverse=descending)
            )
        for (orig_idx, _), (_, sorted_bc) in zip(targets, sorted_targets):
            self.boundaries[orig_idx] = sorted_bc

    def sort_flow_hydrographs(self, *, descending: bool = True) -> None:
        """Sort :class:`FlowHydrograph` entries by river station.

        Parameters
        ----------
        descending:
            If ``True`` (default), highest station first
            (upstream → downstream for standard RAS numbering).
            Pass ``False`` for ascending order (lowest station first).
        """
        self._sort_type(FlowHydrograph, descending=descending)

    def sort_gate_boundaries(self, *, descending: bool = True) -> None:
        """Sort :class:`GateBoundary` entries by river station.

        Other boundary types remain at their original positions.

        Parameters
        ----------
        descending:
            If ``True`` (default), highest station first
            (upstream → downstream for standard RAS numbering).
            Pass ``False`` for ascending order (lowest station first).
        """
        self._sort_type(GateBoundary, descending=descending)

    def sort_lateral_inflows(self, *, descending: bool = True) -> None:
        """Sort :class:`LateralInflow` entries by river station.

        Parameters
        ----------
        descending:
            If ``True`` (default), highest station first
            (upstream → downstream for standard RAS numbering).
            Pass ``False`` for ascending order (lowest station first).
        """
        self._sort_type(LateralInflow, descending=descending)

    # ------------------------------------------------------------------
    # Set by index (works naturally after sorting)
    # ------------------------------------------------------------------

    def set_flow_hydrograph(
        self, index: int, values: _Values, q_min: float = 0.0, q_mult: float = 1.0
    ) -> None:
        """Set flow hydrograph values by position in :attr:`flow_hydrographs`.

        Parameters
        ----------
        index:
            Position in the filtered flow-hydrograph list.
        values:
            New flow values.  A scalar is broadcast to the length of
            the existing time series.
        q_min:
            New ``Flow Hydrograph QMin`` value.  Always applied, overwriting
            any previously set value.
        q_mult:
            New ``Flow Hydrograph QMult`` value.  Always applied, overwriting
            any previously set value.
        """
        bc = self.flow_hydrographs[index]
        bc.values = _coerce_values(values, len(bc.values))
        bc.q_min = float(q_min)
        bc.q_mult = float(q_mult)
        self._modified = True

    def set_lateral_inflow(
        self, index: int, values: _Values, q_min: float = 0.0, q_mult: float = 1.0
    ) -> None:
        """Set lateral inflow values by position in :attr:`lateral_inflows`.

        Parameters
        ----------
        values:
            New flow values.  A scalar is broadcast to the length of
            the existing time series.
        q_min:
            New ``QMin`` value.  Always applied, overwriting any previously
            set value.
        q_mult:
            New ``QMult`` value.  Always applied, overwriting any previously
            set value.
        """
        bc = self.lateral_inflows[index]
        bc.values = _coerce_values(values, len(bc.values))
        bc.q_min = float(q_min)
        bc.q_mult = float(q_mult)
        self._modified = True

    def set_all_lateral_inflows(self, values: list[float | list[float]]) -> None:
        """Set lateral inflow values across all :class:`LateralInflow` boundaries.

        Parameters
        ----------
        values:
            One entry per lateral inflow (in file order).
            Each entry is either a scalar ``float`` (broadcast to the
            boundary's existing time-series length) or a ``list[float]``
            (used as-is).  If ``values`` is shorter than the total number
            of lateral inflows, the remaining boundaries are left unchanged.
        """
        for bc, v in zip(self.lateral_inflows, values, strict=False):
            bc.values = _coerce_values(v, len(bc.values))
        self._modified = True

    def set_gate_opening(
        self, index: int, values: _Values, gate_index: int = 0
    ) -> None:
        """Set gate opening values by position in :attr:`gate_boundaries`.

        Parameters
        ----------
        index:
            Position in the filtered gate-boundary list.
        values:
            New opening values.  A scalar is broadcast to the length
            of the existing gate opening time series.
        gate_index:
            Which gate within the boundary (default 0).
        """
        gate = self.gate_boundaries[index].gates[gate_index]
        gate.values = _coerce_values(values, len(gate.values))
        self._modified = True

    def set_all_gate_openings(self, values: list[float | list[float]]) -> None:
        """Set gate opening values across all gates in all :class:`GateBoundary`.

        Parameters
        ----------
        values:
            One entry per gate (in order across all boundaries).
            Each entry is either a scalar ``float`` (broadcast to the
            gate's existing time-series length) or a ``list[float]``
            (used as-is).  If ``values`` is shorter than the total number
            of gates, the remaining gates are left unchanged.

        """
        all_gates = [gate for gb in self.gate_boundaries for gate in gb.gates]
        for gate, v in zip(all_gates, values, strict=False):
            gate.values = _coerce_values(v, len(gate.values))
        self._modified = True

    # ------------------------------------------------------------------
    # Bulk window operations (adapting a file to a new run period)
    # ------------------------------------------------------------------

    def _apply_atomically(
        self,
        boundary_types: _TimeSeriesBoundaryClass | tuple[_TimeSeriesBoundaryClass, ...],
        apply_fn: Callable[[BoundaryType], None],
    ) -> None:
        """Apply *apply_fn* to every boundary of *boundary_types*, all-or-nothing.

        *apply_fn* is first run against a :func:`copy.deepcopy` of each
        matching boundary. If it raises for any boundary, the exception
        propagates immediately and :attr:`boundaries` is left completely
        untouched. Only once every boundary's copy has succeeded are the
        copies swapped back into :attr:`boundaries`, so a batch call either
        fully applies or has no effect at all.

        Notes
        -----
        On success, matching entries in :attr:`boundaries` are *replaced*
        by new (deep-copied) objects — unlike every other mutator in this
        module, which edits a boundary in place. Any reference to a
        boundary object held from before the call becomes stale; re-fetch
        it via :attr:`boundaries` / :attr:`flow_hydrographs` /
        :attr:`lateral_inflows` / :attr:`stage_hydrographs` afterward.
        """
        targets = [
            (i, b)
            for i, b in enumerate(self.boundaries)
            if isinstance(b, boundary_types)
        ]
        updated: list[tuple[int, BoundaryType]] = []
        for i, b in targets:
            new_b = copy.deepcopy(b)
            apply_fn(new_b)
            updated.append((i, new_b))
        for i, new_b in updated:
            self.boundaries[i] = new_b
        if updated:
            self._modified = True

    def redefine_all_flow_time_series(
        self,
        window: tuple[dt.datetime, dt.datetime],
        interval: str | float | int | dt.timedelta,
        value: float | int = 0.0,
        *,
        q_min: float = 0.0,
        q_mult: float = 1.0,
    ) -> None:
        """Redefine every flow-type boundary to a constant value over a new window.

        Applies :meth:`FlowHydrograph.set_time_series_window` /
        :meth:`LateralInflow.set_time_series_window` with a scalar *value*
        to every :class:`FlowHydrograph` and :class:`LateralInflow`
        boundary in the file. Useful when adapting an existing unsteady
        flow file to a new simulation run that doesn't yet have real flow
        data for its period: rather than trying to preserve the old
        hydrograph shapes, every flow-type boundary is redefined to a flat
        baseline (e.g. ``0``) spanning the new window, ready for the
        caller to overwrite individually (e.g. via
        :meth:`set_flow_hydrograph_at` / :meth:`set_lateral_inflow_at`) as
        real data becomes available. See :meth:`resize_all_flow_time_series`
        for the alternative that preserves existing data instead.

        Parameters
        ----------
        window:
            ``(start, end)`` for the new run, inclusive of both endpoints,
            applied identically to every flow-type boundary.
        interval:
            Spacing between timesteps, applied to every boundary. One of a
            HEC-RAS interval string (e.g. ``"15MIN"``), a
            :class:`datetime.timedelta`, or a bare ``int``/``float``
            interpreted as seconds.
        value:
            Constant value every flow-type boundary is redefined to
            (default ``0.0``).
        q_min:
            ``QMin`` applied to every boundary redefined (default ``0.0``).
        q_mult:
            ``QMult`` applied to every boundary redefined (default ``1.0``).

        Raises
        ------
        ValueError
            *window*/*interval* is invalid for any boundary (see
            :meth:`_TimeSeriesBoundary.set_time_series_window`). No
            boundary is changed if any of them would fail.

        Notes
        -----
        All-or-nothing: either every targeted boundary is updated, or (on
        error) none of them are. On success, updated boundaries are
        replaced by new objects (see :meth:`_apply_atomically`) — re-fetch
        via :attr:`flow_hydrographs` / :attr:`lateral_inflows` afterward
        rather than relying on references held from before the call.
        """
        self._apply_atomically(
            (FlowHydrograph, LateralInflow),
            lambda bc: bc.set_time_series_window(
                window, interval, value, q_min=q_min, q_mult=q_mult
            ),
        )

    def resize_all_flow_time_series(
        self,
        window: tuple[dt.datetime | None, dt.datetime | None],
        *,
        start_datetime: dt.datetime | None = None,
    ) -> None:
        """Clip/extend every flow-type boundary to a new window.

        Applies :meth:`FlowHydrograph.resize_window` /
        :meth:`LateralInflow.resize_window` to every
        :class:`FlowHydrograph` and :class:`LateralInflow` boundary in the
        file, in place. Companion to :meth:`redefine_all_flow_time_series`:
        use this one instead when real flow data already exists and should
        be clipped/extended rather than reset to a constant — existing
        data inside the new window is kept, data outside it is clipped,
        and new timesteps beyond either end are filled by repeating the
        boundary's own first/last value.

        Parameters
        ----------
        window:
            ``(new_start, new_end)``, applied identically to every
            flow-type boundary. Either side may be ``None`` to leave that
            side untouched.
        start_datetime:
            Current start, needed only for boundaries with
            ``use_fixed_start=False`` (see
            :meth:`_TimeSeriesBoundary.resize_window`). Boundaries with
            ``use_fixed_start=True`` ignore this and use their own
            ``fixed_start`` instead, so a single call can mix both kinds.

        Raises
        ------
        ValueError
            Any boundary's window/alignment is invalid, or *start_datetime*
            is required but missing, for any boundary (see
            :meth:`_TimeSeriesBoundary.resize_window`). No boundary is
            changed if any of them would fail.
        NotImplementedError
            Any boundary has ``use_dss=True``.

        Notes
        -----
        All-or-nothing: either every flow-type boundary is updated, or (on
        error) none of them are. On success, updated boundaries are
        replaced by new objects (see :meth:`_apply_atomically`) — re-fetch
        via :attr:`flow_hydrographs` / :attr:`lateral_inflows` afterward
        rather than relying on references held from before the call.
        """
        self._apply_atomically(
            (FlowHydrograph, LateralInflow),
            lambda bc: bc.resize_window(window, start_datetime=start_datetime),
        )

    def resize_all_stage_time_series(
        self,
        window: tuple[dt.datetime | None, dt.datetime | None],
        *,
        start_datetime: dt.datetime | None = None,
    ) -> None:
        """Clip/extend every :class:`StageHydrograph` boundary to a new window.

        Applies :meth:`StageHydrograph.resize_window` to every stage
        boundary in the file, in place. Appropriate for stage/tailwater
        boundaries, where resetting to a constant (as
        :meth:`redefine_all_flow_time_series` does for flow-type
        boundaries) usually isn't physically meaningful: existing data
        inside the new window is kept, data outside it is clipped, and new
        timesteps beyond either end are filled by repeating the boundary's
        own first/last value.

        Parameters
        ----------
        window:
            ``(new_start, new_end)``, applied identically to every stage
            boundary. Either side may be ``None`` to leave that side
            untouched.
        start_datetime:
            Current start, needed only for boundaries with
            ``use_fixed_start=False`` (see
            :meth:`_TimeSeriesBoundary.resize_window`). Boundaries with
            ``use_fixed_start=True`` ignore this and use their own
            ``fixed_start`` instead, so a single call can mix both kinds.

        Raises
        ------
        ValueError
            Any boundary's window/alignment is invalid, or *start_datetime*
            is required but missing, for any boundary (see
            :meth:`_TimeSeriesBoundary.resize_window`). No boundary is
            changed if any of them would fail.
        NotImplementedError
            Any boundary has ``use_dss=True``.

        Notes
        -----
        All-or-nothing: either every stage boundary is updated, or (on
        error) none of them are. On success, updated boundaries are
        replaced by new objects (see :meth:`_apply_atomically`) — re-fetch
        via :attr:`stage_hydrographs` afterward rather than relying on
        references held from before the call.
        """
        self._apply_atomically(
            StageHydrograph,
            lambda bc: bc.resize_window(window, start_datetime=start_datetime),
        )

    def resize_all_time_series_by_type(
        self,
        window: tuple[dt.datetime | None, dt.datetime | None],
        *,
        start_datetime: dt.datetime | None = None,
        boundary_types: tuple[_TimeSeriesBoundaryClass, ...] = (
            FlowHydrograph,
            LateralInflow,
            StageHydrograph,
        ),
    ) -> None:
        """Clip/extend every matching time-series boundary to a new window.

        Generic counterpart to :meth:`resize_all_flow_time_series` /
        :meth:`resize_all_stage_time_series`: applies
        :meth:`_TimeSeriesBoundary.resize_window` to every boundary whose
        type is in *boundary_types* (default: all three time-series
        boundary types). Useful when real flow data already exists and
        should be clipped/extended rather than reset to a constant (see
        :meth:`redefine_all_flow_time_series`) — pass
        ``boundary_types=(FlowHydrograph, LateralInflow)`` to resize only
        flow-type boundaries, for example (equivalent to calling
        :meth:`resize_all_flow_time_series` directly).

        Parameters
        ----------
        window:
            ``(new_start, new_end)``, applied identically to every matching
            boundary. Either side may be ``None`` to leave that side
            untouched.
        start_datetime:
            Current start, needed only for boundaries with
            ``use_fixed_start=False`` (see
            :meth:`_TimeSeriesBoundary.resize_window`).
        boundary_types:
            Which boundary classes to include. Defaults to all three
            time-series boundary types (:class:`FlowHydrograph`,
            :class:`LateralInflow`, :class:`StageHydrograph`).

        Raises
        ------
        ValueError
            Any matching boundary's window/alignment is invalid, or
            *start_datetime* is required but missing, for any boundary
            (see :meth:`_TimeSeriesBoundary.resize_window`). No boundary is
            changed if any of them would fail.
        NotImplementedError
            Any matching boundary has ``use_dss=True``.

        Notes
        -----
        All-or-nothing: either every matching boundary is updated, or (on
        error) none of them are. On success, updated boundaries are
        replaced by new objects (see :meth:`_apply_atomically`) — re-fetch
        via :attr:`boundaries` (or the relevant typed property) afterward
        rather than relying on references held from before the call.
        """
        self._apply_atomically(
            boundary_types,
            lambda bc: bc.resize_window(window, start_datetime=start_datetime),
        )

    def reset_all_values_by_type(
        self,
        data: float | int | Sequence[float | int] | dict[float, float],
        *,
        q_min: float = 0.0,
        q_mult: float = 1.0,
        boundary_types: tuple[_TimeSeriesBoundaryClass, ...] = (
            FlowHydrograph,
            LateralInflow,
        ),
    ) -> None:
        """Reset every matching boundary's values, keeping window/interval fixed.

        Bulk counterpart to :meth:`_TimeSeriesBoundary.reset_values`, the
        same way :meth:`resize_all_time_series_by_type` is the bulk
        counterpart to :meth:`_TimeSeriesBoundary.resize_window`: unlike
        :meth:`redefine_all_flow_time_series` /
        :meth:`resize_all_flow_time_series` /
        :meth:`resize_all_stage_time_series` /
        :meth:`resize_all_time_series_by_type`, this never changes any
        boundary's window or interval — only
        :attr:`~_TimeSeriesBoundary.values` (and, where present,
        ``q_min``/``q_mult``) are replaced, each using that boundary's own
        existing number of timesteps.

        Parameters
        ----------
        data:
            New values, applied identically to every matching boundary.
            One of:

            * a scalar — broadcast to each boundary's own existing number
              of timesteps.
            * a sequence of numbers — used as-is for every boundary; its
              length must exactly match every matching boundary's existing
              number of timesteps (so this only works cleanly when all
              matching boundaries already share the same length).
            * a ``dict[float, float]`` mapping elapsed minutes to a step
              value (see :meth:`_TimeSeriesBoundary.reset_values`) — expanded
              independently against each boundary's own interval and length.
        q_min:
            ``QMin`` applied to every boundary reset (default ``0.0``).
        q_mult:
            ``QMult`` applied to every boundary reset (default ``1.0``).
        boundary_types:
            Which boundary classes to include. Defaults to the two
            flow-type boundaries (:class:`FlowHydrograph`,
            :class:`LateralInflow`) — bulk-resetting stage/tailwater
            boundaries to a shared value is less commonly meaningful.
            Pass ``boundary_types=(StageHydrograph,)`` (or include it
            alongside the flow types) to also cover stage boundaries.

        Raises
        ------
        ValueError
            *data* is invalid for any matching boundary (see
            :meth:`_TimeSeriesBoundary.reset_values`) — e.g. a sequence
            whose length doesn't match one of the boundaries, or any
            matching boundary currently has no values. No boundary is
            changed if any of them would fail.

        Notes
        -----
        All-or-nothing: either every matching boundary is updated, or (on
        error) none of them are. On success, updated boundaries are
        replaced by new objects (see :meth:`_apply_atomically`) — re-fetch
        via :attr:`boundaries` (or the relevant typed property) afterward
        rather than relying on references held from before the call.
        """
        self._apply_atomically(
            boundary_types,
            lambda bc: bc.reset_values(data, q_min=q_min, q_mult=q_mult),
        )

    # ------------------------------------------------------------------
    # Set by location (river / reach / rs)
    # ------------------------------------------------------------------

    def boundaries_at(self, river: str, reach: str, rs: str) -> list[BoundaryType]:
        """Return every boundary condition at the given location, in file order.

        More than one boundary can share the same river/reach/station (e.g.
        two lateral inflows entered at the same cross section).  Use the
        position of an entry in the returned list as the ``occurrence``
        argument to the location-based setters and getters to disambiguate.

        Parameters
        ----------
        river:
            River name (case-insensitive match).
        reach:
            Reach name (case-insensitive match).
        rs:
            River station string.

        Returns
        -------
        list[BoundaryType]
            All matching boundaries, in file order.  Empty if none match.
        """
        r = river.strip().lower()
        rc = reach.strip().lower()
        s = str(rs).strip().lower()
        return [
            b
            for b in self.boundaries
            if b.river.lower() == r
            and b.reach.lower() == rc
            and b.river_station.lower() == s
        ]

    def _find_boundary(
        self, river: str, reach: str, rs: str, occurrence: int = 0
    ) -> BoundaryType | None:
        matches = self.boundaries_at(river, reach, rs)
        if not matches:
            return None
        try:
            return matches[occurrence]
        except IndexError as exc:
            raise IndexError(
                f"occurrence {occurrence} out of range; {len(matches)} "
                f"boundary(ies) found at {river!r}, {reach!r}, {rs!r}"
            ) from exc

    def set_flow_hydrograph_at(
        self,
        river: str,
        reach: str,
        rs: str,
        values: _Values,
        q_min: float = 0.0,
        q_mult: float = 1.0,
        occurrence: int = 0,
    ) -> None:
        """Set flow hydrograph values by location.

        Parameters
        ----------
        values:
            A scalar is broadcast to the existing time-series length.
        q_min:
            New ``Flow Hydrograph QMin`` value.  Always applied, overwriting
            any previously set value.
        q_mult:
            New ``Flow Hydrograph QMult`` value.  Always applied, overwriting
            any previously set value.
        occurrence:
            Zero-based position among boundaries sharing this river/reach/
            station, in file order.  Use :meth:`boundaries_at` to see all
            matches when more than one boundary shares a location.

        Raises
        ------
        IndexError
            *occurrence* is out of range for the number of matches found.
        """
        b = self._find_boundary(river, reach, rs, occurrence)
        if not isinstance(b, FlowHydrograph):
            raise KeyError(f"No FlowHydrograph at {river!r}, {reach!r}, {rs!r}")
        b.values = _coerce_values(values, len(b.values))
        b.q_min = float(q_min)
        b.q_mult = float(q_mult)
        self._modified = True

    def set_lateral_inflow_at(
        self,
        river: str,
        reach: str,
        rs: str,
        values: _Values,
        q_min: float = 0.0,
        q_mult: float = 1.0,
        occurrence: int = 0,
    ) -> None:
        """Set lateral inflow values by location.

        Parameters
        ----------
        values:
            A scalar is broadcast to the existing time-series length.
        q_min:
            New ``QMin`` value.  Always applied, overwriting any previously
            set value.
        q_mult:
            New ``QMult`` value.  Always applied, overwriting any previously
            set value.
        occurrence:
            Zero-based position among boundaries sharing this river/reach/
            station, in file order.  Use :meth:`boundaries_at` to see all
            matches when more than one lateral inflow shares a location.

        Raises
        ------
        IndexError
            *occurrence* is out of range for the number of matches found.
        """
        b = self._find_boundary(river, reach, rs, occurrence)
        if not isinstance(b, LateralInflow):
            raise KeyError(f"No LateralInflow at {river!r}, {reach!r}, {rs!r}")
        b.values = _coerce_values(values, len(b.values))
        b.q_min = float(q_min)
        b.q_mult = float(q_mult)
        self._modified = True

    def reset_values_at(
        self,
        river: str,
        reach: str,
        rs: str,
        data: float | int | Sequence[float | int] | dict[float, float],
        *,
        q_min: float = 0.0,
        q_mult: float = 1.0,
        occurrence: int = 0,
    ) -> None:
        """Reset a single time-series boundary's values by location.

        Location-based counterpart to
        :meth:`_TimeSeriesBoundary.reset_values`: finds the boundary at
        the given river/reach/station and resets its values in place,
        without changing its window or interval. Works on any of
        :class:`FlowHydrograph`, :class:`LateralInflow`, or
        :class:`StageHydrograph` — unlike :meth:`set_flow_hydrograph_at` /
        :meth:`set_lateral_inflow_at`, which each only accept one specific
        boundary type.

        Parameters
        ----------
        data:
            New values (see :meth:`_TimeSeriesBoundary.reset_values`): a
            scalar, an exact-length sequence, or a
            ``{elapsed_minutes: value}`` step dict.
        q_min:
            New ``QMin`` value. Always applied, overwriting any previously
            set value. Has no effect if the boundary at this location is a
            :class:`StageHydrograph` (no such field).
        q_mult:
            New ``QMult`` value. Always applied, overwriting any previously
            set value. Has no effect if the boundary at this location is a
            :class:`StageHydrograph` (no such field).
        occurrence:
            Zero-based position among boundaries sharing this river/reach/
            station, in file order. Use :meth:`boundaries_at` to see all
            matches when more than one boundary shares a location.

        Raises
        ------
        KeyError
            No :class:`FlowHydrograph`, :class:`LateralInflow`, or
            :class:`StageHydrograph` boundary is found at this location.
        IndexError
            *occurrence* is out of range for the number of matches found.
        ValueError
            *data* is invalid for the found boundary (see
            :meth:`_TimeSeriesBoundary.reset_values`).
        """
        b = self._find_boundary(river, reach, rs, occurrence)
        if not isinstance(b, (FlowHydrograph, LateralInflow, StageHydrograph)):
            raise KeyError(
                f"No FlowHydrograph, LateralInflow, or StageHydrograph at "
                f"{river!r}, {reach!r}, {rs!r}"
            )
        b.reset_values(data, q_min=q_min, q_mult=q_mult)
        self._modified = True

    def set_gate_opening_at(
        self,
        river: str,
        reach: str,
        rs: str,
        gate: str | int,
        values: _Values,
        occurrence: int = 0,
    ) -> None:
        """Set gate opening values by location and gate name or index.

        Parameters
        ----------
        gate:
            Gate name string, or a zero-based integer index into
            the boundary's gate list.
        values:
            A scalar is broadcast to the existing time-series length.
        occurrence:
            Zero-based position among boundaries sharing this river/reach/
            station, in file order.  Use :meth:`boundaries_at` to see all
            matches when more than one boundary shares a location.

        Raises
        ------
        IndexError
            *occurrence* is out of range for the number of matches found.
        """
        b = self._find_boundary(river, reach, rs, occurrence)
        if not isinstance(b, GateBoundary):
            raise KeyError(f"No GateBoundary at {river!r}, {reach!r}, {rs!r}")
        if isinstance(gate, int):
            try:
                g = b.gates[gate]
            except IndexError as exc:
                raise IndexError(
                    f"Gate index {gate} out of range; "
                    f"{len(b.gates)} gate(s) at {river!r}, {reach!r}, {rs!r}"
                ) from exc
            g.values = _coerce_values(values, len(g.values))
            self._modified = True
            return
        gn = gate.strip().lower()
        for g in b.gates:
            if g.gate_name.strip().lower() == gn:
                g.values = _coerce_values(values, len(g.values))
                self._modified = True
                return
        raise KeyError(f"Gate {gate!r} not found at {river!r}, {reach!r}, {rs!r}")

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------

    def set_initial_flow(self, index: int, flow: float) -> None:
        """Update the initial flow at *index* in :attr:`initial_flow_locs`.

        Parameters
        ----------
        index:
            Zero-based position in ``initial_flow_locs``.
        flow:
            New flow value.

        Raises
        ------
        IndexError
            *index* is out of range.
        """
        self.initial_flow_locs[index].flow = flow
        self._modified = True

    def set_initial_flow_at(self, river: str, reach: str, rs: str, flow: float) -> None:
        """Update the initial flow at the given location.

        Parameters
        ----------
        river:
            River name (case-insensitive match).
        reach:
            Reach name (case-insensitive match).
        rs:
            River station string.
        flow:
            New flow value.

        Raises
        ------
        KeyError
            No matching ``Initial Flow Loc`` entry found.
        """
        r = river.strip().lower()
        rc = reach.strip().lower()
        s = str(rs).strip().lower()
        for loc in self.initial_flow_locs:
            if (
                loc.river.lower() == r
                and loc.reach.lower() == rc
                and loc.river_station.lower() == s
            ):
                loc.flow = flow
                self._modified = True
                return
        raise KeyError(f"Initial Flow Loc not found for {river!r}, {reach!r}, {rs!r}")

    # ------------------------------------------------------------------
    # Get by location
    # ------------------------------------------------------------------

    def get_flow_hydrograph(
        self, river: str, reach: str, rs: str, occurrence: int = 0
    ) -> list[float] | None:
        """Return flow hydrograph values for the given location, or ``None``.

        Parameters
        ----------
        occurrence:
            Zero-based position among boundaries sharing this river/reach/
            station, in file order.  Use :meth:`boundaries_at` to see all
            matches when more than one boundary shares a location.

        Raises
        ------
        IndexError
            *occurrence* is out of range for the number of matches found.
        """
        b = self._find_boundary(river, reach, rs, occurrence)
        if not isinstance(b, FlowHydrograph):
            return None
        return list(b.values)

    def get_lateral_inflow(
        self, river: str, reach: str, rs: str, occurrence: int = 0
    ) -> list[float] | None:
        """Return lateral inflow values for the given location, or ``None``.

        Parameters
        ----------
        occurrence:
            Zero-based position among boundaries sharing this river/reach/
            station, in file order.  Use :meth:`boundaries_at` to see all
            matches when more than one lateral inflow shares a location.

        Raises
        ------
        IndexError
            *occurrence* is out of range for the number of matches found.
        """
        b = self._find_boundary(river, reach, rs, occurrence)
        if not isinstance(b, LateralInflow):
            return None
        return list(b.values)

    def get_gate_openings(
        self, river: str, reach: str, rs: str, gate_name: str, occurrence: int = 0
    ) -> list[float] | None:
        """Return gate opening values for the given location and gate name, or ``None``.

        Parameters
        ----------
        occurrence:
            Zero-based position among boundaries sharing this river/reach/
            station, in file order.  Use :meth:`boundaries_at` to see all
            matches when more than one boundary shares a location.

        Raises
        ------
        IndexError
            *occurrence* is out of range for the number of matches found.
        """
        b = self._find_boundary(river, reach, rs, occurrence)
        if not isinstance(b, GateBoundary):
            return None
        gn = gate_name.strip().lower()
        for g in b.gates:
            if g.gate_name.strip().lower() == gn:
                return list(g.values)
        return None

    def get_initial_flow(self, river: str, reach: str, rs: str) -> float | None:
        """Return the initial flow at the given location, or ``None``."""
        r = river.strip().lower()
        rc = reach.strip().lower()
        s = str(rs).strip().lower()
        for loc in self.initial_flow_locs:
            if (
                loc.river.lower() == r
                and loc.reach.lower() == rc
                and loc.river_station.lower() == s
            ):
                return loc.flow
        return None

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def _boundary_to_lines(self, bc: BoundaryType) -> list[str]:
        """Serialise a single boundary object to a list of text lines."""
        out: list[str] = [bc._location_line() + "\n"]

        if isinstance(bc, (FlowHydrograph, LateralInflow)):
            out.append(f"Interval={bc.interval}\n")
            if isinstance(bc, FlowHydrograph):
                keyword = "Flow Hydrograph"
            else:
                keyword = (
                    "Uniform Lateral Inflow Hydrograph"
                    if bc.is_uniform
                    else "Lateral Inflow Hydrograph"
                )
            count = len(bc.values)
            out.append(f"{keyword}= {count} \n")
            for dl in _format_data_block(bc.values):
                out.append(dl + "\n")
            if isinstance(bc, FlowHydrograph):
                out.append(f"Stage Hydrograph TW Check={bc.stage_tw_check}\n")
                if bc.flow_hydrograph_slope is not None:
                    out.append(f"Flow Hydrograph Slope= {bc.flow_hydrograph_slope}\n")
            if bc.q_min is not None:
                out.append(f"Flow Hydrograph QMin= {bc.q_min}\n")
            if bc.q_mult is not None:
                out.append(f"Flow Hydrograph QMult= {bc.q_mult}\n")
            if bc.dss_file:
                out.append(f"DSS File={bc.dss_file}\n")
            out.append(f"DSS Path={bc.dss_path}\n")
            out.append(f"Use DSS={str(bc.use_dss)}\n")
            out.append(f"Use Fixed Start Time={str(bc.use_fixed_start)}\n")
            out.append(f"Fixed Start Date/Time={bc.fixed_start}\n")
            out.append(f"Is Critical Boundary={str(bc.is_critical)}\n")
            out.append(f"Critical Boundary Flow={bc.critical_boundary_flow}\n")
            for el in bc._extra_lines:
                out.append(el if el.endswith("\n") else el + "\n")

        elif isinstance(bc, StageHydrograph):
            out.append(f"Interval={bc.interval}\n")
            count = len(bc.values)
            out.append(f"Stage Hydrograph= {count} \n")
            for dl in _format_data_block(bc.values):
                out.append(dl + "\n")
            out.append(f"DSS Path={bc.dss_path}\n")
            out.append(f"Use DSS={str(bc.use_dss)}\n")
            out.append(f"Use Fixed Start Time={str(bc.use_fixed_start)}\n")
            out.append(f"Fixed Start Date/Time={bc.fixed_start}\n")
            for el in bc._extra_lines:
                out.append(el if el.endswith("\n") else el + "\n")

        elif isinstance(bc, RatingCurve):
            flat = [v for pair in bc.pairs for v in pair]
            count = len(bc.pairs)
            out.append(f"Rating Curve= {count} \n")
            for dl in _format_data_block(flat):
                out.append(dl + "\n")
            out.append(f"DSS Path={bc.dss_path}\n")
            out.append(f"Use DSS={str(bc.use_dss)}\n")
            out.append(f"Use Fixed Start Time={str(bc.use_fixed_start)}\n")
            out.append(f"Fixed Start Date/Time={bc.fixed_start}\n")
            out.append(f"Is Critical Boundary={str(bc.is_critical)}\n")
            out.append(f"Critical Boundary Flow={bc.critical_boundary_flow}\n")
            for el in bc._extra_lines:
                out.append(el if el.endswith("\n") else el + "\n")

        elif isinstance(bc, FrictionSlope):
            out.append(f"Friction Slope={bc.slope},{int(bc.value2)}\n")

        elif isinstance(bc, NormalDepth):
            out.append(f"Normal Depth={bc.slope}\n")

        elif isinstance(bc, GateBoundary):
            for gate in bc.gates:
                out.append(f"Gate Name={gate.gate_name}\n")
                out.append(f"Gate DSS Path={gate.dss_path}\n")
                out.append(f"Gate Use DSS={str(gate.use_dss)}\n")
                out.append(f"Gate Time Interval={gate.time_interval}\n")
                out.append(f"Gate Use Fixed Start Time={str(gate.use_fixed_start)}\n")
                out.append(f"Gate Fixed Start Date/Time={gate.fixed_start}\n")
                count = len(gate.values)
                out.append(f"Gate Openings= {count} \n")
                for dl in _format_data_block(gate.values):
                    out.append(dl + "\n")

        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> None:
        """Reconstruct and write the unsteady flow file.

        The file is built as:
        1. Header lines (verbatim from parse)
        2. Initial condition lines (reconstructed from objects)
        3. Boundary section (reconstructed from :attr:`boundaries`)
        4. Trailing lines (verbatim from parse)

        Parameters
        ----------
        path:
            Destination path.  Overwrites the source file if omitted.
        """
        dest = Path(path) if path is not None else self._path

        out: list[str] = []

        # 1. Header
        out.extend(self._header_lines)

        # 2. Initial conditions
        for loc in self.initial_flow_locs:
            out.append(f"Initial Flow Loc={loc._to_raw()}\n")
        for se in self.initial_storage_elevs:
            out.append(f"Initial Storage Elev={se._to_raw()}\n")
        for re_ in self.initial_rainfall_runoff_elevs:
            out.append(f"Initial RRR Elev={re_._to_raw()}\n")

        # 3. Boundaries
        for bc in self.boundaries:
            out.extend(self._boundary_to_lines(bc))

        # 4. Trailing
        out.extend(self._trailing_lines)

        with open(dest, "w", encoding="utf-8") as fh:
            fh.writelines(out)
        self._modified = False
