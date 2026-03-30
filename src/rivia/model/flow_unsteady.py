"""Read/write HEC-RAS unsteady flow files (.u**).

Two classes are provided:

- :class:`UnsteadyFlowFile` — verbatim-line editor.  All lines are stored
  verbatim; only the specific data blocks that are mutated are reformatted.
  ``save()`` is byte-faithful for every unmodified line.  Boundary ordering
  is always preserved.  Use this for targeted, one-off edits.

- :class:`UnsteadyFlowEditor` — structured editor.  Boundary conditions are
  parsed into typed dataclass objects and may be sorted by river station.
  ``save()`` reconstructs the boundary section from the objects; trailing
  meteorological / non-Newtonian lines are still written verbatim.  Use this
  when you need to reorder boundaries.

Derived from: ``archive/ras_tools/unsteadyFlowParser.py``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Literal

logger = logging.getLogger("rivia.model")

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
class InitialRRRElev:
    """Initial water surface elevation for a reservoir / RRR."""

    river: str
    reach: str
    river_station: str
    elevation: float

    @classmethod
    def _from_raw(cls, raw: str) -> "InitialRRRElev":
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

        Args:
            rs_float: If ``True``, river_station is returned as ``float``
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
class FlowHydrograph(_Boundary):
    """Upstream / internal flow hydrograph boundary."""

    interval: str = "1HOUR"
    values: list[float] = field(default_factory=list)
    flow_hydrograph_slope: str | None = None
    stage_tw_check: int = 0
    dss_file: str = ""
    dss_path: str = ""
    use_dss: bool = False
    use_fixed_start: bool = False
    fixed_start: str = ","
    is_critical: bool = False
    critical_boundary_flow: str = ""
    # Extra lines between the standard fields and next Boundary Location
    # that we don't model explicitly (e.g. CWMS InputPosition).
    _extra_lines: list[str] = field(default_factory=list, repr=False)


@dataclass
class LateralInflow(_Boundary):
    """Lateral or uniform lateral inflow hydrograph."""

    interval: str = "1HOUR"
    values: list[float] = field(default_factory=list)
    is_uniform: bool = False
    dss_file: str = ""
    dss_path: str = ""
    use_dss: bool = False
    use_fixed_start: bool = False
    fixed_start: str = ","
    is_critical: bool = False
    critical_boundary_flow: str = ""
    _extra_lines: list[str] = field(default_factory=list, repr=False)


@dataclass
class StageHydrograph(_Boundary):
    """Stage (water-surface) hydrograph boundary."""

    interval: str = "1HOUR"
    values: list[float] = field(default_factory=list)
    dss_path: str = ""
    use_dss: bool = False
    use_fixed_start: bool = False
    fixed_start: str = ","
    _extra_lines: list[str] = field(default_factory=list, repr=False)


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
                        if k == "DSS File":
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
# UnsteadyFlowFile — verbatim-line editor
# ---------------------------------------------------------------------------


class UnsteadyFlowFile:
    """Verbatim-line editor for HEC-RAS unsteady flow files (.u**).

    All lines are loaded into memory verbatim.  Targeted edits (e.g.
    replacing a flow hydrograph) splice new formatted lines into the list
    while leaving every other line byte-identical.  ``save()`` writes the
    list back; a no-op parse+save produces an identical file.

    Boundary ordering is always preserved.  If you need to reorder
    boundaries by river station, use :class:`UnsteadyFlowEditor` instead.

    Derived from: ``archive/ras_tools/unsteadyFlowParser.py``
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.is_file():
            raise FileNotFoundError(f"Unsteady flow file not found: {self._path}")
        with open(self._path, encoding="utf-8", errors="replace") as fh:
            self._lines: list[str] = fh.readlines()

    # ------------------------------------------------------------------
    # Internal helpers (same pattern as PlanFile)
    # ------------------------------------------------------------------

    def _get(self, key: str) -> str | None:
        prefix = key + "="
        for line in self._lines:
            if line.startswith(prefix):
                value = line[len(prefix) :].strip()
                return value if value else None
        return None

    def _set(self, key: str, raw_value: str) -> None:
        prefix = key + "="
        for i, line in enumerate(self._lines):
            if line.startswith(prefix):
                self._lines[i] = f"{prefix}{raw_value}\n"
                return
        raise KeyError(f"Key not found in unsteady flow file: {key!r}")

    def _find_boundary_location(self, river: str, reach: str, rs: str) -> int | None:
        """Return the line index of the matching ``Boundary Location=`` line.

        Matching is done by stripping and case-insensitive comparison of
        river, reach, and river-station fields.
        """
        prefix = "Boundary Location="
        for i, line in enumerate(self._lines):
            if not line.startswith(prefix):
                continue
            tail = line[len(prefix) :]
            parts = tail.split(",", 3)
            if len(parts) < 3:
                continue
            if (
                parts[0].strip().lower() == river.strip().lower()
                and parts[1].strip().lower() == reach.strip().lower()
                and parts[2].strip().lower() == str(rs).strip().lower()
            ):
                return i
        return None

    def _splice(self, start: int, old_count: int, new_lines: list[str]) -> None:
        """Replace *old_count* lines starting at *start* with *new_lines*."""
        self._lines[start : start + old_count] = [
            (ln if ln.endswith("\n") else ln + "\n") for ln in new_lines
        ]

    # ------------------------------------------------------------------
    # Generic escape hatch
    # ------------------------------------------------------------------

    def get(self, key: str) -> str | None:
        """Return the raw stripped value for *key*, or ``None`` if absent."""
        return self._get(key)

    def set(self, key: str, value: str) -> None:
        """Set *key* to *value* verbatim.  Raises ``KeyError`` if absent."""
        self._set(key, value)

    # ------------------------------------------------------------------
    # Scalar properties
    # ------------------------------------------------------------------

    @property
    def flow_title(self) -> str | None:
        """Flow title (``Flow Title=``)."""
        return self._get("Flow Title")

    @flow_title.setter
    def flow_title(self, value: str) -> None:
        self._set("Flow Title", value)

    @property
    def program_version(self) -> str | None:
        """HEC-RAS version string (``Program Version=``)."""
        return self._get("Program Version")

    def _set_restart_filename(self, filename: str) -> None:
        prefix = "Restart Filename="
        for i, line in enumerate(self._lines):
            if line.startswith(prefix):
                self._lines[i] = f"{prefix}{filename}\n"
                return
        ur_prefix = "Use Restart="
        for i, line in enumerate(self._lines):
            if line.startswith(ur_prefix):
                self._lines.insert(i + 1, f"{prefix}{filename}\n")
                return
        raise KeyError("'Use Restart' not found; cannot insert 'Restart Filename'")

    @property
    def restart(self) -> tuple[int, str | None]:
        """Return ``(flag, filename)`` for the restart configuration.

        *flag* is the ``Use Restart`` value (``0`` = disabled, ``1`` = enabled).
        *filename* is the ``Restart Filename`` value, or ``None`` if absent.
        """
        raw = self._get("Use Restart")
        flag = int(raw.strip()) if raw is not None else 0
        filename = self._get("Restart Filename")
        return (flag, filename)

    @restart.setter
    def restart(self, value: int | bool | str | None) -> None:
        if value is None or value is False:
            self._set("Use Restart", " 0 ")
        elif value is True:
            self._set("Use Restart", " 1 ")
        elif isinstance(value, str):
            self._set_restart_filename(value)
            self._set("Use Restart", " 1 ")
        else:
            self._set("Use Restart", " 1 " if value else " 0 ")

    # ------------------------------------------------------------------
    # Flow hydrograph
    # ------------------------------------------------------------------

    def get_flow_hydrograph(
        self, river: str, reach: str, rs: str
    ) -> list[float] | None:
        """Return flow hydrograph values for the given location.

        Returns ``None`` if no matching boundary is found.
        """
        loc_i = self._find_boundary_location(river, reach, rs)
        if loc_i is None:
            return None

        n = len(self._lines)
        i = loc_i + 1
        while i < n:
            line = self._lines[i]
            if line.startswith("Boundary Location="):
                break
            if line.startswith("Flow Hydrograph="):
                count = int(line.split("=", 1)[1].strip())
                nlines = _data_line_count(count)
                data_lines = [
                    l.rstrip("\n") for l in self._lines[i + 1 : i + 1 + nlines]
                ]
                return _parse_data_block(data_lines, count)
            i += 1
        return None

    def set_flow_hydrograph(
        self, river: str, reach: str, rs: str, values: _Values
    ) -> None:
        """Replace the flow hydrograph at the given location.

        Args:
            river: River name (case-insensitive match).
            reach: Reach name (case-insensitive match).
            rs: River station string.
            values: New flow values.  A scalar is broadcast to the length of
                the existing time series.

        Raises:
            KeyError: No matching ``Boundary Location`` line.
            ValueError: Boundary found but is not a flow hydrograph.
        """
        loc_i = self._find_boundary_location(river, reach, rs)
        if loc_i is None:
            raise KeyError(
                f"No Boundary Location found for {river!r}, {reach!r}, {rs!r}"
            )

        n = len(self._lines)
        i = loc_i + 1
        while i < n:
            line = self._lines[i]
            if line.startswith("Boundary Location="):
                break
            if line.startswith("Flow Hydrograph="):
                old_count = int(line.split("=", 1)[1].strip())
                old_nlines = _data_line_count(old_count)
                resolved = _coerce_values(values, old_count)
                new_count = len(resolved)
                self._lines[i] = f"Flow Hydrograph= {new_count} \n"
                self._splice(i + 1, old_nlines, _format_data_block(resolved))
                return
            i += 1
        raise ValueError(
            f"Boundary at {river!r}, {reach!r}, {rs!r} has no Flow Hydrograph"
        )

    # ------------------------------------------------------------------
    # Lateral inflow hydrograph
    # ------------------------------------------------------------------

    def get_lateral_inflow(self, river: str, reach: str, rs: str) -> list[float] | None:
        """Return lateral inflow values for the given location."""
        loc_i = self._find_boundary_location(river, reach, rs)
        if loc_i is None:
            return None

        n = len(self._lines)
        i = loc_i + 1
        while i < n:
            line = self._lines[i]
            if line.startswith("Boundary Location="):
                break
            key = line.split("=", 1)[0]
            if key in (
                "Lateral Inflow Hydrograph",
                "Uniform Lateral Inflow Hydrograph",
            ):
                count = int(line.split("=", 1)[1].strip())
                nlines = _data_line_count(count)
                data_lines = [
                    l.rstrip("\n") for l in self._lines[i + 1 : i + 1 + nlines]
                ]
                return _parse_data_block(data_lines, count)
            i += 1
        return None

    def set_lateral_inflow(
        self, river: str, reach: str, rs: str, values: _Values
    ) -> None:
        """Replace the lateral inflow at the given location.

        Args:
            values: New flow values.  A scalar is broadcast to the length of
                the existing time series.

        Raises:
            KeyError: No matching ``Boundary Location`` line.
            ValueError: Boundary found but has no lateral inflow hydrograph.
        """
        loc_i = self._find_boundary_location(river, reach, rs)
        if loc_i is None:
            raise KeyError(
                f"No Boundary Location found for {river!r}, {reach!r}, {rs!r}"
            )

        n = len(self._lines)
        i = loc_i + 1
        while i < n:
            line = self._lines[i]
            if line.startswith("Boundary Location="):
                break
            key = line.split("=", 1)[0]
            if key in (
                "Lateral Inflow Hydrograph",
                "Uniform Lateral Inflow Hydrograph",
            ):
                old_count = int(line.split("=", 1)[1].strip())
                old_nlines = _data_line_count(old_count)
                resolved = _coerce_values(values, old_count)
                new_count = len(resolved)
                self._lines[i] = f"{key}= {new_count} \n"
                self._splice(i + 1, old_nlines, _format_data_block(resolved))
                return
            i += 1
        raise ValueError(
            f"Boundary at {river!r}, {reach!r}, {rs!r} has no Lateral Inflow Hydrograph"
        )

    # ------------------------------------------------------------------
    # Gate openings
    # ------------------------------------------------------------------

    def get_gate_openings(
        self, river: str, reach: str, rs: str, gate_name: str
    ) -> list[float] | None:
        """Return gate opening values for the given location and gate name."""
        loc_i = self._find_boundary_location(river, reach, rs)
        if loc_i is None:
            return None

        n = len(self._lines)
        i = loc_i + 1
        current_gate: str | None = None
        while i < n:
            line = self._lines[i]
            if line.startswith("Boundary Location="):
                break
            if line.startswith("Gate Name="):
                current_gate = line.split("=", 1)[1].strip()
            elif line.startswith("Gate Openings=") and current_gate is not None:
                if current_gate.lower() == gate_name.strip().lower():
                    count = int(line.split("=", 1)[1].strip())
                    nlines = _data_line_count(count)
                    data_lines = [
                        l.rstrip("\n") for l in self._lines[i + 1 : i + 1 + nlines]
                    ]
                    return _parse_data_block(data_lines, count)
            i += 1
        return None

    def set_gate_opening(
        self, river: str, reach: str, rs: str, gate_name: str, values: _Values
    ) -> None:
        """Replace gate opening values for the given location and gate name.

        Args:
            values: New opening values.  A scalar is broadcast to the length
                of the existing gate opening time series.

        Raises:
            KeyError: Boundary or gate name not found.
        """
        loc_i = self._find_boundary_location(river, reach, rs)
        if loc_i is None:
            raise KeyError(
                f"No Boundary Location found for {river!r}, {reach!r}, {rs!r}"
            )

        n = len(self._lines)
        i = loc_i + 1
        current_gate: str | None = None
        while i < n:
            line = self._lines[i]
            if line.startswith("Boundary Location="):
                break
            if line.startswith("Gate Name="):
                current_gate = line.split("=", 1)[1].strip()
            elif line.startswith("Gate Openings=") and current_gate is not None:
                if current_gate.lower() == gate_name.strip().lower():
                    old_count = int(line.split("=", 1)[1].strip())
                    old_nlines = _data_line_count(old_count)
                    resolved = _coerce_values(values, old_count)
                    new_count = len(resolved)
                    self._lines[i] = f"Gate Openings= {new_count} \n"
                    self._splice(i + 1, old_nlines, _format_data_block(resolved))
                    return
            i += 1
        raise KeyError(f"Gate {gate_name!r} not found at {river!r}, {reach!r}, {rs!r}")

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------

    def get_initial_flow(self, river: str, reach: str, rs: str) -> float | None:
        """Return the initial flow at the given location, or ``None``."""
        for line in self._lines:
            if not line.startswith("Initial Flow Loc="):
                continue
            raw = line[len("Initial Flow Loc=") :].strip()
            parts = raw.split(",")
            if (
                len(parts) >= 4
                and parts[0].strip().lower() == river.strip().lower()
                and parts[1].strip().lower() == reach.strip().lower()
                and parts[2].strip().lower() == str(rs).strip().lower()
            ):
                return float(parts[3].strip())
        return None

    def set_initial_flow(self, river: str, reach: str, rs: str, flow: float) -> None:
        """Update the initial flow at the given location.

        Raises ``KeyError`` if not found.
        """
        for i, line in enumerate(self._lines):
            if not line.startswith("Initial Flow Loc="):
                continue
            raw = line[len("Initial Flow Loc=") :].rstrip("\n")
            parts = raw.split(",")
            if (
                len(parts) >= 4
                and parts[0].strip().lower() == river.strip().lower()
                and parts[1].strip().lower() == reach.strip().lower()
                and parts[2].strip().lower() == str(rs).strip().lower()
            ):
                parts[3] = str(flow)
                self._lines[i] = "Initial Flow Loc=" + ",".join(parts) + "\n"
                return
        raise KeyError(f"Initial Flow Loc not found for {river!r}, {reach!r}, {rs!r}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> None:
        """Write all in-memory lines back to disk.

        If *path* is omitted the source file is overwritten.
        """
        dest = Path(path) if path is not None else self._path
        with open(dest, "w", encoding="utf-8") as fh:
            fh.writelines(self._lines)


# ---------------------------------------------------------------------------
# UnsteadyFlowEditor — structured, sortable editor
# ---------------------------------------------------------------------------


class UnsteadyFlowEditor:
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
        from parsed objects.  Use :class:`UnsteadyFlowFile` when you need a
        faithful roundtrip.

    Derived from: ``archive/ras_tools/unsteadyFlowParser.py``
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
        self.initial_rrr_elevs: list[InitialRRRElev] = []
        for line in self._initial_lines:
            if line.startswith("Initial Flow Loc="):
                raw = line[len("Initial Flow Loc=") :].strip()
                self.initial_flow_locs.append(InitialFlowLoc._from_raw(raw))
            elif line.startswith("Initial Storage Elev="):
                raw = line[len("Initial Storage Elev=") :].strip()
                self.initial_storage_elevs.append(InitialStorageElev._from_raw(raw))
            elif line.startswith("Initial RRR Elev="):
                raw = line[len("Initial RRR Elev=") :].strip()
                self.initial_rrr_elevs.append(InitialRRRElev._from_raw(raw))

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
        return self._header_get("Flow Title")

    @flow_title.setter
    def flow_title(self, value: str) -> None:
        self._header_set("Flow Title", value)

    @property
    def program_version(self) -> str | None:
        return self._header_get("Program Version")

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
        return [b for b in self.boundaries if isinstance(b, FlowHydrograph)]

    @property
    def lateral_inflows(self) -> list[LateralInflow]:
        return [b for b in self.boundaries if isinstance(b, LateralInflow)]

    @property
    def gate_boundaries(self) -> list[GateBoundary]:
        return [b for b in self.boundaries if isinstance(b, GateBoundary)]

    @property
    def friction_slopes(self) -> list[FrictionSlope]:
        return [b for b in self.boundaries if isinstance(b, FrictionSlope)]

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------

    def _sort_type(self, bc_type: type, *, ascending: bool) -> None:
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
                sorted(group, key=lambda t: t[1]._rs_float(), reverse=not ascending)
            )
        for (orig_idx, _), (_, sorted_bc) in zip(targets, sorted_targets):
            self.boundaries[orig_idx] = sorted_bc

    def sort_flow_hydrographs(self, *, ascending: bool = False) -> None:
        """Sort :class:`FlowHydrograph` entries by river station."""
        self._sort_type(FlowHydrograph, ascending=ascending)

    def sort_gate_boundaries(self, *, ascending: bool = False) -> None:
        """Sort :class:`GateBoundary` entries by river station.

        Other boundary types remain at their original positions.

        Args:
            ascending: If ``False`` (default), highest station first
                       (upstream -> downstream for standard RAS numbering).
                       Set to ``True`` for ascending (downstream -> upstream).
        """
        self._sort_type(GateBoundary, ascending=ascending)

    def sort_lateral_inflows(self, *, ascending: bool = False) -> None:
        """Sort :class:`LateralInflow` entries by river station."""
        self._sort_type(LateralInflow, ascending=ascending)

    # ------------------------------------------------------------------
    # Set by index (works naturally after sorting)
    # ------------------------------------------------------------------

    def set_flow_hydrograph(self, index: int, values: _Values) -> None:
        """Set flow hydrograph values by position in :attr:`flow_hydrographs`.

        Args:
            index: Position in the filtered flow-hydrograph list.
            values: New flow values.  A scalar is broadcast to the length of
                the existing time series.
        """
        bc = self.flow_hydrographs[index]
        bc.values = _coerce_values(values, len(bc.values))
        self._modified = True

    def set_lateral_inflow(self, index: int, values: _Values) -> None:
        """Set lateral inflow values by position in :attr:`lateral_inflows`.

        Args:
            values: New flow values.  A scalar is broadcast to the length of
                the existing time series.
        """
        bc = self.lateral_inflows[index]
        bc.values = _coerce_values(values, len(bc.values))
        self._modified = True

    def set_all_lateral_inflows(self, values: list[float | list[float]]) -> None:
        """Set lateral inflow values across all :class:`LateralInflow` boundaries.

        Args:
            values: One entry per lateral inflow (in file order).
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

        Args:
            index: Position in the filtered gate-boundary list.
            values: New opening values.  A scalar is broadcast to the length
                of the existing gate opening time series.
            gate_index: Which gate within the boundary (default 0).
        """
        gate = self.gate_boundaries[index].gates[gate_index]
        gate.values = _coerce_values(values, len(gate.values))
        self._modified = True

    def set_all_gate_opening(self, values: list[float | list[float]]) -> None:
        """Set gate opening values across all gates in all :class:`GateBoundary`.

        Args:
            values: One entry per gate (in order across all boundaries).
                Each entry is either a scalar ``float`` (broadcast to the
                gate's existing time-series length) or a ``list[float]``
                (used as-is).  If ``values`` is shorter than the total number
                of gates, the remaining gates are left unchanged.

        Derived from: ``archive/ras_tools/unsteadyFlowParser.py``
        """
        all_gates = [gate for gb in self.gate_boundaries for gate in gb.gates]
        for gate, v in zip(all_gates, values, strict=False):
            gate.values = _coerce_values(v, len(gate.values))
        self._modified = True

    # ------------------------------------------------------------------
    # Set by location (river / reach / rs)
    # ------------------------------------------------------------------

    def _find_boundary(self, river: str, reach: str, rs: str) -> BoundaryType | None:
        r = river.strip().lower()
        rc = reach.strip().lower()
        s = str(rs).strip().lower()
        for b in self.boundaries:
            if (
                b.river.lower() == r
                and b.reach.lower() == rc
                and b.river_station.lower() == s
            ):
                return b
        return None

    def set_flow_hydrograph_at(
        self, river: str, reach: str, rs: str, values: _Values
    ) -> None:
        """Set flow hydrograph values by location.

        Args:
            values: A scalar is broadcast to the existing time-series length.
        """
        b = self._find_boundary(river, reach, rs)
        if not isinstance(b, FlowHydrograph):
            raise KeyError(f"No FlowHydrograph at {river!r}, {reach!r}, {rs!r}")
        b.values = _coerce_values(values, len(b.values))
        self._modified = True

    def set_lateral_inflow_at(
        self, river: str, reach: str, rs: str, values: _Values
    ) -> None:
        """Set lateral inflow values by location.

        Args:
            values: A scalar is broadcast to the existing time-series length.
        """
        b = self._find_boundary(river, reach, rs)
        if not isinstance(b, LateralInflow):
            raise KeyError(f"No LateralInflow at {river!r}, {reach!r}, {rs!r}")
        b.values = _coerce_values(values, len(b.values))
        self._modified = True

    def set_gate_opening_at(
        self, river: str, reach: str, rs: str, gate: str | int, values: _Values
    ) -> None:
        """Set gate opening values by location and gate name or index.

        Args:
            gate: Gate name string, or a zero-based integer index into
                the boundary's gate list.
            values: A scalar is broadcast to the existing time-series length.
        """
        b = self._find_boundary(river, reach, rs)
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

        Args:
            index: Zero-based position in ``initial_flow_locs``.
            flow: New flow value.

        Raises:
            IndexError: *index* is out of range.
        """
        self.initial_flow_locs[index].flow = flow
        self._modified = True

    def set_initial_flow_at(self, river: str, reach: str, rs: str, flow: float) -> None:
        """Update the initial flow at the given location.

        Args:
            river: River name (case-insensitive match).
            reach: Reach name (case-insensitive match).
            rs: River station string.
            flow: New flow value.

        Raises:
            KeyError: No matching ``Initial Flow Loc`` entry found.
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

        Args:
            path: Destination path.  Overwrites the source file if omitted.
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
        for re_ in self.initial_rrr_elevs:
            out.append(f"Initial RRR Elev={re_._to_raw()}\n")

        # 3. Boundaries
        for bc in self.boundaries:
            out.extend(self._boundary_to_lines(bc))

        # 4. Trailing
        out.extend(self._trailing_lines)

        with open(dest, "w", encoding="utf-8") as fh:
            fh.writelines(out)
        self._modified = False
