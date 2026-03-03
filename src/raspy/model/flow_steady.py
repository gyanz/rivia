"""Read/write HEC-RAS steady flow files (.f**).

:class:`SteadyFlowFile` — verbatim-line editor.  All lines are stored
verbatim; only the specific data blocks that are mutated are reformatted.
``save()`` is byte-faithful for every unmodified line.

File format notes:

- Flow data: each ``River Rch & RM=River,Reach,RS`` line is immediately
  followed by *N* flow values (one per profile) in 8-char fixed-width
  columns, up to 10 per row.  No ``Num Of Flows=`` or ``Flow=`` prefix.
- Boundary conditions: one ``Boundary for River Rch & Prof#=River,Reach,N``
  block **per reach endpoint per profile**.  Up/Dn type codes:

  .. code-block:: text

      0 — None
      1 — Known WS        (``Up/Dn Known WS=<value>``)
      2 — Critical Depth  (no additional data)
      3 — Normal Depth    (``Up/Dn Slope=<value>``)
      4 — Rating Curve    (``Up/Dn Nval=<count>`` + interleaved stage/flow pairs)

Derived from format inspection of HEC-RAS 6.6 example files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from pathlib import Path


# ---------------------------------------------------------------------------
# Formatting helpers (same algorithm as flow_unsteady)
# ---------------------------------------------------------------------------

_COL_WIDTH = 8
_COLS_PER_ROW = 10


def _fit_width(value: float, width: int = _COL_WIDTH) -> str:
    """Right-justify *value* inside *width* characters.

    Tries integer, then progressively fewer decimal places, then scientific
    notation.  Truncates as a last resort.
    """
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
    values: list[float],
    cols: int = _COLS_PER_ROW,
    width: int = _COL_WIDTH,
) -> list[str]:
    """Return fixed-width data lines (no trailing newline)."""
    lines: list[str] = []
    for i in range(0, len(values), cols):
        chunk = values[i : i + cols]
        lines.append("".join(_fit_width(v, width) for v in chunk))
    return lines


def _parse_data_block(
    lines: list[str],
    count: int,
    width: int = _COL_WIDTH,
) -> list[float]:
    """Parse up to *count* fixed-width values from *lines*."""
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
    return ceil(n / cols) if n > 0 else 0


# ---------------------------------------------------------------------------
# File-section keys
# ---------------------------------------------------------------------------

_KEY_FLOW_LOC = "River Rch & RM"
_KEY_BOUNDARY = "Boundary for River Rch & Prof#"

# Lines that signal the start of trailing (non-boundary, non-flow) content.
_TRAILING_STARTS = (
    "DSS Import",
    "Storage Area Elev=",
    "Observed WS=",
)

# BC type codes
_BC_NONE = 0
_BC_KNOWN_WS = 1
_BC_CRITICAL_DEPTH = 2
_BC_NORMAL_DEPTH = 3
_BC_RATING_CURVE = 4


# ---------------------------------------------------------------------------
# Boundary condition dataclass
# ---------------------------------------------------------------------------

@dataclass
class SteadyBoundary:
    """Boundary conditions for one river/reach endpoint for one flow profile.

    ``up_type`` / ``dn_type`` encode the condition type::

        0 — None           (no BC)
        1 — Known WS       (``up_known_ws`` / ``dn_known_ws``)
        2 — Critical Depth (no additional data required)
        3 — Normal Depth   (``up_slope`` / ``dn_slope``)
        4 — Rating Curve   (``up_rating_curve`` / ``dn_rating_curve``)

    ``profile`` is 1-based and matches the profile index in the steady flow
    file (``Boundary for River Rch & Prof#=..., 1``).
    """

    river: str
    reach: str
    profile: int
    up_type: int = 0
    dn_type: int = 0
    up_known_ws: float | None = None
    dn_known_ws: float | None = None
    up_slope: float | None = None
    dn_slope: float | None = None
    up_rating_curve: list[tuple[float, float]] = field(default_factory=list)
    dn_rating_curve: list[tuple[float, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SteadyFlowFile — verbatim-line editor
# ---------------------------------------------------------------------------

class SteadyFlowFile:
    """Verbatim-line editor for HEC-RAS steady flow files (.f**).

    All lines are loaded into memory verbatim.  Targeted edits (e.g.
    replacing flow values at a cross-section or updating a boundary
    condition) splice new formatted lines into the list while leaving every
    other line byte-identical.  ``save()`` writes the list back; a no-op
    parse+save produces an identical file.

    Flow data is accessed by river / reach / river-station string
    (``get_flows`` / ``set_flows``).  Boundary conditions are accessed by
    river / reach / 1-based profile number (``get_boundary`` /
    ``set_boundary``).

    Derived from format inspection of HEC-RAS 6.6 example files.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.is_file():
            raise FileNotFoundError(f"Steady flow file not found: {self._path}")
        with open(self._path, encoding="utf-8", errors="replace") as fh:
            self._lines: list[str] = fh.readlines()

    # ------------------------------------------------------------------
    # Internal helpers
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
        raise KeyError(f"Key not found in steady flow file: {key!r}")

    def _splice(self, start: int, old_count: int, new_lines: list[str]) -> None:
        """Replace *old_count* lines starting at *start* with *new_lines*."""
        self._lines[start : start + old_count] = [
            (ln if ln.endswith("\n") else ln + "\n") for ln in new_lines
        ]

    @staticmethod
    def _match_fields(tail: str, river: str, reach: str, third: str | int) -> bool:
        """Return True if the first three comma-separated fields of *tail*
        match (*river*, *reach*, *third*) after stripping and lowercasing."""
        parts = tail.split(",", 3)
        if len(parts) < 3:
            return False
        return (
            parts[0].strip().lower() == river.strip().lower()
            and parts[1].strip().lower() == reach.strip().lower()
            and parts[2].strip() == str(third).strip()
        )

    @staticmethod
    def _match_fields_ci(tail: str, river: str, reach: str, third: str) -> bool:
        """Case-insensitive version of _match_fields for RS comparisons."""
        parts = tail.split(",", 3)
        if len(parts) < 3:
            return False
        return (
            parts[0].strip().lower() == river.strip().lower()
            and parts[1].strip().lower() == reach.strip().lower()
            and parts[2].strip().lower() == third.strip().lower()
        )

    def _find_flow_location(self, river: str, reach: str, rs: str) -> int | None:
        """Return the line index of the matching ``River Rch & RM=`` line."""
        prefix = _KEY_FLOW_LOC + "="
        for i, line in enumerate(self._lines):
            if line.startswith(prefix):
                if self._match_fields_ci(line[len(prefix) :], river, reach, rs):
                    return i
        return None

    def _find_flow_end(self, loc_i: int) -> int:
        """Return the index of the first line *after* the flow data block."""
        n = len(self._lines)
        i = loc_i + 1
        while i < n:
            line = self._lines[i]
            if line.startswith(_KEY_FLOW_LOC + "=") or line.startswith(
                _KEY_BOUNDARY + "="
            ):
                return i
            i += 1
        return n

    def _find_boundary_location(
        self, river: str, reach: str, profile: int
    ) -> int | None:
        """Return the line index of the matching ``Boundary for River Rch & Prof#=``
        line for the given profile number."""
        prefix = _KEY_BOUNDARY + "="
        for i, line in enumerate(self._lines):
            if not line.startswith(prefix):
                continue
            tail = line[len(prefix) :]
            parts = tail.split(",", 3)
            if len(parts) < 3:
                continue
            try:
                file_profile = int(parts[2].strip())
            except ValueError:
                continue
            if (
                parts[0].strip().lower() == river.strip().lower()
                and parts[1].strip().lower() == reach.strip().lower()
                and file_profile == profile
            ):
                return i
        return None

    def _find_boundary_end(self, loc_i: int) -> int:
        """Return the index of the first line *after* the boundary block."""
        n = len(self._lines)
        i = loc_i + 1
        while i < n:
            line = self._lines[i]
            if line.startswith(_KEY_BOUNDARY + "=") or line.startswith(
                _KEY_FLOW_LOC + "="
            ):
                return i
            if any(line.startswith(t) for t in _TRAILING_STARTS):
                return i
            i += 1
        return n

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
        """HEC-RAS version string.

        Checks ``Program Version=`` first (modern format), then falls back to
        ``Version=`` (pre-v4 files such as older WAILUPE-style projects).
        Treat as read-only; HEC-RAS manages this field.
        """
        v = self._get("Program Version")
        return v if v is not None else self._get("Version")

    @property
    def n_profiles(self) -> int | None:
        """Number of flow profiles (``Number of Profiles=``)."""
        raw = self._get("Number of Profiles")
        if raw is None:
            return None
        try:
            return int(raw.strip())
        except ValueError:
            return None

    @n_profiles.setter
    def n_profiles(self, value: int) -> None:
        self._set("Number of Profiles", f" {value} ")

    @property
    def profile_names(self) -> list[str]:
        """Profile names from the header ``Profile Names=`` line, stripped.

        Returns an empty list if the key is absent.
        """
        raw = self._get("Profile Names")
        if raw is None:
            return []
        return [name.strip() for name in raw.split(",") if name.strip()]

    @profile_names.setter
    def profile_names(self, names: list[str]) -> None:
        """Replace the ``Profile Names=`` line.

        Raises ``KeyError`` if the key is not present in the file.
        """
        self._set("Profile Names", ",".join(names))

    # ------------------------------------------------------------------
    # Flow data
    # ------------------------------------------------------------------

    def get_flows(self, river: str, reach: str, rs: str) -> list[float] | None:
        """Return flow values (one per profile) at the given location.

        Returns ``None`` if no matching ``River Rch & RM`` line is found.
        Returns ``[]`` if the location exists but contains no parseable values.
        """
        loc_i = self._find_flow_location(river, reach, rs)
        if loc_i is None:
            return None
        end_i = self._find_flow_end(loc_i)
        data_lines = [ln.rstrip("\n") for ln in self._lines[loc_i + 1 : end_i]]
        n = self.n_profiles
        if n is None:
            # Parse as many values as the data lines can provide.
            n = _COLS_PER_ROW * len(data_lines)
        return _parse_data_block(data_lines, n)

    def set_flows(
        self, river: str, reach: str, rs: str, values: list[float]
    ) -> None:
        """Replace the flow values at the given location.

        The number of values in *values* need not match the current count;
        the data lines are replaced wholesale.

        Args:
            river: River name (case-insensitive match).
            reach: Reach name (case-insensitive match).
            rs: River station string (stripped comparison).
            values: New flow values, one per profile.

        Raises:
            KeyError: No matching ``River Rch & RM`` line found.
        """
        loc_i = self._find_flow_location(river, reach, rs)
        if loc_i is None:
            raise KeyError(
                f"No flow location found for {river!r}, {reach!r}, {rs!r}"
            )
        end_i = self._find_flow_end(loc_i)
        old_line_count = end_i - (loc_i + 1)
        self._splice(loc_i + 1, old_line_count, _format_data_block(values))

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def get_boundary(
        self, river: str, reach: str, profile: int
    ) -> SteadyBoundary | None:
        """Return the boundary conditions for the given reach endpoint and profile.

        Returns ``None`` if no matching ``Boundary for River Rch & Prof#`` line
        is found.
        """
        loc_i = self._find_boundary_location(river, reach, profile)
        if loc_i is None:
            return None

        bc = SteadyBoundary(river=river, reach=reach, profile=profile)
        end_i = self._find_boundary_end(loc_i)
        i = loc_i + 1

        while i < end_i:
            line = self._lines[i]
            stripped = line.strip()
            if not stripped or "=" not in stripped:
                i += 1
                continue

            key, _, val = stripped.partition("=")
            key = key.strip()
            val = val.strip()

            if key == "Up Type":
                bc.up_type = int(val)
            elif key == "Dn Type":
                bc.dn_type = int(val)
            elif key == "Up Known WS":
                bc.up_known_ws = float(val)
            elif key == "Dn Known WS":
                bc.dn_known_ws = float(val)
            elif key == "Up Slope":
                bc.up_slope = float(val)
            elif key == "Dn Slope":
                bc.dn_slope = float(val)
            elif key == "Up Nval":
                count = int(val)
                nlines = _data_line_count(count * 2)
                data_lines = [
                    ln.rstrip("\n")
                    for ln in self._lines[i + 1 : i + 1 + nlines]
                ]
                flat = _parse_data_block(data_lines, count * 2)
                bc.up_rating_curve = [
                    (flat[j], flat[j + 1]) for j in range(0, len(flat), 2)
                ]
                i += nlines
            elif key == "Dn Nval":
                count = int(val)
                nlines = _data_line_count(count * 2)
                data_lines = [
                    ln.rstrip("\n")
                    for ln in self._lines[i + 1 : i + 1 + nlines]
                ]
                flat = _parse_data_block(data_lines, count * 2)
                bc.dn_rating_curve = [
                    (flat[j], flat[j + 1]) for j in range(0, len(flat), 2)
                ]
                i += nlines
            i += 1

        return bc

    def set_boundary(self, boundary: SteadyBoundary) -> None:
        """Replace the boundary conditions at the given location.

        The entire boundary block (from ``Boundary for River Rch & Prof#=``
        through its fields) is rebuilt from *boundary*.  Fields irrelevant
        to the set type are omitted (e.g. ``Dn Slope=`` is not written when
        ``dn_type != 3``).

        Raises:
            KeyError: No matching ``Boundary for River Rch & Prof#`` line found.
        """
        river, reach, profile = boundary.river, boundary.reach, boundary.profile
        loc_i = self._find_boundary_location(river, reach, profile)
        if loc_i is None:
            raise KeyError(
                f"No boundary found for {river!r}, {reach!r}, profile {profile}"
            )
        end_i = self._find_boundary_end(loc_i)

        # Preserve the original location header line verbatim.
        new_lines: list[str] = [self._lines[loc_i].rstrip("\n")]

        new_lines.append(f"Up Type= {boundary.up_type} ")
        if boundary.up_type == _BC_KNOWN_WS and boundary.up_known_ws is not None:
            new_lines.append(f"Up Known WS={boundary.up_known_ws}")
        elif boundary.up_type == _BC_NORMAL_DEPTH and boundary.up_slope is not None:
            new_lines.append(f"Up Slope={boundary.up_slope}")
        elif boundary.up_type == _BC_RATING_CURVE and boundary.up_rating_curve:
            count = len(boundary.up_rating_curve)
            new_lines.append(f"Up Nval= {count}")
            flat = [v for pair in boundary.up_rating_curve for v in pair]
            new_lines.extend(_format_data_block(flat))

        new_lines.append(f"Dn Type= {boundary.dn_type} ")
        if boundary.dn_type == _BC_KNOWN_WS and boundary.dn_known_ws is not None:
            new_lines.append(f"Dn Known WS={boundary.dn_known_ws}")
        elif boundary.dn_type == _BC_NORMAL_DEPTH and boundary.dn_slope is not None:
            new_lines.append(f"Dn Slope={boundary.dn_slope}")
        elif boundary.dn_type == _BC_RATING_CURVE and boundary.dn_rating_curve:
            count = len(boundary.dn_rating_curve)
            new_lines.append(f"Dn Nval= {count}")
            flat = [v for pair in boundary.dn_rating_curve for v in pair]
            new_lines.extend(_format_data_block(flat))

        self._splice(loc_i, end_i - loc_i, new_lines)

    def get_boundaries(self, river: str, reach: str) -> list[SteadyBoundary]:
        """Return all boundary conditions for a river/reach, sorted by profile.

        Returns an empty list if no matching boundaries are found.
        """
        prefix = _KEY_BOUNDARY + "="
        profiles: list[int] = []
        for line in self._lines:
            if not line.startswith(prefix):
                continue
            tail = line[len(prefix) :]
            parts = tail.split(",", 3)
            if len(parts) < 3:
                continue
            if (
                parts[0].strip().lower() == river.strip().lower()
                and parts[1].strip().lower() == reach.strip().lower()
            ):
                try:
                    profiles.append(int(parts[2].strip()))
                except ValueError:
                    pass
        result = []
        for p in sorted(profiles):
            bc = self.get_boundary(river, reach, p)
            if bc is not None:
                result.append(bc)
        return result

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
