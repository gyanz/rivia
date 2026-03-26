"""Read/write HEC-RAS geometry files (.g**).

:class:`GeometryFile` — verbatim-line editor for HEC-RAS 1-D geometry files
(.g01, .g02, …).  All lines are stored verbatim.  Typed access is provided
for:

- File metadata (``geom_title``, ``program_version``, ``viewing_rectangle``)
- Reach / node inventory (``reaches``, ``junctions``, ``node_rs_list``)
- Cross-section data: ``#Sta/Elev``, ``#Mann``, ``Bank Sta``,
  ``#XS Ineff``, ``Exp/Cntr``, ``Levee``, ``#Block Obstruct``,
  ``XS HTab Starting El and Incr``
  (read via :meth:`get_cross_section`, write via targeted setters)
- Structure nodes (bridge, culvert, inline/lateral structure) preserved
  verbatim and accessible via :meth:`get_node_lines`

``save()`` is byte-faithful for every unmodified line.

Node type codes in ``Type RM Length L Ch R = TYPE, ...``::

    1 — Cross Section
    2 — Culvert (single or twin-pipe)
    3 — Bridge
    4 — Multiple Opening
    5 — Inline Structure
    6 — Lateral Structure

Cross-section fixed-width format (8-char columns):

.. code-block:: text

    #Sta/Elev= N          alternating station/elevation pairs, 10 per row
    #Mann= N,t,a          triplets (station, n-value, variation), 9 per row
    #XS Ineff= N          triplets (x_start, x_end, elevation),  9 per row
                          followed by Permanent Ineff= flags (8-char, 10/row)
    #Block Obstruct= N,t  triplets (x_start, x_end, elevation),  9 per row
    Levee=lf,ls,le,rf,rs,re[,lfe,rfe]
                          left/right flag(-1=active), station, elevation,
                          optional failure elevations
    XS HTab Starting El and Incr=el,incr,count

Vertical (depth/flow-varying) Manning's n — appears between ``XS Rating Curve=``
and ``Exp/Cntr=`` when active:

.. code-block:: text

    Vertical n Elevations= N    N WSE or flow breakpoints (8-char cols, 10/row)
    Vertical n for Station=S    per-station entry; N n-values follow (8-char, 10/row)
    ...
    Vertical n Flow= F          F=0 → WSE breakpoints; F=-1 → flow breakpoints

When vertical n is active, ``#Mann= N , 0 , 0`` stores zone boundary stations
only; n-values in that block are all zero (placeholders).

Derived from format inspection of HEC-RAS 6.6 example files and
``archive/ras_tools/geomParser.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COL = 8  # fixed-width column for numerical data
_COLS_STAE = 10  # values per row in #Sta/Elev blocks  (5 pairs)
_COLS_MANN = 9  # values per row in #Mann / #XS Ineff  (3 triplets)
_COLS_FLAGS = 10  # flags per row in Permanent Ineff blocks

#: Node type: cross section
NODE_XS = 1
#: Node type: culvert
NODE_CULVERT = 2
#: Node type: bridge
NODE_BRIDGE = 3
#: Node type: multiple opening
NODE_MULTIPLE_OPENING = 4
#: Node type: inline structure
NODE_INLINE_STRUCTURE = 5
#: Node type: lateral structure
NODE_LATERAL_STRUCTURE = 6

_NODE_TYPE_NAMES: dict[int, str] = {
    NODE_XS: "CrossSection",
    NODE_CULVERT: "Culvert",
    NODE_BRIDGE: "Bridge",
    NODE_MULTIPLE_OPENING: "MultipleOpening",
    NODE_INLINE_STRUCTURE: "InlineStructure",
    NODE_LATERAL_STRUCTURE: "LateralStructure",
}

_KEY_NODE = "Type RM Length L Ch R"
_KEY_REACH = "River Reach"
_KEY_JUNCT = "Junct Name"

# ---------------------------------------------------------------------------
# Formatting helpers (same algorithm as flow_steady)
# ---------------------------------------------------------------------------


def _fit_width(value: float, width: int = _COL) -> str:
    """Right-justify *value* inside *width* chars.

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


def _format_block(values: list[float], cols: int, width: int = _COL) -> list[str]:
    """Return fixed-width data rows (no trailing newline)."""
    rows: list[str] = []
    for i in range(0, len(values), cols):
        rows.append("".join(_fit_width(v, width) for v in values[i : i + cols]))
    return rows


def _parse_block(lines: list[str], count: int, width: int = _COL) -> list[float]:
    """Parse up to *count* fixed-width values from *lines*, skipping blanks."""
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


def _parse_block_fixed(
    lines: list[str], count: int, width: int = _COL
) -> list[float]:
    """Read exactly *count* fixed-width positions; blank fields become 0.0.

    Unlike :func:`_parse_block`, blank columns are NOT skipped — they
    contribute a ``0.0``.  Required for ``#Block Obstruct`` data where
    absent endpoints are left blank rather than omitted.
    """
    values: list[float] = []
    for line in lines:
        pos = 0
        while pos + width <= len(line) and len(values) < count:
            token = line[pos : pos + width].strip()
            try:
                values.append(float(token) if token else 0.0)
            except ValueError:
                values.append(0.0)
            pos += width
        # Pad if line is shorter than expected
        while len(values) < count:
            values.append(0.0)
    return values[:count]


def _row_count(n: int, cols: int) -> int:
    return ceil(n / cols) if n > 0 else 0


def _fmt_levee_val(v: float | None) -> str:
    """Format a levee field value: integer when whole, else decimal, else ''."""
    if v is None:
        return ""
    return str(int(v)) if v == int(v) else str(v)


def _fmt_levee_line(left: LeveeData | None, right: LeveeData | None) -> str:
    """Build the ``Levee=`` line from left/right :class:`LeveeData`."""
    if left is not None:
        lf = f"-1,{_fmt_levee_val(left.station)},{_fmt_levee_val(left.elevation)}"
        l_fail = _fmt_levee_val(left.failure_elevation)
    else:
        lf, l_fail = "0,,", ""
    if right is not None:
        rf = f"-1,{_fmt_levee_val(right.station)},{_fmt_levee_val(right.elevation)}"
        r_fail = _fmt_levee_val(right.failure_elevation)
    else:
        rf, r_fail = "0,,", ""
    return f"Levee={lf},{rf},{l_fail},{r_fail}"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ManningEntry:
    """One horizontal Manning's n zone.

    Attributes:
        station:   Left boundary station of this n zone.
        n_value:   Manning's roughness coefficient.
        variation: Third column in the HEC-RAS file.  Usually ``0``; used
                   for vertical-n or alternative-n assignments.
    """

    station: float
    n_value: float
    variation: float = 0.0


@dataclass
class IneffArea:
    """One ineffective flow area interval.

    Attributes:
        x_start:   Left boundary station.
        x_end:     Right boundary station.
        elevation: Activation elevation (area is ineffective below this).
        permanent: ``True`` if always active (``T`` flag), ``False`` if
                   elevation-triggered (``F`` flag).
    """

    x_start: float
    x_end: float
    elevation: float
    permanent: bool = False


@dataclass
class LeveeData:
    """Levee definition for one bank of a cross section.

    Encoded on the ``Levee=`` line as a ``-1`` active flag followed by
    station and elevation.

    Attributes:
        station:           Lateral station of the levee crest.
        elevation:         Levee crest elevation.
        failure_elevation: Elevation at which the levee fails; ``None`` if
                           not specified.
    """

    station: float
    elevation: float
    failure_elevation: float | None = None


@dataclass
class BlockedObstruction:
    """One blocked-obstruction interval in a cross section (``#Block Obstruct``).

    Flow area between *x_start* and *x_end* is blocked up to *elevation*.
    Use a very large elevation (e.g. ``999``) for a permanently blocked zone.

    Attributes:
        x_start:   Left boundary station.
        x_end:     Right boundary station.
        elevation: Elevation ceiling of the blocked zone.
    """

    x_start: float
    x_end: float
    elevation: float


@dataclass
class VerticalNStation:
    """Manning's n values at one cross-section station, varying by depth or flow.

    Attributes:
        station:  Lateral station (same coordinate system as ``#Sta/Elev``).
        n_values: n-value at each breakpoint in :class:`VerticalN`.
    """

    station: float
    n_values: list[float]


@dataclass
class VerticalN:
    """Vertical (depth/flow-varying) Manning's n for a cross section.

    HEC-RAS stores this block between ``XS Rating Curve=`` and
    ``Exp/Cntr=`` when vertical variation is active.  The ``#Mann``
    block in the same XS still defines zone boundaries but carries
    placeholder zero n-values.

    Attributes:
        breakpoints: Water-surface elevations (``by_flow=False``) or
                     flow values (``by_flow=True``) at which n is
                     tabulated.  Length N.
        by_flow:     ``True`` when breakpoints are flows
                     (``Vertical n Flow=-1``); ``False`` when they
                     are WSE (``Vertical n Flow= 0``).
        stations:    Per-station n-value curves, each with N entries.
    """

    breakpoints: list[float]
    by_flow: bool
    stations: list[VerticalNStation]


@dataclass
class CrossSection:
    """Parsed data for one HEC-RAS cross section (node type 1).

    Returned by :meth:`GeometryFile.get_cross_section`.  Write changes back
    with the targeted setters on :class:`GeometryFile` (``set_mannings``,
    ``set_stations``, ``set_bank_stations``, ``set_exp_cntr``).

    Attributes:
        river:          River name.
        reach:          Reach name.
        rs:             River station string (normalised: no trailing
                        whitespace or ``*`` interpolation marker).
        description:    Node description from ``BEGIN/END DESCRIPTION``.
        stations:       Station values from ``#Sta/Elev``.
        elevations:     Elevation values from ``#Sta/Elev``.
        mann_entries:   Manning's n zones from ``#Mann``.
        mann_type:      Type flag from ``#Mann= N , type , alt`` header.
        mann_alt:       Alt flag from ``#Mann= N , type , alt`` header.
        bank_left:      Left bank station (``Bank Sta``).
        bank_right:     Right bank station (``Bank Sta``).
        ineff_areas:    Ineffective flow areas (``#XS Ineff``).
        expansion:      Expansion loss coefficient (``Exp/Cntr``).
        contraction:    Contraction loss coefficient (``Exp/Cntr``).
        left_length:    Left overbank reach length from node header.
        channel_length: Channel reach length from node header.
        right_length:   Right overbank reach length from node header.
        interpolated:           ``True`` if the RS string had a trailing ``*``
                                (HEC-RAS interpolated cross section).
        vertical_n:             Vertical (depth/flow-varying) Manning's n, or
                                ``None`` if the cross section uses flat n-values.
        levee_left:             Left-bank levee (``Levee=`` line), or ``None``.
        levee_right:            Right-bank levee (``Levee=`` line), or ``None``.
        blocked_obstructions:   Blocked-obstruction intervals
                                (``#Block Obstruct``).
        htab_starting_elevation: Starting elevation for the hydraulic table
                                 (``XS HTab Starting El and Incr``).
        htab_increment:         Elevation increment for the hydraulic table.
        htab_count:             Number of entries in the hydraulic table.
    """

    river: str
    reach: str
    rs: str
    description: str = ""
    stations: list[float] = field(default_factory=list)
    elevations: list[float] = field(default_factory=list)
    mann_entries: list[ManningEntry] = field(default_factory=list)
    mann_type: int = 0
    mann_alt: int = 0
    bank_left: float | None = None
    bank_right: float | None = None
    ineff_areas: list[IneffArea] = field(default_factory=list)
    expansion: float = 0.3
    contraction: float = 0.1
    left_length: float | None = None
    channel_length: float | None = None
    right_length: float | None = None
    interpolated: bool = False
    vertical_n: VerticalN | None = None
    levee_left: LeveeData | None = None
    levee_right: LeveeData | None = None
    blocked_obstructions: list[BlockedObstruction] = field(default_factory=list)
    htab_starting_elevation: float | None = None
    htab_increment: float | None = None
    htab_count: int | None = None


logger = logging.getLogger("raspy.model")

# ---------------------------------------------------------------------------
# GeometryFile
# ---------------------------------------------------------------------------


class GeometryFile:
    """Verbatim-line editor for HEC-RAS geometry files (.g**).

    All lines are loaded verbatim.  Structured cross-section data can be
    read (:meth:`get_cross_section`, :meth:`cross_sections`) and written
    (:meth:`set_mannings`, :meth:`set_stations`, :meth:`set_bank_stations`,
    :meth:`set_exp_cntr`).  Structure nodes (bridges, culverts, etc.) are
    accessible as raw lines via :meth:`get_node_lines`.

    ``save()`` writes all in-memory lines back to disk; a no-op parse+save
    produces a byte-identical file.

    Derived from format inspection of HEC-RAS 6.6 example files and
    ``archive/ras_tools/geomParser.py``.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.is_file():
            raise FileNotFoundError(f"Geometry file not found: {self._path}")
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
        raise KeyError(f"Key not found in geometry file: {key!r}")

    def _splice(self, start: int, old_count: int, new_lines: list[str]) -> None:
        """Replace *old_count* lines beginning at *start* with *new_lines*."""
        self._lines[start : start + old_count] = [
            (ln if ln.endswith("\n") else ln + "\n") for ln in new_lines
        ]

    # ------------------------------------------------------------------
    # Static parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_rs(rs: str) -> str:
        """Strip whitespace and trailing ``*`` (interpolated XS marker)."""
        return rs.strip().rstrip("*").strip()

    @staticmethod
    def _parse_node_header(
        line: str,
    ) -> tuple[int, str, float | None, float | None, float | None] | None:
        """Parse ``Type RM Length L Ch R = TYPE,RS,L,Ch,R``.

        Returns ``(node_type, rs_normalised, left, ch, right)`` or ``None``.
        """
        prefix = _KEY_NODE + " ="
        if not line.startswith(prefix):
            return None
        tail = line[len(prefix) :].strip()
        parts = tail.split(",", 4)
        if len(parts) < 2:
            return None
        try:
            node_type = int(parts[0].strip())
        except ValueError:
            return None
        rs = GeometryFile._normalize_rs(parts[1]) if len(parts) > 1 else ""

        def _opt_float(s: str) -> float | None:
            s = s.strip()
            return float(s) if s else None

        left = _opt_float(parts[2]) if len(parts) > 2 else None
        ch = _opt_float(parts[3]) if len(parts) > 3 else None
        right = _opt_float(parts[4]) if len(parts) > 4 else None
        return node_type, rs, left, ch, right

    @staticmethod
    def _parse_reach_header(line: str) -> tuple[str, str] | None:
        """Parse ``River Reach=river,reach``. Returns ``(river, reach)``."""
        prefix = _KEY_REACH + "="
        if not line.startswith(prefix):
            return None
        parts = line[len(prefix) :].split(",", 1)
        if len(parts) != 2:
            return None
        return parts[0].strip(), parts[1].strip()

    # ------------------------------------------------------------------
    # Node location
    # ------------------------------------------------------------------

    def _find_node_start(self, river: str, reach: str, rs: str) -> int | None:
        """Return the line index of the matching node header, or ``None``."""
        prefix = _KEY_NODE + " ="
        river_l = river.strip().lower()
        reach_l = reach.strip().lower()
        rs_norm = self._normalize_rs(rs)
        in_reach = False
        for i, line in enumerate(self._lines):
            if line.startswith(_KEY_REACH + "="):
                rh = self._parse_reach_header(line)
                if rh:
                    in_reach = rh[0].lower() == river_l and rh[1].lower() == reach_l
            if in_reach and line.startswith(prefix):
                parsed = self._parse_node_header(line)
                if parsed and self._normalize_rs(parsed[1]) == rs_norm:
                    return i
        return None

    def _find_node_end(self, start: int) -> int:
        """Return the index of the first line *after* the node block at *start*.

        A new node begins with ``Type RM Length L Ch R =``.  A new reach
        begins with ``River Reach=`` or ``Junct Name=``.
        """
        prefix = _KEY_NODE + " ="
        n = len(self._lines)
        i = start + 1
        while i < n:
            line = self._lines[i]
            if (
                line.startswith(prefix)
                or line.startswith(_KEY_REACH + "=")
                or line.startswith(_KEY_JUNCT + "=")
            ):
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
    # Metadata properties
    # ------------------------------------------------------------------

    @property
    def geom_title(self) -> str | None:
        """Geometry title (``Geom Title=``)."""
        return self._get("Geom Title")

    @geom_title.setter
    def geom_title(self, value: str) -> None:
        self._set("Geom Title", value)

    @property
    def program_version(self) -> str | None:
        """HEC-RAS version that wrote this file (``Program Version=``).

        Treat as read-only; HEC-RAS manages this field.
        """
        return self._get("Program Version")

    @property
    def viewing_rectangle(self) -> tuple[float, float, float, float] | None:
        """Map viewport as ``(min_x, min_y, max_x, max_y)``, or ``None``.

        HEC-RAS writes this as ``Viewing Rectangle= x1 , y1 , x2 , y2``.
        """
        raw = self._get("Viewing Rectangle")
        if raw is None:
            return None
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 4:
            return None
        try:
            return (
                float(parts[0]),
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
            )
        except ValueError:
            return None

    @viewing_rectangle.setter
    def viewing_rectangle(
        self, value: tuple[float, float, float, float]
    ) -> None:
        self._set("Viewing Rectangle", " , ".join(str(v) for v in value))

    # ------------------------------------------------------------------
    # Reach / node inventory
    # ------------------------------------------------------------------

    @property
    def reaches(self) -> list[tuple[str, str]]:
        """Ordered list of ``(river, reach)`` pairs in file order."""
        result: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for line in self._lines:
            if line.startswith(_KEY_REACH + "="):
                rh = self._parse_reach_header(line)
                if rh and rh not in seen:
                    result.append(rh)
                    seen.add(rh)
        return result

    @property
    def junctions(self) -> list[str]:
        """Junction names defined in the file, in order of appearance."""
        result: list[str] = []
        prefix = _KEY_JUNCT + "="
        for line in self._lines:
            if line.startswith(prefix):
                result.append(line[len(prefix) :].strip())
        return result

    def node_rs_list(self, river: str, reach: str) -> list[tuple[int, str]]:
        """Return ``(node_type, rs)`` pairs for every node in *reach*.

        Useful for surveying what cross sections and structures exist.
        Results are in file order (upstream to downstream for standard
        HEC-RAS convention).
        """
        result: list[tuple[int, str]] = []
        prefix = _KEY_NODE + " ="
        river_l = river.strip().lower()
        reach_l = reach.strip().lower()
        in_reach = False
        for line in self._lines:
            if line.startswith(_KEY_REACH + "="):
                rh = self._parse_reach_header(line)
                if rh:
                    in_reach = rh[0].lower() == river_l and rh[1].lower() == reach_l
            if in_reach and line.startswith(prefix):
                parsed = self._parse_node_header(line)
                if parsed:
                    result.append((parsed[0], parsed[1]))
        return result

    # ------------------------------------------------------------------
    # Cross-section parsing
    # ------------------------------------------------------------------

    def _parse_xs_from_lines(
        self,
        river: str,
        reach: str,
        start: int,
        end: int,
    ) -> CrossSection:
        """Parse a cross section from ``self._lines[start:end]``."""
        header_line = self._lines[start]
        parsed_hdr = self._parse_node_header(header_line)

        # Detect interpolated XS (RS had trailing '*')
        raw_rs = header_line.split("=", 1)[1].split(",")[1] if parsed_hdr else ""
        interpolated = raw_rs.strip().endswith("*")

        rs = parsed_hdr[1] if parsed_hdr else ""
        left_len = parsed_hdr[2] if parsed_hdr else None
        ch_len = parsed_hdr[3] if parsed_hdr else None
        right_len = parsed_hdr[4] if parsed_hdr else None

        xs = CrossSection(
            river=river,
            reach=reach,
            rs=rs,
            left_length=left_len,
            channel_length=ch_len,
            right_length=right_len,
            interpolated=interpolated,
        )

        # Collect block lines (stripped of newline for easy processing)
        block = [ln.rstrip("\n") for ln in self._lines[start + 1 : end]]

        # --- Pass 1: extract description ---
        desc_lines: list[str] = []
        in_desc = False
        for ln in block:
            stripped = ln.strip()
            if stripped == "BEGIN DESCRIPTION:":
                in_desc = True
                continue
            if stripped == "END DESCRIPTION:":
                in_desc = False
                continue
            if in_desc:
                desc_lines.append(ln)
        xs.description = "\n".join(desc_lines)

        # --- Pass 2: extract keyed fields ---
        i = 0
        while i < len(block):
            ln = block[i]

            # #Sta/Elev= N
            if ln.startswith("#Sta/Elev="):
                n_pts = int(ln.split("=", 1)[1].strip())
                n_rows = _row_count(n_pts * 2, _COLS_STAE)
                flat = _parse_block(block[i + 1 : i + 1 + n_rows], n_pts * 2)
                xs.stations = flat[0::2]
                xs.elevations = flat[1::2]
                i += 1 + n_rows
                continue

            # #Mann= N , type , alt
            if ln.startswith("#Mann="):
                parts = ln.split("=", 1)[1].split(",")
                n_zones = int(parts[0].strip())
                xs.mann_type = int(parts[1].strip()) if len(parts) > 1 else 0
                xs.mann_alt = int(parts[2].strip()) if len(parts) > 2 else 0
                n_rows = _row_count(n_zones * 3, _COLS_MANN)
                flat = _parse_block(block[i + 1 : i + 1 + n_rows], n_zones * 3)
                xs.mann_entries = [
                    ManningEntry(
                        station=flat[j],
                        n_value=flat[j + 1],
                        variation=flat[j + 2],
                    )
                    for j in range(0, len(flat), 3)
                ]
                i += 1 + n_rows
                continue

            # #XS Ineff= N , type
            if ln.startswith("#XS Ineff="):
                parts = ln.split("=", 1)[1].split(",")
                n_ineff = int(parts[0].strip())
                n_rows = _row_count(n_ineff * 3, _COLS_MANN)
                flat = _parse_block(block[i + 1 : i + 1 + n_rows], n_ineff * 3)
                areas: list[IneffArea] = [
                    IneffArea(
                        x_start=flat[j],
                        x_end=flat[j + 1],
                        elevation=flat[j + 2],
                    )
                    for j in range(0, len(flat), 3)
                ]
                i += 1 + n_rows
                # Permanent Ineff= flags (marker line + flag lines)
                if i < len(block) and block[i].startswith("Permanent Ineff="):
                    i += 1  # skip marker
                    n_flag_rows = _row_count(n_ineff, _COLS_FLAGS)
                    flags: list[str] = []
                    for _ in range(n_flag_rows):
                        if i < len(block):
                            # Flags are 8-char right-justified; use split() safely
                            flags.extend(block[i].split())
                            i += 1
                    for k, area in enumerate(areas):
                        if k < len(flags):
                            area.permanent = flags[k].strip().upper() == "T"
                xs.ineff_areas = areas
                continue

            # Bank Sta=LB,RB
            if ln.startswith("Bank Sta="):
                parts = ln.split("=", 1)[1].split(",")
                if len(parts) >= 2:
                    try:
                        xs.bank_left = float(parts[0].strip())
                        xs.bank_right = float(parts[1].strip())
                    except ValueError:
                        pass
                i += 1
                continue

            # Exp/Cntr=exp,cntr
            if ln.startswith("Exp/Cntr="):
                parts = ln.split("=", 1)[1].split(",")
                if len(parts) >= 2:
                    try:
                        xs.expansion = float(parts[0].strip())
                        xs.contraction = float(parts[1].strip())
                    except ValueError:
                        pass
                i += 1
                continue

            # Levee=L_flag,L_sta,L_elev,R_flag,R_sta,R_elev[,L_fail,R_fail]
            if ln.startswith("Levee="):
                lp = ln.split("=", 1)[1].split(",")

                def _gp(p: list[str], idx: int) -> str:
                    return p[idx].strip() if idx < len(p) else ""

                if _gp(lp, 0) == "-1" and _gp(lp, 1):
                    sta = _gp(lp, 1)
                    elv = _gp(lp, 2)
                    fai = _gp(lp, 6)
                    if sta:
                        xs.levee_left = LeveeData(
                            station=float(sta),
                            elevation=float(elv) if elv else 0.0,
                            failure_elevation=float(fai) if fai else None,
                        )
                if _gp(lp, 3) == "-1" and _gp(lp, 4):
                    sta = _gp(lp, 4)
                    elv = _gp(lp, 5)
                    fai = _gp(lp, 7)
                    if sta:
                        xs.levee_right = LeveeData(
                            station=float(sta),
                            elevation=float(elv) if elv else 0.0,
                            failure_elevation=float(fai) if fai else None,
                        )
                i += 1
                continue

            # #Block Obstruct= N , type
            if ln.startswith("#Block Obstruct="):
                parts = ln.split("=", 1)[1].split(",")
                n_obs = int(parts[0].strip())
                n_rows = _row_count(n_obs * 3, _COLS_MANN)
                # Use fixed-position reader: blank columns = 0.0 (absent
                # endpoints are left blank in interpolated XS).
                flat = _parse_block_fixed(
                    block[i + 1 : i + 1 + n_rows], n_obs * 3
                )
                xs.blocked_obstructions = [
                    BlockedObstruction(
                        x_start=flat[j],
                        x_end=flat[j + 1],
                        elevation=flat[j + 2],
                    )
                    for j in range(0, len(flat), 3)
                ]
                i += 1 + n_rows
                continue

            # XS HTab Starting El and Incr=el,incr,count
            if ln.startswith("XS HTab Starting El and Incr="):
                parts = ln.split("=", 1)[1].split(",")
                try:
                    xs.htab_starting_elevation = float(parts[0].strip())
                    if len(parts) > 1:
                        xs.htab_increment = float(parts[1].strip())
                    if len(parts) > 2:
                        xs.htab_count = int(parts[2].strip())
                except (ValueError, IndexError):
                    pass
                i += 1
                continue

            # Vertical n Elevations= N
            if ln.startswith("Vertical n Elevations="):
                n_bp = int(ln.split("=", 1)[1].strip())
                n_bp_rows = _row_count(n_bp, _COLS_STAE)
                breakpoints = _parse_block(
                    block[i + 1 : i + 1 + n_bp_rows], n_bp
                )
                i += 1 + n_bp_rows
                vn_stations: list[VerticalNStation] = []
                by_flow = False
                while i < len(block):
                    sln = block[i]
                    if sln.startswith("Vertical n for Station="):
                        sta = float(sln.split("=", 1)[1].strip())
                        n_val_rows = _row_count(n_bp, _COLS_STAE)
                        n_vals = _parse_block(
                            block[i + 1 : i + 1 + n_val_rows], n_bp
                        )
                        vn_stations.append(
                            VerticalNStation(station=sta, n_values=n_vals)
                        )
                        i += 1 + n_val_rows
                    elif sln.startswith("Vertical n Flow="):
                        flow_val = int(sln.split("=", 1)[1].strip())
                        by_flow = flow_val == -1
                        i += 1
                        break
                    else:
                        i += 1
                xs.vertical_n = VerticalN(
                    breakpoints=breakpoints,
                    by_flow=by_flow,
                    stations=vn_stations,
                )
                continue

            i += 1

        return xs

    # ------------------------------------------------------------------
    # Cross-section access
    # ------------------------------------------------------------------

    def get_cross_section(self, river: str, reach: str, rs: str) -> CrossSection | None:
        """Parse and return the cross section at *(river, reach, rs)*.

        Returns ``None`` if not found or if the node is not a cross section
        (type != 1).
        """
        start = self._find_node_start(river, reach, rs)
        if start is None:
            return None
        parsed = self._parse_node_header(self._lines[start])
        if parsed is None or parsed[0] != NODE_XS:
            return None
        end = self._find_node_end(start)
        return self._parse_xs_from_lines(river, reach, start, end)

    def cross_sections(self, river: str, reach: str) -> list[CrossSection]:
        """Return all cross sections in *reach*, in file order.

        Structure nodes (bridges, culverts, etc.) are skipped.
        """
        prefix = _KEY_NODE + " ="
        river_l = river.strip().lower()
        reach_l = reach.strip().lower()
        result: list[CrossSection] = []
        in_reach = False
        i = 0
        n = len(self._lines)
        while i < n:
            line = self._lines[i]
            if line.startswith(_KEY_REACH + "="):
                rh = self._parse_reach_header(line)
                if rh:
                    in_reach = rh[0].lower() == river_l and rh[1].lower() == reach_l
            if in_reach and line.startswith(prefix):
                parsed = self._parse_node_header(line)
                if parsed and parsed[0] == NODE_XS:
                    end = self._find_node_end(i)
                    xs = self._parse_xs_from_lines(river, reach, i, end)
                    result.append(xs)
                    i = end
                    continue
            i += 1
        return result

    # ------------------------------------------------------------------
    # Cross-section write helpers
    # ------------------------------------------------------------------

    def _find_key_in_block(self, start: int, end: int, key: str) -> int | None:
        """Return index of first line in ``[start, end)`` starting with *key*."""
        for i in range(start, end):
            if self._lines[i].startswith(key):
                return i
        return None

    def set_mannings(
        self,
        river: str,
        reach: str,
        rs: str,
        entries: list[ManningEntry],
        mann_type: int | None = None,
        mann_alt: int | None = None,
    ) -> None:
        """Replace the Manning's n data for the given cross section.

        The ``#Mann=`` header and data rows are rebuilt from *entries*.
        If *mann_type* or *mann_alt* are ``None``, the existing values from
        the file are preserved.

        Args:
            river:     River name (case-insensitive).
            reach:     Reach name (case-insensitive).
            rs:        River station string (leading/trailing whitespace and
                       trailing ``*`` are ignored in comparisons).
            entries:   New Manning's n zones (station, n_value, variation).
            mann_type: Type flag for the ``#Mann=`` header.  ``None`` keeps
                       the existing value.
            mann_alt:  Alt flag for the ``#Mann=`` header.  ``None`` keeps
                       the existing value.

        Raises:
            KeyError: No matching node or no ``#Mann=`` line found.
        """
        start = self._find_node_start(river, reach, rs)
        if start is None:
            raise KeyError(f"No node found for {river!r}, {reach!r}, {rs!r}")
        end = self._find_node_end(start)
        mann_i = self._find_key_in_block(start, end, "#Mann=")
        if mann_i is None:
            raise KeyError(f"No #Mann= line found for {river!r}, {reach!r}, {rs!r}")

        # Preserve existing type/alt if caller did not supply them
        existing = self._lines[mann_i]
        parts = existing.split("=", 1)[1].split(",")
        existing_type = int(parts[1].strip()) if len(parts) > 1 else 0
        existing_alt = int(parts[2].strip()) if len(parts) > 2 else 0
        if mann_type is None:
            mann_type = existing_type
        if mann_alt is None:
            mann_alt = existing_alt

        n_old_zones = int(parts[0].strip())
        n_old_rows = _row_count(n_old_zones * 3, _COLS_MANN)

        n = len(entries)
        header = f"#Mann= {n} , {mann_type} , {mann_alt} "
        flat = [v for e in entries for v in (e.station, e.n_value, e.variation)]
        new_lines = [header] + _format_block(flat, _COLS_MANN)
        self._splice(mann_i, 1 + n_old_rows, new_lines)

    def set_stations(
        self,
        river: str,
        reach: str,
        rs: str,
        stations: list[float],
        elevations: list[float],
    ) -> None:
        """Replace the station/elevation data for the given cross section.

        Args:
            river:      River name (case-insensitive).
            reach:      Reach name (case-insensitive).
            rs:         River station string.
            stations:   New station values.
            elevations: New elevation values (must match length of *stations*).

        Raises:
            ValueError: *stations* and *elevations* have different lengths.
            KeyError:   No matching node or no ``#Sta/Elev=`` line found.
        """
        if len(stations) != len(elevations):
            raise ValueError(
                f"stations ({len(stations)}) and elevations ({len(elevations)})"
                " must have the same length"
            )
        start = self._find_node_start(river, reach, rs)
        if start is None:
            raise KeyError(f"No node found for {river!r}, {reach!r}, {rs!r}")
        end = self._find_node_end(start)
        sta_i = self._find_key_in_block(start, end, "#Sta/Elev=")
        if sta_i is None:
            raise KeyError(f"No #Sta/Elev= line found for {river!r}, {reach!r}, {rs!r}")

        n_old = int(self._lines[sta_i].split("=", 1)[1].strip())
        n_old_rows = _row_count(n_old * 2, _COLS_STAE)

        n = len(stations)
        header = f"#Sta/Elev= {n} "
        flat = [v for pair in zip(stations, elevations) for v in pair]
        new_lines = [header] + _format_block(flat, _COLS_STAE)
        self._splice(sta_i, 1 + n_old_rows, new_lines)

    def set_bank_stations(
        self,
        river: str,
        reach: str,
        rs: str,
        left: float,
        right: float,
    ) -> None:
        """Set the left and right bank stations for the given cross section.

        Raises:
            KeyError: No matching node or no ``Bank Sta=`` line found.
        """
        start = self._find_node_start(river, reach, rs)
        if start is None:
            raise KeyError(f"No node found for {river!r}, {reach!r}, {rs!r}")
        end = self._find_node_end(start)
        bank_i = self._find_key_in_block(start, end, "Bank Sta=")
        if bank_i is None:
            raise KeyError(f"No Bank Sta= line found for {river!r}, {reach!r}, {rs!r}")
        self._splice(bank_i, 1, [f"Bank Sta={left},{right}"])

    def set_exp_cntr(
        self,
        river: str,
        reach: str,
        rs: str,
        expansion: float,
        contraction: float,
    ) -> None:
        """Set the expansion/contraction loss coefficients.

        Raises:
            KeyError: No matching node or no ``Exp/Cntr=`` line found.
        """
        start = self._find_node_start(river, reach, rs)
        if start is None:
            raise KeyError(f"No node found for {river!r}, {reach!r}, {rs!r}")
        end = self._find_node_end(start)
        ec_i = self._find_key_in_block(start, end, "Exp/Cntr=")
        if ec_i is None:
            raise KeyError(f"No Exp/Cntr= line found for {river!r}, {reach!r}, {rs!r}")
        self._splice(ec_i, 1, [f"Exp/Cntr={expansion},{contraction}"])

    def set_vertical_n(
        self,
        river: str,
        reach: str,
        rs: str,
        vertical_n: VerticalN | None,
    ) -> None:
        """Replace or remove the vertical n block for the given cross section.

        When *vertical_n* is not ``None`` the existing block (if any) is
        replaced in-place; if none exists, the block is inserted after the
        ``XS Rating Curve=`` line.  Passing ``None`` removes any existing
        block.

        The caller is responsible for ensuring the ``#Mann`` block still
        contains valid zone boundary stations.  When vertical n is active
        HEC-RAS expects those n-values to be ``0`` (placeholders).

        Args:
            river:      River name (case-insensitive).
            reach:      Reach name (case-insensitive).
            rs:         River station string.
            vertical_n: New vertical n data, or ``None`` to remove.

        Raises:
            KeyError: No matching cross-section node found.
        """
        start = self._find_node_start(river, reach, rs)
        if start is None:
            raise KeyError(f"No node found for {river!r}, {reach!r}, {rs!r}")
        end = self._find_node_end(start)

        vn_elev_i = self._find_key_in_block(
            start, end, "Vertical n Elevations="
        )
        if vn_elev_i is not None:
            vn_flow_i = self._find_key_in_block(
                vn_elev_i, end, "Vertical n Flow="
            )
            old_count = (
                (vn_flow_i - vn_elev_i + 1) if vn_flow_i is not None else 0
            )
            insert_at = vn_elev_i
        else:
            rc_i = self._find_key_in_block(start, end, "XS Rating Curve=")
            insert_at = (rc_i + 1) if rc_i is not None else end
            old_count = 0

        if vertical_n is None:
            if old_count > 0:
                self._splice(insert_at, old_count, [])
            return

        n_bp = len(vertical_n.breakpoints)
        new_lines: list[str] = [f"Vertical n Elevations= {n_bp} "]
        new_lines += _format_block(vertical_n.breakpoints, _COLS_STAE)
        for vs in vertical_n.stations:
            sta_val = vs.station
            sta_str = (
                str(int(sta_val))
                if sta_val == int(sta_val)
                else str(sta_val)
            )
            new_lines.append(f"Vertical n for Station={sta_str}")
            new_lines += _format_block(vs.n_values, _COLS_STAE)
        flow_val = -1 if vertical_n.by_flow else 0
        new_lines.append(f"Vertical n Flow={flow_val} ")

        self._splice(insert_at, old_count, new_lines)

    def set_levee(
        self,
        river: str,
        reach: str,
        rs: str,
        left: LeveeData | None,
        right: LeveeData | None,
    ) -> None:
        """Set or remove levee data for the given cross section.

        Pass ``None`` for both *left* and *right* to remove any existing
        ``Levee=`` line.  The line is replaced in-place when it already
        exists; otherwise it is inserted before ``#XS Ineff=`` or
        ``Bank Sta=``.

        Args:
            river: River name (case-insensitive).
            reach: Reach name (case-insensitive).
            rs:    River station string.
            left:  Left-bank levee definition, or ``None`` to clear.
            right: Right-bank levee definition, or ``None`` to clear.

        Raises:
            KeyError: No matching cross-section node found.
        """
        start = self._find_node_start(river, reach, rs)
        if start is None:
            raise KeyError(f"No node found for {river!r}, {reach!r}, {rs!r}")
        end = self._find_node_end(start)
        levee_i = self._find_key_in_block(start, end, "Levee=")

        if left is None and right is None:
            if levee_i is not None:
                self._splice(levee_i, 1, [])
            return

        line = _fmt_levee_line(left, right)
        if levee_i is not None:
            self._splice(levee_i, 1, [line])
        else:
            insert_at = end
            for key in ("#XS Ineff=", "Bank Sta="):
                idx = self._find_key_in_block(start, end, key)
                if idx is not None:
                    insert_at = idx
                    break
            self._splice(insert_at, 0, [line])

    def set_blocked_obstructions(
        self,
        river: str,
        reach: str,
        rs: str,
        obstructions: list[BlockedObstruction],
    ) -> None:
        """Replace or remove the blocked-obstruction block for the given XS.

        Pass an empty list to remove any existing ``#Block Obstruct=`` block.
        When no block exists, the new block is inserted before ``Bank Sta=``.

        Args:
            river:        River name (case-insensitive).
            reach:        Reach name (case-insensitive).
            rs:           River station string.
            obstructions: New obstruction list (empty = remove).

        Raises:
            KeyError: No matching cross-section node found.
        """
        start = self._find_node_start(river, reach, rs)
        if start is None:
            raise KeyError(f"No node found for {river!r}, {reach!r}, {rs!r}")
        end = self._find_node_end(start)

        bo_i = self._find_key_in_block(start, end, "#Block Obstruct=")
        if bo_i is not None:
            n_old = int(
                self._lines[bo_i].split("=", 1)[1].split(",")[0].strip()
            )
            old_count = 1 + _row_count(n_old * 3, _COLS_MANN)
        else:
            old_count = 0

        if not obstructions:
            if bo_i is not None:
                self._splice(bo_i, old_count, [])
            return

        n = len(obstructions)
        flat = [v for o in obstructions for v in (o.x_start, o.x_end, o.elevation)]
        new_lines = [f"#Block Obstruct= {n} , 0 "] + _format_block(
            flat, _COLS_MANN
        )

        if bo_i is not None:
            self._splice(bo_i, old_count, new_lines)
        else:
            bank_i = self._find_key_in_block(start, end, "Bank Sta=")
            insert_at = bank_i if bank_i is not None else end
            self._splice(insert_at, 0, new_lines)

    def set_htab(
        self,
        river: str,
        reach: str,
        rs: str,
        starting_elevation: float,
        increment: float,
        count: int,
    ) -> None:
        """Set the hydraulic-table parameters for the given cross section.

        Replaces the ``XS HTab Starting El and Incr=`` line in-place, or
        inserts it after ``XS Rating Curve=`` if absent.

        Args:
            river:              River name (case-insensitive).
            reach:              Reach name (case-insensitive).
            rs:                 River station string.
            starting_elevation: First elevation in the hydraulic table.
            increment:          Elevation increment between table entries.
            count:              Number of entries in the hydraulic table.

        Raises:
            KeyError: No matching cross-section node found.
        """
        start = self._find_node_start(river, reach, rs)
        if start is None:
            raise KeyError(f"No node found for {river!r}, {reach!r}, {rs!r}")
        end = self._find_node_end(start)

        htab_i = self._find_key_in_block(
            start, end, "XS HTab Starting El and Incr="
        )
        line = (
            f"XS HTab Starting El and Incr="
            f"{starting_elevation},{increment}, {count} "
        )
        if htab_i is not None:
            self._splice(htab_i, 1, [line])
        else:
            rc_i = self._find_key_in_block(start, end, "XS Rating Curve=")
            insert_at = (rc_i + 1) if rc_i is not None else end
            self._splice(insert_at, 0, [line])

    # ------------------------------------------------------------------
    # Raw node access (for structures)
    # ------------------------------------------------------------------

    def get_node_lines(self, river: str, reach: str, rs: str) -> list[str] | None:
        """Return the raw lines for a node block (header inclusive).

        Useful for inspecting structure nodes (bridges, culverts, etc.) that
        are not fully parsed.  Returns ``None`` if not found.
        """
        start = self._find_node_start(river, reach, rs)
        if start is None:
            return None
        end = self._find_node_end(start)
        return [ln.rstrip("\n") for ln in self._lines[start:end]]

    def node_type(self, river: str, reach: str, rs: str) -> int | None:
        """Return the node type code for *(river, reach, rs)*, or ``None``.

        Returns one of :data:`NODE_XS`, :data:`NODE_CULVERT`,
        :data:`NODE_BRIDGE`, :data:`NODE_MULTIPLE_OPENING`,
        :data:`NODE_INLINE_STRUCTURE`, :data:`NODE_LATERAL_STRUCTURE`.
        """
        start = self._find_node_start(river, reach, rs)
        if start is None:
            return None
        parsed = self._parse_node_header(self._lines[start])
        return parsed[0] if parsed else None

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
