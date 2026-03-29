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

import contextlib
import logging
from dataclasses import dataclass, field
from math import ceil, isnan, nan
from pathlib import Path
from typing import Generic, TypeVar, overload

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

#: Culvert shape codes from ``Culvert=`` data line (index 0).
_CULVERT_SHAPES: dict[int, str] = {
    1: "Circular",
    2: "Box",
    3: "Pipe Arch",
    4: "Ellipse",
    5: "Arch",
    6: "Semi-Circle",
    7: "Low Profile Arch",
    8: "High Profile Arch",
    9: "Con Span",
}

_KEY_NODE = "Type RM Length L Ch R"
_KEY_REACH = "River Reach"
_KEY_JUNCT = "Junct Name"

_T = TypeVar("_T")

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


# ---------------------------------------------------------------------------
# Structure dataclasses (inline structures from text geometry files)
# ---------------------------------------------------------------------------


@dataclass
class GateOpening:
    """One gate opening within a gate group.

    Attributes:
        name:    Opening name from ``IW Gate Opening=`` line; empty string
                 when the line is absent.
        station: Lateral station of the gate opening (8-char fixed-width column).
        gis:     GIS (x, y) coordinate pairs from the data line following the
                 ``IW Gate Opening=`` line (16-char fixed-width columns).
                 Empty list when the GIS point count is zero.
    """

    name: str
    station: float
    gis: list[tuple[float, float]] = field(default_factory=list)


_GATE_TYPE_NAMES: dict[int, str] = {
    0: "sluice",
    1: "radial",
    2: "overflow_closed_top",
    3: "overflow_open",
    4: "user_defined_curves",
}


@dataclass
class GateGroup:
    """One gate group in an inline structure (``IW Gate Name`` block).

    Field order follows the comma-delimited ``IW Gate Name`` data line::

        name,Wd,H,Inv,GCoef,Exp_T,Exp_O,Exp_H,Type,WCoef,Is_Ogee,
        SpillHt,DesHd,#Openings,trunnion_height,orifice_coef,
        head_ref,radial_coef,,is_sharp_crested,weir_p1,weir_p2,weir_p3

    Attributes:
        name:                     Group name (index 0).
        width:                    Gate width — ``Wd`` (index 1).
        height:                   Gate height — ``H`` (index 2).
        invert:                   Gate invert elevation — ``Inv`` (index 3).
        gate_coefficient:         Gate discharge coefficient (applies to all
                                  gate types) — ``GCoef`` (index 4).  The HDF
                                  version names this field
                                  ``sluice_coefficient``.
        trunnion_exponent:        Trunnion arm exponent — ``Exp_T`` (index 5).
        opening_exponent:         Gate opening exponent — ``Exp_O`` (index 6).
        height_exponent:          Gate height exponent — ``Exp_H`` (index 7).
        gate_type:                Gate type string — ``Type`` (index 8):
                                  ``'sluice'``, ``'radial'``,
                                  ``'overflow_closed_top'``,
                                  ``'overflow_open'``, or
                                  ``'user_defined_curves'``.
        weir_coefficient:         Overflow weir coefficient — ``WCoef``
                                  (index 9).
        is_ogee:                  ``True`` when ``Is_Ogee`` (index 10) is
                                  ``-1`` (ogee crest shape).
        spillway_approach_height: Spillway approach height — ``SpillHt``
                                  (index 11).
        design_energy_head:       Design energy head — ``DesHd`` (index 12).
        trunnion_height:          Trunnion height (index 14).
        orifice_coefficient:      Orifice coefficient (index 15).
        head_reference:           Head reference point — 0 = sill,
                                  1 = centre of opening (index 16).
        radial_coefficient:       Radial (Tainter) gate discharge coefficient
                                  (index 17).
        is_sharp_crested:         ``True`` when index 19 is ``-1``.
        use_weir_param1:          ``True`` when index 20 is ``-1``.
        use_weir_param2:          ``True`` when index 21 is ``-1``.
        use_weir_param3:          ``True`` when index 22 is ``-1``.
        openings:                 Individual gate openings (stations + names).
    """

    name: str
    width: float
    height: float
    invert: float
    gate_coefficient: float
    trunnion_exponent: float
    opening_exponent: float
    height_exponent: float
    gate_type: str
    weir_coefficient: float
    is_ogee: bool
    spillway_approach_height: float
    design_energy_head: float
    trunnion_height: float
    orifice_coefficient: float
    head_reference: int
    radial_coefficient: float
    is_sharp_crested: bool
    use_weir_param1: bool
    use_weir_param2: bool
    use_weir_param3: bool
    openings: list[GateOpening] = field(default_factory=list)

    @property
    def weir_shape(self) -> str:
        """Shape of the overflow weir crest.

        Returns ``'Ogee'``, ``'Sharp Crested'``, or ``'Broad Crested'``
        based on the ``is_ogee`` and ``is_sharp_crested`` flags.
        """
        if self.is_ogee:
            return "Ogee"
        if self.is_sharp_crested:
            return "Sharp Crested"
        return "Broad Crested"

    @property
    def Cu(self) -> float:
        """Unsubmerged discharge coefficient (alias for ``gate_coefficient``).

        Typical range 0.5–0.7 for sluice and radial gates.
        """
        return self.gate_coefficient

    @property
    def Cs(self) -> float:
        """Submerged discharge coefficient (alias for ``orifice_coefficient``).

        Typical value ~0.8; applied when tailwater submerges the gate opening.
        """
        return self.orifice_coefficient


@dataclass
class Weir:
    """Overflow weir parameters from the ``IW Dist`` block.

    Mirrors :class:`raspy.hdf._geometry.Weir` in field names.  Key
    differences vs the HDF version:

    - Always populated when an ``IW Dist`` line exists, even when *mode* is
      empty (the HDF version returns ``None`` for ``mode=''``).
    - ``us_slope``, ``ds_slope``: not present in the ``IW Dist`` text line;
      stored as ``nan``.

      .. TODO: Check whether HEC-RAS stores US/DS slope elsewhere in the
         text geometry file (e.g. a secondary ``IW Dist`` continuation line
         or a ``Spillway`` block) and parse them if found.

    - ``use_water_surface``: not stored in the text geometry format; always
      ``False``.

      .. TODO: Identify whether this flag is persisted anywhere in the
         ``.g**`` file and add parsing when found.

    Attributes:
        width:                    Weir width (``WD``, index 1).
        coefficient:              Weir discharge coefficient (``Coef``, index 2).
        shape:                    ``'Broad Crested'`` or ``'Ogee'``
                                  (``Is_Ogee``, index 6).
        max_submergence:          Maximum submergence ratio (``MaxSub``,
                                  index 4).
        min_elevation:            Minimum weir crest elevation (``Min_El``,
                                  index 5); ``nan`` when blank.
        skew:                     Weir skew angle in degrees (``Skew``,
                                  index 3).
        dist:                     Distance from the upstream XS to the weir
                                  face (index 0 of the ``IW Dist`` data line);
                                  ``nan`` for bridges and laterals (not stored
                                  in those formats).
        spillway_approach_height: Spillway approach height — ``SpillHt``
                                  (index 7 of the ``IW Dist`` data line);
                                  ``nan`` for bridges and laterals.
        design_energy_head:       Design energy head — ``DesHd`` (index 8 of
                                  the ``IW Dist`` data line); ``nan`` for
                                  bridges and laterals.
        us_slope:                 Upstream face slope H:V; ``nan`` — not in
                                  text format.
        ds_slope:                 Downstream face slope H:V; ``nan`` — not in
                                  text format.
        use_water_surface:        Use water surface as weir reference head;
                                  always ``False`` — not in text format.
    """

    width: float
    coefficient: float
    shape: str
    max_submergence: float
    min_elevation: float
    skew: float
    dist: float = nan
    spillway_approach_height: float = nan
    design_energy_head: float = nan
    us_slope: float = nan
    ds_slope: float = nan
    use_water_surface: bool = False


@dataclass
class CulvertGroup:
    """One culvert barrel definition from a ``Culvert=`` line.

    Field order matches the comma-separated data on the ``Culvert=`` line::

        shape, span, rise, length, n, Ke, exit_loss, inlet_type, outlet_type,
        us_invert, us_station, ds_invert, ds_station, name, ?, chart_number

    Attributes:
        name:           Culvert identifier (index 13), e.g. ``'Culvert # 1'``.
        shape_code:     Shape code (index 0) — see :data:`_CULVERT_SHAPES`.
        shape_name:     Human-readable shape, e.g. ``'Circular'``, ``'Box'``.
        span:           Width or diameter (index 1).
        rise:           Height (index 2); ``0.0`` when blank.
        length:         Barrel length (index 3).
        n_top:          Manning's roughness coefficient (index 4).
        entrance_loss:  Entrance loss coefficient Ke (index 5).
        exit_loss:      Exit loss coefficient (index 6).
        inlet_type:     Inlet control type code (index 7).
        outlet_type:    Outlet control type code (index 8).
        upstream_invert:   Upstream invert elevation (index 9).
        upstream_station:  Upstream station location (index 10).
        downstream_invert: Downstream invert elevation (index 11).
        downstream_station: Downstream station location (index 12).
        chart_number:   Inlet control chart number (index 15).
        num_barrels:    Number of barrels — from ``BC Culvert Barrel=`` line;
                        ``1`` when that line is absent.
        n_bottom:       Bottom Manning's n — from ``Culvert Bottom n=``;
                        ``None`` when absent.
        depth_n_bottom: Bottom fill depth — from ``Culvert Bottom Depth=``;
                        ``None`` when absent.

    .. TODO: ``Multiple Barrel Culv=`` uses a different field layout
       (no ``us_station`` / ``ds_station`` columns; barrel stations appear on
       the next line as ``num_barrels`` upstream + ``num_barrels`` downstream
       values in 8-char fixed-width columns).  These station values are
       currently not parsed.
    """

    name: str
    shape_code: int
    shape_name: str
    span: float
    rise: float
    length: float
    n_top: float
    entrance_loss: float
    exit_loss: float
    inlet_type: int
    outlet_type: int
    upstream_invert: float
    upstream_station: float
    downstream_invert: float
    downstream_station: float
    chart_number: int
    num_barrels: int = 1
    n_bottom: float | None = None
    depth_n_bottom: float | None = None


@dataclass
class Pier:
    """One pier group from a ``Pier Skew, UpSta & Num, DnSta & Num=`` block.

    The header line carries the skew, station, and count; four fixed-width
    blocks (8-char columns, up to 10 per row) immediately follow::

        upstream_count   widths
        upstream_count   elevations
        downstream_count widths
        downstream_count elevations

    Attributes:
        skew:                 Pier skew angle in degrees; ``0.0`` when blank.
        upstream_station:     Station of the upstream pier face.
        upstream_count:       Number of upstream pier elements.
        downstream_station:   Station of the downstream pier face.
        downstream_count:     Number of downstream pier elements.
        upstream_widths:      Width of each upstream pier element.
        upstream_elevations:  Top-of-pier elevation for each upstream element.
        downstream_widths:    Width of each downstream pier element.
        downstream_elevations: Top-of-pier elevation for each downstream element.
    """

    skew: float
    upstream_station: float
    upstream_count: int
    downstream_station: float
    downstream_count: int
    upstream_widths: list[float]
    upstream_elevations: list[float]
    downstream_widths: list[float]
    downstream_elevations: list[float]


@dataclass
class Roadway:
    """Deck/roadway geometry from the ``Deck Dist Width WeirC ...`` block.

    The header ``Deck Dist Width WeirC Skew NumUp NumDn MinLoCord MaxHiCord
    MaxSubmerge Is_Ogee`` is followed by a comma-separated data line and then
    six fixed-width blocks (8-char columns, up to 10 values per row):

    - *numUp* upstream deck stations
    - *numUp* upstream high-chord (top of deck) elevations
    - *numUp* upstream low-chord (soffit) elevations
    - *numDn* downstream versions of the above (same layout)

    ``lo_chord_up`` / ``lo_chord_dn`` will be an empty list when the
    corresponding block is all-blank (some culvert files omit low-chord data).

    Attributes:
        dist:             Distance from the upstream cross section to the
                          bridge face (index 0).
        width:            Deck/roadway width (index 1).
        weir_coefficient: Overflow weir discharge coefficient (index 2).
        skew:             Bridge skew angle in degrees (index 3).
        max_submergence:  Maximum submergence ratio (index 8); ``nan`` when
                          blank.
        shape:            Overflow weir crest shape — ``'Broad Crested'`` or
                          ``'Ogee'`` (from Is_Ogee flag, index 9).
        min_lo_chord:     Minimum low-chord elevation (index 6); ``nan`` when
                          blank.
        max_hi_chord:     Maximum high-chord elevation (index 7); ``nan`` when
                          blank.
        stations_up:      Upstream deck station values.
        hi_chord_up:      Upstream high-chord (top of deck) elevations.
        lo_chord_up:      Upstream low-chord (soffit) elevations; empty list
                          when the block is all-blank.
        stations_dn:      Downstream deck station values.
        hi_chord_dn:      Downstream high-chord elevations.
        lo_chord_dn:      Downstream low-chord elevations; empty list when
                          all-blank.

    .. TODO: The ``Bridge Culvert-`` flags line (immediately before
       ``Deck Dist``) encodes open-deck / culvert-only options and is not yet
       parsed.
    .. TODO: ``BR Coef=`` (bridge loss coefficients — momentum/energy method,
       Yarnell K factor, etc.) is not yet parsed.
    .. TODO: ``WSPro=`` (water surface profile method options) is not yet
       parsed.
    .. TODO: ``BC Design=`` (design flow parameters) is not yet parsed.
    """

    dist: float
    width: float
    weir_coefficient: float
    skew: float
    max_submergence: float
    shape: str
    min_lo_chord: float
    max_hi_chord: float
    stations_up: list[float] = field(default_factory=list)
    hi_chord_up: list[float] = field(default_factory=list)
    lo_chord_up: list[float] = field(default_factory=list)
    stations_dn: list[float] = field(default_factory=list)
    hi_chord_dn: list[float] = field(default_factory=list)
    lo_chord_dn: list[float] = field(default_factory=list)


@dataclass
class Structure:
    """Base class for one HEC-RAS structure parsed from a text geometry file.

    Mirrors :class:`raspy.hdf._geometry.Structure` in field names.  Key
    difference:

    - ``centerline``: the text geometry file carries no GIS centreline
      coordinates for structures; always an empty list.  The HDF version
      stores an ``(n_pts, 2)`` numpy array.

      .. TODO: Investigate whether centreline coordinates can be recovered
         from the ``*.rasmap`` file or the HDF geometry file alongside the
         text file, and populate this field when available.

    Attributes:
        mode:           HEC-RAS mode string (e.g. ``'Weir/Gate/Culverts'``).
                        Always ``''`` for inline structures — not stored in
                        the text format.

                        .. TODO: Check whether ``Mode=`` or equivalent is
                           written to the ``.g**`` file for any structure type
                           and parse it when found.

        upstream_type:  Connection type on the upstream side (``'XS'``,
                        ``'SA'``, ``'2D'``, or ``'--'``).  Always ``'XS'``
                        for inline structures.
        downstream_type: Connection type on the downstream side.
        centerline:     GIS centreline as ``[(x, y), ...]``; always ``[]``
                        from text files.
    """

    mode: str
    upstream_type: str
    downstream_type: str
    centerline: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class Inline(Structure):
    """Inline structure (node type 5) parsed from a ``.g**`` text geometry file.

    Mirrors :class:`raspy.hdf._geometry.Inline`.  Inherits ``mode``,
    ``upstream_type``, ``downstream_type``, and ``centerline`` from
    :class:`Structure`.

    Differences vs the HDF version:

    - ``upstream_node`` / ``downstream_node``: use ``("", "", "")`` as the
      no-value sentinel (same as HDF), populated by walking ``node_rs_list``
      to find the nearest flanking cross sections.
    - ``weir``: populated from ``IW Dist`` even when ``mode`` is ``''``
      (HDF returns ``None`` for ``mode=''``).

      .. TODO: Align weir-presence logic with HDF once ``mode`` parsing
         is implemented (see :class:`Structure` TODO).

    - ``description``: text-file-specific field from the
      ``BEGIN/END DESCRIPTION`` block; no equivalent in HDF.
    - ``pilot_flow``: minimum flow through the structure when all gates are
      fully closed — from ``IW Pilot Flow=``; no equivalent in HDF.
    - ``crest_profile``: station/elevation pairs defining the weir crest
      geometry — from the ``#Inline Weir SE=`` block; no equivalent in HDF.

    Attributes:
        location:        ``(river, reach, rs)`` of this structure.
        upstream_node:   ``(river, reach, rs)`` of the nearest upstream XS;
                         ``("", "", "")`` when none found.
        downstream_node: ``(river, reach, rs)`` of the nearest downstream XS;
                         ``("", "", "")`` when none found.
        weir:            Overflow weir data from ``IW Dist`` block; ``None``
                         if no ``IW Dist`` line is present.
        gate_groups:     Gate group definitions from ``IW Gate Name`` blocks.
        description:     Node description from ``BEGIN/END DESCRIPTION``.
        pilot_flow:      Minimum (pilot) flow through the structure —
                         ``IW Pilot Flow=``; ``0.0`` when absent.
        crest_profile:   Weir crest station/elevation pairs from the
                         ``#Inline Weir SE=`` block; empty list when absent.
    """

    location: tuple[str, str, str] = ("", "", "")
    upstream_node: tuple[str, str, str] = ("", "", "")
    downstream_node: tuple[str, str, str] = ("", "", "")
    weir: Weir | None = None
    gate_groups: list[GateGroup] = field(default_factory=list)
    description: str = ""
    pilot_flow: float = 0.0
    crest_profile: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class Bridge(Structure):
    """Bridge or culvert (node types 3 / 2) parsed from a ``.g**`` text geometry file.

    Mirrors :class:`raspy.hdf._geometry.Bridge`.  Inherits ``mode``,
    ``upstream_type``, ``downstream_type``, and ``centerline`` from
    :class:`Structure`.

    Differences vs the HDF version:

    - ``weir``: populated from the ``Deck Dist`` data line scalar fields
      (width, coefficient, skew, max_submergence, shape); ``us_slope``,
      ``ds_slope``, and ``use_water_surface`` are always ``nan`` / ``False``
      — not stored in the text format.  Redundant with ``roadway`` scalars.
    - ``roadway``: full deck geometry including upstream/downstream station,
      high-chord, and low-chord arrays; ``None`` when no ``Deck Dist`` line
      is present.
    - ``culvert_groups``: barrel definitions from ``Culvert=`` lines.
    - ``piers``: pier groups from ``Pier Skew, ...`` blocks.
    - ``gate_groups``: always ``[]`` — bridge/culvert format uses
      ``Bridge Culvert`` hydraulics, not ``IW Gate Name`` blocks.
    - ``description``: text-file-specific field from
      ``BEGIN/END DESCRIPTION``; no equivalent in HDF.

    Attributes:
        location:        ``(river, reach, rs)`` of this structure.
        upstream_node:   ``(river, reach, rs)`` of the nearest upstream XS;
                         ``("", "", "")`` when none found.
        downstream_node: ``(river, reach, rs)`` of the nearest downstream XS;
                         ``("", "", "")`` when none found.
        weir:            Overflow weir scalars from the ``Deck Dist`` line;
                         ``None`` if the line is absent.
        gate_groups:     Always ``[]``.
        description:     Node description from ``BEGIN/END DESCRIPTION``.
        roadway:         Full deck geometry (stations, hi/lo chords) from the
                         ``Deck Dist`` block; ``None`` when absent.
        culvert_groups:  Barrel definitions from ``Culvert=`` lines.
        piers:           Pier groups from ``Pier Skew, ...`` blocks.
        node_name:       Node name from ``Node Name=``; empty string when
                         absent.
        htab_hw_max:     Max headwater elevation from ``BC HTab HWMax=``;
                         ``None`` when absent.
        htab_tw_max:     Max tailwater elevation from ``BC HTab TWMax=``;
                         ``None`` when absent.
        htab_max_flow:   Max flow from ``BC HTab MaxFlow=``; ``None`` when
                         absent.
    """

    location: tuple[str, str, str] = ("", "", "")
    upstream_node: tuple[str, str, str] = ("", "", "")
    downstream_node: tuple[str, str, str] = ("", "", "")
    weir: Weir | None = None
    gate_groups: list[GateGroup] = field(default_factory=list)
    description: str = ""
    roadway: Roadway | None = None
    culvert_groups: list[CulvertGroup] = field(default_factory=list)
    piers: list[Pier] = field(default_factory=list)
    node_name: str = ""
    htab_hw_max: float | None = None
    htab_tw_max: float | None = None
    htab_max_flow: float | None = None


@dataclass
class Lateral(Structure):
    """Lateral structure (node type 6) parsed from a ``.g**`` text geometry file.

    Mirrors :class:`raspy.hdf._geometry.Lateral`.  Inherits ``mode``,
    ``upstream_type``, ``downstream_type``, and ``centerline`` from
    :class:`Structure`.

    Differences vs the HDF version:

    - ``downstream_node``: HDF stores the name of the connected Storage Area
      or 2-D Flow Area.  The text format stores a connected river+reach via
      ``Lateral Weir End=river,reach,rs,...``; this field holds
      ``"river reach"`` (stripped, space-joined).  Empty string when the
      ``Lateral Weir End=`` line is absent.
    - ``weir``: built from individual ``Lateral Weir WD=``,
      ``Lateral Weir Coef=``, and ``Lateral Weir WSCriteria=`` lines;
      ``skew=0``, ``min_elevation=nan``, ``us_slope/ds_slope=nan``.
    - ``gate_groups``: always ``[]`` — lateral structures in the text format
      use ``Lateral Weir`` hydraulics, not ``IW Gate Name`` blocks.
    - ``description``: text-file-specific field from
      ``BEGIN/END DESCRIPTION``; no equivalent in HDF.

    Attributes:
        location:        ``(river, reach, rs)`` of this structure.
        upstream_node:   ``(river, reach, rs)`` of the nearest upstream XS;
                         ``("", "", "")`` when none found.
        downstream_node: Connected river+reach as ``"river reach"``
                         (from ``Lateral Weir End=``); empty string when
                         absent.
        weir:            Overflow weir data from ``Lateral Weir`` lines;
                         ``None`` if no ``Lateral Weir WD=`` line is present.
        gate_groups:     Always ``[]``.
        description:     Node description from ``BEGIN/END DESCRIPTION``.
        pos:             Bank side of the weir — ``0`` = left, ``1`` = right
                         (from ``Lateral Weir Pos=``).
        distance:        Setback distance from the upstream cross section
                         (from ``Lateral Weir Distance=``); ``nan`` when
                         absent.
        crest_profile:   Weir crest station/elevation pairs from the
                         ``Lateral Weir SE=`` block; empty list when absent.
        flap_gates:      ``True`` when ``Lateral Weir Flap Gates= -1`` or
                         ``1``; ``False`` when ``0`` or absent.
        tw_multiple_xs:  ``True`` when tailwater uses multiple XS averaging
                         (``Lateral Weir TW Multiple XS=`` non-zero).

    .. TODO: ``Lateral Weir Hagers EQN=`` (Hager's weir equation on/off flag
       and six coefficients) is not yet parsed.
    .. TODO: ``Lateral Weir SS=`` (upstream and downstream side slopes) is not
       yet parsed.
    .. TODO: ``Lateral Weir Connection Pos and Dist=`` (connection-point
       position code and distance) is not yet parsed.
    .. TODO: ``Lateral Weir Centerline=`` (GIS centreline point count and
       coordinate block) is not yet parsed.
    .. TODO: ``LW Div RC=`` (diversion rating-curve flag and label) is not yet
       parsed.
    """

    location: tuple[str, str, str] = ("", "", "")
    upstream_node: tuple[str, str, str] = ("", "", "")
    downstream_node: str = ""
    weir: Weir | None = None
    gate_groups: list[GateGroup] = field(default_factory=list)
    description: str = ""
    pos: int = 0
    distance: float = nan
    crest_profile: list[tuple[float, float]] = field(default_factory=list)
    flap_gates: bool = False
    tw_multiple_xs: bool = False


# ---------------------------------------------------------------------------
# Structure containers
# ---------------------------------------------------------------------------


class StructureIndex(Generic[_T]):
    """Read-only ordered mapping from a string key to a structure object.

    Supports both string key (``"River Reach RS"``) and integer positional
    index.  Mirrors the interface of :class:`~raspy.hdf._geometry.StructureIndex`.
    """

    def __init__(self, items: list[tuple[str, _T]]) -> None:
        self._keys: list[str] = [k for k, _ in items]
        self._values: list[_T] = [v for _, v in items]
        self._map: dict[str, _T] = {k: v for k, v in items}

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self):
        return iter(self._keys)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, int):
            return 0 <= key < len(self._values)
        return key in self._map

    @overload
    def __getitem__(self, key: str) -> _T: ...

    @overload
    def __getitem__(self, key: int) -> _T: ...

    def __getitem__(self, key):  # type: ignore[override]
        if isinstance(key, int):
            return self._values[key]
        return self._map[key]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"({list(zip(self._keys, self._values, strict=True))})"
        )

    def keys(self) -> list[str]:
        """Ordered list of string keys."""
        return list(self._keys)

    def values(self) -> list[_T]:
        """Ordered list of structure objects."""
        return list(self._values)

    def items(self) -> list[tuple[str, _T]]:
        """Ordered ``(key, structure)`` pairs."""
        return list(zip(self._keys, self._values, strict=True))


class ModelStructureCollection:
    """Structure collection parsed from a HEC-RAS text geometry file.

    Covers inline structures (node type 5), bridges/culverts (types 3/2),
    and lateral structures (type 6).

    Access structures via the typed properties:

    .. code-block:: python

        g = GeometryFile("model.g01")
        for key, iw in g.structures.inlines.items():
            print(key, iw.gate_groups)
        for key, br in g.structures.bridges.items():
            print(key, br.weir)
        for key, lat in g.structures.laterals.items():
            print(key, lat.downstream_node)
    """

    def __init__(
        self,
        inlines: list[tuple[str, Inline]],
        bridges: list[tuple[str, Bridge]],
        laterals: list[tuple[str, Lateral]],
    ) -> None:
        self._inlines: StructureIndex[Inline] = StructureIndex(inlines)
        self._bridges: StructureIndex[Bridge] = StructureIndex(bridges)
        self._laterals: StructureIndex[Lateral] = StructureIndex(laterals)

    @property
    def inlines(self) -> StructureIndex[Inline]:
        """All inline structures (type 5) keyed by ``'River Reach RS'``."""
        return self._inlines

    @property
    def bridges(self) -> StructureIndex[Bridge]:
        """All bridges and culverts (types 3 and 2) keyed by ``'River Reach RS'``."""
        return self._bridges

    @property
    def laterals(self) -> StructureIndex[Lateral]:
        """All lateral structures (type 6) keyed by ``'River Reach RS'``."""
        return self._laterals

    @property
    def summary(self) -> dict[str, int]:
        """Count of each parsed structure type."""
        return {
            "inlines": len(self._inlines),
            "bridges": len(self._bridges),
            "laterals": len(self._laterals),
        }

    def __repr__(self) -> str:
        return (
            f"ModelStructureCollection("
            f"inlines={len(self._inlines)}, "
            f"bridges={len(self._bridges)}, "
            f"laterals={len(self._laterals)})"
        )


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
        self._modified: bool = False
        self._structures_cache: ModelStructureCollection | None = None

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
                self._modified = True
                return
        raise KeyError(f"Key not found in geometry file: {key!r}")

    def _splice(self, start: int, old_count: int, new_lines: list[str]) -> None:
        """Replace *old_count* lines beginning at *start* with *new_lines*."""
        self._lines[start : start + old_count] = [
            (ln if ln.endswith("\n") else ln + "\n") for ln in new_lines
        ]
        self._modified = True

    # ------------------------------------------------------------------
    # Modification state
    # ------------------------------------------------------------------

    @property
    def is_modified(self) -> bool:
        """``True`` if any value has been changed since the last :meth:`save`."""
        return self._modified

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
    # Structure parsing (inline structures)
    # ------------------------------------------------------------------

    @property
    def structures(self) -> ModelStructureCollection:
        """Structure collection parsed from this geometry file.

        Returns a :class:`ModelStructureCollection` whose ``inlines`` property
        holds all inline structures keyed by ``'River Reach RS'``.
        The result is cached; invalidated automatically on :meth:`save`.

        Example::

            g = GeometryFile("model.g01")
            for key, iw in g.structures.inlines.items():
                print(key, [gg.name for gg in iw.gate_groups])
        """
        if self._structures_cache is None:
            self._structures_cache = self._build_structures()
        return self._structures_cache

    def _build_structures(self) -> ModelStructureCollection:
        """Scan all reaches and build the :class:`ModelStructureCollection`."""
        inlines: list[tuple[str, Inline]] = []
        bridges: list[tuple[str, Bridge]] = []
        laterals: list[tuple[str, Lateral]] = []
        for river, reach in self.reaches:
            for node_type, rs in self.node_rs_list(river, reach):
                if node_type not in (
                    NODE_CULVERT,
                    NODE_BRIDGE,
                    NODE_INLINE_STRUCTURE,
                    NODE_LATERAL_STRUCTURE,
                ):
                    continue
                start = self._find_node_start(river, reach, rs)
                if start is None:
                    continue
                end = self._find_node_end(start)
                upstream_node, downstream_node = self._adjacent_xs_nodes(
                    river, reach, rs
                )
                key = f"{river} {reach} {rs}"
                if node_type == NODE_INLINE_STRUCTURE:
                    iw = self._parse_inline_structure(
                        river, reach, rs, start, end, upstream_node, downstream_node
                    )
                    inlines.append((key, iw))
                elif node_type in (NODE_CULVERT, NODE_BRIDGE):
                    br = self._parse_bridge(
                        river, reach, rs, start, end, upstream_node, downstream_node
                    )
                    bridges.append((key, br))
                elif node_type == NODE_LATERAL_STRUCTURE:
                    lat = self._parse_lateral(
                        river, reach, rs, start, end, upstream_node
                    )
                    laterals.append((key, lat))
        return ModelStructureCollection(inlines, bridges, laterals)

    def _adjacent_xs_nodes(
        self, river: str, reach: str, rs: str
    ) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
        """Return the nearest upstream and downstream XS nodes flanking *rs*.

        HEC-RAS lists nodes from upstream (high RS) to downstream (low RS).
        Walking backward in the list finds the upstream XS; forward finds the
        downstream XS.

        Returns ``(upstream, downstream)`` tuples of ``(river, reach, rs)``.
        Each tuple is ``("", "", "")`` when no adjacent XS exists on that side.
        """
        _empty: tuple[str, str, str] = ("", "", "")
        nodes = self.node_rs_list(river, reach)
        rs_norm = self._normalize_rs(rs)
        idx = next(
            (
                i
                for i, (nt, r) in enumerate(nodes)
                if self._normalize_rs(r) == rs_norm
            ),
            None,
        )
        if idx is None:
            return _empty, _empty
        upstream: tuple[str, str, str] = _empty
        for i in range(idx - 1, -1, -1):
            if nodes[i][0] == NODE_XS:
                upstream = (river, reach, nodes[i][1])
                break
        downstream: tuple[str, str, str] = _empty
        for i in range(idx + 1, len(nodes)):
            if nodes[i][0] == NODE_XS:
                downstream = (river, reach, nodes[i][1])
                break
        return upstream, downstream

    def _parse_inline_structure(
        self,
        river: str,
        reach: str,
        rs: str,
        start: int,
        end: int,
        upstream_node: tuple[str, str, str],
        downstream_node: tuple[str, str, str],
    ) -> Inline:
        """Parse one inline-structure node block into an :class:`Inline`.

        Parameters
        ----------
        river, reach, rs:
            Node identity.
        start, end:
            Line-index range of the node block (``self._lines[start:end]``).
        upstream_node, downstream_node:
            Adjacent XS tuples from :meth:`_adjacent_xs_nodes`.
        """
        lines = [ln.rstrip("\n") for ln in self._lines[start:end]]

        # -- description ---------------------------------------------------
        description = ""
        desc_s = next(
            (i for i, ln in enumerate(lines) if ln.strip() == "BEGIN DESCRIPTION:"),
            None,
        )
        desc_e = next(
            (i for i, ln in enumerate(lines) if ln.strip() == "END DESCRIPTION:"),
            None,
        )
        if desc_s is not None and desc_e is not None and desc_e > desc_s:
            description = "\n".join(lines[desc_s + 1 : desc_e]).strip()

        # -- pilot flow (IW Pilot Flow=) -----------------------------------
        pilot_flow = 0.0
        for ln in lines:
            if ln.startswith("IW Pilot Flow="):
                val = ln[len("IW Pilot Flow="):].strip()
                try:
                    pilot_flow = float(val)
                except ValueError:
                    pass
                break

        # -- weir crest profile (#Inline Weir SE=) -------------------------
        crest_profile: list[tuple[float, float]] = []
        iw_se_i = next(
            (i for i, ln in enumerate(lines) if ln.startswith("#Inline Weir SE=")),
            None,
        )
        if iw_se_i is not None and iw_se_i + 1 < len(lines):
            count_str = lines[iw_se_i][len("#Inline Weir SE="):].strip()
            try:
                n_pairs = int(count_str)
            except ValueError:
                n_pairs = 0
            if n_pairs > 0:
                flat = _parse_block(lines[iw_se_i + 1:], n_pairs * 2, _COL)
                crest_profile = [
                    (flat[j], flat[j + 1]) for j in range(0, len(flat) - 1, 2)
                ]

        # -- helper --------------------------------------------------------
        def _cf(parts: list[str], idx: int, default: float = 0.0) -> float:
            """Parse comma-split field at *idx* to float; return *default* on blank."""
            if idx >= len(parts):
                return default
            s = parts[idx].strip()
            if not s:
                return default
            try:
                return float(s)
            except ValueError:
                return default

        def _cf_nan(parts: list[str], idx: int) -> float:
            """Like ``_cf`` but returns ``nan`` for blank fields."""
            if idx >= len(parts):
                return nan
            s = parts[idx].strip()
            if not s:
                return nan
            try:
                return float(s)
            except ValueError:
                return nan

        # -- overflow weir (IW Dist) ---------------------------------------
        weir: Weir | None = None
        iw_dist_i = next(
            (i for i, ln in enumerate(lines) if ln.startswith("IW Dist,")), None
        )
        if iw_dist_i is not None and iw_dist_i + 1 < len(lines):
            parts = lines[iw_dist_i + 1].split(",")
            is_ogee = int(_cf(parts, 6)) == -1
            weir = Weir(
                width=_cf(parts, 1),
                coefficient=_cf(parts, 2),
                skew=_cf(parts, 3),
                max_submergence=_cf(parts, 4),
                min_elevation=_cf_nan(parts, 5),
                shape="Ogee" if is_ogee else "Broad Crested",
                dist=_cf(parts, 0),
                spillway_approach_height=_cf(parts, 7),
                design_energy_head=_cf(parts, 8),
            )

        # -- gate openings (one IW Gate Opening= line per opening) ------------
        # Format: IW Gate Opening=<index>,<name>,<n_gis_points>
        # When n_gis_points > 0 the next line holds n_gis_points (x, y) pairs
        # in 16-char fixed-width columns.
        _GIS_COL = 16
        all_openings: list[tuple[str, list[tuple[float, float]]]] = []
        for i, ln in enumerate(lines):
            if ln.startswith("IW Gate Opening="):
                op_parts = ln[len("IW Gate Opening="):].split(",")
                oname = op_parts[1].strip() if len(op_parts) >= 2 else ""
                n_gis = int(op_parts[2].strip()) if len(op_parts) >= 3 and op_parts[2].strip() else 0
                gis: list[tuple[float, float]] = []
                if n_gis > 0 and i + 1 < len(lines):
                    flat = _parse_block([lines[i + 1]], n_gis * 2, _GIS_COL)
                    gis = [(flat[j], flat[j + 1]) for j in range(0, len(flat) - 1, 2)]
                all_openings.append((oname, gis))

        # -- gate groups (IW Gate Name blocks) -----------------------------
        gate_groups: list[GateGroup] = []
        gate_header_indices = [
            i for i, ln in enumerate(lines) if ln.startswith("IW Gate Name")
        ]
        opening_iter = iter(all_openings)
        for gate_i in gate_header_indices:
            if gate_i + 1 >= len(lines):
                continue
            data_parts = lines[gate_i + 1].split(",")
            name = data_parts[0].strip() if data_parts else ""
            n_openings = int(_cf(data_parts, 13))

            # Station line: fixed-width 8-char columns, immediately after data line
            stations: list[float] = []
            if n_openings > 0 and gate_i + 2 < len(lines):
                stations = _parse_block([lines[gate_i + 2]], n_openings, _COL)

            openings: list[GateOpening] = []
            for k in range(n_openings):
                oname, ogis = next(opening_iter, ("", []))
                st = stations[k] if k < len(stations) else 0.0
                openings.append(GateOpening(name=oname, station=st, gis=ogis))

            gate_type_code = int(_cf(data_parts, 8))
            gate_groups.append(
                GateGroup(
                    name=name,
                    width=_cf(data_parts, 1),
                    height=_cf(data_parts, 2),
                    invert=_cf(data_parts, 3),
                    gate_coefficient=_cf(data_parts, 4),
                    trunnion_exponent=_cf(data_parts, 5),
                    opening_exponent=_cf(data_parts, 6),
                    height_exponent=_cf(data_parts, 7),
                    gate_type=_GATE_TYPE_NAMES.get(gate_type_code, str(gate_type_code)),
                    weir_coefficient=_cf(data_parts, 9),
                    is_ogee=int(_cf(data_parts, 10)) == -1,
                    spillway_approach_height=_cf(data_parts, 11),
                    design_energy_head=_cf(data_parts, 12),
                    trunnion_height=_cf(data_parts, 14),
                    orifice_coefficient=_cf(data_parts, 15),
                    head_reference=int(_cf(data_parts, 16)),
                    radial_coefficient=_cf(data_parts, 17),
                    is_sharp_crested=int(_cf(data_parts, 19)) == -1,
                    use_weir_param1=int(_cf(data_parts, 20)) == -1,
                    use_weir_param2=int(_cf(data_parts, 21)) == -1,
                    use_weir_param3=int(_cf(data_parts, 22)) == -1,
                    openings=openings,
                )
            )

        _empty: tuple[str, str, str] = ("", "", "")
        return Inline(
            mode="",
            upstream_type="XS" if upstream_node != _empty else "",
            downstream_type="XS" if downstream_node != _empty else "",
            location=(river, reach, rs),
            upstream_node=upstream_node,
            downstream_node=downstream_node,
            weir=weir,
            gate_groups=gate_groups,
            description=description,
            pilot_flow=pilot_flow,
            crest_profile=crest_profile,
        )

    def _parse_bridge(
        self,
        river: str,
        reach: str,
        rs: str,
        start: int,
        end: int,
        upstream_node: tuple[str, str, str],
        downstream_node: tuple[str, str, str],
    ) -> Bridge:
        """Parse one bridge or culvert node block into a :class:`Bridge`.

        Parameters
        ----------
        river, reach, rs:
            Node identity.
        start, end:
            Line-index range of the node block (``self._lines[start:end]``).
        upstream_node, downstream_node:
            Adjacent XS tuples from :meth:`_adjacent_xs_nodes`.
        """
        lines = [ln.rstrip("\n") for ln in self._lines[start:end]]

        # Shared float-parsing helper (takes the current parts list as arg
        # to avoid closure rebinding issues across multiple blocks).
        def _f(parts: list[str], idx: int, default: float = 0.0) -> float:
            if idx >= len(parts):
                return default
            s = parts[idx].strip()
            try:
                return float(s) if s else default
            except ValueError:
                return default

        def _fn(parts: list[str], idx: int) -> float:
            """Like _f but returns nan for blank/missing."""
            if idx >= len(parts):
                return nan
            s = parts[idx].strip()
            try:
                return float(s)
            except (ValueError, TypeError):
                return nan

        # -- description ---------------------------------------------------
        description = ""
        desc_s = next(
            (i for i, ln in enumerate(lines) if ln.strip() == "BEGIN DESCRIPTION:"),
            None,
        )
        desc_e = next(
            (i for i, ln in enumerate(lines) if ln.strip() == "END DESCRIPTION:"),
            None,
        )
        if desc_s is not None and desc_e is not None and desc_e > desc_s:
            description = "\n".join(lines[desc_s + 1 : desc_e]).strip()

        # -- node name (Node Name= line, if present) -----------------------
        node_name = ""
        for ln in lines:
            if ln.startswith("Node Name="):
                node_name = ln[len("Node Name="):].strip()
                break

        # -- roadway / deck geometry (Deck Dist ... block) -----------------
        # Header: "Deck Dist Width WeirC Skew NumUp NumDn MinLoCord MaxHiCord
        #          MaxSubmerge Is_Ogee"
        # Data line columns (0-based):
        #   0=dist, 1=width, 2=coef, 3=skew, 4=numUp, 5=numDn,
        #   6=minLoCord, 7=maxHiCord, 8=maxSubmerge, 9=is_ogee
        # After the data line: numUp stations, numUp hi-chords, numUp lo-chords,
        #   then numDn stations, numDn hi-chords, numDn lo-chords
        #   (all 8-char fixed-width columns, up to 10 per row).
        roadway: Roadway | None = None
        weir: Weir | None = None
        deck_hdr_i = next(
            (i for i, ln in enumerate(lines) if ln.startswith("Deck Dist")),
            None,
        )
        if deck_hdr_i is not None and deck_hdr_i + 1 < len(lines):
            dp = lines[deck_hdr_i + 1].split(",")
            dist = _f(dp, 0)
            width = _f(dp, 1)
            coef = _f(dp, 2)
            skew = _f(dp, 3)
            num_up = int(_f(dp, 4))
            num_dn = int(_f(dp, 5))
            min_lo = _fn(dp, 6)
            max_hi = _fn(dp, 7)
            max_sub = _fn(dp, 8)
            is_ogee = int(_f(dp, 9)) == -1
            shape = "Ogee" if is_ogee else "Broad Crested"

            row_off = deck_hdr_i + 2  # first fixed-width row
            rows_up = _row_count(num_up, _COLS_STAE)
            stations_up = _parse_block(lines[row_off: row_off + rows_up], num_up)
            row_off += rows_up
            hi_up = _parse_block(lines[row_off: row_off + rows_up], num_up)
            row_off += rows_up
            lo_up = _parse_block(lines[row_off: row_off + rows_up], num_up)
            row_off += rows_up

            rows_dn = _row_count(num_dn, _COLS_STAE)
            stations_dn = _parse_block(lines[row_off: row_off + rows_dn], num_dn)
            row_off += rows_dn
            hi_dn = _parse_block(lines[row_off: row_off + rows_dn], num_dn)
            row_off += rows_dn
            lo_dn = _parse_block(lines[row_off: row_off + rows_dn], num_dn)

            roadway = Roadway(
                dist=dist,
                width=width,
                weir_coefficient=coef,
                skew=skew,
                max_submergence=max_sub,
                shape=shape,
                min_lo_chord=min_lo,
                max_hi_chord=max_hi,
                stations_up=stations_up,
                hi_chord_up=hi_up,
                lo_chord_up=lo_up,
                stations_dn=stations_dn,
                hi_chord_dn=hi_dn,
                lo_chord_dn=lo_dn,
            )
            weir = Weir(
                width=width,
                coefficient=coef,
                skew=skew,
                max_submergence=max_sub,
                min_elevation=nan,
                shape=shape,
            )

        # -- culvert_groups (Culvert= and Multiple Barrel Culv= lines) ------
        culvert_groups: list[CulvertGroup] = []
        i = 0
        while i < len(lines):
            ln = lines[i]
            if ln.startswith("Culvert="):
                cp = ln[len("Culvert="):].split(",")
                shape_code = int(_f(cp, 0)) if cp[0].strip() else 0
                sname = _CULVERT_SHAPES.get(shape_code, f"Unknown ({shape_code})")
                chart = (
                    int(_f(cp, 15))
                    if len(cp) > 15 and cp[15].strip()
                    else 0
                )
                culvert_group = CulvertGroup(
                    name=cp[13].strip() if len(cp) > 13 else "",
                    shape_code=shape_code,
                    shape_name=sname,
                    span=_f(cp, 1),
                    rise=_f(cp, 2),
                    length=_f(cp, 3),
                    n_top=_f(cp, 4),
                    entrance_loss=_f(cp, 5),
                    exit_loss=_f(cp, 6),
                    inlet_type=int(_f(cp, 7)),
                    outlet_type=int(_f(cp, 8)),
                    upstream_invert=_f(cp, 9),
                    upstream_station=_f(cp, 10),
                    downstream_invert=_f(cp, 11),
                    downstream_station=_f(cp, 12),
                    chart_number=chart,
                )
                # Consume immediately-following optional parameter lines
                j = i + 1
                while j < i + 5 and j < len(lines):
                    nxt = lines[j]
                    if nxt.startswith("Culvert Bottom n="):
                        with contextlib.suppress(ValueError):
                            culvert_group.n_bottom = float(
                                nxt[len("Culvert Bottom n="):].strip()
                            )
                        j += 1
                    elif nxt.startswith("Culvert Bottom Depth="):
                        with contextlib.suppress(ValueError):
                            culvert_group.depth_n_bottom = float(
                                nxt[len("Culvert Bottom Depth="):].strip()
                            )
                        j += 1
                    elif nxt.startswith("BC Culvert Barrel="):
                        bp = nxt[len("BC Culvert Barrel="):].split(",")
                        with contextlib.suppress(ValueError, IndexError):
                            culvert_group.num_barrels = int(bp[0].strip())
                        j += 1
                    else:
                        break
                culvert_groups.append(culvert_group)
                i = j
            elif ln.startswith("Multiple Barrel Culv="):
                # Field layout differs from Culvert=: no us_station/ds_station
                # columns; num_barrels is at index 11; the next line contains
                # num_barrels upstream + num_barrels downstream station values
                # in 8-char fixed-width columns (not yet parsed — see TODO in
                # CulvertGroup docstring).
                mp = ln[len("Multiple Barrel Culv="):].split(",")
                shape_code = int(_f(mp, 0)) if mp[0].strip() else 0
                sname = _CULVERT_SHAPES.get(shape_code, f"Unknown ({shape_code})")
                num_barrels_m = (
                    int(_f(mp, 11)) if len(mp) > 11 and mp[11].strip() else 1
                )
                chart = (
                    int(_f(mp, 14)) if len(mp) > 14 and mp[14].strip() else 0
                )
                culvert_group = CulvertGroup(
                    name=mp[12].strip() if len(mp) > 12 else "",
                    shape_code=shape_code,
                    shape_name=sname,
                    span=_f(mp, 1),
                    rise=_f(mp, 2),
                    length=_f(mp, 3),
                    n_top=_f(mp, 4),
                    entrance_loss=_f(mp, 5),
                    exit_loss=_f(mp, 6),
                    inlet_type=int(_f(mp, 7)),
                    outlet_type=int(_f(mp, 8)),
                    upstream_invert=_f(mp, 9),
                    upstream_station=0.0,   # station data is on next line
                    downstream_invert=_f(mp, 10),
                    downstream_station=0.0,  # station data is on next line
                    chart_number=chart,
                    num_barrels=num_barrels_m,
                )
                # Skip the station line (num_barrels * 2 values) and optional
                # Culvert Bottom n= line
                j = i + 1
                non_data = ("Culvert", "BC ", "Pier ", "BR ")
                if j < len(lines) and not lines[j].startswith(non_data):
                    j += 1  # station data line
                while j < i + 5 and j < len(lines):
                    nxt = lines[j]
                    if nxt.startswith("Culvert Bottom n="):
                        with contextlib.suppress(ValueError):
                            culvert_group.n_bottom = float(
                                nxt[len("Culvert Bottom n="):].strip()
                            )
                        j += 1
                    elif nxt.startswith("Culvert Bottom Depth="):
                        with contextlib.suppress(ValueError):
                            culvert_group.depth_n_bottom = float(
                                nxt[len("Culvert Bottom Depth="):].strip()
                            )
                        j += 1
                    else:
                        break
                culvert_groups.append(culvert_group)
                i = j
            else:
                i += 1

        # -- piers (Pier Skew, UpSta & Num, DnSta & Num= blocks) ----------
        piers: list[Pier] = []
        i = 0
        while i < len(lines):
            ln = lines[i]
            if ln.startswith("Pier Skew, UpSta & Num, DnSta & Num="):
                pp = ln[ln.index("=") + 1:].split(",")
                skew_p = _f(pp, 0)
                up_sta = _f(pp, 1)
                up_num = int(_f(pp, 2))
                dn_sta = _f(pp, 3)
                dn_num = int(_f(pp, 4))

                row_i = i + 1
                rows_up = _row_count(up_num, _COLS_STAE)
                up_widths = _parse_block(lines[row_i: row_i + rows_up], up_num)
                row_i += rows_up
                up_elev = _parse_block(lines[row_i: row_i + rows_up], up_num)
                row_i += rows_up
                rows_dn = _row_count(dn_num, _COLS_STAE)
                dn_widths = _parse_block(lines[row_i: row_i + rows_dn], dn_num)
                row_i += rows_dn
                dn_elev = _parse_block(lines[row_i: row_i + rows_dn], dn_num)
                row_i += rows_dn

                piers.append(Pier(
                    skew=skew_p,
                    upstream_station=up_sta,
                    upstream_count=up_num,
                    downstream_station=dn_sta,
                    downstream_count=dn_num,
                    upstream_widths=up_widths,
                    upstream_elevations=up_elev,
                    downstream_widths=dn_widths,
                    downstream_elevations=dn_elev,
                ))
                i = row_i
            else:
                i += 1

        # -- HTab parameters -----------------------------------------------
        htab_hw_max: float | None = None
        htab_tw_max: float | None = None
        htab_max_flow: float | None = None
        for ln in lines:
            if ln.startswith("BC HTab HWMax="):
                with contextlib.suppress(ValueError):
                    htab_hw_max = float(ln[len("BC HTab HWMax="):].strip())
            elif ln.startswith("BC HTab TWMax="):
                with contextlib.suppress(ValueError):
                    htab_tw_max = float(ln[len("BC HTab TWMax="):].strip())
            elif ln.startswith("BC HTab MaxFlow="):
                with contextlib.suppress(ValueError):
                    htab_max_flow = float(ln[len("BC HTab MaxFlow="):].strip())

        _empty: tuple[str, str, str] = ("", "", "")
        return Bridge(
            mode="",
            upstream_type="XS" if upstream_node != _empty else "",
            downstream_type="XS" if downstream_node != _empty else "",
            location=(river, reach, rs),
            upstream_node=upstream_node,
            downstream_node=downstream_node,
            weir=weir,
            description=description,
            roadway=roadway,
            culvert_groups=culvert_groups,
            piers=piers,
            node_name=node_name,
            htab_hw_max=htab_hw_max,
            htab_tw_max=htab_tw_max,
            htab_max_flow=htab_max_flow,
        )

    def _parse_lateral(
        self,
        river: str,
        reach: str,
        rs: str,
        start: int,
        end: int,
        upstream_node: tuple[str, str, str],
    ) -> Lateral:
        """Parse one lateral structure node block into a :class:`Lateral`.

        Parameters
        ----------
        river, reach, rs:
            Node identity.
        start, end:
            Line-index range of the node block (``self._lines[start:end]``).
        upstream_node:
            Adjacent upstream XS tuple from :meth:`_adjacent_xs_nodes`.
        """
        lines = [ln.rstrip("\n") for ln in self._lines[start:end]]

        # -- description ---------------------------------------------------
        description = ""
        desc_s = next(
            (i for i, ln in enumerate(lines) if ln.strip() == "BEGIN DESCRIPTION:"),
            None,
        )
        desc_e = next(
            (i for i, ln in enumerate(lines) if ln.strip() == "END DESCRIPTION:"),
            None,
        )
        if desc_s is not None and desc_e is not None and desc_e > desc_s:
            description = "\n".join(lines[desc_s + 1 : desc_e]).strip()

        # -- downstream connection (Lateral Weir End=river,reach,rs,...) ---
        downstream_node = ""
        for ln in lines:
            if ln.startswith("Lateral Weir End="):
                parts = ln[len("Lateral Weir End="):].split(",")
                if len(parts) >= 2:
                    lat_river = parts[0].strip()
                    lat_reach = parts[1].strip()
                    downstream_node = f"{lat_river} {lat_reach}".strip()
                break

        # -- scalar Lateral Weir fields ------------------------------------
        def _get_float(prefix: str, default: float = nan) -> float:
            for ln in lines:
                if ln.startswith(prefix):
                    s = ln[len(prefix):].strip().rstrip(",").strip()
                    try:
                        return float(s)
                    except ValueError:
                        return default
            return default

        pos = int(_get_float("Lateral Weir Pos=", default=0.0))
        distance = _get_float("Lateral Weir Distance=")
        flap_val = _get_float("Lateral Weir Flap Gates=", default=0.0)
        flap_gates = not (isnan(flap_val) or flap_val == 0.0)
        tw_val = _get_float("Lateral Weir TW Multiple XS=", default=0.0)
        tw_multiple_xs = not (isnan(tw_val) or tw_val == 0.0)

        # -- weir from individual Lateral Weir lines -----------------------
        weir: Weir | None = None
        wd = _get_float("Lateral Weir WD=")
        coef = _get_float("Lateral Weir Coef=")
        if not isnan(wd):  # Lateral Weir WD= line was found
            ws_criteria = _get_float("Lateral Weir WSCriteria=", default=0.0)
            use_ws = (ws_criteria == -1)
            # Lateral Weir Type= 0=broad crested, 1=ogee (not commonly set)
            lat_type = _get_float("Lateral Weir Type=", default=0.0)
            shape = "Ogee" if int(lat_type) == 1 else "Broad Crested"
            weir = Weir(
                width=wd,
                coefficient=coef if not isnan(coef) else 0.0,
                skew=0.0,
                max_submergence=nan,
                min_elevation=nan,
                shape=shape,
                use_water_surface=use_ws,
            )

        # -- weir crest profile (Lateral Weir SE= N block) -----------------
        # Format: "Lateral Weir SE= N" then N pairs of (station, elevation)
        # in 8-char fixed-width columns, up to 10 values per row.
        crest_profile: list[tuple[float, float]] = []
        for i, ln in enumerate(lines):
            if ln.startswith("Lateral Weir SE="):
                n_str = ln[len("Lateral Weir SE="):].strip()
                try:
                    n_pairs = int(n_str)
                except ValueError:
                    n_pairs = 0
                if n_pairs > 0:
                    n_rows = _row_count(n_pairs * 2, _COLS_STAE)
                    flat = _parse_block(lines[i + 1: i + 1 + n_rows], n_pairs * 2)
                    crest_profile = [
                        (flat[j], flat[j + 1])
                        for j in range(0, len(flat) - 1, 2)
                    ]
                break

        _empty: tuple[str, str, str] = ("", "", "")
        return Lateral(
            mode="",
            upstream_type="XS" if upstream_node != _empty else "",
            downstream_type="XS" if downstream_node else "",
            location=(river, reach, rs),
            upstream_node=upstream_node,
            downstream_node=downstream_node,
            weir=weir,
            description=description,
            pos=pos,
            distance=distance,
            crest_profile=crest_profile,
            flap_gates=flap_gates,
            tw_multiple_xs=tw_multiple_xs,
        )

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

    def inline_gate_groups(self, river: str, reach: str, rs: str) -> list[str]:
        """Return gate group names for an inline structure, in file order.

        Parses ``IW Gate Name`` header lines from the node block.  The name
        is the first comma-separated field on the data line that immediately
        follows each header line.

        Parameters
        ----------
        river:
            River name.
        reach:
            Reach name.
        rs:
            River station of the inline structure.

        Returns
        -------
        list[str]
            Gate group names, e.g. ``["Left Group", "Center Group", "Right Group"]``.

        Raises
        ------
        KeyError
            If the node is not found or is not an inline structure (type 5).
        """
        lines = self.get_node_lines(river, reach, rs)
        if lines is None:
            raise KeyError(
                f"Node not found: river={river!r}, reach={reach!r}, rs={rs!r}"
            )
        node_type = self.node_type(river, reach, rs)
        if node_type != NODE_INLINE_STRUCTURE:
            raise KeyError(
                f"Node at river={river!r}, reach={reach!r}, rs={rs!r} "
                f"is not an inline structure (type={node_type!r})"
            )
        names: list[str] = []
        for i, line in enumerate(lines):
            if line.startswith("IW Gate Name") and i + 1 < len(lines):
                names.append(lines[i + 1].split(",", 1)[0].strip())
        return names

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
        self._modified = False
        self._structures_cache = None
