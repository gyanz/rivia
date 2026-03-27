"""Read HEC-RAS DSS output files (.dss) for plan results.

Provides :class:`DssReader`, accessible via :attr:`Model.dss`, which reads
time-series data from the DSS file written alongside an HEC-RAS unsteady
simulation.

DSS path convention used by HEC-RAS::

    /A            /B                /C      /D /E        /F
    /<river reach>/<location>/<output>//<interval>/<plan_short_id>/

Where:

- **A** — ``"<river> <reach>"`` (river and reach names joined by a space)
- **B** — depends on node type and gate selector:

  - Cross section → RS bare (e.g. ``"7"``)
  - Inline structure, no gate → ``"<RS> INL STRUCT"``
  - Inline structure, gate total → ``"<RS> INL STRUCT GATE TOTAL"``
  - Inline structure, gate *N* → ``"<RS> INL STRUCT Gate #N"``

- **C** — output variable:

  - Cross section: ``"FLOW"``, ``"FLOW-CUM"``, ``"STAGE"``
  - Inline structure: ``"FLOW-TOTAL"``, ``"FLOW-WEIR"``, ``"STAGE-HW"``,
    ``"STAGE-TW"``, ``"FLOW-GATE"``, ``"Gate Opening"``

- **D** — date block (empty — let pydsstools select via time window)
- **E** — output interval taken from ``plan.dss_interval``
- **F** — plan short identifier from ``plan.short_id``

Reading time-series directly from the DSS file is particularly useful when a
model is run as a sequence of short-duration simulations using restart files.
HEC-RAS appends each run's output to the DSS file, so the DSS record spans the
full period across all restarts.  The plan HDF file, by contrast, is overwritten
on each run and therefore only contains results for the most recent duration.

Requires the ``pydsstools`` optional dependency (pre-release version supports
Python 3.10+).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .geometry import NODE_INLINE_STRUCTURE, NODE_XS

if TYPE_CHECKING:
    from . import Model

logger = logging.getLogger("raspy.model")

__all__ = ["DssReader"]

# ---------------------------------------------------------------------------
# Output variable name constants
# ---------------------------------------------------------------------------

#: Cross-section flow rate
XS_FLOW = "FLOW"
#: Cross-section cumulative flow volume
XS_FLOW_CUM = "FLOW-CUM"
#: Cross-section stage (water surface elevation)
XS_STAGE = "STAGE"

#: Inline structure total flow (weir + gates)
INL_FLOW_TOTAL = "FLOW-TOTAL"
#: Inline structure weir flow
INL_FLOW_WEIR = "FLOW-WEIR"
#: Inline structure headwater stage
INL_STAGE_HW = "STAGE-HW"
#: Inline structure tailwater stage
INL_STAGE_TW = "STAGE-TW"
#: Inline structure gate flow (structure-total or per-gate)
INL_FLOW_GATE = "FLOW-GATE"
#: Inline structure gate opening
INL_GATE_OPENING = "Gate Opening"


# ---------------------------------------------------------------------------
# DssReader
# ---------------------------------------------------------------------------


class DssReader:
    """DSS output reader for an HEC-RAS model.

    Obtained via :attr:`Model.dss` rather than instantiated directly.  Reads
    time-series results from the ``.dss`` file written alongside an unsteady
    simulation.

    Requires the ``pydsstools`` package::

        pip install pydsstools

    Usage::

        model = Model("path/to/project.prj")
        flow  = model.dss.flow("Canal 1", "Pool 1-4", "7")
        hw    = model.dss.stage_hw("Canal 1", "Pool 1-4", "6.9")
        gate1 = model.dss.gate_opening(1, "Canal 1", "Pool 1-4", "6.9")
    """

    def __init__(self, model: Model) -> None:
        self._model = model

    # ------------------------------------------------------------------
    # File path
    # ------------------------------------------------------------------

    @property
    def dss_file(self) -> Path:
        """Path to the DSS output file (project file with ``.dss`` extension)."""
        return self._model.project_file.with_suffix(".dss")

    # ------------------------------------------------------------------
    # Generic timeseries
    # ------------------------------------------------------------------

    def timeseries(
        self,
        river: str,
        reach: str,
        rs: str,
        output: str,
        *,
        gate: str | int | None = None,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        trim_missing: bool = False,
    ) -> pd.Series:
        """Return a time-series for a cross-section or inline structure.

        Determines the node type from the geometry file, constructs the DSS
        pathname, and returns the result as a :class:`pandas.Series` indexed
        by datetime.

        Parameters
        ----------
        river:
            River name.
        reach:
            Reach name.
        rs:
            River station.
        output:
            DSS Part C variable name.

            Cross sections: ``"FLOW"``, ``"FLOW-CUM"``, ``"STAGE"``.

            Inline structures: ``"FLOW-TOTAL"``, ``"FLOW-WEIR"``,
            ``"STAGE-HW"``, ``"STAGE-TW"``, ``"FLOW-GATE"``,
            ``"Gate Opening"``.

            Module-level constants ``XS_FLOW``, ``INL_FLOW_TOTAL``, etc. are
            provided for convenience.
        gate:
            Gate selector for inline structure gate outputs (ignored for cross
            sections).

            ``None`` (default) — structure-level path
            (``"<RS> INL STRUCT"``); use for ``FLOW-TOTAL``, ``FLOW-WEIR``,
            ``STAGE-HW``, ``STAGE-TW``.

            ``"GATE TOTAL"`` — aggregate gate path
            (``"<RS> INL STRUCT GATE TOTAL"``).

            ``str`` — gate group name as defined in the geometry file (e.g.
            ``"Left Group"``); produces ``"<RS> INL STRUCT Left Group"``.

            ``int`` — zero-based index into the gate group list parsed from
            the geometry file; resolved to the group name automatically.
        start:
            Start of the time window as a :class:`datetime` or a DSS-style
            date string (e.g. ``"01Jan2020"``).  ``None`` defaults to the
            plan simulation start.
        end:
            End of the time window as a :class:`datetime` or a DSS-style
            date string.  ``None`` defaults to the plan simulation end.
        trim_missing:
            Passed to ``pydsstools`` ``read_ts``.  When ``False`` (default),
            missing values are kept; when ``True``, leading and trailing
            missing values are removed.

        Returns
        -------
        pd.Series
            Time-series values indexed by :class:`pandas.DatetimeIndex`,
            with ``name`` set to *output*.

        Raises
        ------
        ImportError
            If ``pydsstools`` is not installed.
        FileNotFoundError
            If the DSS file does not exist.
        ValueError
            If the node at *(river, reach, rs)* is not found in the geometry,
            or its type is not supported (only cross sections and inline
            structures are currently handled).
        """
        try:
            from pydsstools.heclib.dss import HecDss
        except ImportError as exc:
            raise ImportError(
                "pydsstools is required for DSS reading. "
                "Install it with: pip install pydsstools"
            ) from exc

        if not self.dss_file.is_file():
            raise FileNotFoundError(f"DSS file not found: {self.dss_file}")

        node_type = self._model.geom.node_type(river, reach, rs)
        if node_type is None:
            raise ValueError(
                f"Node not found in geometry: river={river!r}, "
                f"reach={reach!r}, rs={rs!r}"
            )
        if node_type not in (NODE_XS, NODE_INLINE_STRUCTURE):
            raise ValueError(
                f"Node type {node_type!r} at river={river!r}, reach={reach!r}, "
                f"rs={rs!r} is not supported; only cross sections "
                "(NODE_XS) and inline structures (NODE_INLINE_STRUCTURE) "
                "are currently handled."
            )

        if gate is not None and node_type == NODE_XS:
            raise ValueError(
                f"gate selector is not valid for cross sections; "
                f"node at river={river!r}, reach={reach!r}, rs={rs!r} "
                f"is a cross section."
            )

        gate_name: str | None
        if isinstance(gate, int):
            groups = self._model.geom.inline_gate_groups(river, reach, rs)
            try:
                gate_name = groups[gate]
            except IndexError as exc:
                raise IndexError(
                    f"Gate index {gate} out of range; "
                    f"{len(groups)} gate group(s) at {river!r}, {reach!r}, {rs!r}"
                ) from exc
        else:
            gate_name = gate

        plan = self._model.plan
        part_a = f"{river} {reach}"
        part_b = _part_b(node_type, rs, gate_name)
        part_e = plan.dss_interval or ""
        part_f = plan.short_id or ""
        pathname = f"/{part_a}/{part_b}/{output}//{part_e}/{part_f}/"

        logger.debug("Reading DSS pathname: %s", pathname)

        sim_window = plan.simulation_window
        if sim_window is not None:
            if start is None:
                start = sim_window[0].replace(",", " ")
            if end is None:
                end = sim_window[1].replace(",", " ")
        window = (start, end) if (start is not None or end is not None) else None

        logger.debug("DSS time window: %s", window)

        with HecDss.Open(str(self.dss_file)) as fid:
            ts = fid.read_ts(pathname, window=window, trim_missing=trim_missing)

        times = pd.DatetimeIndex([t.datetime() for t in ts.times])
        values = ts.values
        values[ts.nodata] = np.nan
        return pd.Series(values, index=times, name=output)

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def flow(
        self,
        river: str,
        reach: str,
        rs: str,
        *,
        window: tuple[str | datetime | None, str | datetime | None] | None = None,
    ) -> pd.Series:
        """Return the flow time-series for a cross-section or inline structure.

        The DSS Part C variable is selected automatically based on node type:

        - Cross section → ``"FLOW"``
        - Inline structure → ``"FLOW-TOTAL"``

        Parameters
        ----------
        river:
            River name.
        reach:
            Reach name.
        rs:
            River station.
        window:
            Optional ``(start, end)`` time window.  Each bound is a
            :class:`datetime` or a DSS-style date string (e.g.
            ``"01Jan2020"``), or ``None`` to use the plan simulation window.

        Returns
        -------
        pd.Series
            Flow values indexed by :class:`pandas.DatetimeIndex`.

        Raises
        ------
        ValueError
            If the node is not found or its type is not a cross section or
            inline structure.
        """
        node_type = self._model.geom.node_type(river, reach, rs)
        if node_type is None:
            raise ValueError(
                f"Node not found in geometry: river={river!r}, "
                f"reach={reach!r}, rs={rs!r}"
            )
        if node_type == NODE_XS:
            output = XS_FLOW
        elif node_type == NODE_INLINE_STRUCTURE:
            output = INL_FLOW_TOTAL
        else:
            raise ValueError(
                f"flow() requires a cross section or inline structure; "
                f"node at river={river!r}, reach={reach!r}, rs={rs!r} "
                f"has type {node_type!r}."
            )
        start, end = window if window is not None else (None, None)
        return self.timeseries(river, reach, rs, output, start=start, end=end)

    def stage(
        self,
        river: str,
        reach: str,
        rs: str,
        *,
        window: tuple[str | datetime | None, str | datetime | None] | None = None,
    ) -> pd.Series:
        """Return the stage (water surface elevation) time-series for a cross section.

        Parameters
        ----------
        river:
            River name.
        reach:
            Reach name.
        rs:
            River station of the cross section.
        window:
            Optional ``(start, end)`` time window.  Each bound is a
            :class:`datetime` or a DSS-style date string (e.g.
            ``"01Jan2020"``), or ``None`` to use the plan simulation window.

        Returns
        -------
        pd.Series
            Stage values indexed by :class:`pandas.DatetimeIndex`.

        Raises
        ------
        ValueError
            If the node at *(river, reach, rs)* is not a cross section.
        """
        node_type = self._model.geom.node_type(river, reach, rs)
        if node_type is None:
            raise ValueError(
                f"Node not found in geometry: river={river!r}, "
                f"reach={reach!r}, rs={rs!r}"
            )
        if node_type != NODE_XS:
            raise ValueError(
                f"stage() requires a cross section; "
                f"node at river={river!r}, reach={reach!r}, rs={rs!r} "
                f"has type {node_type!r} (expected NODE_XS={NODE_XS})."
            )
        start, end = window if window is not None else (None, None)
        return self.timeseries(river, reach, rs, XS_STAGE, start=start, end=end)

    def stage_hw(
        self,
        river: str,
        reach: str,
        rs: str,
        *,
        window: tuple[str | datetime | None, str | datetime | None] | None = None,
    ) -> pd.Series:
        """Return the headwater stage time-series for an inline structure.

        Parameters
        ----------
        river:
            River name.
        reach:
            Reach name.
        rs:
            River station of the inline structure.
        window:
            Optional ``(start, end)`` time window.  Each bound is a
            :class:`datetime` or a DSS-style date string (e.g.
            ``"01Jan2020"``), or ``None`` to use the plan simulation window.

        Returns
        -------
        pd.Series
            Headwater stage values indexed by :class:`pandas.DatetimeIndex`.

        Raises
        ------
        ValueError
            If the node at *(river, reach, rs)* is not an inline structure.
        """
        _assert_inline(self._model.geom.node_type(river, reach, rs), river, reach, rs)
        start, end = window if window is not None else (None, None)
        return self.timeseries(river, reach, rs, INL_STAGE_HW, start=start, end=end)

    def stage_tw(
        self,
        river: str,
        reach: str,
        rs: str,
        *,
        window: tuple[str | datetime | None, str | datetime | None] | None = None,
    ) -> pd.Series:
        """Return the tailwater stage time-series for an inline structure.

        Parameters
        ----------
        river:
            River name.
        reach:
            Reach name.
        rs:
            River station of the inline structure.
        window:
            Optional ``(start, end)`` time window.  Each bound is a
            :class:`datetime` or a DSS-style date string (e.g.
            ``"01Jan2020"``), or ``None`` to use the plan simulation window.

        Returns
        -------
        pd.Series
            Tailwater stage values indexed by :class:`pandas.DatetimeIndex`.

        Raises
        ------
        ValueError
            If the node at *(river, reach, rs)* is not an inline structure.
        """
        _assert_inline(self._model.geom.node_type(river, reach, rs), river, reach, rs)
        start, end = window if window is not None else (None, None)
        return self.timeseries(river, reach, rs, INL_STAGE_TW, start=start, end=end)

    def gate_opening(
        self,
        gate: str | int,
        river: str,
        reach: str,
        rs: str,
        *,
        window: tuple[str | datetime | None, str | datetime | None] | None = None,
    ) -> pd.Series:
        """Return the gate opening time-series for a specific gate on an
        inline structure.

        Parameters
        ----------
        gate:
            Gate group name (str) as defined in the geometry file, or a
            zero-based integer index into the gate group list.
        river:
            River name.
        reach:
            Reach name.
        rs:
            River station of the inline structure.
        window:
            Optional ``(start, end)`` time window.  Each bound is a
            :class:`datetime` or a DSS-style date string (e.g.
            ``"01Jan2020"``), or ``None`` to use the plan simulation window.

        Returns
        -------
        pd.Series
            Gate opening values indexed by :class:`pandas.DatetimeIndex`.

        Raises
        ------
        ValueError
            If the node at *(river, reach, rs)* is not an inline structure.
        """
        _assert_inline(self._model.geom.node_type(river, reach, rs), river, reach, rs)
        start, end = window if window is not None else (None, None)
        return self.timeseries(
            river, reach, rs, INL_GATE_OPENING, gate=gate, start=start, end=end
        )

    def gate_flow_total(
        self,
        river: str,
        reach: str,
        rs: str,
        *,
        window: tuple[str | datetime | None, str | datetime | None] | None = None,
    ) -> pd.Series:
        """Return the total gate flow time-series for an inline structure.

        Reads the ``FLOW-GATE`` variable at the ``GATE TOTAL`` path
        (``"<RS> INL STRUCT GATE TOTAL"``), which is the sum of flow across
        all gates.

        Parameters
        ----------
        river:
            River name.
        reach:
            Reach name.
        rs:
            River station of the inline structure.
        window:
            Optional ``(start, end)`` time window.  Each bound is a
            :class:`datetime` or a DSS-style date string (e.g.
            ``"01Jan2020"``), or ``None`` to use the plan simulation window.

        Returns
        -------
        pd.Series
            Total gate flow values indexed by :class:`pandas.DatetimeIndex`.

        Raises
        ------
        ValueError
            If the node at *(river, reach, rs)* is not an inline structure.
        """
        _assert_inline(self._model.geom.node_type(river, reach, rs), river, reach, rs)
        start, end = window if window is not None else (None, None)
        return self.timeseries(
            river, reach, rs, INL_FLOW_GATE, gate="GATE TOTAL", start=start, end=end
        )

    def weir_flow(
        self,
        river: str,
        reach: str,
        rs: str,
        *,
        window: tuple[str | datetime | None, str | datetime | None] | None = None,
    ) -> pd.Series:
        """Return the weir flow time-series for an inline structure.

        Parameters
        ----------
        river:
            River name.
        reach:
            Reach name.
        rs:
            River station of the inline structure.
        window:
            Optional ``(start, end)`` time window.  Each bound is a
            :class:`datetime` or a DSS-style date string (e.g.
            ``"01Jan2020"``), or ``None`` to use the plan simulation window.

        Returns
        -------
        pd.Series
            Weir flow values indexed by :class:`pandas.DatetimeIndex`.

        Raises
        ------
        ValueError
            If the node at *(river, reach, rs)* is not an inline structure.
        """
        _assert_inline(self._model.geom.node_type(river, reach, rs), river, reach, rs)
        start, end = window if window is not None else (None, None)
        return self.timeseries(river, reach, rs, INL_FLOW_WEIR, start=start, end=end)

    def flow_cum(
        self,
        river: str,
        reach: str,
        rs: str,
        *,
        window: tuple[str | datetime | None, str | datetime | None] | None = None,
    ) -> pd.Series:
        """Return the cumulative flow volume time-series for a cross section.

        Parameters
        ----------
        river:
            River name.
        reach:
            Reach name.
        rs:
            River station of the cross section.
        window:
            Optional ``(start, end)`` time window.  Each bound is a
            :class:`datetime` or a DSS-style date string (e.g.
            ``"01Jan2020"``), or ``None`` to use the plan simulation window.

        Returns
        -------
        pd.Series
            Cumulative flow volume values indexed by :class:`pandas.DatetimeIndex`.

        Raises
        ------
        ValueError
            If the node at *(river, reach, rs)* is not a cross section.
        """
        node_type = self._model.geom.node_type(river, reach, rs)
        if node_type is None:
            raise ValueError(
                f"Node not found in geometry: river={river!r}, "
                f"reach={reach!r}, rs={rs!r}"
            )
        if node_type != NODE_XS:
            raise ValueError(
                f"flow_cum() requires a cross section; "
                f"node at river={river!r}, reach={reach!r}, rs={rs!r} "
                f"has type {node_type!r} (expected NODE_XS={NODE_XS})."
            )
        start, end = window if window is not None else (None, None)
        return self.timeseries(river, reach, rs, XS_FLOW_CUM, start=start, end=end)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assert_inline(node_type: int | None, river: str, reach: str, rs: str) -> None:
    """Raise ``ValueError`` if *node_type* is not ``NODE_INLINE_STRUCTURE``."""
    if node_type is None:
        raise ValueError(
            f"Node not found in geometry: river={river!r}, "
            f"reach={reach!r}, rs={rs!r}"
        )
    if node_type != NODE_INLINE_STRUCTURE:
        raise ValueError(
            f"Method requires an inline structure; "
            f"node at river={river!r}, reach={reach!r}, rs={rs!r} "
            f"has type {node_type!r} "
            f"(expected NODE_INLINE_STRUCTURE={NODE_INLINE_STRUCTURE})."
        )


def _part_b(node_type: int, rs: str, gate: str | None) -> str:
    """Return the DSS Part B string for a given node type and gate selector.

    RS is normalised (stripped of whitespace and trailing ``*``) to match the
    convention used in the DSS file.

    *gate* must already be resolved to a string name (or ``None``).  Integer
    gate indexes are resolved by the caller before reaching this function.
    """
    rs_norm = rs.strip().rstrip("*").strip()
    if node_type == NODE_XS:
        return rs_norm
    # Inline structure
    if gate is None:
        return f"{rs_norm} INL STRUCT"
    return f"{rs_norm} INL STRUCT {gate}"
