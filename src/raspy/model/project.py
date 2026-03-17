"""Read HEC-RAS project files (.prj).

Note: HEC-RAS .prj files are unrelated to ESRI projection (.prj) files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger("raspy.model")


class ProjectFile:
    """Read-only parser for a HEC-RAS project file (.prj).

    Parses the project file to expose project metadata, unit system, and
    lists of associated file extensions (geometry, plan, flow, etc.).

    Example
    -------
    >>> prj = ProjectFile("Baxter.prj")
    >>> prj.title
    'Baxter River GIS Example'
    >>> prj.units
    'English'
    >>> prj.current_plan_file
    PosixPath('Baxter.p01')

    Derived from: HEC-RAS project file format (no archive equivalent).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.is_file():
            raise FileNotFoundError(f"Project file not found: {self._path}")
        with open(self._path, encoding="utf-8", errors="replace") as fh:
            self._lines: list[str] = fh.readlines()
        self._parse()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_first(self, key: str) -> str | None:
        """Return the stripped value for the first matching *key=*, or None."""
        prefix = key + "="
        for line in self._lines:
            if line.startswith(prefix):
                value = line[len(prefix):].strip()
                return value if value else None
        return None

    def _get_all(self, key: str) -> list[str]:
        """Return all stripped values for *key=* lines (for repeated keys)."""
        prefix = key + "="
        return [
            line[len(prefix):].strip()
            for line in self._lines
            if line.startswith(prefix) and line[len(prefix):].strip()
        ]

    def _ext_to_path(self, ext: str) -> Path:
        """Convert a file extension token (e.g. 'p01') to an absolute Path."""
        return self._path.with_suffix(f".{ext}")

    def _parse(self) -> None:
        """Parse all fields from the file lines into cached attributes."""
        # --- title ---
        self._title = self._get_first("Proj Title")

        # --- units: a bare flag line, no '=' ---
        self._units: Literal["English", "SI"] = "English"
        for line in self._lines:
            stripped = line.strip()
            if stripped == "SI Units":
                self._units = "SI"
                break
            if stripped == "English Units":
                self._units = "English"
                break

        # --- current plan / geom ---
        self._current_plan = self._get_first("Current Plan")
        self._current_geom = self._get_first("Current Geom")

        # --- default expansion/contraction ---
        raw_expcontr = self._get_first("Default Exp/Contr")
        if raw_expcontr is not None:
            try:
                parts = raw_expcontr.split(",")
                self._default_exp_contr: tuple[float, float] | None = (
                    float(parts[0]),
                    float(parts[1]),
                )
            except (ValueError, IndexError):
                logger.warning(
                    "Could not parse Default Exp/Contr=%r in %s", raw_expcontr, self._path
                )
                self._default_exp_contr = None
        else:
            self._default_exp_contr = None

        # --- associated file lists ---
        self._geom_files = [self._ext_to_path(e) for e in self._get_all("Geom File")]
        self._plan_files = [self._ext_to_path(e) for e in self._get_all("Plan File")]
        self._steady_flow_files = [self._ext_to_path(e) for e in self._get_all("Flow File")]
        self._unsteady_flow_files = [self._ext_to_path(e) for e in self._get_all("Unsteady File")]
        self._sediment_files = [self._ext_to_path(e) for e in self._get_all("Sediment File")]
        self._quasi_steady_files = [self._ext_to_path(e) for e in self._get_all("QuasiSteady File")]

        # --- description block ---
        self._description = _parse_description(self._lines)

        # --- cache ---
        self._plans_cache: list[dict[str, str | Path | None]] | None = None

        # --- DSS ---
        self._dss_start_date = self._get_first("DSS Start Date")
        self._dss_start_time = self._get_first("DSS Start Time")
        self._dss_end_date = self._get_first("DSS End Date")
        self._dss_end_time = self._get_first("DSS End Time")

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        """Absolute path to the project file."""
        return self._path

    @property
    def title(self) -> str | None:
        """Project title (``Proj Title=``)."""
        return self._title

    @property
    def units(self) -> Literal["English", "SI"]:
        """Unit system: ``'English'`` or ``'SI'``."""
        return self._units

    @property
    def current_plan_ext(self) -> str | None:
        """Extension of the current plan file, e.g. ``'p01'``.

        Returns ``None`` when the project has no current plan set.
        """
        return self._current_plan

    @property
    def current_plan_file(self) -> Path | None:
        """Full path to the current plan file, or ``None`` if unset."""
        if self._current_plan is None:
            return None
        return self._ext_to_path(self._current_plan)


    @property
    def default_exp_contr(self) -> tuple[float, float] | None:
        """Default expansion/contraction coefficients as ``(expansion, contraction)``.

        Returns ``None`` if the field is absent or malformed.
        """
        return self._default_exp_contr

    @property
    def description(self) -> str:
        """Project description text (between ``BEGIN DESCRIPTION:`` and ``END DESCRIPTION:``)."""
        return self._description

    @property
    def geom_files(self) -> list[Path]:
        """Full paths to all geometry files listed in the project."""
        return list(self._geom_files)

    @property
    def plan_files(self) -> list[Path]:
        """Full paths to all plan files listed in the project."""
        return list(self._plan_files)

    @property
    def steady_flow_files(self) -> list[Path]:
        """Full paths to all steady flow files (``Flow File=``) listed in the project."""
        return list(self._steady_flow_files)

    @property
    def unsteady_flow_files(self) -> list[Path]:
        """Full paths to all unsteady flow files listed in the project."""
        return list(self._unsteady_flow_files)

    @property
    def sediment_files(self) -> list[Path]:
        """Full paths to all sediment files listed in the project."""
        return list(self._sediment_files)

    @property
    def quasi_steady_files(self) -> list[Path]:
        """Full paths to all quasi-steady files listed in the project."""
        return list(self._quasi_steady_files)

    @property
    def dss_start_date(self) -> str | None:
        """DSS simulation start date string, or ``None`` if unset."""
        return self._dss_start_date

    @property
    def dss_start_time(self) -> str | None:
        """DSS simulation start time string, or ``None`` if unset."""
        return self._dss_start_time

    @property
    def dss_end_date(self) -> str | None:
        """DSS simulation end date string, or ``None`` if unset."""
        return self._dss_end_date

    @property
    def dss_end_time(self) -> str | None:
        """DSS simulation end time string, or ``None`` if unset."""
        return self._dss_end_time

    # ------------------------------------------------------------------
    # Raw escape hatch
    # ------------------------------------------------------------------

    def get(self, key: str) -> str | None:
        """Return the raw stripped value for *key*, or ``None`` if absent/empty.

        Use this for fields not exposed as typed properties (e.g. ``'Y Axis Title'``).
        """
        return self._get_first(key)

    def get_all(self, key: str) -> list[str]:
        """Return all raw stripped values for repeated *key* lines.

        Use this for repeated fields not exposed as typed properties.
        """
        return self._get_all(key)

    # ------------------------------------------------------------------
    # Plan metadata helpers
    # ------------------------------------------------------------------

    def plan_titles(self) -> list[str | None]:
        """Return the ``Plan Title`` for each plan file, in project order.

        A ``None`` entry means the plan file does not exist or has no
        ``Plan Title=`` line.
        """
        return [_read_plan_field(p, "Plan Title") for p in self._plan_files]

    def plan_short_ids(self) -> list[str | None]:
        """Return the ``Short Identifier`` for each plan file, in project order.

        A ``None`` entry means the plan file does not exist or has no
        ``Short Identifier=`` line.
        """
        return [_read_plan_field(p, "Short Identifier") for p in self._plan_files]

    def plans(self) -> list[dict[str, str | Path | None]]:
        """Return metadata for each plan file in project order.

        Each entry is a dict with keys:

        - ``"ext"``      — file extension token, e.g. ``'p01'``
        - ``"path"``     — full :class:`~pathlib.Path` to the plan file
        - ``"title"``    — value of ``Plan Title=``, or ``None``
        - ``"short_id"`` — value of ``Short Identifier=``, or ``None``

        Results are cached after the first call.
        """
        if self._plans_cache is None:
            self._plans_cache = [
                {
                    "ext": p.suffix.lstrip("."),
                    "path": p,
                    "title": _read_plan_field(p, "Plan Title"),
                    "short_id": _read_plan_field(p, "Short Identifier"),
                }
                for p in self._plan_files
            ]
        return list(self._plans_cache)

    def __repr__(self) -> str:
        return f"ProjectFile({self._path!r})"


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _read_plan_field(plan_path: Path, key: str) -> str | None:
    """Read the first ``key=value`` line from *plan_path*, or return ``None``.

    Reads only until the key is found (plan headers are always near the top),
    so this is efficient even for large plan files.
    """
    if not plan_path.is_file():
        return None
    prefix = key + "="
    try:
        with open(plan_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith(prefix):
                    value = line[len(prefix):].strip()
                    return value if value else None
    except OSError:
        return None
    return None


def _parse_description(lines: list[str]) -> str:
    """Extract the text between ``BEGIN DESCRIPTION:`` and ``END DESCRIPTION:``."""
    collecting = False
    body: list[str] = []
    for line in lines:
        stripped = line.rstrip("\n")
        if stripped.strip() == "BEGIN DESCRIPTION:":
            collecting = True
            continue
        if stripped.strip() == "END DESCRIPTION:":
            break
        if collecting:
            body.append(stripped)
    return "\n".join(body).strip()
