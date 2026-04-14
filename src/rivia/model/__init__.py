"""High-level HEC-RAS project interface and text file I/O.

:class:`Project` is the primary entry point — it opens a project via COM,
manages plan switching and reloading, and provides lazy access to plan files,
geometry files, flow files, and HDF results.

Individual file classes (:class:`Geometry`, :class:`Plan`, etc.) can
also be used standalone without a :class:`Project` instance.
"""

import atexit
import collections
import contextlib
import dataclasses
import datetime as dt
import logging
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from rivia.hdf.staleness import PlanStalenessReport

from rivia.hdf import SteadyPlan, UnsteadyPlan

from .. import controller
from ..utils import normalize_sim_end_time, normalize_sim_start_time
from ._dss import DssReader
from ._mapper import MapperExtension
from .geometry import (  # noqa: F401
    CrossSection,
    Geometry,
    IneffArea,
    ManningEntry,
    NodeType,
)
from .plan import Plan
from .project import Proj
from .steady_flow import SteadyBoundary, SteadyFlow
from .unsteady_flow import (
    FlowHydrograph,
    FrictionSlope,
    GateBoundary,
    GateOpening,
    InitialFlowLoc,
    InitialRainfallRunoffElev,
    InitialStorageElev,
    LateralInflow,
    NormalDepth,
    RatingCurve,
    StageHydrograph,
    UnsteadyFlow,
)

__all__ = [
    # Entry points
    "Project",
    "Proj",
    "Geometry",
    "Plan",
    "SteadyFlow",
    "UnsteadyFlow",
    # Enum
    "NodeType",
    # Unsteady-flow boundary classes (user-constructed)
    "FlowHydrograph",
    "LateralInflow",
    "StageHydrograph",
    "RatingCurve",
    "FrictionSlope",
    "NormalDepth",
    "GateBoundary",
    "GateOpening",
    "InitialFlowLoc",
    "InitialStorageElev",
    "InitialRainfallRunoffElev",
    # Steady-flow boundary classes
    "SteadyBoundary",
]

logger = logging.getLogger("rivia.model")

EXT_BACKUP_FILE = "rivia_bkup"


@dataclasses.dataclass
class PlanSummary:
    """Lightweight summary of one plan in a project.

    Returned by :attr:`Project.plans`.

    Attributes
    ----------
    index:
        Zero-based position in the project's plan list.
    title:
        ``Plan Title=`` string from the plan file, or ``None``.
    short_id:
        ``Short Identifier=`` string from the plan file, or ``None``.
    path:
        Full path to the plan file.
    active:
        ``True`` for the plan currently loaded by HEC-RAS.
    flow_type:
        ``'steady'``, ``'unsteady'``, or ``'quasi_steady'`` based on the
        ``Flow File=`` extension; ``None`` if the key is absent or unrecognised.
    sediment_path:
        Full path to the sediment file, or ``None`` if the plan has no
        ``Sediment File=`` entry.
    water_quality_path:
        Full path to the water quality file, or ``None`` if the plan has no
        ``Water Quality File=`` entry.
    """

    index: int
    title: str | None
    short_id: str | None
    path: Path
    active: bool
    flow_type: Literal["steady", "unsteady", "quasi_steady"] | None
    sediment_path: Path | None
    water_quality_path: Path | None

    def __repr__(self) -> str:
        active = "*" if self.active else ""
        parts = [
            f"PlanSummary({active}{self.index}: {self.title!r}",
            f"short_id={self.short_id!r}",
            f"flow_type={self.flow_type!r}",
            f"path={self.path.name!r}",
        ]
        if self.sediment_path is not None:
            parts.append(f"sediment={self.sediment_path.name!r}")
        if self.water_quality_path is not None:
            parts.append(f"water_quality={self.water_quality_path.name!r}")
        return ", ".join(parts) + ")"



class Project(MapperExtension):
    """High-level interface for working with an HEC-RAS project via the COM object.

    Use this class in preference to `com.open`. While `com.open` returns a raw HEC-RAS
    controller instance that is not associated with any project, `Project` binds the COM
    object to a specific HEC-RAS project file and provides project-aware operations.
    """

    def __init__(
        self,
        project_file: str | Path,
        ras_version: str | int | None = None,
        backup: bool = False,
    ):
        self._project_path = Path(project_file)
        self._backup = backup

        # restore if there are any backup files
        model_files = _get_project_files(project_file)
        _restore_backups(model_files)

        if ras_version is None:
            ras_version = _get_ras_version_from_project_file(project_file)

        if backup:
            _create_backups(model_files)
            # Bypassing __del__ which is unreliable during interpreter teardown
            atexit.register(_restore_backups, model_files)

        self._controller = controller.connect(ras_version)
        self._ras_version = self._controller.ras_version()
        self._controller.Project_Open(str(self._project_path))
        self._plan: Plan | None = None
        self._project: Proj | None = None
        self._geometry: Geometry | None = None
        self._flow: SteadyFlow | UnsteadyFlow | None = None
        self._hdf = None
        self._dss: DssReader | None = None
        self._run_history: collections.deque[dict] = collections.deque(maxlen=5)
        self._chaining = _ChainingState()
        self._plans_file_cache: list[PlanSummary] | None = None

    @property
    def version(self) -> int:
        return self._ras_version

    @property
    def controller(self):
        return self._controller

    @property
    def path(self) -> Path:
        """Return the project file path."""
        return self._project_path

    @property
    def geometry_path(self) -> Path:
        """Return the current geometry file path."""
        return Path(self.controller.CurrentGeomFile())

    @property
    def geometry_hdf_path(self) -> Path:
        """Return the current geometry HDF file path."""
        return self.geometry_path.with_name(self.geometry_path.name + ".hdf")

    @property
    def plan_path(self) -> Path:
        """Return the current plan file path."""
        return Path(self.controller.CurrentPlanFile())

    @property
    def plan_hdf_path(self) -> Path:
        """Return the current plan HDF file path."""
        return self.plan_path.with_name(self.plan_path.name + ".hdf")

    @property
    def flow_path(self) -> Path:
        """Return the current flow file path.

        Raises
        ------
        ValueError
            If the plan file has no ``Flow File=`` entry or the entry is blank.
        """
        plan_file = self.plan_path
        with open(plan_file) as fid:
            for line in fid:
                if line.startswith("Flow File"):
                    ext = line.split("=")[1].strip()
                    if ext:
                        return plan_file.with_suffix(f".{ext}")
        raise ValueError(
            f"Plan file {plan_file.name!r} has no 'Flow File=' entry."
        )

    @property
    def dss_path(self) -> Path:
        """Path to the DSS output file (project file with ``.dss`` extension)."""
        return self._project_path.with_suffix(".dss")

    @property
    def description(self) -> str:
        return self.project.description

    @property
    def project(self) -> Proj:
        """Lazily parsed project file.

        """
        if self._project is None:
            self._project = Proj(self.path)
        return self._project

    @property
    def plan(self) -> Plan:
        """Lazily parsed plan file.

        Cached after first access.  Call ``plan.save()`` then ``reload()`` to
        write changes back to disk and refresh the cache.
        """
        if self._plan is None:
            self._plan = Plan(self.plan_path)
        return self._plan

    @property
    def geometry(self) -> Geometry:
        """Lazily parsed geometry file for the current plan.

        Cached after first access.  Call ``geom.save()`` then ``reload()`` to
        write changes back to disk and refresh the cache.
        """
        if self._geometry is None:
            self._geometry = Geometry(self.geometry_path)
        return self._geometry

    @property
    def flow(self) -> SteadyFlow | UnsteadyFlow:
        """Lazily parsed flow file for the current plan.

        Cached after first access.  Call ``flow.save()`` then ``reload()`` to
        write changes back to disk and refresh the cache.

        Returns a :class:`SteadyFlow` when the plan references a steady
        flow file (extension starting with ``f``), or an
        :class:`UnsteadyFlow` when it references an unsteady flow file
        (extension starting with ``u``).

        Raises
        ------
        ValueError
            If the plan has no ``Flow File=`` entry, or the extension is not a
            recognised steady (``f*``) or unsteady (``u*``) type.
        FileNotFoundError
            If the flow file path does not exist on disk.
        """
        if self._flow is None:
            path = self.flow_path
            if path is None:
                raise ValueError(
                    f"Plan file {self.plan_path.name!r} has no 'Flow File=' entry."
                )
            if self.plan.is_steady:
                self._flow = SteadyFlow(path)
            elif self.plan.is_unsteady:
                self._flow = UnsteadyFlow(path)
            else:
                ext = self.plan.flow_file
                raise ValueError(
                    f"Unrecognised flow file extension {ext!r}; "
                    "expected an 'f*' (steady) or 'u*' (unsteady) extension."
                )
        return self._flow

    @property
    def results(self) -> SteadyPlan | UnsteadyPlan:
        """Lazily opened HDF results file for the current plan.

        Dispatches to the appropriate class based on plan type:

        * Steady flow (``plan.is_steady``) → :class:`~rivia.hdf.SteadyPlan`
        * Unsteady flow (``plan.is_unsteady``) → :class:`~rivia.hdf.UnsteadyPlan`

        The handle is kept open until :meth:`reload` is called or :meth:`close`
        is invoked.  For geometry-only access (no results), use
        ``hdf.Geometry(model.geometry_hdf_path)`` directly.

        Raises
        ------
        FileNotFoundError
            If the plan HDF does not exist — run the model first, or use
            ``hdf.Geometry(model.geometry_hdf_path)`` for geometry-only access.
        ValueError
            If the plan type cannot be determined from the flow file extension.
        """
        if self._hdf is None:
            plan_path = self.plan_hdf_path
            if not plan_path.exists():
                raise FileNotFoundError(
                    f"Plan HDF {plan_path.name!r} does not exist. "
                    "Run the model first with model.run(), or use "
                    "hdf.Geometry(model.geometry_hdf_path) for geometry-only access."
                )
            if self.plan.is_steady:
                self._hdf = SteadyPlan(plan_path)
            elif self.plan.is_unsteady:
                self._hdf = UnsteadyPlan(plan_path)
            else:
                raise ValueError(
                    f"Cannot determine plan type from flow file "
                    f"{self.plan.flow_file!r}."
                )
        return self._hdf

    @property
    def dss(self) -> DssReader:
        """Lazily created :class:`DssReader` for this plan's DSS output file.

        Provides time-series access for cross-sections and inline structures::

            flow  = model.dss.flow("Canal 1", "Pool 1-4", "7")
            hw    = model.dss.stage_hw("Canal 1", "Pool 1-4", "6.9")
            gate1 = model.dss.gate_opening(1, "Canal 1", "Pool 1-4", "6.9")
        """
        if self._dss is None:
            self._dss = DssReader(self)
        return self._dss

    def plans(self, invalidate_cache: bool = False) -> list[PlanSummary]:
        """All plans in the project with index, title, short_id, path, and active flag.

        Plan file data (title, short_id, flow_type, sediment_path,
        water_quality_path) is read once and cached.  The ``active`` flag is
        recomputed on every call without re-reading files.

        Parameters
        ----------
        invalidate_cache:
            If ``True``, discard the cached plan file data and re-read all
            plan files from disk.  Use this after editing plan files outside
            of a :meth:`editing` block or after other external changes.

        Example::

            for p in model.plans():
                active = "*" if p.active else " "
                print(f"[{active}] {p.index}: {p.title} ({p.short_id})")
        """
        if invalidate_cache or self._plans_file_cache is None:
            self._plans_file_cache = self._build_plans_file_cache()
        active_name = self.plan_path.name
        plans = [
            dataclasses.replace(d, active=(d.path.name == active_name))
            for d in self._plans_file_cache
        ]
        return sorted(plans, key=lambda p: not p.active)

    def _build_plans_file_cache(self) -> list[PlanSummary]:
        """Read all plan files and build the file-data cache.

        ``active`` is stored as ``False`` throughout — it is a placeholder
        that :attr:`plans` overwrites via ``dataclasses.replace`` on each
        access.
        """
        proj_path = self._project_path
        cache = []
        for i, p in enumerate(self.project.plans):
            plan_path = p["path"]
            plan = Plan(plan_path)
            if plan.is_steady:
                flow_type: (
                    Literal["steady", "unsteady", "quasi_steady"] | None
                ) = "steady"
            elif plan.is_unsteady:
                flow_type = "unsteady"
            elif plan.is_quasi_steady:
                flow_type = "quasi_steady"
            else:
                flow_type = None
            sed_ext = plan.sediment_file
            wq_ext = plan.water_quality_file
            cache.append(PlanSummary(
                index=i,
                title=p["title"],
                short_id=p["short_id"],
                path=plan_path,
                active=False,
                flow_type=flow_type,
                sediment_path=(
                    proj_path.with_suffix(f".{sed_ext}") if sed_ext else None
                ),
                water_quality_path=(
                    proj_path.with_suffix(f".{wq_ext}") if wq_ext else None
                ),
            ))
        return cache

    def _resolve_plan_path(self, plan: "PlanSummary | int | str | None") -> Path:
        """Resolve *plan* (PlanSummary, index, short_id, or None) to a plan file Path.

        ``None`` resolves to the currently active plan via the COM controller.
        """
        if plan is None:
            return self.plan_path
        all_plans = self.plans()
        if isinstance(plan, PlanSummary):
            return plan.path
        if isinstance(plan, int):
            matches = [p for p in all_plans if p.index == plan]
            if not matches:
                raise IndexError(f"No plan with index {plan!r}")
            return matches[0].path
        # str → match by short_id
        matches = [p for p in all_plans if p.short_id == plan]
        if not matches:
            raise KeyError(f"No plan with short_id={plan!r}")
        return matches[0].path

    def plan_staleness(
        self, plan: "PlanSummary | int | str | None" = None
    ) -> "PlanStalenessReport":
        """Return a detailed staleness diagnostic report for one plan.

        Inspects the plan text file, plan HDF, and geometry HDF to report
        whether geometry layers are stale, whether the geometry was
        re-preprocessed since the plan was run, and whether the simulation
        results appear complete.

        No active HEC-RAS simulation is required — the check is purely
        file-based.

        Parameters
        ----------
        plan:
            Identifies the plan to inspect.  Accepts a
            :class:`PlanSummary` (from :meth:`plans`), an integer index,
            a ``short_id`` string, or ``None`` (default).  When ``None``,
            the currently active plan (as reported by the COM controller)
            is used.

        Returns
        -------
        PlanStalenessReport

        Raises
        ------
        IndexError
            If *plan* is an integer that does not match any plan index.
        KeyError
            If *plan* is a string that does not match any ``short_id``.

        Examples
        --------
        ::

            # active plan
            report = model.plan_staleness()
            print(report)
            if not report.run_appears_complete:
                print("Results may be from a prior or incomplete run.")

            # specific plan by index
            report = model.plan_staleness(2)
        """
        from rivia.hdf.staleness import check_plan_staleness

        path = self._resolve_plan_path(plan)
        return check_plan_staleness(path)

    @contextlib.contextmanager
    def editing(self):
        """Context manager for batch file edits with automatic save and reload.

        All modifications made inside the block are saved and the project is
        reloaded on exit::

            with model.editing():
                model.plan.simulation_window = (start, end)
                model.plan.computation_interval = "30SEC"
            # files saved, project reloaded
        """
        yield self
        self.reload(save_if_modified=True)

    @contextlib.contextmanager
    def chaining(self, cleanup: bool = False):
        """Context manager that enables run-chaining for the duration of the block.

        Chaining is automatically disabled (and optionally restart files
        deleted) when the block exits, even if an exception is raised::

            with model.chaining():
                for window in windows:
                    with model.editing():
                        model.plan.simulation_window = window
                    model.run()

        Parameters
        ----------
        cleanup:
            If ``True``, delete all restart files after chaining ends.
        """
        self.enable_chaining(True)
        try:
            yield self
        finally:
            self.enable_chaining(False)
            if cleanup:
                self.delete_restart_files()

    @property
    def plan_index(self) -> int:
        for i, plan_info in enumerate(self.project.plans):
            if plan_info["path"].name == self.plan_path.name:
                return i

    def set_plan(
        self,
        *,
        index: int | None = None,
        title: str | None = None,
        short_id: str | None = None,
    ) -> None:
        """Set the active plan, then reload.

        Exactly one keyword argument must be supplied.

        Parameters
        ----------
        index:
            Zero-based position in the project's plan list.  No-op if
            already the current plan.
        title:
            ``Plan Title=`` string from the plan file.
        short_id:
            ``Short Identifier=`` string from the plan file.

        Raises
        ------
        ValueError
            If not exactly one argument is given, the requested plan is not
            found, or the COM plan list does not match the project file.
        RuntimeError
            If HEC-RAS fails to switch the active plan.
        """
        given = sum(x is not None for x in (index, title, short_id))
        if given != 1:
            raise ValueError(
                "Exactly one of index, title, or short_id must be provided "
                f"({given} given)."
            )

        plans = self.project.plans

        # --- validate COM plan list matches project file ---
        _, com_titles = self.controller.Plan_Names(IncludeOnlyPlansInBaseDirectory=True)
        if len(com_titles) != len(plans):
            logger.warning(
                "COM reports %d plans but project file lists %d.",
                len(com_titles),
                len(plans),
            )
        else:
            mismatches = [
                (i, com_titles[i], plans[i]["title"])
                for i in range(len(plans))
                if com_titles[i] != plans[i]["title"]
            ]
            if mismatches:
                logger.warning(
                    "Plan title mismatches between COM and project file: %s",
                    [(i, com, prj) for i, com, prj in mismatches],
                )

        # --- resolve target plan entry ---
        if index is not None:
            if index < 0 or index >= len(plans):
                raise ValueError(
                    f"Plan index {index} out of range; "
                    f"valid range is 0 to {len(plans) - 1}."
                )
            if index == self.plan_index:
                logger.debug("Plan index %d is already active — no-op.", index)
                return
            target = plans[index]
        elif title is not None:
            matches = [p for p in plans if p["title"] == title]
            if not matches:
                raise ValueError(
                    f"No plan with title {title!r}; "
                    f"available titles: {[p['title'] for p in plans]}"
                )
            target = matches[0]
        else:  # short_id
            matches = [p for p in plans if p["short_id"] == short_id]
            if not matches:
                raise ValueError(
                    f"No plan with short_id {short_id!r}; "
                    f"available short_ids: {[p['short_id'] for p in plans]}"
                )
            target = matches[0]

        plan_title = target["title"]
        if plan_title is None:
            raise ValueError(
                f"Plan file {target['path'].name} has no Plan Title — "
                "cannot pass to HEC-RAS COM."
            )

        logger.debug("Setting current plan to %r", plan_title)
        success = self.controller.Plan_SetCurrent(plan_title)
        if not success:
            raise RuntimeError(
                f"HEC-RAS failed to set current plan to {plan_title!r}."
            )
        self.controller.Project_Save()
        self.reload()

    def reset(self):
        if not self._backup:
            raise ValueError(
                "Project instance does not have back files to perform reset."
            )
        model_files = _get_project_files(self._project_path)
        _restore_backups(model_files)
        self.reload(save_if_modified=False)

    def reload(self, save_if_modified: bool = True):
        if save_if_modified:
            if self._plan is not None and self._plan.is_modified:
                self._plan.save()
            if self._geometry is not None and self._geometry.is_modified:
                self._geometry.save()
            if (
                self._flow is not None
                and hasattr(self._flow, "is_modified")
                and self._flow.is_modified
            ):
                self._flow.save()
        self._plan = None  # invalidate cached Plan so next access re-parses
        self._geometry = None
        self._flow = None
        self._dss = None
        if self._hdf is not None:
            self._hdf.close()
            self._hdf = None
        # v503+: Project_Close + Project_Open reloads without restarting COM.
        # Older versions: restart the COM process entirely.
        try:
            self.controller.Project_Close()
        except NotImplementedError:
            self.controller.close()
            self._controller = controller.connect(self._ras_version)
        finally:
            self.controller.Project_Open(str(self._project_path))

    def show(self):
        self.controller.show()

    def hide(self):
        self.controller.hide()

    def run(
        self, blocking: bool = True, hide_window: bool = False, reload: bool = False
    ) -> tuple[bool, tuple[str, ...]]:
        """Run HEC-RAS computations for the current plan.

        Parameters
        ----------
        blocking : bool, optional
            If ``True`` (default), block until computations complete.
            If ``False``, return immediately while HEC-RAS computes in
            the background.  Not supported in HEC-RAS 4.x (always blocking).
        hide_window : bool, optional
            If ``True``, hide the computation window before running and
            restore it afterward. Default is ``False``.
        reload : bool, optional
            If ``True``, call :meth:`reload` before running to refresh the
            project from disk. Default is ``False``.

        Returns
        -------
        success : bool
            ``True`` if the computation completed successfully.
        messages : tuple[str, ...]
            Messages returned by HEC-RAS. Empty for versions below 5.0.3.

        Raises
        ------
        ~rivia.controller.HecRasComputeError
            If HEC-RAS reports a computation failure or a COM error occurs.
        """
        if reload:
            self.reload(True)
        if self._hdf is not None:
            self._hdf.close()
            self._hdf = None
        if self._plan is not None and self._plan.is_modified:
            logger.warning("Plan file %s is modified but not saved.", self.plan_path.name)
        if self._geometry is not None and self._geometry.is_modified:
            logger.warning("Geometry file %s is modified but not saved.", self.geometry_path.name)
        if (
            self._flow is not None
            and hasattr(self._flow, "is_modified")
            and self._flow.is_modified
        ):
            logger.warning("Flow file %s is modified but not saved.", self.flow_path.name)
        if self._chaining.enabled:
            if self._plan is not None and self._plan.is_modified:
                raise RuntimeError(
                    f"Run-chaining is active (run #{self._chaining.run_number}) "
                    f"but plan file {self.plan_path.name!r} has unsaved modifications. "
                    "Save or discard changes before calling run()."
                )
            if (
                self._flow is not None
                and hasattr(self._flow, "is_modified")
                and self._flow.is_modified
            ):
                raise RuntimeError(
                    f"Run-chaining is active (run #{self._chaining.run_number}) "
                    f"but flow file {self.flow_path.name!r} has unsaved modifications. "
                    "Save or discard changes before calling run()."
                )
        if self._chaining.enabled and self._chaining.run_number > 0:
            rst_flag, _ = self.flow.restart
            if rst_flag != 1:
                raise RuntimeError(
                    f"Run-chaining is active (run #{self._chaining.run_number}) "
                    "but the restart flag is not enabled in the flow file. "
                    "This should have been set automatically; check that the "
                    "flow file was not replaced or reloaded after chaining was enabled."
                )
            rst_path = self.plan_path.parent / self._chaining_ic_filename(end=False)
            if not rst_path.exists():
                raise FileNotFoundError(
                    f"Chained run #{self._chaining.run_number} expects restart file "
                    f"{rst_path.name!r} but it does not exist in {rst_path.parent}."
                )
        if self.plan.is_unsteady:
            rst_flag, rst_filename = self.flow.restart
            if rst_flag == 1 and rst_filename is not None:
                window = self.plan.simulation_window
                if window is not None:
                    m = re.search(
                        r"\.(\d{2}[A-Za-z]{3}\d{4}) (\d{4,6})\.rst$",
                        rst_filename,
                    )
                    if m is None:
                        logger.warning(
                            "Cannot validate restart file datetime: filename %r "
                            "does not match the expected "
                            "'<plan>.<date> <time>.rst' pattern.",
                            rst_filename,
                        )
                    else:
                        rst_date, rst_time = normalize_sim_start_time(
                            m.group(1), m.group(2)
                        )
                        (sim_date, sim_time), _ = window
                        sim_date, sim_time = normalize_sim_start_time(
                            sim_date, sim_time
                        )
                        if (rst_date, rst_time) != (sim_date, sim_time):
                            logger.warning(
                                "Restart file datetime (%s %s) does not match "
                                "simulation start (%s %s) in plan %r — verify "
                                "the correct restart file is configured in the "
                                "flow file.",
                                rst_date, rst_time, sim_date, sim_time,
                                self.plan_path.name,
                            )
        if hide_window:
            self.controller.Compute_HideComputationWindow()
        try:
            result = self.controller.Compute_CurrentPlan(BlockingMode=blocking)
        except Exception:
            if hide_window:
                self.controller.Compute_ShowComputationWindow()
            raise
        finally:
            _ts = dt.datetime.now().isoformat(timespec="seconds")
            _entry: dict = {
                "plan": self.plan_hdf_path.name,
                "sim_window": None,
                "timestamp": _ts,
                "summary": None,
            }
            try:
                _entry["sim_window"] = self.plan.simulation_window
            except Exception as exc:
                logger.debug("Could not capture simulation window: %s", exc)
            try:
                _entry["summary"] = self.results.compute_summary().to_dict()
            except Exception as exc:
                logger.debug("Could not capture run summary: %s", exc)
            self._run_history.append(_entry)
        success, messages = result
        logger.debug("Compute_CurrentPlan: success=%s, messages=%s", success, messages)
        if self._chaining.enabled:
            rst_name = self._chaining_ic_filename(end=True)
            self.flow.restart = rst_name
            self._flow.save()
            self._chaining.run_number += 1
            logger.debug(
                "Chaining: restart set to %r; run_number now %d.",
                rst_name,
                self._chaining.run_number,
            )
        return success, messages

    @property
    def run_history(self) -> list[dict]:
        """Last up to 5 run summaries, oldest first.

        Each entry is a dict with keys:

        - ``"plan"``: plan HDF filename (e.g. ``"MyModel.p01.hdf"``)
        - ``"timestamp"``: ISO-8601 wall-clock time the run completed
          (e.g. ``"2026-04-02T09:27:24"``)
        - ``"summary"``: :meth:`~rivia.hdf.UnsteadyPlan.compute_summary`
          output as a dict, or ``None`` if the summary could not be read
          (e.g. run failed before writing HDF output, or steady-flow plan).

        The container holds at most 5 entries; the oldest is evicted when
        a sixth run completes.
        """
        return list(self._run_history)

    def enable_chaining(self, enabled: bool) -> None:
        """Enable or disable run-chaining for sequential simulations.

        When chaining is enabled the model automatically writes a restart file
        at the end of each run (via ``plan.write_ic_at_end``) and configures
        the flow file to use that file as the initial condition for the next
        run.

        Parameters
        ----------
        enabled:
            ``True`` to activate chaining; ``False`` to deactivate and reset
            all chaining state.

        Raises
        ------
        RuntimeError
            If the current plan is not an unsteady-flow plan.

        Notes
        -----
        Calling ``enable_chaining(True)`` multiple times is safe — the run
        counter is not reset on repeated calls, so chaining can be
        re-configured mid-sequence without losing track of which run is next.
        """
        if enabled:
            if not self.plan.is_unsteady:
                raise RuntimeError(
                    "Run-chaining requires an unsteady-flow plan; "
                    f"current plan {self.plan_path.name!r} is not unsteady."
                )
            self._chaining.enabled = True
            if not self.plan.write_ic_at_end:
                self.plan.write_ic_at_end = True
                self.plan.save()
                logger.debug("write_ic_at_end enabled and plan saved.")
            if self._chaining.run_number is None:
                self._chaining.run_number = 0
                logger.debug("Chaining initialised; run_number=0.")
            else:
                logger.debug(
                    "Chaining re-enabled; run_number preserved at %d.",
                    self._chaining.run_number,
                )
        else:
            self._chaining.enabled = False
            self._chaining.run_number = None
            logger.debug("Chaining disabled and state reset.")

    def _chaining_ic_filename(self, end: bool = True) -> str:
        """Return the IC restart filename derived from the simulation window.

        The filename follows HEC-RAS convention::

            <plan_file_name>.<date> <time>.rst

        Parameters
        ----------
        end:
            If ``True`` (default), the filename is based on the simulation
            *end* datetime — used to name the restart file written by the
            current run.  Midnight is expressed as ``"2400"`` on the ending
            day via :func:`~rivia.utils.normalize_sim_end_time`.

            If ``False``, the filename is based on the simulation *start*
            datetime — useful for verifying which restart file the next run
            expects.  Midnight is expressed as ``"0000"`` on the starting day
            via :func:`~rivia.utils.normalize_sim_start_time`.

        Returns
        -------
        str
            e.g. ``"MyModel.p01.01JAN2026 2400.rst"`` (end=True)
            or   ``"MyModel.p01.02JAN2026 0000.rst"`` (end=False, same instant)

        Raises
        ------
        RuntimeError
            If the simulation window has not been set on the plan file.
        """
        window = self.plan.simulation_window
        if window is None:
            raise RuntimeError(
                f"Plan {self.plan_path.name!r} has no simulation window set; "
                "cannot determine IC filename."
            )
        if end:
            (_, _), (date, time) = window
            date, time = normalize_sim_end_time(date, time)
        else:
            (date, time), (_, _) = window
            date, time = normalize_sim_end_time(date, time)
        return f"{self.plan_path.name}.{date} {time}.rst"

    def delete_restart_files(self) -> list[Path]:
        """Delete all restart files associated with the current plan.

        Restart files follow the pattern ``<plan_file>.<datestamp>.rst``
        where the plan file extension is ``.p<digits>`` (e.g. ``.p01``).

        Returns
        -------
        list[Path]
            Paths of the files that were deleted. Empty if none were found.

        Raises
        ------
        ValueError
            If the current plan file does not have the expected ``.p<digits>``
            extension.
        """
        plan_file = self.plan_path
        if not re.fullmatch(r"\.p\d+", plan_file.suffix):
            raise ValueError(
                f"Unexpected plan file extension {plan_file.suffix!r}; "
                "expected .p<digits> (e.g. .p01)."
            )
        pattern = re.compile(re.escape(plan_file.name) + r"\..+\.rst$")
        deleted = []
        for path in plan_file.parent.iterdir():
            if pattern.fullmatch(path.name):
                path.unlink()
                logger.debug("Deleted restart file: %s", path.name)
                deleted.append(path)
        return deleted

    # ------------------------------------------------------------------
    # Resource lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the COM connection and release all cached file handles.

        Safe to call multiple times.  Called automatically on ``__exit__``
        when used as a context manager.
        """
        with contextlib.suppress(Exception):
            if self._hdf is not None:
                self._hdf.close()
                self._hdf = None
        with contextlib.suppress(Exception):
            self.controller.close()

    def __enter__(self) -> "Project":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def __repr__(self) -> str:
        return (
            f"Project({self.path.name!r}, plan={self.plan_path.name!r})"
        )

    def __del__(self):
        with contextlib.suppress(Exception):
            logger.debug("Executing Project destructor.")
        self.close()


@dataclasses.dataclass
class _ChainingState:
    """Private state that tracks run-chaining for :class:`Project`.

    Run-chaining is the pattern where a sequence of short HEC-RAS simulations
    are linked end-to-end: each run writes a restart (``.rst``) file at its
    simulation end time, and the following run reads that file as its initial
    condition.  This allows long simulations to be broken into smaller windows
    while preserving hydraulic continuity across the boundaries.

    This dataclass is stored as ``Project._chaining`` and is only mutated by
    :meth:`~Project.enable_chaining` and :meth:`~Project.run`.  All other
    ``Project`` methods treat it as read-only.

    Attributes
    ----------
    enabled : bool
        ``True`` while chaining is active.  Set to ``True`` by
        ``enable_chaining(True)`` and back to ``False`` by
        ``enable_chaining(False)``.  When ``False`` no chaining logic runs
        inside :meth:`~Project.run`.
    run_number : int or None
        Tracks how many chained runs have completed in the current sequence.

        - ``None`` — chaining has never been enabled on this ``Project``
          instance, or has been fully reset by ``enable_chaining(False)``.
        - ``0`` — the seed run: no restart file exists yet, so only the
          unsaved-modifications guard is active; the restart-file existence
          check is skipped.
        - ``≥ 1`` — a subsequent chained run: :meth:`~Project.run` verifies
          that the restart file produced by the previous run exists on disk
          and that the flow file has the restart flag enabled before
          launching HEC-RAS.

        The counter is initialised to ``0`` on the first ``enable_chaining(True)``
        call and is never reset by subsequent ``enable_chaining(True)`` calls,
        so chaining configuration can be adjusted mid-sequence without losing
        track of the run position.  It is reset to ``None`` only by
        ``enable_chaining(False)``.
    """

    enabled: bool = False
    run_number: int | None = None


def _get_ras_version_from_project_file(project_file: str | Path):
    path = Path(project_file)
    if not path.is_file():
        raise OSError(f"HEC-RAS Project not found: {project_file}")
    plan_file = None
    with open(project_file) as fid:
        for line in fid:
            if line.startswith("Current Plan"):
                ext = line.split("=")[1].strip()
                plan_file = path.parent / f"{path.stem}.{ext}"
    if plan_file is None:
        raise RuntimeError(
            f"The HEC-RAS project file does not have current plan specified: "
            f"{project_file}"
        )
    with open(plan_file) as fid:
        for line in fid:
            if line.startswith("Program Version"):
                return line.split("=")[1].strip()
    raise OSError(f"HEC-RAS version info not found in current plan: {plan_file}")


def _get_project_files(project_file: str | Path) -> list[Path]:
    path = Path(project_file)
    if not path.is_file():
        raise OSError(f"HEC-RAS Project not found: {project_file}")
    keys = ("Geom File", "Plan File", "Unsteady File", "Steady File")
    files = []
    with open(project_file) as fid:
        for line in fid:
            if line.startswith(keys):
                ext = line.split("=")[1].strip()
                files.append(path.parent / f"{path.stem}.{ext}")
    return files


def _create_backups(project_files: list[Path]) -> None:
    for src in project_files:
        dst = src.with_suffix(f"{src.suffix}.{EXT_BACKUP_FILE}")
        tmp = dst.with_suffix(".tmp")
        shutil.copyfile(src, tmp)
        tmp.replace(dst)


def _restore_backups(project_files: list[Path]) -> None:
    for dst in project_files:
        src = dst.with_suffix(f"{dst.suffix}.{EXT_BACKUP_FILE}")
        if src.exists():
            src.replace(dst)

