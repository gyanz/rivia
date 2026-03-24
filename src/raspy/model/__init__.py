"""High-level HEC-RAS project interface and text file I/O.

:class:`Model` is the primary entry point — it opens a project via COM,
manages plan switching and reloading, and provides lazy access to plan files,
geometry files, flow files, and HDF results.

Individual file classes (:class:`GeometryFile`, :class:`PlanFile`, etc.) can
also be used standalone without a :class:`Model` instance.
"""

import atexit
import contextlib
import logging
import re
import shutil
from pathlib import Path

from .. import com
from ._mapper import MapperExtension
from .flow_steady import SteadyBoundary, SteadyFlowFile
from .flow_unsteady import (
    FlowHydrograph,
    FrictionSlope,
    GateBoundary,
    GateOpening,
    InitialFlowLoc,
    InitialRRRElev,
    InitialStorageElev,
    LateralInflow,
    NormalDepth,
    RatingCurve,
    StageHydrograph,
    UnsteadyFlowEditor,
    UnsteadyFlowFile,
)
from .geometry import (
    NODE_BRIDGE,
    NODE_CULVERT,
    NODE_INLINE_STRUCTURE,
    NODE_LATERAL_STRUCTURE,
    NODE_MULTIPLE_OPENING,
    NODE_XS,
    CrossSection,
    GeometryFile,
    IneffArea,
    ManningEntry,
)
from .plan import PlanFile
from .project import ProjectFile

__all__ = [
    "Model",
    "PlanFile",
    "ProjectFile",
    "GeometryFile",
    "CrossSection",
    "ManningEntry",
    "IneffArea",
    "NODE_XS",
    "NODE_CULVERT",
    "NODE_BRIDGE",
    "NODE_MULTIPLE_OPENING",
    "NODE_INLINE_STRUCTURE",
    "NODE_LATERAL_STRUCTURE",
    "UnsteadyFlowFile",
    "UnsteadyFlowEditor",
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
    "InitialRRRElev",
    "SteadyFlowFile",
    "SteadyBoundary",
]

logger = logging.getLogger("raspy.model")

EXT_BACKUP_FILE = "raspy_bkup"


class Model(MapperExtension):
    """High-level interface for working with an HEC-RAS project via the COM object.

    Use this class in preference to `com.open`. While `com.open` returns a raw HEC-RAS
    controller instance that is not associated with any project, `Model` binds the COM
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

        self._rc = com.open(ras_version)
        self._ras_version = self._rc.ras_version()
        self._rc.Project_Open(str(self._project_path))
        self._plan: PlanFile | None = None
        self._project: ProjectFile | None = None
        self._flow: SteadyFlowFile | UnsteadyFlowEditor | None = None
        self._hdf = None

    @property
    def version(self) -> int:
        return self._ras_version

    @property
    def controller(self):
        return self._rc

    @property
    def project_file(self) -> Path:
        """Return the project file path."""
        return self._project_path

    @property
    def geom_file(self) -> Path:
        """Return the current geometry file path."""
        return Path(self.controller.CurrentGeomFile())

    @property
    def geom_hdf_file(self) -> Path:
        """Return the current geometry HDF file path."""
        return self.geom_file.with_name(self.geom_file.name + ".hdf")

    @property
    def plan_file(self) -> Path:
        """Return the current plan file path."""
        return Path(self.controller.CurrentPlanFile())

    @property
    def plan_hdf_file(self) -> Path:
        """Return the current plan HDF file path."""
        return self.plan_file.with_name(self.plan_file.name + ".hdf")

    @property
    def flow_file(self) -> Path:
        """Return the current flow file path."""
        plan_file = self.plan_file
        with open(plan_file) as fid:
            for line in fid:
                if line.startswith("Flow File"):
                    ext = line.split("=")[1].strip()
                    if ext:
                        return plan_file.with_suffix(f".{ext}")

    @property
    def project(self) -> ProjectFile:
        """Lazily parsed project file.

        """
        if self._project is None:
            self._project = ProjectFile(self.project_file)
        return self._project

    @property
    def plan(self) -> PlanFile:
        """Lazily parsed plan file.

        Call ``plan.save()`` then ``reload()`` to activate changes.
        """
        if self._plan is None:
            self._plan = PlanFile(self.plan_file)
        return self._plan

    @property
    def flow(self) -> SteadyFlowFile | UnsteadyFlowEditor:
        """Lazily parsed flow file for the current plan.

        Returns a :class:`SteadyFlowFile` when the plan references a steady
        flow file (extension starting with ``f``), or an
        :class:`UnsteadyFlowEditor` when it references an unsteady flow file
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
            path = self.flow_file
            if path is None:
                raise ValueError(
                    f"Plan file {self.plan_file.name!r} has no 'Flow File=' entry."
                )
            if self.plan.is_steady:
                self._flow = SteadyFlowFile(path)
            elif self.plan.is_unsteady:
                self._flow = UnsteadyFlowEditor(path)
            else:
                ext = self.plan.flow_file
                raise ValueError(
                    f"Unrecognised flow file extension {ext!r}; "
                    "expected an 'f*' (steady) or 'u*' (unsteady) extension."
                )
        return self._flow

    @property
    def hdf(self):
        """Lazily opened plan HDF file as a :class:`raspy.hdf.PlanHdf` instance.

        The handle is kept open until :meth:`reload` is called (which closes
        and discards it) or until the ``PlanHdf`` object is closed directly.
        Use as a context manager for explicit lifetime control::

            with model.hdf as h:
                depth = h.flow_areas["spillway"].depth(10)

        Or keep it open across multiple calls::

            area = model.hdf.flow_areas["spillway"]
            wse  = area.water_surface[5]
            depth = area.depth(5)
        """
        from raspy.hdf import PlanHdf

        if self._hdf is None:
            self._hdf = PlanHdf(self.plan_hdf_file)
        return self._hdf

    @property
    def plan_index(self) -> int:
        for i, plan_info in enumerate(self.project.plans()):
            if plan_info["path"].name == self.plan_file.name:
                return i

    def change_plan(
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

        plans = self.project.plans()

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
                "Model instance does not have back files to perform reset."
            )
        model_files = _get_project_files(self._project_path)
        _restore_backups(model_files)
        self.reload()

    def reload(self):
        self._plan = None  # invalidate cached PlanFile so next access re-parses
        self._flow = None
        if self._hdf is not None:
            self._hdf.close()
            self._hdf = None
        # v503+: Project_Close + Project_Open reloads without restarting COM.
        # Older versions: restart the COM process entirely.
        try:
            self.controller.Project_Close()
        except NotImplementedError:
            self.controller.close()
            self._rc = com.open(self._ras_version)
        finally:
            self.controller.Project_Open(str(self._project_path))

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
        plan_file = self.plan_file
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

    def show(self):
        self.controller.show()

    def hide(self):
        self.controller.hide()

    def run(
        self, blocking: bool = True, hide_window: bool = False
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

        Returns
        -------
        success : bool
            ``True`` if the computation completed successfully.
        messages : tuple[str, ...]
            Messages returned by HEC-RAS. Empty for versions below 5.0.3.

        Raises
        ------
        HecRasComputeError
            If HEC-RAS reports a computation failure or a COM error occurs.
        """
        if hide_window:
            self.controller.Compute_HideComputationWindow()
        try:
            result = self.controller.Compute_CurrentPlan(BlockingMode=blocking)
        except Exception:
            if hide_window:
                self.controller.Compute_ShowComputationWindow()
            raise
        success, messages = result
        logger.debug("Compute_CurrentPlan: success=%s, messages=%s", success, messages)
        return success, messages

    def __del__(self):
        with contextlib.suppress(Exception):
            logger.debug("Executing Model destructor.")
        with contextlib.suppress(Exception):
            self.controller.close()


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

