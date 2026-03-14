"""Read/write HEC-RAS text input files (.prj, .g*, .f*, etc.)."""

import atexit
import logging
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

__all__ = [
    "Model",
    "PlanFile",
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
        self._compute_blocking = 1
        self._plan: PlanFile | None = None
        self._hdf = None

    @property
    def version(self):
        return self._ras_version

    @property
    def project_file(self) -> Path:
        """Return the project file path."""
        return self._project_path

    @property
    def geom_file(self) -> Path:
        """Return the plan file path."""
        return Path(self._rc.CurrentGeomFile())

    @property
    def geom_hdf_file(self) -> Path:
        """Return the plan file path."""
        return self.geom_file.with_name(self.geom_file.name + ".hdf")

    @property
    def plan_file(self) -> Path:
        """Return the plan file path."""
        return Path(self._rc.CurrentPlanFile())

    @property
    def plan_hdf_file(self) -> Path:
        """Return the plan file path."""
        return self.plan_file.with_name(self.plan_file.name + ".hdf")

    @property
    def flow_file(self) -> Path:
        """Return the flow file path."""
        plan_file = Path(self._rc.CurrentPlanFile())
        with open(plan_file) as fid:
            for line in fid:
                if line.startswith("Flow File"):
                    ext = line.split("=")[1].strip()
                    if ext:
                        return plan_file.with_suffix(f".{ext}")

    @property
    def plan(self) -> PlanFile:
        """Lazily parsed plan file.

        Call ``plan.save()`` then ``reload()`` to activate changes.
        """
        if self._plan is None:
            self._plan = PlanFile(self.plan_file)
        return self._plan

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

    def reset(self):
        if not self._backup:
            raise ValueError(
                "Model instance does not have back files to perform reset."
            )
        model_files = _get_project_files(self._project_path)
        _restore_backups(model_files)
        self.reload()

    def change_plan(self, plan: str | int) -> None:
        """Set the active plan by name or zero-based index, then reload.

        Parameters
        ----------
        plan:
            Plan title (str) or zero-based index into the project's plan list (int).
        """
        _, plan_names = self._rc.Plan_Names(IncludeOnlyPlansInBaseDirectory=True)
        if isinstance(plan, int):
            if plan < 0 or plan >= len(plan_names):
                raise IndexError(
                    f"Plan index {plan} out of range; "
                    f"available range is 0 to {len(plan_names) - 1}"
                )
            plan_name = plan_names[plan]
        else:
            if plan not in plan_names:
                raise ValueError(
                    f"Plan '{plan}' not found; available plans: {plan_names}"
                )
            plan_name = plan
        success = self._rc.Plan_SetCurrent(plan_name)
        if not success:
            raise RuntimeError(f"HEC-RAS failed to set current plan to '{plan_name}'")
        self.reload()

    def reload(self):
        self._plan = None  # invalidate cached PlanFile so next access re-parses
        if self._hdf is not None:
            self._hdf.close()
            self._hdf = None
        # v503+: Project_Close + Project_Open reloads without restarting COM.
        # Older versions: restart the COM process entirely.
        if self._ras_version >= 5030:
            self._rc.Project_Close()
            self._rc.Project_Open(str(self._project_path))
        else:
            self._rc.close()
            self._rc = com.open(self._ras_version)
            self._rc.Project_Open(str(self._project_path))

    def show(self):
        self._rc.show()

    def hide(self):
        self._rc.hide()

    def show_compute(self, flag: bool):
        if flag:
            self._rc.Compute_ShowComputationWindow()
        else:
            self._rc.Compute_HideComputationWindow()

    @property
    def compute_blocking(self) -> bool:
        """Return ``True`` if compute runs are blocking (synchronous)."""
        return bool(self._compute_blocking)

    @compute_blocking.setter
    def compute_blocking(self, flag: bool) -> None:
        self._compute_blocking = 1 if flag else 0

    def __del__(self):
        import contextlib
        with contextlib.suppress(Exception):
            logger.debug("Executing Model destructor.")
        with contextlib.suppress(Exception):
            self._rc.close()


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

