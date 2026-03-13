"""Read/write HEC-RAS text input files (.prj, .g*, .f*, etc.)."""

import atexit
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal
from xml.sax.saxutils import escape

from .. import com
from ..com.ras import installed_ras_directory
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

EXT_BACKUP_FILE = "raspy_bkup"


class Model:
    """High-level interface for working with an HEC-RAS project via the wrapped COM object.

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
        model_files = Model._get_project_files(project_file)
        Model._restore_backups(model_files)

        if ras_version is None:
            ras_version = Model._get_ras_version_from_project_file(project_file)

        if backup:
            Model._create_backups(model_files)
            # Bypassing this in __del__ which is unreliable for system operation during interpretor teardown
            atexit.register(Model._restore_backups, model_files)

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
        model_files = Model._get_project_files(self._project_path)
        Model._restore_backups(model_files)
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

    def compute_blocking(self, flag: bool):
        if flag:
            self._compute_blocking = 1
        else:
            self._compute_blocking = 0

    def __del__(self):
        try:
            logging.debug("Executing Model destructor.")
        except Exception:
            pass
        try:
            self._rc.close()
        except Exception:
            pass

    @staticmethod
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
                f"The HEC-RAS project file does not have current plan specified: {project_file}"
            )

        with open(plan_file) as fid:
            for line in fid:
                if line.startswith("Program Version"):
                    version = line.split("=")[1].strip()
                    return version

        raise OSError(f"HEC-RAS version info not found in current plan: {plan_file}")

    @staticmethod
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

    @staticmethod
    def _create_backups(project_files: list[Path]) -> None:
        for src in project_files:
            dst = src.with_suffix(f"{src.suffix}.{EXT_BACKUP_FILE}")
            tmp = dst.with_suffix(".tmp")
            shutil.copyfile(src, tmp)
            tmp.replace(dst)

    @staticmethod
    def _restore_backups(project_files: list[Path]) -> None:
        for dst in project_files:
            src = dst.with_suffix(f"{dst.suffix}.{EXT_BACKUP_FILE}")
            if src.exists():
                src.replace(dst)

    def export_using_mapper( 
        self,
        variable: Literal["water_surface", "depth", "cell_speed"],
        output_path: str | Path,
        timestep: int | None = None,
    ):
        """Legacy export method with hardcoded parameters for quick raster export.

        See :meth:`export_raster` for the new flexible method with full
        parameterization.  This is retained for quick-and-dirty exports using
        the most common settings, but all parameters are fixed and it delegates
        to :meth:`export_raster` internally.

        Parameters
        ----------
        variable:
            ``"water_surface"``, ``"depth"``, ``"cell_speed"``, or
            ``"cell_velocity"``.
        timestep:
            0-based time index.  Pass ``None`` to use maximum values (only
            valid for ``"water_surface"`` and ``"depth"``).
        output_path:
            Destination ``.tif`` file path.  ``None`` returns an in-memory
            ``rasterio.DatasetReader``; the caller must close it.

        Returns
        -------
        Path
            Written GeoTIFF path (when *output_path* is given).
        rasterio.io.DatasetReader
            Open in-memory dataset (when *output_path* is ``None``).
        """
        return self.store_map(variable=variable, output_path=output_path, timestep=timestep)

    def _store_map_profile_name(self, timestep: int | None) -> str:
        if timestep is None:
            return "Max"
        if timestep < 0:
            raise ValueError("timestep must be >= 0 or None")

        ts = self.hdf.time_stamps
        if timestep >= len(ts):
            raise IndexError(
                f"timestep index {timestep} out of range; available range is 0 to {len(ts) - 1}"
            )
        return ts[timestep].strftime("%d%b%Y %H:%M:%S")

    def store_map(
        self,
        variable: Literal[
            "wse",
            "water_surface",
            "depth",
            "velocity",
            "froude",
            "shear_stress",
            "depth_x_velocity",
            "dv"
            "depth_x_velocity_sq",
            "dv2"
        ],
        output_path: str | Path,
        timestep: int | None = None,
        timeout: int = 600,
    ) -> Path:
        """Store one map with RasProcess.exe using a single variable/timestep.

        Parameters
        ----------
        variable:
            Map variable name. Supported values:
            ``"wse"``/``"water_surface"``, ``"depth"``, ``"velocity"``/``"cell_speed"``,
            ``"froude"``, ``"shear_stress"``, ``"depth_x_velocity"``,
            ``"depth_x_velocity_sq"``.
        output_path:
            Output base path for the generated map.
        timestep:
            0-based timestep index. ``None`` uses ``"Max"`` profile.
        timeout:
            RasProcess timeout in seconds.
        """
        variable_key = str(variable).strip().lower()
        map_type_by_variable = {
            "wse": "elevation",
            "water_surface": "elevation",
            "depth": "depth",
            "velocity": "velocity",
            "froude": "froude",
            "shear_stress": "Shear",
            "depth_x_velocity": "depth and velocity",
            "dv": "depth and velocity",
            "depth_x_velocity_sq": "depth and velocity squared",
            "dv2": "depth and velocity squared",
        }
        map_type = map_type_by_variable.get(variable_key)
        if map_type is None:
            supported = ", ".join(map_type_by_variable.keys())
            raise ValueError(f"Unsupported variable '{variable}'. Supported: {supported}")

        program_dir = installed_ras_directory(self.version)
        if not program_dir:
            raise FileNotFoundError(
                f"Could not find installed HEC-RAS directory for version {self.version}"
            )

        ras_process = Path(program_dir) / "RasProcess.exe"
        if not ras_process.exists():
            raise FileNotFoundError(f"RasProcess.exe not found: {ras_process}")

        result_hdf = self.plan_hdf_file
        if not result_hdf.exists():
            raise FileNotFoundError(f"Plan HDF file not found: {result_hdf}")

        output_base = Path(output_path)
        output_base.parent.mkdir(parents=True, exist_ok=True)
        profile_name = self._store_map_profile_name(timestep)

        xml_text = (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<Command Type="StoreMap">\n'
            f"  <MapType>{escape(map_type)}</MapType>\n"
            f"  <Result>{escape(str(result_hdf))}</Result>\n"
            f"  <ProfileName>{escape(profile_name)}</ProfileName>\n"
            f"  <OutputBaseFilename>{escape(str(output_base))}</OutputBaseFilename>\n"
            "</Command>\n"
        )

        tmp_xml = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".xml",
                delete=False,
                encoding="utf-8",
            ) as f:
                f.write(xml_text)
                tmp_xml = Path(f.name)

            result = subprocess.run(
                [str(ras_process), f"-CommandFile={tmp_xml}"],
                cwd=str(self.project_file.parent),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        finally:
            if tmp_xml is not None and tmp_xml.exists():
                tmp_xml.unlink(missing_ok=True)

        if result.returncode != 0:
            raise RuntimeError(
                "RasProcess StoreMap failed.\n"
                f"Return code: {result.returncode}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        return output_base
