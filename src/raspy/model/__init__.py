"""Read/write HEC-RAS text input files (.prj, .g*, .f*, etc.)."""

import logging
import shutil
from pathlib import Path

from .. import com

EXT_BACKUP_FILE = "raspy_bkup"

class Model:
    """ High-level interface for working with an HEC-RAS project via the wrapped COM object.

    Use this class in preference to `com.open`. While `com.open` returns a raw HEC-RAS
    controller instance that is not associated with any project, `Model` binds the COM
    object to a specific HEC-RAS project file and provides project-aware operations.
    """
    def __init__(self, project_file: str | Path, ras_version: str | int | None = None,
                 backup: bool = False):
        self._project_path = Path(project_file)
        self._backup = backup

        # restore if there are any backup files
        model_files = Model._get_project_files(project_file)
        Model._restore_backups(model_files)

        if ras_version is None:
            ras_version= Model._get_ras_version_from_project_file(project_file)

        if backup:
            Model._create_backups(model_files)

        self._rc = com.open(ras_version)
        self._ras_version = self._rc.ras_version()
        self._rc.Project_Open(str(self._project_path))
        self._compute_blocking = 1
    
    @property
    def project_file(self) -> Path:
        """Return the project file path."""
        return self._project_path
    
    def reset(self):
        if not self._backup:
            raise ValueError("Model instance does not have back files to perform reset.")
        model_files = Model._get_project_files(self._project_path)
        Model._restore_backups(model_files)
        self.reload()

    def reload(self):
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
    
    def show_compute(self, flag:bool):
        if flag:
            self._rc.Compute_ShowComputationWindow()
        else:
            self._rc.Compute_HideComputationWindow()

    def compute_blocking(self, flag:bool):
        if flag:
            self._compute_blocking = 1
        else:
            self._compute_blocking = 0
    
    def __del__(self):
        logging.debug("Executing Model destructor.")
        self._rc.close()
        if self._backup:
            model_files = Model._get_project_files(self._project_path)
            Model._restore_backups(model_files)
    
    @staticmethod
    def _get_ras_version_from_project_file(project_file: str | Path):
        path = Path(project_file)

        if not path.is_file():
            raise IOError(f"HEC-RAS Project not found: {project_file}")

        plan_file = None
        with open(project_file) as fid:
            for line in fid:
                if line.startswith("Current Plan"):
                    ext = line.split("=")[1].strip()
                    plan_file = path.parent / f"{path.stem}.{ext}"
        if plan_file is None:
            raise RuntimeError(f"The HEC-RAS project file does not have current plan specified: {project_file}")
        
        with open(plan_file) as fid:
            for line in fid:
                if line.startswith("Program Version"):
                    version = line.split("=")[1].strip()
                    return version

        raise IOError(f"HEC-RAS version info not found in current plan: {plan_file}")

    @staticmethod
    def _get_project_files(project_file: str | Path) -> list[Path]:
        path = Path(project_file)

        if not path.is_file():
            raise IOError(f"HEC-RAS Project not found: {project_file}")

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
