"""Read/write HEC-RAS text input files (.prj, .g*, .f*, etc.)."""

import atexit
import logging
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import rasterio.io

from ..com.ras import installed_ras_directory

logger = logging.getLogger("raspy.model")

__all__ = [
    "MapperExtension",
    "VrtMap",
]

_temp_vrts: set[Path] = set()


def _cleanup_temp_vrts() -> None:
    """Delete any VRT files and their source rasters left over from ``read_map`` calls.

    Registered with :func:`atexit` so it runs on normal interpreter shutdown,
    catching cases where the ``with`` block was not exited cleanly (e.g. a
    Jupyter kernel restart). Has no effect if the set is already empty.
    """
    for vrt_path in list(_temp_vrts):
        import contextlib
        with contextlib.suppress(Exception):
            VrtMap(vrt_path).delete()
        _temp_vrts.discard(vrt_path)


atexit.register(_cleanup_temp_vrts)


class VrtMap:
    """A stored RAS map VRT file and its source raster tiles.

    Returned by :meth:`MapperExtension.store_map`. Provides access to the VRT
    path, the source files referenced inside it, and deletion helpers. After
    either delete method is called the instance is invalidated; any further
    access raises ``RuntimeError``.
    """

    def __init__(self, vrt_path: Path) -> None:
        self._path = vrt_path
        self._deleted = False

    def _check_valid(self) -> None:
        if self._deleted:
            raise RuntimeError(
                "This VrtMap has already been deleted and can no longer be used."
            )

    @property
    def path(self) -> Path:
        """Absolute path to the VRT file."""
        self._check_valid()
        return self._path

    @property
    def source_files(self) -> list[Path]:
        """Resolved paths of all source rasters referenced in the VRT."""
        self._check_valid()
        tree = ET.parse(self._path)
        root = tree.getroot()
        sources: list[Path] = []
        for elem in root.iter("SourceFilename"):
            if elem.text:
                src = Path(elem.text.strip())
                if not src.is_absolute():
                    src = self._path.parent / src
                sources.append(src.resolve())
        return sources

    def delete(self, include_sources: bool = True) -> None:
        """Delete the VRT file and, optionally, all source rasters.

        Parameters
        ----------
        include_sources:
            When ``True`` (default) every source file listed in the VRT is
            also deleted. Pass ``False`` to remove only the VRT itself.
        """
        self._check_valid()
        sources = self.source_files if include_sources else []
        self._path.unlink(missing_ok=True)
        for src in sources:
            src.unlink(missing_ok=True)
        self._deleted = True

    def exists(self, include_sources: bool = False) -> bool:
        """Return ``True`` if the VRT file (and optionally all source rasters) exist.

        Parameters
        ----------
        include_sources:
            When ``True``, also checks that every source raster listed in the
            VRT exists. Returns ``False`` if any source file is
            missing.
        """
        self._check_valid()
        if not self._path.exists():
            return False
        if include_sources:
            return all(src.exists() for src in self.source_files)
        return True

    def is_locked(self) -> bool:
        """Return ``True`` if the VRT or any source raster is locked by another process.

        On Windows, a file is considered locked when it cannot be opened
        for writing (e.g. because QGIS or another application holds it open).
        """
        import contextlib

        self._check_valid()
        candidates = [self._path] if self._path.exists() else []
        if candidates:
            with contextlib.suppress(Exception):
                candidates.extend(src for src in self.source_files if src.exists())
        for path in candidates:
            try:
                with open(path, "r+b"):
                    pass
            except PermissionError:
                return True
        return False

    def delete_vrt(self) -> None:
        """Delete only the VRT file, leaving the source rasters on disk."""
        self._check_valid()
        self._path.unlink(missing_ok=True)
        self._deleted = True

    def __repr__(self) -> str:
        if self._deleted:
            return "VrtMap(<deleted>)"
        return f"VrtMap({self._path!r})"


class MapperExtension:
    """Extends raspy.model.Model with RasMapper functions"""

    def timestep_to_profile_name(self, timestep: int | None) -> str:
        """Return the HEC-RAS profile name for a given timestep index.

        Parameters
        ----------
        timestep:
            Zero-based index into the plan's time series. ``None`` returns
            ``"Max"``, which selects the maximum-value profile.
        """
        if timestep is None:
            return "Max"
        if timestep < 0:
            raise ValueError("timestep must be >= 0 or None")

        ts = self.hdf.time_stamps
        if timestep >= len(ts):
            raise IndexError(
                f"timestep index {timestep} out of range; "
                f"available range is 0 to {len(ts) - 1}"
            )
        return ts[timestep].strftime("%d%b%Y %H:%M:%S")

    def _locate_project_rasmap(self) -> Path:
        rasmap = self.project_file.with_suffix(".rasmap")
        if rasmap.exists():
            return rasmap

        candidates = sorted(self.plan_file.parent.glob("*.rasmap"))
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise FileNotFoundError(
                f"No .rasmap file found in project folder: {self.plan_file.parent}"
            )

        raise FileNotFoundError(
            "Multiple .rasmap files found in project folder; "
            f"could not infer which one to use: {[str(p) for p in candidates]}"
        )

    def store_map(
        self,
        variable: Literal[
            "wse",
            "water_surface",
            "depth",
            "velocity",
            "froude",
            "shear_stress",
            "dv",
            "depth_x_velocity",
            "dv2",
            "depth_x_velocity_sq",
        ],
        timestep: int | None = None,
        timeout: int | None = None,
        raster_name: str | None = None,
        stream_output: bool = True,
    ) -> "VrtMap":
        """Store one map by editing a temporary copy of the project's .rasmap file.

        Output is always written to ``{project_dir}/{plan_short_id}/`` — RasProcess.exe
        reads the Plan ShortID from the plan file and uses it as the output folder
        regardless of any path set in the rasmap XML.
        """
        variable_key = str(variable).strip().lower()
        map_type_by_variable = {
            "wse": ("elevation", "WSE"),
            "water_surface": ("elevation", "WSE"),
            "depth": ("depth", "Depth"),
            "velocity": ("velocity", "Velocity"),
            "froude": ("froude", "Froude"),
            "shear_stress": ("Shear", "Shear Stress"),
            "dv": ("depth and velocity", "D x V"),
            "depth_x_velocity": ("depth and velocity", "D x V"),
            "dv2": ("depth and velocity squared", "D x V2"),
            "depth_x_velocity_sq": ("depth and velocity squared", "D x V2"),
        }
        map_type_info = map_type_by_variable.get(variable_key)
        if map_type_info is None:
            supported = ", ".join(map_type_by_variable.keys())
            raise ValueError(
                f"Unsupported variable '{variable}'. Supported: {supported}"
            )
        map_type, display_name = map_type_info

        if raster_name is not None:
            if not isinstance(raster_name, str):
                raise TypeError("raster_name must be a string or None")
            if raster_name.strip() == "":
                raise ValueError("raster_name cannot be empty")
            if any(sep in raster_name for sep in ("/", "\\", ".")):
                raise ValueError(
                    "name of the raster cannot contain separator or extension name"
                )

        plan_short_id = self.plan.short_id
        if not plan_short_id:
            raise ValueError(
                "self.plan.short_id is empty; set a plan short id before storing maps"
            )

        project_dir = self.plan_file.parent
        absolute_output_dir = project_dir / plan_short_id
        relative_output_dir = f"./{plan_short_id}"
        absolute_output_dir.mkdir(parents=True, exist_ok=True)
        output_raster = None

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

        profile_name = self.timestep_to_profile_name(timestep)
        profile_index = 2147483647 if timestep is None else timestep
        safe_profile = profile_name.replace(":", " ").replace("/", "_")

        rasmap_src = self._locate_project_rasmap()
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".rasmap",
            delete=False,
            dir=project_dir,
            encoding="utf-8",
        ) as f:
            temp_rasmap = Path(f.name)

        shutil.copy2(rasmap_src, temp_rasmap)

        try:
            tree = ET.parse(temp_rasmap)
            root = tree.getroot()

            results_elem = root.find(".//Results")
            if results_elem is None:
                results_elem = ET.SubElement(root, "Results")
                results_elem.set("Checked", "True")
                results_elem.set("Expanded", "True")

            plan_basename = result_hdf.name.lower()
            plan_layer = None
            for layer in results_elem.findall("Layer"):
                filename = layer.get("Filename", "")
                if Path(filename).name.lower() == plan_basename:
                    plan_layer = layer
                    break

            if plan_layer is None:
                plan_layer = ET.SubElement(results_elem, "Layer")
                plan_layer.set("Name", plan_short_id)
                plan_layer.set("Type", "RASResults")
                plan_layer.set("Filename", f".\\{result_hdf.name}")

            to_remove = []
            for layer in plan_layer.findall("Layer"):
                if layer.get("Type") == "RASResultsMap":
                    params = layer.find("MapParameters")
                    if params is not None and "Stored" in params.get("OutputMode", ""):
                        to_remove.append(layer)
            for layer in to_remove:
                plan_layer.remove(layer)

            stored_vrt_rel = (
                f"{relative_output_dir}/{display_name} ({safe_profile}).vrt"
            )
            stored_vrt_attr = stored_vrt_rel.replace("/", "\\")

            layer_elem = ET.SubElement(plan_layer, "Layer")
            layer_elem.set("Name", display_name)
            layer_elem.set("Type", "RASResultsMap")
            layer_elem.set("Checked", "True")
            layer_elem.set("Filename", stored_vrt_attr)

            params = ET.SubElement(layer_elem, "MapParameters")
            params.set("MapType", map_type)
            params.set("OutputMode", "Stored Current Terrain")
            params.set("StoredFilename", stored_vrt_attr)
            params.set("ProfileIndex", str(profile_index))
            params.set("ProfileName", profile_name)
            if raster_name:
                params.set("OverwriteOutputFilename", raster_name)
            else:
                raster_name = f"{display_name} ({safe_profile})"

            tree.write(temp_rasmap, encoding="utf-8", xml_declaration=True)

            expected_vrt = absolute_output_dir / f"{raster_name}.vrt"
            if expected_vrt.exists():
                candidate = VrtMap(expected_vrt)
                if candidate.is_locked():
                    raise PermissionError(
                        f"Output file is locked by another process (e.g. QGIS): "
                        f"{expected_vrt}\n"
                        "Close the file before calling store_map."
                    )

            logger.info(
                "Pre RasProcess: output raster expected at: %s",
                absolute_output_dir / f"{raster_name}.vrt",
            )

            cmd = [
                str(ras_process),
                "-Command=StoreAllMaps",
                f"-RasMapFilename={temp_rasmap}",
                f"-ResultFilename={result_hdf}",
            ]
            logger.debug("RasProcess command: %s", " ".join(cmd))
            if stream_output:
                stdout_lines: list[str] = []
                stderr_lines: list[str] = []
                with subprocess.Popen(
                    cmd,
                    cwd=str(project_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ) as proc:
                    for line in proc.stdout:
                        line = line.rstrip()
                        logger.info("RasProcess stdout: %s", line)
                        stdout_lines.append(line)
                    for line in proc.stderr:
                        line = line.rstrip()
                        logger.warning("RasProcess stderr: %s", line)
                        stderr_lines.append(line)
                    proc.wait(timeout=timeout)
                result = subprocess.CompletedProcess(
                    args=cmd,
                    returncode=proc.returncode,
                    stdout="\n".join(stdout_lines),
                    stderr="\n".join(stderr_lines),
                )
            else:
                result = subprocess.run(
                    cmd,
                    cwd=str(project_dir),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )

            output_raster = absolute_output_dir / f"{raster_name}.vrt"
            logger.info("RasProcess output raster expected at: %s", output_raster)

        finally:
            shutil.copy2(temp_rasmap, "gbtest.rasmap")
            temp_rasmap.unlink(missing_ok=True)

        if result.returncode != 0:
            raise RuntimeError(
                "RasProcess StoreAllMaps failed for temporary rasmap copy.\n"
                f"Return code: {result.returncode}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        if output_raster is None:
            raise Exception(
                "Unexpected error: RasProcess completed but "
                "output raster path is not set."
            )

        vrt = VrtMap(output_raster)
        if not vrt.exists():
            raise FileNotFoundError(
                "RasProcess completed but no output file was found \n"
                f"Output folder: {absolute_output_dir}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )
        missing_sources = [src for src in vrt.source_files if not src.exists()]
        if missing_sources:
            missing_list = "\n  ".join(str(p) for p in missing_sources)
            raise FileNotFoundError(
                "RasProcess completed but the following source rasters are missing:\n"
                f"  {missing_list}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        return vrt

    @contextmanager
    def open_map(
        self,
        variable: Literal[
            "wse",
            "water_surface",
            "depth",
            "velocity",
            "froude",
            "shear_stress",
            "dv",
            "depth_x_velocity",
            "dv2",
            "depth_x_velocity_sq",
        ],
        timestep: int | None = None,
        timeout: int | None = None,
        raster_name: str | None = None,
    ) -> Generator["rasterio.io.DatasetReader", None, None]:
        """Store a map, yield an open rasterio dataset, then delete it.

        RasProcess.exe always writes to ``{project_dir}/{plan_short_id}/``
        regardless of any path set in the rasmap XML — the output location
        cannot be redirected. Files are deleted on context exit, and also
        registered for cleanup on interpreter shutdown via ``atexit``.

        Usage::

            with model.open_map("wse") as ds:
                data = ds.read(1)
        """
        import rasterio

        vrt = self.store_map(
            variable,
            timestep=timestep,
            raster_name=raster_name,
            timeout=timeout,
        )
        vrt_path = vrt.path
        _temp_vrts.add(vrt_path)
        try:
            with rasterio.open(vrt_path) as ds:
                yield ds
        finally:
            vrt.delete()
            _temp_vrts.discard(vrt_path)

    def export_wse(
        self,
        timestep: int | None = None,
        timeout: int | None = None,
        raster_name: str | None = None,
    ) -> "VrtMap":
        return self.store_map(
            "wse", timestep=timestep, raster_name=raster_name, timeout=timeout
        )

    def open_wse(
        self,
        timestep: int | None = None,
        timeout: int | None = None,
    ) -> "AbstractContextManager[rasterio.io.DatasetReader]":
        import secrets
        name = secrets.token_hex(8)
        return self.open_map(
            "wse", timestep=timestep, raster_name=name, timeout=timeout
        )

    def export_depth(
        self,
        timestep: int | None = None,
        timeout: int | None = None,
        raster_name: str | None = None,
    ) -> "VrtMap":
        return self.store_map(
            "depth", timestep=timestep, raster_name=raster_name, timeout=timeout
        )

    def open_depth(
        self,
        timestep: int | None = None,
        timeout: int | None = None,
    ) -> "AbstractContextManager[rasterio.io.DatasetReader]":
        import secrets
        name = secrets.token_hex(8)
        return self.open_map(
            "depth", timestep=timestep, raster_name=name, timeout=timeout
        )

    def export_velocity(
        self,
        timestep: int | None = None,
        timeout: int | None = None,
        raster_name: str | None = None,
    ) -> "VrtMap":
        return self.store_map(
            "velocity", timestep=timestep, raster_name=raster_name, timeout=timeout
        )

    def open_velocity(
        self,
        timestep: int | None = None,
        timeout: int | None = None,
    ) -> "AbstractContextManager[rasterio.io.DatasetReader]":
        import secrets
        name = secrets.token_hex(8)
        return self.open_map(
            "velocity", timestep=timestep, raster_name=name, timeout=timeout
        )

    def export_froude(
        self,
        timestep: int | None = None,
        timeout: int | None = None,
        raster_name: str | None = None,
    ) -> "VrtMap":
        return self.store_map(
            "froude", timestep=timestep, raster_name=raster_name, timeout=timeout
        )

    def open_froude(
        self,
        timestep: int | None = None,
        timeout: int | None = None,
    ) -> "AbstractContextManager[rasterio.io.DatasetReader]":
        import secrets
        name = secrets.token_hex(8)
        return self.open_map(
            "froude", timestep=timestep, raster_name=name, timeout=timeout
        )

    def export_shear_stress(
        self,
        timestep: int | None = None,
        timeout: int | None = None,
        raster_name: str | None = None,
    ) -> "VrtMap":
        return self.store_map(
            "shear_stress", timestep=timestep, raster_name=raster_name, timeout=timeout
        )

    def open_shear_stress(
        self,
        timestep: int | None = None,
        timeout: int | None = None,
    ) -> "AbstractContextManager[rasterio.io.DatasetReader]":
        import secrets
        name = secrets.token_hex(8)
        return self.open_map(
            "shear_stress", timestep=timestep, raster_name=name, timeout=timeout
        )

    def export_dv(
        self,
        timestep: int | None = None,
        timeout: int | None = None,
        raster_name: str | None = None,
    ) -> "VrtMap":
        return self.store_map(
            "dv", timestep=timestep, raster_name=raster_name, timeout=timeout
        )

    def open_dv(
        self,
        timestep: int | None = None,
        timeout: int | None = None,
    ) -> "AbstractContextManager[rasterio.io.DatasetReader]":
        import secrets
        name = secrets.token_hex(8)
        return self.open_map(
            "dv", timestep=timestep, raster_name=name, timeout=timeout
        )

    def export_dv2(
        self,
        timestep: int | None = None,
        timeout: int | None = None,
        raster_name: str | None = None,
    ) -> "VrtMap":
        return self.store_map(
            "dv2", timestep=timestep, raster_name=raster_name, timeout=timeout
        )

    def open_dv2(
        self,
        timestep: int | None = None,
        timeout: int | None = None,
    ) -> "AbstractContextManager[rasterio.io.DatasetReader]":
        import secrets
        name = secrets.token_hex(8)
        return self.open_map(
            "dv2", timestep=timestep, raster_name=name, timeout=timeout
        )
