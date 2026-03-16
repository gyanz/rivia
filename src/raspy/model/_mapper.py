"""RasMapper integration for exporting hydraulic result rasters.

Provides :class:`VrtMap` — a handle to a VRT raster exported by RasProcess.exe —
and :class:`MapperExtension`, a mixin that adds ``store_map`` / ``open_map`` and
per-variable convenience wrappers (``export_wse``, ``open_wse``, etc.) to the
Model class.

RasProcess.exe always writes output to ``{project_dir}/{plan_short_id}/``
regardless of any path configured in the rasmap XML.
"""

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
from ..utils.helpers import log_call, timed

logger = logging.getLogger("raspy.model")

__all__ = [
    "MapperExtension",
    "VrtMap",
]

_temp_dirs: set[Path] = set()


def _cleanup_temp_dirs() -> None:
    """Delete temp directories created by ``open_map`` calls.

    Registered with :func:`atexit` so it runs on normal interpreter shutdown,
    catching cases where the ``with`` block was not exited cleanly (e.g. a
    Jupyter kernel restart). Has no effect if the set is already empty.
    """
    import contextlib

    for tmp_dir in list(_temp_dirs):
        with contextlib.suppress(Exception):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        _temp_dirs.discard(tmp_dir)


atexit.register(_cleanup_temp_dirs)


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
        logger.debug("Deleting VRT: %s", self._path)
        self._path.unlink(missing_ok=True)
        for src in sources:
            logger.debug("Deleting source: %s", src)
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

    @staticmethod
    def _run_rasprocess(
        cmd: list[str],
        cwd: "Path",
        timeout: "int | None",
        stream_output: bool,
    ) -> "subprocess.CompletedProcess[str]":
        """Run a RasProcess.exe command and return a CompletedProcess.

        Parameters
        ----------
        cmd:
            Full command list, e.g. ``[str(ras_process), "-Command=..."]``.
        cwd:
            Working directory passed to the subprocess.
        timeout:
            Timeout in seconds, or ``None`` for no limit.
        stream_output:
            When ``True``, lines are read and logged in real time via
            ``subprocess.Popen``; stdout lines at INFO, stderr at WARNING.
            When ``False``, output is captured silently via ``subprocess.run``.
        """
        logger.debug("RasProcess command: %s", " ".join(cmd))
        if stream_output:
            stdout_lines: list[str] = []
            stderr_lines: list[str] = []
            with subprocess.Popen(
                cmd,
                cwd=str(cwd),
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
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=proc.returncode,
                stdout="\n".join(stdout_lines),
                stderr="\n".join(stderr_lines),
            )
        return subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    @log_call(logging.INFO)
    @timed(logging.INFO)
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
        raster_name: str | None = None,
        output_path: "str | Path | None" = None,
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "VrtMap":
        """Store one hydraulic result map via RasProcess.exe.

        When ``output_path`` is ``None``, uses ``StoreAllMaps`` (output goes to
        ``{project_dir}/{plan_short_id}/``).  When ``output_path`` is provided,
        uses a ``StoreMap`` XML command (``-CommandFile``) with an absolute
        ``OutputBaseFilename`` so output lands directly in that directory.

        Parameters
        ----------
        variable:
            Hydraulic variable to export.
        timestep:
            Zero-based timestep index. ``None`` exports the maximum profile.
        timeout:
            Subprocess timeout in seconds.
        raster_name:
            Stem of the output VRT (no path, no extension). Defaults to
            ``"{display_name} ({profile_name})"``.
        output_path:
            Directory to write output into. Must already exist. When ``None``,
            output goes to ``{project_dir}/{plan_short_id}/``.
        stream_output:
            When ``True`` (default) stdout/stderr are logged line-by-line in
            real time. When ``False`` output is captured silently.
        """
        # -- Variable lookup --
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
            raise ValueError(
                f"Unsupported variable '{variable}'. "
                f"Supported: {', '.join(map_type_by_variable)}"
            )
        map_type, display_name = map_type_info

        # -- raster_name validation --
        if raster_name is not None:
            if not isinstance(raster_name, str):
                raise TypeError("raster_name must be a string or None")
            if raster_name.strip() == "":
                raise ValueError("raster_name cannot be empty")
            if any(sep in raster_name for sep in ("/", "\\", ".")):
                raise ValueError(
                    "raster_name cannot contain path separators or an extension"
                )

        # -- Common setup --
        plan_short_id = self.plan.short_id
        if not plan_short_id:
            raise ValueError(
                "self.plan.short_id is empty; set a plan short id before storing maps"
            )

        project_dir = self.plan_file.parent

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

        custom_raster_name = raster_name  # None if caller did not supply one
        if raster_name is None:
            raster_name = f"{display_name} ({safe_profile})"

        rasmap_src = self._locate_project_rasmap()

        # -- Branch: build command, manage temp file, run --
        if output_path is None:
            # StoreAllMaps: inject map layer into a temp rasmap copy; output
            # always goes to {project_dir}/{plan_short_id}/ regardless of XML.
            abs_output_dir = project_dir / plan_short_id
            abs_output_dir.mkdir(parents=True, exist_ok=True)

            stored_vrt_attr = f".\\{plan_short_id}\\{display_name} ({safe_profile}).vrt"

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

                plan_layer = None
                plan_basename = result_hdf.name.lower()
                for layer in results_elem.findall("Layer"):
                    if Path(layer.get("Filename", "")).name.lower() == plan_basename:
                        plan_layer = layer
                        break
                if plan_layer is None:
                    plan_layer = ET.SubElement(results_elem, "Layer")
                    plan_layer.set("Name", plan_short_id)
                    plan_layer.set("Type", "RASResults")
                    plan_layer.set("Filename", f".\\{result_hdf.name}")

                to_remove = [
                    layer for layer in plan_layer.findall("Layer")
                    if layer.get("Type") == "RASResultsMap"
                    and layer.find("MapParameters") is not None
                    and "Stored" in layer.find("MapParameters").get("OutputMode", "")
                ]
                for layer in to_remove:
                    plan_layer.remove(layer)

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
                if custom_raster_name:
                    params.set("OverwriteOutputFilename", raster_name)

                tree.write(temp_rasmap, encoding="utf-8", xml_declaration=True)

                expected_vrt = abs_output_dir / f"{raster_name}.vrt"
                if expected_vrt.exists() and VrtMap(expected_vrt).is_locked():
                    raise PermissionError(
                        f"Output file is locked by another process (e.g. QGIS): "
                        f"{expected_vrt}\nClose the file before calling store_map."
                    )

                logger.info("StoreAllMaps: output expected at %s", expected_vrt)
                cmd = [
                    str(ras_process),
                    "-Command=StoreAllMaps",
                    f"-RasMapFilename={temp_rasmap}",
                    f"-ResultFilename={result_hdf}",
                ]
                result = self._run_rasprocess(cmd, project_dir, timeout, stream_output)
            finally:
                temp_rasmap.unlink(missing_ok=True)

        else:
            # StoreMap XML: original rasmap, output to caller-supplied directory.
            abs_output_dir = Path(output_path)
            if not abs_output_dir.exists():
                raise FileNotFoundError(
                    f"output_path does not exist: {abs_output_dir}"
                )
            abs_output_dir = abs_output_dir.resolve()

            # OutputBaseFilename has no extension — RasProcess.exe appends .vrt
            out_base = abs_output_dir / raster_name
            expected_vrt = abs_output_dir / f"{raster_name}.vrt"

            if expected_vrt.exists() and VrtMap(expected_vrt).is_locked():
                raise PermissionError(
                    f"Output file is locked by another process (e.g. QGIS): "
                    f"{expected_vrt}\nClose the file before calling store_map."
                )

            xml = (
                '<?xml version="1.0" encoding="utf-8"?>\n'
                '<Command Type="StoreMap">\n'
                f"  <RasMapFilename>{rasmap_src}</RasMapFilename>\n"
                f"  <MapType>{map_type}</MapType>\n"
                f"  <Result>{result_hdf}</Result>\n"
                f"  <ProfileName>{profile_name}</ProfileName>\n"
                f"  <ProfileIndex>{profile_index}</ProfileIndex>\n"
                "  <OutputMode>Stored Current Terrain</OutputMode>\n"
                f"  <OutputBaseFilename>{out_base}</OutputBaseFilename>\n"
                "</Command>"
            )
            logger.debug("StoreMap XML:\n%s", xml)
            logger.info("StoreMap: output expected at %s", expected_vrt)

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".xml",
                delete=False,
                dir=project_dir,
                encoding="utf-8",
            ) as f:
                temp_xml = Path(f.name)
                f.write(xml)

            try:
                cmd = [str(ras_process), f"-CommandFile={temp_xml}"]
                result = self._run_rasprocess(cmd, project_dir, timeout, stream_output)
            finally:
                temp_xml.unlink(missing_ok=True)

        # -- Common tail: error check and VrtMap validation --
        stderr_lower = result.stderr.lower()
        if result.returncode != 0 or "error" in stderr_lower:
            stdout_detail = (
                "(stdout/stderr already streamed to logger)"
                if stream_output
                else f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
            raise RuntimeError(
                f"RasProcess failed (return code {result.returncode}).\n"
                + stdout_detail
            )

        vrt = VrtMap(expected_vrt)
        if not vrt.exists():
            raise FileNotFoundError(
                f"RasProcess completed but output VRT was not found: {expected_vrt}\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

        missing_sources = [src for src in vrt.source_files if not src.exists()]
        if missing_sources:
            missing_list = "\n  ".join(str(p) for p in missing_sources)
            raise FileNotFoundError(
                "RasProcess completed but source rasters are missing:\n"
                f"  {missing_list}\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

        return vrt

    @log_call(logging.INFO)
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
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> Generator["rasterio.io.DatasetReader", None, None]:
        """Store a map into a temp dir, yield an open rasterio dataset, then delete.

        Output is written to a freshly-created system temp directory that is
        removed (including all companion raster tiles) on context exit. The
        directory is also registered with :func:`atexit` so it is cleaned up on
        interpreter shutdown if the ``with`` block is not exited cleanly (e.g.
        a Jupyter kernel restart).

        Usage::

            with model.open_map("wse") as ds:
                data = ds.read(1)
        """
        import secrets

        import rasterio

        raster_name = secrets.token_hex(8)
        tmp_dir = Path(tempfile.mkdtemp())
        logger.debug("open_map: temp dir %s, raster name %s", tmp_dir, raster_name)
        _temp_dirs.add(tmp_dir)
        try:
            vrt = self.store_map(
                variable,
                timestep=timestep,
                raster_name=raster_name,
                output_path=tmp_dir,
                stream_output=stream_output,
                timeout=timeout,
            )
            logger.debug("open_map: created %s", vrt.path)
            with rasterio.open(vrt.path) as ds:
                yield ds
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            _temp_dirs.discard(tmp_dir)

    @log_call(logging.INFO)
    def export_wse(
        self,
        timestep: int | None,
        output_vrt: "str | Path",
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "VrtMap":
        output_vrt = Path(output_vrt)
        if output_vrt.suffix != ".vrt":
            raise ValueError(
                f"output_vrt must have a .vrt extension, got: {output_vrt}"
            )
        return self.store_map(
            "wse",
            timestep=timestep,
            raster_name=output_vrt.stem,
            output_path=output_vrt.parent,
            stream_output=stream_output,
            timeout=timeout,
        )

    @log_call(logging.INFO)
    def open_wse(
        self,
        timestep: int | None = None,
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "AbstractContextManager[rasterio.io.DatasetReader]":
        return self.open_map(
            "wse", timestep=timestep, stream_output=stream_output, timeout=timeout
        )

    @log_call(logging.INFO)
    def export_depth(
        self,
        timestep: int | None,
        output_vrt: "str | Path",
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "VrtMap":
        output_vrt = Path(output_vrt)
        if output_vrt.suffix != ".vrt":
            raise ValueError(
                f"output_vrt must have a .vrt extension, got: {output_vrt}"
            )
        return self.store_map(
            "depth",
            timestep=timestep,
            raster_name=output_vrt.stem,
            output_path=output_vrt.parent,
            stream_output=stream_output,
            timeout=timeout,
        )

    @log_call(logging.INFO)
    def open_depth(
        self,
        timestep: int | None = None,
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "AbstractContextManager[rasterio.io.DatasetReader]":
        return self.open_map(
            "depth", timestep=timestep, stream_output=stream_output, timeout=timeout
        )

    @log_call(logging.INFO)
    def export_velocity(
        self,
        timestep: int | None,
        output_vrt: "str | Path",
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "VrtMap":
        output_vrt = Path(output_vrt)
        if output_vrt.suffix != ".vrt":
            raise ValueError(
                f"output_vrt must have a .vrt extension, got: {output_vrt}"
            )
        return self.store_map(
            "velocity",
            timestep=timestep,
            raster_name=output_vrt.stem,
            output_path=output_vrt.parent,
            stream_output=stream_output,
            timeout=timeout,
        )

    @log_call(logging.INFO)
    def open_velocity(
        self,
        timestep: int | None = None,
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "AbstractContextManager[rasterio.io.DatasetReader]":
        return self.open_map(
            "velocity", timestep=timestep, stream_output=stream_output, timeout=timeout
        )

    @log_call(logging.INFO)
    def export_froude(
        self,
        timestep: int | None,
        output_vrt: "str | Path",
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "VrtMap":
        output_vrt = Path(output_vrt)
        if output_vrt.suffix != ".vrt":
            raise ValueError(
                f"output_vrt must have a .vrt extension, got: {output_vrt}"
            )
        return self.store_map(
            "froude",
            timestep=timestep,
            raster_name=output_vrt.stem,
            output_path=output_vrt.parent,
            stream_output=stream_output,
            timeout=timeout,
        )

    @log_call(logging.INFO)
    def open_froude(
        self,
        timestep: int | None = None,
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "AbstractContextManager[rasterio.io.DatasetReader]":
        return self.open_map(
            "froude", timestep=timestep, stream_output=stream_output, timeout=timeout
        )

    @log_call(logging.INFO)
    def export_shear_stress(
        self,
        timestep: int | None,
        output_vrt: "str | Path",
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "VrtMap":
        output_vrt = Path(output_vrt)
        if output_vrt.suffix != ".vrt":
            raise ValueError(
                f"output_vrt must have a .vrt extension, got: {output_vrt}"
            )
        return self.store_map(
            "shear_stress",
            timestep=timestep,
            raster_name=output_vrt.stem,
            output_path=output_vrt.parent,
            stream_output=stream_output,
            timeout=timeout,
        )

    @log_call(logging.INFO)
    def open_shear_stress(
        self,
        timestep: int | None = None,
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "AbstractContextManager[rasterio.io.DatasetReader]":
        return self.open_map(
            "shear_stress",
            timestep=timestep,
            stream_output=stream_output,
            timeout=timeout,
        )

    @log_call(logging.INFO)
    def export_dv(
        self,
        timestep: int | None,
        output_vrt: "str | Path",
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "VrtMap":
        output_vrt = Path(output_vrt)
        if output_vrt.suffix != ".vrt":
            raise ValueError(
                f"output_vrt must have a .vrt extension, got: {output_vrt}"
            )
        return self.store_map(
            "dv",
            timestep=timestep,
            raster_name=output_vrt.stem,
            output_path=output_vrt.parent,
            stream_output=stream_output,
            timeout=timeout,
        )

    @log_call(logging.INFO)
    def open_dv(
        self,
        timestep: int | None = None,
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "AbstractContextManager[rasterio.io.DatasetReader]":
        return self.open_map(
            "dv", timestep=timestep, stream_output=stream_output, timeout=timeout
        )

    @log_call(logging.INFO)
    def export_dv2(
        self,
        timestep: int | None,
        output_vrt: "str | Path",
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "VrtMap":
        output_vrt = Path(output_vrt)
        if output_vrt.suffix != ".vrt":
            raise ValueError(
                f"output_vrt must have a .vrt extension, got: {output_vrt}"
            )
        return self.store_map(
            "dv2",
            timestep=timestep,
            raster_name=output_vrt.stem,
            output_path=output_vrt.parent,
            stream_output=stream_output,
            timeout=timeout,
        )

    @log_call(logging.INFO)
    def open_dv2(
        self,
        timestep: int | None = None,
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "AbstractContextManager[rasterio.io.DatasetReader]":
        return self.open_map(
            "dv2", timestep=timestep, stream_output=stream_output, timeout=timeout
        )
