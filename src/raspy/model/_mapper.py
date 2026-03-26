"""RasMapper integration for exporting hydraulic result rasters.

Provides :class:`VrtMap` — a handle to a VRT raster exported by the map-store
executables — and :class:`MapperExtension`, a mixin that adds ``store_map`` /
``open_map`` and per-variable convenience wrappers (``export_wse``,
``open_wse``, etc.) to the Model class.

Two executables are used depending on the ``render_mode`` argument:

- ``RasMapperStoreMap.exe`` (``render_mode`` is ``"sloping"``, ``"hybrid"``, or
  ``"horizontal"``): the primary map-store tool, shipped with ``raspy`` in
  ``src/raspy/bin/``.  Properly initialises RasMapperLib ``SharedData``
  render-mode state before rendering, enabling deterministic, fully-specified
  output.  The render mode that matches RasMapper's output depends on the
  ``<RenderMode>`` configured in the project's ``.rasmap`` file.  Also supports
  ``tight_extent=True`` (default), which clips output tiles to the model
  geometry footprint (2D flow areas + cross sections + storage areas), matching
  RasMapper's "Original Extent" behaviour.

- ``RasProcess.exe`` (``render_mode=None``): the legacy HEC-RAS tool, used only
  as a fallback.  Always renders with its built-in defaults
  (sloping/JustFacepoints) regardless of the ``<RenderMode>`` element in the
  ``.rasmap`` file.  ``tight_extent`` is not supported in this mode.

Both executables write output to ``{project_dir}/{plan_short_id}/`` by default;
supply ``output_path`` to direct output to a different directory.

Author: Gyan Basyal
Year: 2026
"""

import atexit
import logging
import re
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import rasterio.io

from ..com.ras import installed_ras_directory
from ..utils.fs import assert_path_writable
from ..utils.helpers import log_call, timed

logger = logging.getLogger("raspy.model")

__all__ = [
    "MapperExtension",
    "TerrainLayer",
    "TerrainSubLayer",
    "VrtMap",
]


def _run_subprocess(
    cmd: list[str],
    cwd: "Path | None",
    timeout: "int | None",
    stream_output: bool,
) -> "subprocess.CompletedProcess[str]":
    """Run a subprocess command and return a CompletedProcess.

    Used for both ``RasProcess.exe`` and ``RasMapperStoreMap.exe`` invocations.
    ``cmd[0]`` is the executable; remaining elements are its arguments.

    Parameters
    ----------
    cmd:
        Full command list, e.g. ``[str(exe), "-Arg=value", ...]``.
    cwd:
        Working directory passed to the subprocess, or ``None`` to
        inherit the calling process's CWD.
    timeout:
        Timeout in seconds, or ``None`` for no limit.
    stream_output:
        When ``True``, lines are read and logged in real time via
        ``subprocess.Popen``; stdout lines prefixed ``DEBUG:``/``INFO:`` by
        the stub are logged at the matching level; unprefixed lines at INFO;
        stderr at WARNING.
        When ``False``, output is captured silently via ``subprocess.run``.
    """
    exe_name = Path(cmd[0]).name
    logger.debug("%s command: %s", exe_name, " ".join(cmd))
    if stream_output:
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        with subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as proc:
            for line in proc.stdout:
                line = line.rstrip()
                if line.startswith("DEBUG: "):
                    logger.debug("%s: %s", exe_name, line[7:])
                elif line.startswith("INFO: "):
                    logger.info("%s: %s", exe_name, line[6:])
                else:
                    logger.info("%s stdout: %s", exe_name, line)
                stdout_lines.append(line)
            for line in proc.stderr:
                line = line.rstrip()
                logger.warning("%s stderr: %s", exe_name, line)
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


@dataclass
class TerrainSubLayer:
    """A modification sub-layer nested inside a :class:`TerrainLayer`.

    Represents any ``<Layer>`` child node under a terrain layer, including
    ``ElevationModificationGroup``, ``GroundLineModificationLayer``, and
    ``ElevationControlPointLayer`` nodes.  The tree structure mirrors the XML.
    """

    name: str
    type: str
    filename: Path
    children: list["TerrainSubLayer"] = field(default_factory=list)


@dataclass
class TerrainLayer:
    """A terrain layer entry from the ``<Terrains>`` section of a .rasmap file."""

    name: str
    type: str
    filename: Path
    resample_method: str | None = None
    modifications: list[TerrainSubLayer] = field(default_factory=list)


def _parse_terrain_sublayers(
    el: ET.Element, base_dir: Path
) -> list[TerrainSubLayer]:
    """Recursively parse child ``<Layer>`` elements into :class:`TerrainSubLayer`."""
    result = []
    for child in el.findall("Layer"):
        name = child.get("Name", "")
        type_ = child.get("Type", "")
        filename_str = child.get("Filename", "")
        filename = (base_dir / filename_str).resolve() if filename_str else Path()
        children = _parse_terrain_sublayers(child, base_dir)
        result.append(
            TerrainSubLayer(name=name, type=type_, filename=filename, children=children)
        )
    return result

_temp_dirs: set[Path] = set()

_MAC_UNC_RE = re.compile(r"^\\\\Mac\\([A-Za-z])$")


def _resolve(p: Path) -> Path:
    """Resolve *p* to an absolute path, converting Mac Parallels virtual-drive
    UNC paths (``\\\\Mac\\Z\\...``) back to Windows drive-letter paths (``Z:\\...``).

    On Mac with Parallels, ``Z:\\`` resolves as ``//Mac/Z/`` which RasProcess.exe
    cannot handle.  This function re-maps those UNC roots to the drive letter.
    """
    p = p.resolve()
    m = _MAC_UNC_RE.match(p.drive)
    if m:
        p = Path(f"{m.group(1).upper()}:\\") / p.relative_to(p.anchor)
    return p


def _cleanup_temp_dirs() -> None:
    """Delete temp directories created by ``open_map("temp_dir")`` calls.

    Registered with :func:`atexit` so it runs on normal interpreter shutdown,
    catching cases where the ``with`` block was not exited cleanly (e.g. a
    Jupyter kernel restart). Has no effect if the set is already empty.
    """
    for tmp_dir in list(_temp_dirs):
        with suppress(Exception):
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
        self._check_valid()
        candidates = [self._path] if self._path.exists() else []
        if candidates:
            with suppress(Exception):
                candidates.extend(src for src in self.source_files if src.exists())
        for path in candidates:
            try:
                assert_path_writable(path)
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
    """Mixin that adds RasMapper stored-map export to :class:`raspy.model.Model`.

    Renders hydraulic result maps via ``RasMapperStoreMap.exe`` (primary, when
    ``render_mode`` is set) or ``RasProcess.exe`` (fallback, ``render_mode=None``),
    returning results as :class:`VrtMap` handles or open ``rasterio`` datasets.

    **Export workflow** — two families:

    - :meth:`export_wse` / :meth:`export_depth` / … write a persistent VRT to
      a caller-supplied path and return a :class:`VrtMap`.
    - :meth:`open_wse` / :meth:`open_depth` / … are context managers that yield
      an open ``rasterio.DatasetReader`` and clean up on exit.

    Both families delegate to :meth:`store_map`.  A temporary copy of the
    project ``.rasmap`` file is created with the target map layer injected and
    ``OverwriteOutputFilename`` set; the chosen executable is then invoked with
    that file.  Output lands in ``{project_dir}/{plan_short_id}/`` by default,
    or in ``output_path`` when supplied.

    Variables supported: ``wse``, ``depth``, ``velocity``, ``froude``,
    ``shear_stress``, ``dv`` (depth × velocity), ``dv2`` (depth × velocity²).
    """

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

        ts = self.hdf.time_stamps_map
        if timestep >= len(ts):
            raise IndexError(
                f"timestep index {timestep} out of range; "
                f"available range is 0 to {len(ts) - 1}"
            )
        return ts[timestep].strftime("%d%b%Y %H:%M:%S")
    
    def _terrain_layer(self) -> list[TerrainLayer]:
        """Return terrain layers from the ``<Terrains>`` section of the .rasmap file.

        Each :class:`TerrainLayer` carries ``name``, ``type``, ``filename``
        (absolute :class:`~pathlib.Path`), and ``modifications`` — a recursive
        list of :class:`TerrainSubLayer` objects for any modification sub-layers
        (``ElevationModificationGroup``, ``GroundLineModificationLayer``,
        ``ElevationControlPointLayer``) nested inside the terrain layer.
        """
        rasmap = self._locate_project_rasmap()
        tree = ET.parse(rasmap)
        root = tree.getroot()
        terrains_el = root.find("Terrains")
        if terrains_el is None:
            return []
        layers = []
        for el in terrains_el.findall("Layer[@Type='TerrainLayer']"):
            name = el.get("Name", "")
            type_ = el.get("Type", "")
            filename_str = el.get("Filename", "")
            filename = (
                (rasmap.parent / filename_str).resolve() if filename_str else Path()
            )
            resample_el = el.find("ResampleMethod")
            resample_method = resample_el.text if resample_el is not None else None
            modifications = _parse_terrain_sublayers(el, rasmap.parent)
            layers.append(
                TerrainLayer(
                    name=name,
                    type=type_,
                    filename=filename,
                    resample_method=resample_method,
                    modifications=modifications,
                )
            )
        return layers

    def get_plan_terrain(self) -> TerrainLayer:
        """Return the :class:`TerrainLayer` associated with the current plan.

        The terrain name is read from the ``Geometry`` group attribute
        ``Terrain Layername`` in the geometry HDF file, then matched against
        the ``<Terrains>`` section of the ``.rasmap`` file.

        Raises:
            FileNotFoundError: if the geometry HDF file does not exist.
            KeyError: if the HDF has no ``Terrain Layername`` attribute, or the
                name does not match any layer in ``<Terrains>``.
        """
        import h5py

        hdf_path = Path(str(self.geom_file) + ".hdf")
        if not hdf_path.exists():
            raise FileNotFoundError(
                f"Geometry HDF file not found: {hdf_path}"
            )
        with h5py.File(hdf_path, "r") as f:
            geom = f.get("Geometry")
            if geom is None or "Terrain Layername" not in geom.attrs:
                raise KeyError(
                    f"No 'Terrain Layername' attribute in Geometry group of {hdf_path}"
                )
            terrain_name = geom.attrs["Terrain Layername"]
            if isinstance(terrain_name, bytes):
                terrain_name = terrain_name.decode()

        layers = self._terrain_layer()
        for layer in layers:
            if layer.name == terrain_name:
                return layer
        available = [lay.name for lay in layers]
        raise KeyError(
            f"Terrain {terrain_name!r} (from plan HDF) not found in .rasmap. "
            f"Available: {available}"
        )

    def export_plan_terrain(
        self,
        raster_path: "str | Path",
        copy: bool = False,
    ) -> Path:
        """Export the terrain used by the current plan to a GeoTIFF or GDAL VRT.

        Identifies the terrain HDF from :meth:`get_plan_terrain`, mosaics all
        source GeoTIFFs by priority order, applies any ``Levee``- or
        ``Channel``-type ground-line modifications stored in the same HDF,
        and writes the result to *raster_path*.

        When *raster_path* has a ``.vrt`` extension the output is a GDAL VRT
        referencing the original source TIFFs.  If modifications are present
        a sidecar ``<stem>_mods.tif`` is written beside the VRT.

        Parameters
        ----------
        raster_path:
            Destination path (e.g. ``"terrain.tif"`` or ``"terrain.vrt"``).
            Parent directories are created automatically.
        copy:
            Only relevant for ``.vrt`` output.  When ``True``, source TIFFs
            are copied into the VRT's parent directory and the VRT uses
            relative paths.  When ``False`` (default) the VRT uses absolute
            paths and no files are copied.

        Returns
        -------
        Path
            Resolved absolute path of the written file.

        Raises
        ------
        FileNotFoundError
            If the geometry HDF, terrain HDF, or any source TIFF is missing.
        KeyError
            If the geometry HDF has no ``Terrain Layername`` attribute, the
            terrain name is not in the ``.rasmap`` file, or the terrain HDF
            has no source TIFF entries.
        """
        from ..hdf._terrain import export_terrain

        terrain_layer = self.get_plan_terrain()
        return export_terrain(terrain_layer.filename, raster_path, copy=copy)

    def _locate_project_rasmap(self) -> Path:
        """Locate the ``.rasmap`` file for the current project.

        Prefers ``{project_file}.rasmap``; falls back to any single ``.rasmap``
        file found in the plan directory.

        Raises
        ------
        FileNotFoundError
            If no ``.rasmap`` file is found, or if multiple candidates exist
            and the preferred name is absent.
        """
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

    @log_call(logging.INFO)
    @timed()
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
        output_path: "Path | str | None" = None,
        render_mode: Literal["sloping", "hybrid", "horizontal"] | None = None,
        use_depth_weights: bool = False,
        shallow_to_flat: bool = False,
        tight_extent: bool = True,
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "VrtMap":
        """Store one hydraulic result map via RasMapperStoreMap.exe or RasProcess.exe.

        Returns a :class:`VrtMap` handle to the written VRT and source tiles.

        Parameters
        ----------
        variable:
            Hydraulic variable to export.  Accepted values and their aliases:
            ``"wse"`` / ``"water_surface"``, ``"depth"``, ``"velocity"``,
            ``"froude"``, ``"shear_stress"``, ``"dv"`` / ``"depth_x_velocity"``,
            ``"dv2"`` / ``"depth_x_velocity_sq"``.
        timestep:
            Zero-based index into the plan's output time series.  ``None``
            exports the maximum-value profile across all timesteps.
        raster_name:
            Stem of the output VRT file (no path separators, no extension).
            Defaults to ``"{display_name} ({profile_name})"``.
        output_path:
            Directory to write the VRT into.  Must already exist.  When
            ``None``, uses the **StoreAllMaps** strategy and output goes to
            ``{project_dir}/{plan_short_id}/``; when provided, uses the
            **StoreMap XML** strategy and output goes directly into that
            directory.
        render_mode:
            Water-surface interpolation mode.  ``None`` (default) delegates to
            ``RasProcess.exe`` directly with its built-in defaults (basic sloping,
            cell-corner facepoints only).  Any explicit value — ``"sloping"``,
            ``"hybrid"``, or ``"horizontal"`` — routes through
            ``RasMapperStoreMap.exe``, which properly initialises the render-mode
            state before executing the same underlying map-generation engine.
            The mode that matches RasMapper's output depends on the
            ``<RenderMode>`` configured in the project's ``.rasmap`` file.
            Requires ``RasMapperStoreMap.exe`` in ``src/raspy/bin/`` when not
            ``None``.
        use_depth_weights:
            When ``True``, face weights in the ``hybrid`` stencil are
            proportional to the face's water depth (``UseDepthWeightedFaces``).
            Only meaningful with ``render_mode="hybrid"``.
        shallow_to_flat:
            When ``True`` (default), shallow cells are rendered flat
            (``ReduceShallowToHorizontal``).  Only meaningful with
            ``render_mode="hybrid"``.
        tight_extent:
            When ``True`` (default), the output raster extent is clipped to
            the model geometry (2D flow area + cross sections + storage areas),
            pixel-aligned to the terrain grid — matching RasMapper's "Original
            Extent" behaviour.  When ``False``, output tiles cover the full
            terrain tile extent; cells outside the model are NoData.  Requires
            ``render_mode`` to be set (routes through ``RasMapperStoreMap.exe``).
        stream_output:
            When ``True`` (default) subprocess stdout/stderr are logged
            line-by-line in real time.  When ``False`` output is captured
            silently and only shown on error.
        timeout:
            Subprocess timeout in seconds.  ``None`` (default) means no limit.

        Returns
        -------
        VrtMap
            Handle to the written VRT and its source raster tiles.

        Raises
        ------
        ValueError
            If *variable* is not recognised, *raster_name* is invalid, or
            *output_path* exists but is not a directory.
        FileNotFoundError
            If HEC-RAS is not installed, the plan HDF is missing, *output_path*
            does not exist, or RasProcess.exe did not produce the expected VRT.
        PermissionError
            If the output VRT is locked by another process (e.g. QGIS).
        RuntimeError
            If RasProcess.exe exits with a non-zero return code or writes
            ``"error"`` to stderr.
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
        if raster_name is not None and Path(raster_name).parent != Path("."):
            raise ValueError(
                f"raster_name must be a plain filename with no directory component,"
                f" got: {raster_name!r}"
            )

        if raster_name is not None and Path(raster_name).suffix:
            raise ValueError(
                f"raster_name must not include a file extension, got: {raster_name!r}"
            )

        # -- Common setup --
        plan_short_id = self.plan.short_id
        if not plan_short_id:
            raise ValueError(
                "self.plan.short_id is empty; set a plan short id before storing maps"
            )

        project_dir = _resolve(self.plan_file.parent)

        program_dir = _resolve(Path(installed_ras_directory(self.version)))
        if not program_dir:
            raise FileNotFoundError(
                f"Could not find installed HEC-RAS directory for version {self.version}"
            )
        ras_process = Path(program_dir) / "RasProcess.exe"
        if not ras_process.exists():
            raise FileNotFoundError(f"RasProcess.exe not found: {ras_process}")

        if render_mode is not None:
            _VALID_RENDER_MODES = {"sloping", "hybrid", "horizontal"}
            if render_mode not in _VALID_RENDER_MODES:
                raise ValueError(
                    f"render_mode must be one of {sorted(_VALID_RENDER_MODES)}, "
                    f"got: {render_mode!r}"
                )
            if render_mode != "hybrid":
                # Program.cs passes these to SetHorizontalRenderingMode() or
                # SetSlopingRenderingMode(), neither of which accepts parameters
                # — RasMapperLib would silently ignore them.  Force False here
                # so the raster name and logs accurately reflect what is rendered.
                use_depth_weights = False
                shallow_to_flat = False

        # None  → RasProcess.exe (its built-in defaults, no SharedData init)
        # str   → RasMapperStoreMap.exe (properly initialises SharedData first)
        if render_mode is None:
            logger.warning(
                "render_mode=None: using RasProcess.exe, which never initialises "
                "SharedData render-mode state.  The output raster will use "
                "RasMapperLib defaults (sloping/JustFacepoints) regardless of "
                "the <RenderMode> setting in the .rasmap file.  Pass "
                "render_mode='sloping', 'horizontal', or 'hybrid' to get "
                "deterministic, fully-specified results."
            )
        _use_stub = render_mode is not None
        _stub_exe = Path(__file__).parent.parent / "bin" / "RasMapperStoreMap.exe"
        if _use_stub and not _stub_exe.exists():
            raise FileNotFoundError(
                f"RasMapperStoreMap.exe not found at {_stub_exe}.\n"
                "Build tools/RasMapperStoreMap with 'dotnet build -c Release' "
                "and copy RasMapperStoreMap.exe to src/raspy/bin/."
            )

        result_hdf = _resolve(self.plan_hdf_file)
        if not result_hdf.exists():
            raise FileNotFoundError(f"Plan HDF file not found: {result_hdf}")

        profile_name = self.timestep_to_profile_name(timestep)
        profile_index = 2147483647 if timestep is None else timestep
        safe_profile = profile_name.replace(":", " ").replace("/", "_")

        custom_raster_name = raster_name  # None if caller did not supply one
        if raster_name is None:
            raster_name = f"{display_name} ({safe_profile})"

        rasmap_src = _resolve(self._locate_project_rasmap())

        # StoreAllMaps: By default it outputs rasters to relative 'project_dir / plan_short_id'
        # Here we always use absolute path in OverwriteFilname to reduce complexity
        if output_path is None:
            abs_output_dir = project_dir / plan_short_id
            abs_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            abs_output_dir = _resolve(Path(output_path))

            if abs_output_dir.is_file() or abs_output_dir.suffix:
                raise ValueError(
                    f"output_path must be a directory, not a file: {abs_output_dir}"
                )
            if not abs_output_dir.exists():
                raise FileNotFoundError(
                    f"output_path does not exist: {abs_output_dir}"
                )

        abs_output_filename_wo_ext = abs_output_dir / raster_name
        abs_output_filename_w_ext = abs_output_dir / f"{raster_name}.vrt"

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
            params.set("OverwriteOutputFilename", str(abs_output_filename_wo_ext))

            ET.indent(root, space="  ")
            tree.write(temp_rasmap, encoding="utf-8", xml_declaration=False)

            logger.info("StoreAllMaps: output expected at %s", abs_output_filename_w_ext)

            if abs_output_filename_w_ext.exists() and VrtMap(abs_output_filename_w_ext).is_locked():
                raise PermissionError(
                    f"Output file is locked by another process (e.g. QGIS): "
                    f"{abs_output_filename_w_ext}\nClose the file before calling store_map."
                )

            if _use_stub:
                # ── Portability warnings ──────────────────────────────────────
                # RasMapperStoreMap.exe requires HEC-RAS 6.1+.
                #   • StoreAllMapsCommand lives in RasMapperLib.Scripting, a
                #     namespace that did not exist before 6.x.
                #   • The candidate auto-discovery paths only cover 6.1–6.6;
                #     6.0 and 5.x must supply -RasMapperLibDir explicitly and
                #     will still likely fail at the StoreAllMapsCommand lookup.
                if self.version < 6100:
                    logger.warning(
                        "RasMapperStoreMap.exe requires HEC-RAS 6.1+. "
                        "Version %s may not have StoreAllMapsCommand in "
                        "RasMapperLib.Scripting — the stub is likely to fail.",
                        self.version,
                    )
                # SetSlopingPrettyRenderingMode was added to SharedData in
                # roughly HEC-RAS 6.3.  Earlier 6.x builds have
                # SetSlopingRenderingMode and SetHorizontalRenderingMode but
                # not the three-mode pretty variant.
                if render_mode == "hybrid" and self.version < 6300:
                    logger.warning(
                        "render_mode='hybrid' calls "
                        "SharedData.SetSlopingPrettyRenderingMode(), which may "
                        "not exist in HEC-RAS %s (introduced ~6.3). "
                        "The stub will raise InvalidOperationException if the "
                        "method is absent.",
                        self.version,
                    )
                # ConsoleProgressReporter lives in Utility.Core.dll.  In some
                # older HEC-RAS installs the utility assembly is named
                # differently, so the stub falls back to ProgressReporter.None()
                # and progress messages will not appear on stdout.
                utility_core = Path(program_dir) / "Utility.Core.dll"
                if not utility_core.exists():
                    logger.warning(
                        "Utility.Core.dll not found in %s. "
                        "Progress messages from RasMapperStoreMap.exe will not "
                        "appear on stdout (stub falls back to no-op reporter).",
                        program_dir,
                    )
                # Translate raspy's public name back to the RasMapperLib string.
                _render_mode_arg = (
                    "slopingPretty" if render_mode == "hybrid" else render_mode
                )
                cmd = [
                    str(_stub_exe),
                    f"-RasMapFilename={temp_rasmap}",
                    f"-ResultFilename={result_hdf}",
                    f"-RenderMode={_render_mode_arg}",
                    f"-UseDepthWeightedFaces="
                    f"{'true' if use_depth_weights else 'false'}",
                    f"-ReduceShallowToHorizontal="
                    f"{'true' if shallow_to_flat else 'false'}",
                    f"-TightExtent="
                    f"{'true' if tight_extent else 'false'}",
                    f"-RasMapperLibDir={program_dir}",
                ]
            else:
                cmd = [
                    str(ras_process),
                    "-Command=StoreAllMaps",
                    f"-RasMapFilename={temp_rasmap}",
                    f"-ResultFilename={result_hdf}",
                ]
            # Run from the HEC-RAS install directory so Windows finds native
            # GDAL DLLs via the default DLL search path — the same CWD that
            # RasProcess.exe uses.  Without this, P/Invoke calls inside
            # RASResultsMap.StoreMap() can fail with NullReferenceException
            # when processing terrain tiles that trigger GDAL native operations.
            _cwd = program_dir if _use_stub else None

            # Retry the stub on transient .NET exceptions (returncode=0 but
            # a NullReferenceException or similar appears in stderr).  The
            # root cause is concurrent HDF5 access: HEC-RAS COM keeps the
            # plan HDF file open while MapProcessingEngine.StoreMap() runs
            # multi-threaded reads on the same file.  Retrying after a short
            # pause lets the competing HDF5 operation finish.
            #
            # Permanent RasMapperLib errors (e.g. "Error loading facepoint
            # elevations") do NOT contain "Exception" and are not retried.
            # Hard failures (non-zero returncode) are never retried.
            _max_attempts = 3 if _use_stub else 1
            for _attempt in range(1, _max_attempts + 1):
                if _use_stub:
                    logger.debug(
                        "RasMapperStoreMap.exe attempt %d/%d",
                        _attempt,
                        _max_attempts,
                    )
                result = _run_subprocess(cmd, _cwd, timeout, stream_output)
                _is_transient = (
                    result.returncode == 0
                    and "exception" in result.stderr.lower()
                )
                if _is_transient and _attempt < _max_attempts:
                    logger.warning(
                        "RasMapperStoreMap.exe transient error on attempt %d/%d "
                        "(likely concurrent HDF5 access) — retrying in 2 s...",
                        _attempt,
                        _max_attempts,
                    )
                    time.sleep(2)
                    abs_output_filename_w_ext.unlink(missing_ok=True)
                else:
                    break
        finally:
            temp_rasmap.unlink(missing_ok=True)

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

        vrt = VrtMap(abs_output_filename_w_ext)
        if not vrt.exists():
            raise FileNotFoundError(
                f"RasProcess completed but output VRT was not found: {abs_output_filename_w_ext}\n"
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

        logger.info(
            "store_map: folder %s vrt %s",
            vrt.path.parent.as_uri(),
            vrt.path.as_uri(),
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
        timestep: int | None,
        render_mode: (
            Literal["sloping", "hybrid", "horizontal"] | None
        ) = "horizontal",
        use_depth_weights: bool = False,
        shallow_to_flat: bool = False,
        tight_extent: bool = True,
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> Generator["rasterio.io.DatasetReader", None, None]:
        """Store a map in a temporary directory and yield an open
        ``rasterio.DatasetReader``.

        Output is written to a freshly-created system temp directory that is
        removed with :func:`shutil.rmtree` on context exit and registered with
        :func:`atexit` for cleanup on interpreter shutdown.

        Parameters
        ----------
        variable:
            Hydraulic variable — same accepted values as :meth:`store_map`.
        timestep:
            Zero-based timestep index.  ``None`` uses the maximum-value profile.
        render_mode:
            Passed through to :meth:`store_map`.  Defaults to ``"horizontal"``.
        use_depth_weights:
            Passed through to :meth:`store_map`.
        shallow_to_flat:
            Passed through to :meth:`store_map`.
        tight_extent:
            Passed through to :meth:`store_map`.
        stream_output:
            Passed through to :meth:`store_map`.
        timeout:
            Subprocess timeout in seconds.  ``None`` means no limit.

        Yields
        ------
        rasterio.io.DatasetReader
            Open dataset positioned at the exported VRT.

        Usage::

            with model.open_map("wse", 10) as ds:
                data = ds.read(1)
        """
        import secrets

        import rasterio

        out_dir = Path(tempfile.mkdtemp())
        raster_name = secrets.token_hex(8)
        logger.debug("open_map: temp dir %s, raster name %s", out_dir, raster_name)
        _temp_dirs.add(out_dir)
        try:
            vrt = self.store_map(
                variable,
                timestep=timestep,
                raster_name=raster_name,
                output_path=out_dir,
                render_mode=render_mode,
                use_depth_weights=use_depth_weights,
                shallow_to_flat=shallow_to_flat,
                tight_extent=tight_extent,
                stream_output=stream_output,
                timeout=timeout,
            )
            logger.debug(
                "open_map: folder %s created %s",
                vrt.path.parent.as_uri(),
                vrt.path.as_uri(),
            )
            with rasterio.open(vrt.path) as ds:
                yield ds
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
            _temp_dirs.discard(out_dir)

    @log_call(logging.INFO)
    def export_wse(
        self,
        timestep: int | None,
        output_vrt: "str | Path | None" = None,
        render_mode: (
            Literal["sloping", "hybrid", "horizontal"] | None
        ) = "horizontal",
        use_depth_weights: bool = False,
        shallow_to_flat: bool = False,
        tight_extent: bool = True,
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "VrtMap":
        """Export a water-surface elevation (WSE) raster.

        Parameters
        ----------
        timestep:
            Zero-based timestep index.  ``None`` exports the maximum profile.
        output_vrt:
            Destination for the exported VRT.  Three forms accepted:
            ``None`` — written to ``{project_dir}/{plan_short_id}/`` via the
            StoreAllMaps strategy; an existing directory — written into that
            folder with an auto-generated name; a path ending with ``.vrt`` —
            written to that exact file.
        render_mode:
            Water-surface interpolation mode passed to :meth:`store_map`.
            ``"horizontal"`` (default) renders a flat per-cell water surface.
            ``"sloping"`` uses cell-corner facepoints only.  ``"hybrid"`` adds
            face centroids with optional depth-weighted interpolation.  Any
            explicit value routes through ``RasMapperStoreMap.exe``; the mode
            that matches RasMapper's output depends on the project's
            ``.rasmap`` setting.
        use_depth_weights:
            When ``True``, face weights are depth-proportional.  Only
            meaningful with ``render_mode="hybrid"``.
        shallow_to_flat:
            When ``True``, shallow cells are rendered flat.  Defaults to
            ``False``.  Only meaningful with ``render_mode="hybrid"``.
        tight_extent:
            Passed through to :meth:`store_map`.
        stream_output:
            Stream subprocess output to the logger in real time.
        timeout:
            Subprocess timeout in seconds.  ``None`` means no limit.

        Raises
        ------
        ValueError
            If *output_vrt* is not ``None``, not an existing directory, and
            does not end with ``.vrt``.
        """
        if output_vrt is None:
            return self.store_map(
                "wse",
                timestep=timestep,
                render_mode=render_mode,
                use_depth_weights=use_depth_weights,
                shallow_to_flat=shallow_to_flat,
                tight_extent=tight_extent,
                stream_output=stream_output,
                timeout=timeout,
            )
        output_vrt = Path(output_vrt)
        if output_vrt.is_dir():
            return self.store_map(
                "wse",
                timestep=timestep,
                output_path=output_vrt,
                render_mode=render_mode,
                use_depth_weights=use_depth_weights,
                shallow_to_flat=shallow_to_flat,
                tight_extent=tight_extent,
                stream_output=stream_output,
                timeout=timeout,
            )
        if output_vrt.suffix != ".vrt":
            raise ValueError(
                f"output_vrt must be a .vrt file path or an existing directory,"
                f" got: {output_vrt}"
            )
        return self.store_map(
            "wse",
            timestep=timestep,
            raster_name=output_vrt.stem,
            output_path=output_vrt.parent,
            render_mode=render_mode,
            use_depth_weights=use_depth_weights,
            shallow_to_flat=shallow_to_flat,
            tight_extent=tight_extent,
            stream_output=stream_output,
            timeout=timeout,
        )

    @log_call(logging.INFO)
    def open_wse(
        self,
        timestep: int | None,
        render_mode: (
            Literal["sloping", "hybrid", "horizontal"] | None
        ) = "horizontal",
        use_depth_weights: bool = False,
        shallow_to_flat: bool = False,
        tight_extent: bool = True,
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "AbstractContextManager[rasterio.io.DatasetReader]":
        """Context manager: export WSE and yield an open rasterio dataset.

        See :meth:`open_map` for parameter and cleanup details.
        """
        return self.open_map(
            "wse",
            timestep,
            render_mode=render_mode,
            use_depth_weights=use_depth_weights,
            shallow_to_flat=shallow_to_flat,
            tight_extent=tight_extent,
            stream_output=stream_output,
            timeout=timeout,
        )

    @log_call(logging.INFO)
    def export_depth(
        self,
        timestep: int | None,
        output_vrt: "str | Path | None" = None,
        render_mode: (
            Literal["sloping", "hybrid", "horizontal"] | None
        ) = "horizontal",
        use_depth_weights: bool = False,
        shallow_to_flat: bool = False,
        tight_extent: bool = True,
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "VrtMap":
        """Export a water depth raster.

        Parameters
        ----------
        timestep:
            Zero-based timestep index.  ``None`` exports the maximum profile.
        output_vrt:
            Destination for the exported VRT.  Three forms accepted:
            ``None`` — written to ``{project_dir}/{plan_short_id}/`` via the
            StoreAllMaps strategy; an existing directory — written into that
            folder with an auto-generated name; a path ending with ``.vrt`` —
            written to that exact file.
        render_mode:
            Water-surface interpolation mode passed to :meth:`store_map`.
            ``"horizontal"`` (default) renders a flat per-cell water surface.
            ``"sloping"`` uses cell-corner facepoints only.  ``"hybrid"`` adds
            face centroids with optional depth-weighted interpolation.  Any
            explicit value routes through ``RasMapperStoreMap.exe``; the mode
            that matches RasMapper's output depends on the project's
            ``.rasmap`` setting.
        use_depth_weights:
            When ``True``, face weights are depth-proportional.  Only
            meaningful with ``render_mode="hybrid"``.
        shallow_to_flat:
            When ``True``, shallow cells are rendered flat.  Defaults to
            ``False``.  Only meaningful with ``render_mode="hybrid"``.
        tight_extent:
            Passed through to :meth:`store_map`.
        stream_output:
            Stream subprocess output to the logger in real time.
        timeout:
            Subprocess timeout in seconds.  ``None`` means no limit.

        Raises
        ------
        ValueError
            If *output_vrt* is not ``None``, not an existing directory, and
            does not end with ``.vrt``.
        """
        if output_vrt is None:
            return self.store_map(
                "depth",
                timestep=timestep,
                render_mode=render_mode,
                use_depth_weights=use_depth_weights,
                shallow_to_flat=shallow_to_flat,
                tight_extent=tight_extent,
                stream_output=stream_output,
                timeout=timeout,
            )
        output_vrt = Path(output_vrt)
        if output_vrt.is_dir():
            return self.store_map(
                "depth",
                timestep=timestep,
                output_path=output_vrt,
                render_mode=render_mode,
                use_depth_weights=use_depth_weights,
                shallow_to_flat=shallow_to_flat,
                tight_extent=tight_extent,
                stream_output=stream_output,
                timeout=timeout,
            )
        if output_vrt.suffix != ".vrt":
            raise ValueError(
                f"output_vrt must be a .vrt file path or an existing directory,"
                f" got: {output_vrt}"
            )
        return self.store_map(
            "depth",
            timestep=timestep,
            raster_name=output_vrt.stem,
            output_path=output_vrt.parent,
            render_mode=render_mode,
            use_depth_weights=use_depth_weights,
            shallow_to_flat=shallow_to_flat,
            tight_extent=tight_extent,
            stream_output=stream_output,
            timeout=timeout,
        )

    @log_call(logging.INFO)
    def open_depth(
        self,
        timestep: int | None,
        render_mode: (
            Literal["sloping", "hybrid", "horizontal"] | None
        ) = "horizontal",
        use_depth_weights: bool = False,
        shallow_to_flat: bool = False,
        tight_extent: bool = True,
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "AbstractContextManager[rasterio.io.DatasetReader]":
        """Context manager: export depth and yield an open rasterio dataset.

        See :meth:`open_map` for parameter and cleanup details.
        """
        return self.open_map(
            "depth",
            timestep,
            render_mode=render_mode,
            use_depth_weights=use_depth_weights,
            shallow_to_flat=shallow_to_flat,
            tight_extent=tight_extent,
            stream_output=stream_output,
            timeout=timeout,
        )

    @log_call(logging.INFO)
    def export_velocity(
        self,
        timestep: int | None,
        output_vrt: "str | Path | None" = None,
        render_mode: (
            Literal["sloping", "hybrid", "horizontal"] | None
        ) = "horizontal",
        use_depth_weights: bool = False,
        shallow_to_flat: bool = False,
        tight_extent: bool = True,
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "VrtMap":
        """Export a depth-averaged velocity magnitude raster.

        Parameters
        ----------
        timestep:
            Zero-based timestep index.  ``None`` exports the maximum profile.
        output_vrt:
            Destination for the exported VRT.  Three forms accepted:
            ``None`` — written to ``{project_dir}/{plan_short_id}/`` via the
            StoreAllMaps strategy; an existing directory — written into that
            folder with an auto-generated name; a path ending with ``.vrt`` —
            written to that exact file.
        render_mode:
            Water-surface interpolation mode passed to :meth:`store_map`.
            ``"horizontal"`` (default) renders a flat per-cell water surface.
            ``"sloping"`` uses cell-corner facepoints only.  ``"hybrid"`` adds
            face centroids with optional depth-weighted interpolation.  Any
            explicit value routes through ``RasMapperStoreMap.exe``; the mode
            that matches RasMapper's output depends on the project's
            ``.rasmap`` setting.
        use_depth_weights:
            When ``True``, face weights are depth-proportional.  Only
            meaningful with ``render_mode="hybrid"``.
        shallow_to_flat:
            When ``True``, shallow cells are rendered flat.  Defaults to
            ``False``.  Only meaningful with ``render_mode="hybrid"``.
        tight_extent:
            Passed through to :meth:`store_map`.
        stream_output:
            Stream subprocess output to the logger in real time.
        timeout:
            Subprocess timeout in seconds.  ``None`` means no limit.

        Raises
        ------
        ValueError
            If *output_vrt* is not ``None``, not an existing directory, and
            does not end with ``.vrt``.
        """
        if output_vrt is None:
            return self.store_map(
                "velocity",
                timestep=timestep,
                render_mode=render_mode,
                use_depth_weights=use_depth_weights,
                shallow_to_flat=shallow_to_flat,
                tight_extent=tight_extent,
                stream_output=stream_output,
                timeout=timeout,
            )
        output_vrt = Path(output_vrt)
        if output_vrt.is_dir():
            return self.store_map(
                "velocity",
                timestep=timestep,
                output_path=output_vrt,
                render_mode=render_mode,
                use_depth_weights=use_depth_weights,
                shallow_to_flat=shallow_to_flat,
                tight_extent=tight_extent,
                stream_output=stream_output,
                timeout=timeout,
            )
        if output_vrt.suffix != ".vrt":
            raise ValueError(
                f"output_vrt must be a .vrt file path or an existing directory,"
                f" got: {output_vrt}"
            )
        return self.store_map(
            "velocity",
            timestep=timestep,
            raster_name=output_vrt.stem,
            output_path=output_vrt.parent,
            render_mode=render_mode,
            use_depth_weights=use_depth_weights,
            shallow_to_flat=shallow_to_flat,
            tight_extent=tight_extent,
            stream_output=stream_output,
            timeout=timeout,
        )

    @log_call(logging.INFO)
    def open_velocity(
        self,
        timestep: int | None,
        render_mode: (
            Literal["sloping", "hybrid", "horizontal"] | None
        ) = "horizontal",
        use_depth_weights: bool = False,
        shallow_to_flat: bool = False,
        tight_extent: bool = True,
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "AbstractContextManager[rasterio.io.DatasetReader]":
        """Context manager: export velocity and yield an open rasterio dataset.

        See :meth:`open_map` for parameter and cleanup details.
        """
        return self.open_map(
            "velocity",
            timestep,
            render_mode=render_mode,
            use_depth_weights=use_depth_weights,
            shallow_to_flat=shallow_to_flat,
            tight_extent=tight_extent,
            stream_output=stream_output,
            timeout=timeout,
        )

    @log_call(logging.INFO)
    def export_froude(
        self,
        timestep: int | None,
        output_vrt: "str | Path | None" = None,
        render_mode: (
            Literal["sloping", "hybrid", "horizontal"] | None
        ) = "horizontal",
        use_depth_weights: bool = False,
        shallow_to_flat: bool = False,
        tight_extent: bool = True,
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "VrtMap":
        """Export a Froude number raster.

        Parameters
        ----------
        timestep:
            Zero-based timestep index.  ``None`` exports the maximum profile.
        output_vrt:
            Destination for the exported VRT.  Three forms accepted:
            ``None`` — written to ``{project_dir}/{plan_short_id}/`` via the
            StoreAllMaps strategy; an existing directory — written into that
            folder with an auto-generated name; a path ending with ``.vrt`` —
            written to that exact file.
        render_mode:
            Water-surface interpolation mode passed to :meth:`store_map`.
            ``"horizontal"`` (default) renders a flat per-cell water surface.
            ``"sloping"`` uses cell-corner facepoints only.  ``"hybrid"`` adds
            face centroids with optional depth-weighted interpolation.  Any
            explicit value routes through ``RasMapperStoreMap.exe``; the mode
            that matches RasMapper's output depends on the project's
            ``.rasmap`` setting.
        use_depth_weights:
            When ``True``, face weights are depth-proportional.  Only
            meaningful with ``render_mode="hybrid"``.
        shallow_to_flat:
            When ``True``, shallow cells are rendered flat.  Defaults to
            ``False``.  Only meaningful with ``render_mode="hybrid"``.
        tight_extent:
            Passed through to :meth:`store_map`.
        stream_output:
            Stream subprocess output to the logger in real time.
        timeout:
            Subprocess timeout in seconds.  ``None`` means no limit.

        Raises
        ------
        ValueError
            If *output_vrt* is not ``None``, not an existing directory, and
            does not end with ``.vrt``.
        """
        if output_vrt is None:
            return self.store_map(
                "froude",
                timestep=timestep,
                render_mode=render_mode,
                use_depth_weights=use_depth_weights,
                shallow_to_flat=shallow_to_flat,
                tight_extent=tight_extent,
                stream_output=stream_output,
                timeout=timeout,
            )
        output_vrt = Path(output_vrt)
        if output_vrt.is_dir():
            return self.store_map(
                "froude",
                timestep=timestep,
                output_path=output_vrt,
                render_mode=render_mode,
                use_depth_weights=use_depth_weights,
                shallow_to_flat=shallow_to_flat,
                tight_extent=tight_extent,
                stream_output=stream_output,
                timeout=timeout,
            )
        if output_vrt.suffix != ".vrt":
            raise ValueError(
                f"output_vrt must be a .vrt file path or an existing directory,"
                f" got: {output_vrt}"
            )
        return self.store_map(
            "froude",
            timestep=timestep,
            raster_name=output_vrt.stem,
            output_path=output_vrt.parent,
            render_mode=render_mode,
            use_depth_weights=use_depth_weights,
            shallow_to_flat=shallow_to_flat,
            tight_extent=tight_extent,
            stream_output=stream_output,
            timeout=timeout,
        )

    @log_call(logging.INFO)
    def open_froude(
        self,
        timestep: int | None,
        render_mode: (
            Literal["sloping", "hybrid", "horizontal"] | None
        ) = "horizontal",
        use_depth_weights: bool = False,
        shallow_to_flat: bool = False,
        tight_extent: bool = True,
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "AbstractContextManager[rasterio.io.DatasetReader]":
        """Context manager: export Froude number and yield an open rasterio dataset.

        See :meth:`open_map` for parameter and cleanup details.
        """
        return self.open_map(
            "froude",
            timestep,
            render_mode=render_mode,
            use_depth_weights=use_depth_weights,
            shallow_to_flat=shallow_to_flat,
            tight_extent=tight_extent,
            stream_output=stream_output,
            timeout=timeout,
        )

    @log_call(logging.INFO)
    def export_shear_stress(
        self,
        timestep: int | None,
        output_vrt: "str | Path | None" = None,
        render_mode: (
            Literal["sloping", "hybrid", "horizontal"] | None
        ) = "horizontal",
        use_depth_weights: bool = False,
        shallow_to_flat: bool = False,
        tight_extent: bool = True,
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "VrtMap":
        """Export a bed shear stress raster.

        Parameters
        ----------
        timestep:
            Zero-based timestep index.  ``None`` exports the maximum profile.
        output_vrt:
            Destination for the exported VRT.  Three forms accepted:
            ``None`` — written to ``{project_dir}/{plan_short_id}/`` via the
            StoreAllMaps strategy; an existing directory — written into that
            folder with an auto-generated name; a path ending with ``.vrt`` —
            written to that exact file.
        render_mode:
            Water-surface interpolation mode passed to :meth:`store_map`.
            ``"horizontal"`` (default) renders a flat per-cell water surface.
            ``"sloping"`` uses cell-corner facepoints only.  ``"hybrid"`` adds
            face centroids with optional depth-weighted interpolation.  Any
            explicit value routes through ``RasMapperStoreMap.exe``; the mode
            that matches RasMapper's output depends on the project's
            ``.rasmap`` setting.
        use_depth_weights:
            When ``True``, face weights are depth-proportional.  Only
            meaningful with ``render_mode="hybrid"``.
        shallow_to_flat:
            When ``True``, shallow cells are rendered flat.  Defaults to
            ``False``.  Only meaningful with ``render_mode="hybrid"``.
        tight_extent:
            Passed through to :meth:`store_map`.
        stream_output:
            Stream subprocess output to the logger in real time.
        timeout:
            Subprocess timeout in seconds.  ``None`` means no limit.

        Raises
        ------
        ValueError
            If *output_vrt* is not ``None``, not an existing directory, and
            does not end with ``.vrt``.
        """
        if output_vrt is None:
            return self.store_map(
                "shear_stress",
                timestep=timestep,
                render_mode=render_mode,
                use_depth_weights=use_depth_weights,
                shallow_to_flat=shallow_to_flat,
                tight_extent=tight_extent,
                stream_output=stream_output,
                timeout=timeout,
            )
        output_vrt = Path(output_vrt)
        if output_vrt.is_dir():
            return self.store_map(
                "shear_stress",
                timestep=timestep,
                output_path=output_vrt,
                render_mode=render_mode,
                use_depth_weights=use_depth_weights,
                shallow_to_flat=shallow_to_flat,
                tight_extent=tight_extent,
                stream_output=stream_output,
                timeout=timeout,
            )
        if output_vrt.suffix != ".vrt":
            raise ValueError(
                f"output_vrt must be a .vrt file path or an existing directory,"
                f" got: {output_vrt}"
            )
        return self.store_map(
            "shear_stress",
            timestep=timestep,
            raster_name=output_vrt.stem,
            output_path=output_vrt.parent,
            render_mode=render_mode,
            use_depth_weights=use_depth_weights,
            shallow_to_flat=shallow_to_flat,
            tight_extent=tight_extent,
            stream_output=stream_output,
            timeout=timeout,
        )

    @log_call(logging.INFO)
    def open_shear_stress(
        self,
        timestep: int | None,
        render_mode: (
            Literal["sloping", "hybrid", "horizontal"] | None
        ) = "horizontal",
        use_depth_weights: bool = False,
        shallow_to_flat: bool = False,
        tight_extent: bool = True,
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "AbstractContextManager[rasterio.io.DatasetReader]":
        """Context manager: export shear stress and yield an open rasterio dataset.

        See :meth:`open_map` for parameter and cleanup details.
        """
        return self.open_map(
            "shear_stress",
            timestep,
            render_mode=render_mode,
            use_depth_weights=use_depth_weights,
            shallow_to_flat=shallow_to_flat,
            tight_extent=tight_extent,
            stream_output=stream_output,
            timeout=timeout,
        )

    @log_call(logging.INFO)
    def export_dv(
        self,
        timestep: int | None,
        output_vrt: "str | Path | None" = None,
        render_mode: (
            Literal["sloping", "hybrid", "horizontal"] | None
        ) = "horizontal",
        use_depth_weights: bool = False,
        shallow_to_flat: bool = False,
        tight_extent: bool = True,
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "VrtMap":
        """Export a depth × velocity (D×V) raster.

        Parameters
        ----------
        timestep:
            Zero-based timestep index.  ``None`` exports the maximum profile.
        output_vrt:
            Destination for the exported VRT.  Three forms accepted:
            ``None`` — written to ``{project_dir}/{plan_short_id}/`` via the
            StoreAllMaps strategy; an existing directory — written into that
            folder with an auto-generated name; a path ending with ``.vrt`` —
            written to that exact file.
        render_mode:
            Water-surface interpolation mode passed to :meth:`store_map`.
            ``"horizontal"`` (default) renders a flat per-cell water surface.
            ``"sloping"`` uses cell-corner facepoints only.  ``"hybrid"`` adds
            face centroids with optional depth-weighted interpolation.  Any
            explicit value routes through ``RasMapperStoreMap.exe``; the mode
            that matches RasMapper's output depends on the project's
            ``.rasmap`` setting.
        use_depth_weights:
            When ``True``, face weights are depth-proportional.  Only
            meaningful with ``render_mode="hybrid"``.
        shallow_to_flat:
            When ``True``, shallow cells are rendered flat.  Defaults to
            ``False``.  Only meaningful with ``render_mode="hybrid"``.
        tight_extent:
            Passed through to :meth:`store_map`.
        stream_output:
            Stream subprocess output to the logger in real time.
        timeout:
            Subprocess timeout in seconds.  ``None`` means no limit.

        Raises
        ------
        ValueError
            If *output_vrt* is not ``None``, not an existing directory, and
            does not end with ``.vrt``.
        """
        if output_vrt is None:
            return self.store_map(
                "dv",
                timestep=timestep,
                render_mode=render_mode,
                use_depth_weights=use_depth_weights,
                shallow_to_flat=shallow_to_flat,
                tight_extent=tight_extent,
                stream_output=stream_output,
                timeout=timeout,
            )
        output_vrt = Path(output_vrt)
        if output_vrt.is_dir():
            return self.store_map(
                "dv",
                timestep=timestep,
                output_path=output_vrt,
                render_mode=render_mode,
                use_depth_weights=use_depth_weights,
                shallow_to_flat=shallow_to_flat,
                tight_extent=tight_extent,
                stream_output=stream_output,
                timeout=timeout,
            )
        if output_vrt.suffix != ".vrt":
            raise ValueError(
                f"output_vrt must be a .vrt file path or an existing directory,"
                f" got: {output_vrt}"
            )
        return self.store_map(
            "dv",
            timestep=timestep,
            raster_name=output_vrt.stem,
            output_path=output_vrt.parent,
            render_mode=render_mode,
            use_depth_weights=use_depth_weights,
            shallow_to_flat=shallow_to_flat,
            tight_extent=tight_extent,
            stream_output=stream_output,
            timeout=timeout,
        )

    @log_call(logging.INFO)
    def open_dv(
        self,
        timestep: int | None,
        render_mode: (
            Literal["sloping", "hybrid", "horizontal"] | None
        ) = "horizontal",
        use_depth_weights: bool = False,
        shallow_to_flat: bool = False,
        tight_extent: bool = True,
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "AbstractContextManager[rasterio.io.DatasetReader]":
        """Context manager: export D×V and yield an open rasterio dataset.

        See :meth:`open_map` for parameter and cleanup details.
        """
        return self.open_map(
            "dv",
            timestep,
            render_mode=render_mode,
            use_depth_weights=use_depth_weights,
            shallow_to_flat=shallow_to_flat,
            tight_extent=tight_extent,
            stream_output=stream_output,
            timeout=timeout,
        )

    @log_call(logging.INFO)
    def export_dv2(
        self,
        timestep: int | None,
        output_vrt: "str | Path | None" = None,
        render_mode: (
            Literal["sloping", "hybrid", "horizontal"] | None
        ) = "horizontal",
        use_depth_weights: bool = False,
        shallow_to_flat: bool = False,
        tight_extent: bool = True,
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "VrtMap":
        """Export a depth × velocity² (D×V²) raster.

        Parameters
        ----------
        timestep:
            Zero-based timestep index.  ``None`` exports the maximum profile.
        output_vrt:
            Destination for the exported VRT.  Three forms accepted:
            ``None`` — written to ``{project_dir}/{plan_short_id}/`` via the
            StoreAllMaps strategy; an existing directory — written into that
            folder with an auto-generated name; a path ending with ``.vrt`` —
            written to that exact file.
        render_mode:
            Water-surface interpolation mode passed to :meth:`store_map`.
            ``"horizontal"`` (default) renders a flat per-cell water surface.
            ``"sloping"`` uses cell-corner facepoints only.  ``"hybrid"`` adds
            face centroids with optional depth-weighted interpolation.  Any
            explicit value routes through ``RasMapperStoreMap.exe``; the mode
            that matches RasMapper's output depends on the project's
            ``.rasmap`` setting.
        use_depth_weights:
            When ``True``, face weights are depth-proportional.  Only
            meaningful with ``render_mode="hybrid"``.
        shallow_to_flat:
            When ``True``, shallow cells are rendered flat.  Defaults to
            ``False``.  Only meaningful with ``render_mode="hybrid"``.
        tight_extent:
            Passed through to :meth:`store_map`.
        stream_output:
            Stream subprocess output to the logger in real time.
        timeout:
            Subprocess timeout in seconds.  ``None`` means no limit.

        Raises
        ------
        ValueError
            If *output_vrt* is not ``None``, not an existing directory, and
            does not end with ``.vrt``.
        """
        if output_vrt is None:
            return self.store_map(
                "dv2",
                timestep=timestep,
                render_mode=render_mode,
                use_depth_weights=use_depth_weights,
                shallow_to_flat=shallow_to_flat,
                tight_extent=tight_extent,
                stream_output=stream_output,
                timeout=timeout,
            )
        output_vrt = Path(output_vrt)
        if output_vrt.is_dir():
            return self.store_map(
                "dv2",
                timestep=timestep,
                output_path=output_vrt,
                render_mode=render_mode,
                use_depth_weights=use_depth_weights,
                shallow_to_flat=shallow_to_flat,
                tight_extent=tight_extent,
                stream_output=stream_output,
                timeout=timeout,
            )
        if output_vrt.suffix != ".vrt":
            raise ValueError(
                f"output_vrt must be a .vrt file path or an existing directory,"
                f" got: {output_vrt}"
            )
        return self.store_map(
            "dv2",
            timestep=timestep,
            raster_name=output_vrt.stem,
            output_path=output_vrt.parent,
            render_mode=render_mode,
            use_depth_weights=use_depth_weights,
            shallow_to_flat=shallow_to_flat,
            tight_extent=tight_extent,
            stream_output=stream_output,
            timeout=timeout,
        )

    @log_call(logging.INFO)
    def open_dv2(
        self,
        timestep: int | None,
        render_mode: (
            Literal["sloping", "hybrid", "horizontal"] | None
        ) = "horizontal",
        use_depth_weights: bool = False,
        shallow_to_flat: bool = False,
        tight_extent: bool = True,
        stream_output: bool = True,
        timeout: int | None = None,
    ) -> "AbstractContextManager[rasterio.io.DatasetReader]":
        """Context manager: export D×V² and yield an open rasterio dataset.

        See :meth:`open_map` for parameter and cleanup details.
        """
        return self.open_map(
            "dv2",
            timestep,
            render_mode=render_mode,
            use_depth_weights=use_depth_weights,
            shallow_to_flat=shallow_to_flat,
            tight_extent=tight_extent,
            stream_output=stream_output,
            timeout=timeout,
        )
