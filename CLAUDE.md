# raspy — HEC-RAS Python Interface Library
# Status: ACTIVE DEVELOPMENT

## Project Overview
`raspy` is a modern, modular Python library for interacting with HEC-RAS hydraulic
modeling software. It replaces two legacy internal libraries (`pyras` and `ras_tools`)
currently in `archive/`. Goal: clean public PyPI package with a well-designed API.

- Platform: Windows-only (HEC-RAS is Windows-only)
- Python: 3.10+
- Packaging: pyproject.toml
- Distribution: Public PyPI package (design API with external users in mind)

## Architecture

```
raspy/
├── com/       # COM interface to run/control HEC-RAS (derived from archive/pyras)
├── model/     # Read/write HEC-RAS text files (.prj, .g*, .p*, .f*, .u*)
├── hdf/       # Read HEC-RAS HDF5 files (geometry + plan results)
├── geo/       # Geospatial operations: interpolation, raster export (geopandas/rasterio only here)
└── utils/     # Shared helpers (path handling, validation, logging)
```

Archive reference:
- `archive/pyras/`    → informs `com/`
- `archive/ras_tools/` → informs `model/`, `hdf/`, `geo/`

## What Has Been Built

### `com/` — COM Controller
Files: `controller.py`, `ras.py`, `registry.py`, `_geometry.py`, `_runtime.py`, `_ver400.py`, `_ver500.py`, `_ver503.py`
- `RasController` — connects to HEC-RAS via COM, opens/runs projects
- Version-specific bindings for HEC-RAS 4.x, 5.0, 5.03
- Registry helpers for locating HEC-RAS installations

### `model/` — Text File I/O
Files: `project.py`, `plan.py`, `geometry.py`, `flow_steady.py`, `flow_unsteady.py`, `_mapper.py`
- `PlanFile` — plan file (.p**) reader/writer
- `GeometryFile` — geometry file (.g**) reader/writer; verbatim-line editor pattern
  - `get_cross_section(river, reach, rs)` → `CrossSection` dataclass
  - `cross_sections(river, reach)` → list all XS in a reach
  - `set_mannings`, `set_stations`, `set_bank_stations`, `set_exp_cntr` — splice fixed-width blocks
  - Node types: XS=1, Culvert=2, Bridge=3, MultipleOpening=4, InlineStructure=5, LateralStructure=6
  - Structure nodes (type 2–6) stored verbatim — not parsed
- `SteadyFlowFile`, `UnsteadyFlowFile` — flow file readers/writers with dataclass models
- `_mapper.py` — `.rasmap` file reader/writer for RasMapper layer configuration

### `hdf/` — HDF5 Results
Files: `_base.py`, `_geometry.py`, `_plan.py`, `_terrain.py`, `_velocity.py`
- `_geometry.py` — mesh geometry: cell coordinates, face/facepoint data, structure collections
  - `facepoint_face_orientation` — angle-sorted facepoint→face mapping with orientation flags
  - `Structure`, `Bridge`, `Inline`, `Lateral`, `SA2DConnection`, `Weir`, `GateGroup` dataclasses
- `_plan.py` — plan results: `FlowAreaResults` with time-series and summary datasets
  - `water_surface`, `face_velocity`, `max_water_surface`, `max_face_velocity`
  - `export_raster` — rasterize mesh results via triangulation (approximate)
  - `export_raster2` — RASMapper-exact rasterization (pixel-perfect vs RASMapper VRT output)
- `_terrain.py` — terrain HDF export: mosaic source TIFFs + apply Levee/Channel ground-line modifications
- `_velocity.py` — velocity-specific helpers

### `geo/` — Geospatial Operations
Files: `raster.py`, `_rasmap.py`, `mesh_validation.py`
- `_rasmap.py` — RASMapper interpolation pipeline (reverse-engineered from `RasMapperLib.dll`):
  - Step A: hydraulic connectivity (`compute_face_wss`)
  - Step B: PlanarRegressionZ vertex WSE (`compute_facepoint_wse`)
  - Step 2: C-stencil tangential velocity reconstruction (`reconstruct_face_velocities`)
  - Step 3: arc-based inverse-face-length weighted vertex averaging (`compute_vertex_velocities`)
  - Step 3.5: sloped face velocity replacement (`replace_face_velocities_sloped`)
  - Step 4: barycentric+donate pixel loop (`rasterize_rasmap`); Numba-accelerated (~9× speedup)
  - Validated pixel-perfect against RASMapper VRT exports (median |diff| = 0.000000)
- `raster.py` — `rasmap_raster()` public API wrapping `_rasmap.py` pipeline
- `mesh_validation.py` — mesh geometry validation utilities

### `bin/` — Native Helpers
- `RasMapperStoreMap.exe` — C#/.NET stub replacing `RasProcess.exe` for stored-map generation
  - Supports all render modes including `slopingPretty` with depth-weighted faces
  - Works around `RasProcess.exe` limitation where `<RenderMode>` is always ignored
  - See `docs/rasprocess_render_mode_limitation.md` for details

## Key Dependencies
- `numpy` — arrays, numerical data
- `pandas` — tabular results and timeseries
- `h5py` — HDF5 file access for hdf subpackage
- `pywin32` — COM interface (Windows only)
- `psutil` — determine process ID of loaded COM process
- `numba` — optional; ~9× speedup in `_rasmap.py` pixel loop
- `geopandas` — optional; geo subpackage only
- `rasterio` — optional; geo subpackage only

## Design Principles
- `geo/` is the **only** subpackage that imports `geopandas` or `rasterio`
- `com/` is the **only** subpackage that imports `win32com`
- Data returned from `hdf/` and `model/` is plain Python/numpy; geospatial conversion is
  an optional step via helper functions in `geo/`
- Subpackages are independently usable (no forced top-level object)
- HEC-RAS versions 5.x and above are the primary target; 4.x supported in `com/`

## Conventions
- Type hints on all public functions
- Dataclasses for structured data models
- Raise specific exceptions, never bare `except:` or silent failures
- All file paths accept `str | Path`, normalize internally with `pathlib.Path`
- Public API should feel Pythonic — no Hungarian notation, no RAS-style naming
- File parsers use verbatim-line editor pattern (store all lines, splice edits)
- RS matching: strip whitespace and trailing `*`; `interpolated=True` on CrossSection when RS had `*`
- `save(path=None)` overwrites source if path omitted

## Archive Reference
- `archive/DLLs/RasMapperLib/` — decompiled C# source for `RasMapperLib.dll` (HEC-RAS 6.6)
  used to understand RASMapper's internal rendering and scripting pipelines
- Archive code is reference only — do not copy-paste; rewrite cleanly
- Note which archive file an implementation was derived from in docstrings

## Never Do
- Never import `win32com` outside `com/` subpackage
- Never import `geopandas` or `rasterio` outside `geo/` subpackage
- Never use mutable default arguments
- Never bare `except:` or silent failures

## Approach
- Before writing any code, describe your approach and wait for approval
- If requirements are ambiguous, ask clarifying questions before writing any code
- After finishing code, list edge cases and suggest test cases to cover them
- If a task requires changing more than 3 files, stop and break it into smaller tasks first
- When there is a bug, start by writing a test that reproduces it, then fix it until the test passes
- Every time I correct you, reflect on what you did wrong and come up with a plan to not repeat it
- Use the example HEC-RAS projects in `..\HEC-RAS Examples` to understand input/output files

## HDF Key Facts
- Geometry file: `*.gx.hdf` — group `Geometry/2D Flow Areas/<name>/`
- Plan file: `*.px.hdf` — contains Geometry + Results
- Results path: `Results/Unsteady/Output/Output Blocks/Base Output/`
  - Time series: `Unsteady Time Series/2D Flow Areas/<name>/Water Surface|Face Velocity`
  - Summary: `Summary Output/2D Flow Areas/<name>/Maximum Water Surface|Maximum Face Velocity`
- `Faces FacePoint Indexes` shape `(n_faces, 2)` — two corner facepoints per face only
  - Curved faces have additional geometry via `Faces Perimeter Info/Values` (not facepoints)
- `Cells FacePoint Indexes` shape `(n_cells, 8)`, padded `-1` — polygon corners in polygon order
- Terrain HDF: root attribute `File Type = "HEC Terrain"`;
  pixel values are in source TIFFs (not in the HDF), HDF stores pyramid metadata/masks only

## RasMapper Store Map
- `store_map()` delegates to `RasMapperStoreMap.exe` (not `RasProcess.exe`)
- `RasProcess.exe -Command=StoreAllMaps` **always uses basic sloping (JustFacepoints) mode**,
  ignoring `<RenderMode>` in the `.rasmap` — this is a hard-coded C# default
- `RasMapperStoreMap.exe` fixes this: applies correct render mode including `slopingPretty`
  with depth-weighted faces (requires `PostProcessing.hdf` — written automatically)
- See `docs/store_map_execution.md` and `docs/rasprocess_render_mode_limitation.md`

## Test Fixtures
- `tests/model/fixtures/ex1.g01` — multi-reach (Butte Cr./Fall River), 1 junction
- `tests/model/fixtures/conspan.g01` — 1 reach, 10 XS + 1 culvert, ineff areas, interpolated XS
- `tests/model/fixtures/beaver.g01` — 1 reach, many XS + 1 bridge
- `tests/model/fixtures/nit_inline.g01` — 1 reach, many XS + 1 inline structure
- `tests/hdf/conftest.py` — synthetic HDF (minimal 2-cell mesh; all facepoint indexes = 0)

## Environment
- Conda env: `raspy-dev` (activate before Claude Code sessions)

## Commands
```bash
conda run -n raspy-dev python -m pytest tests/ -x --tb=short
conda run -n raspy-dev ruff check src/
conda run -n raspy-dev mypy src/raspy
pip install -e ".[dev]"
python -m build
```

## Known Pre-Existing Test Failures
- `tests/test_raspy.py::test_subpackage_imports` — `raspy.controller` skeleton not yet implemented

## For Deeper Reference
- `docs/export_raster2_plan.md` — full RASMapper interpolation algorithm (Steps A–4)
- `docs/terrain_export.md` — terrain HDF structure and levee/channel modification algorithm
- `docs/store_map_execution.md` — `StoreMapCommand` execution graph and XML format
- `docs/rasprocess_render_mode_limitation.md` — why `RasProcess.exe` ignores `<RenderMode>`
- `docs/channel_modification.md` — channel ground-line modification details
- `archive/DLLs/RasMapperLib/` — decompiled C# source (RasMapperLib.dll, HEC-RAS 6.6)
