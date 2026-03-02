# raspy — HEC-RAS Python Interface Library
# Status: PLANNING PHASE

## Project Overview
`raspy` is a modern, modular Python library for interacting with HEC-RAS hydraulic
modeling software. It replaces two legacy internal libraries (`pyras` and `ras_tools`)
currently in `archive/`. Goal: clean public PyPI package with a well-designed API.

- Platform: Windows-only (HEC-RAS is Windows-only)
- Python: 3.10+
- Packaging: pyproject.toml
- Distribution: Public PyPI package (design API with external users in mind)

## Architecture (Planned Subpackages)

```
raspy/
├── com/              # COM interface to run/control HEC-RAS derived from pyras
├── model/            # Uses com to load HEC-RAS software and load model and read/parse/write  text input and output files including plan, flow and geometry (.prj, .g*, .f*, .u* etc.)
├── hdf/              # Read HEC-RAS HDF5 files (h5py-based) for geometric data esp mesh and results
├── geo/              # Functions and classes for geospatial operation of input and output data such as interpolation, write raster after reading hdf data, etc.
└── utils/            # Shared helpers (path handling, validation, logging)
```

Archive reference:
- `archive/pyras/`    → informs `com/`
- `archive/ras_tools/` → informs `model/`, `hdf/`, `geo/`

## Current Phase: Architecture Design
We are in the PLANNING phase. Do not write production code yet.
Current tasks:
- [ ] Define subpackage responsibilities and boundaries
- [ ] Design public API (class names, method signatures, return types)
- [ ] Identify what to reuse vs rewrite from archive
- [ ] Define data models / shared types used across subpackages
- [ ] Set up pyproject.toml and package skeleton

## Key Dependencies
- `numpy` — arrays, numerical data
- `pandas` — tabular results and timeseries
- `h5py` — HDF5 file access for hdf subpackage
- `pywin32` — COM interface in ras (Windows only)
- `psutil` - Determine process id of loaded COM process
- `geopandas` — optional, but needed for geo subpackage
- `rasterio` — optional, but needed for geo subpackage

## Conventions
- Type hints on all public functions
- Dataclasses or Pydantic for structured data models (decide before coding)
- Raise specific exceptions, never bare `except:` or silent failures
- All file paths accept `str | Path`, normalize internally with `pathlib.Path`
- Public API should feel Pythonic — no Hungarian notation, no RAS-style naming

## Archive Usage Rules
- Archive code is reference only — read it to understand behavior, do not copy-paste
- Archive code has no tests; assume edge cases are unhandled
- When porting logic, rewrite cleanly rather than adapting clunky patterns
- Note which archive file a new implementation was derived from in docstrings

## Never Do
- Never import `win32com` outside `com/` subpackage
- Never use mutable default arguments
- Never commit without updating "Current Phase" task list above

## Design Questions 
- Should subpackages be usable independently, or always via a top-level object?
- Support for HEC-RAS versions 4.x and above
- Error handling philosophy: strict (raise early) vs lenient (warn and continue)?
- Data read from file and hdf file are read in non-geospatial format but give an optional argument to convert them to geopandas or raster object via helper function defined in geo subpackage
- I want only geo subpackage to use libraries such as geopandas and rasterio so that these dependencies are not needed to perform other unrelated task

## Approach
- Before writing any code, describe your approach and wait for approval
- If the requirements I give your are ambiguous, ask clarifying questions before writing any code
- After your finish writing code, list the edge cases and suggest test cases to cover them
- If a task requires changing to more than 3 files, stop and break it into smaller tasks first
- When there is a bug, start by writing a test that reporoduces it, then fix it until the test passes
- Every time I correct you, refect on what you did wrong and come up with a plan to never make the same mistake again

## Environment
- Conda env: `raspy-dev` (activate before Claude Code sessions)

## Commands
- Install dev: `pip install -e ".[dev]"`
- Test: `pytest tests/ -x --tb=short`
- Lint: `ruff check src/`
- Type check: `mypy src/raspy`
- Build: `python -m build`

## For Deeper Reference
- Archive COM logic: see `archive/pyras/` before designing controller/
- Archive HDF logic: see `archive/ras_tools` before designing hdf/
