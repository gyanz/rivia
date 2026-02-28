# raspy

A modern, modular Python library for interacting with [HEC-RAS](https://www.hec.usace.army.mil/software/hec-ras/) hydraulic modeling software.

> **Status:** Pre-alpha / Planning phase. Not yet functional.

## Overview

`raspy` provides a clean, Pythonic interface for:

- **Controlling HEC-RAS** via COM automation (run simulations, modify plans)
- **Reading/writing** HEC-RAS text input files (`.prj`, `.g*`, `.f*`, etc.)
- **Accessing HDF5 output** files produced by HEC-RAS
- **Working with geometry** data (cross sections, reaches, junctions)

## Requirements

- **Windows** (HEC-RAS is Windows-only)
- Python 3.10+
- HEC-RAS installed (for COM automation features)

## Installation

```bash
pip install raspy
```

For development:

```bash
git clone https://github.com/gbasyal/raspy.git
cd raspy
pip install -e ".[dev]"
```

## Package Structure

```
raspy/
├── controller/   # COM interface to run/control HEC-RAS
├── io/           # Read/write HEC-RAS text input files
├── hdf/          # Read/write HEC-RAS HDF5 output files
├── geometry/     # Geometry and mesh data
└── utils/        # Shared helpers
```

## Development

```bash
# Run tests
pytest tests/ -x --tb=short

# Lint
ruff check src/

# Type check
mypy src/raspy

# Build
python -m build
```

## License

MIT
