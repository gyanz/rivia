# RIVIA

**RAS Interface for Visualization, Information, and Automation**

A modern, modular Python library for interacting with [HEC-RAS](https://www.hec.usace.army.mil/software/hec-ras/) hydraulic modeling software.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)](https://www.hec.usace.army.mil/software/hec-ras/)
[![Documentation](https://readthedocs.org/projects/rivia/badge/?version=latest)](https://rivia.readthedocs.io/en/latest/)

## Documentation

Full documentation is available at [rivia.readthedocs.io](https://rivia.readthedocs.io).

## Overview

`rivia` provides a clean, Pythonic interface for working with HEC-RAS projects:

- **Control HEC-RAS** via COM automation — open projects, switch plans, run simulations
- **Read and write** HEC-RAS text input files (`.prj`, `.g*`, `.p*`, `.f*`, `.u*`)
- **Access HDF5 results** — water surface, velocity, depth, and other outputs
- **Export rasters** — pixel-perfect RASMapper-equivalent rasters (WSE, depth, velocity, etc.)
- **Export terrain** — mosaic and modify terrain from HEC-RAS terrain HDF files

## Requirements

- **Windows** — HEC-RAS is Windows-only
- Python 3.10+
- HEC-RAS 5.x or later installed

## Installation

```bash
pip install rivia
```

With geospatial extras (required for raster export):

```bash
pip install rivia[geo]
```

## Quick Example

```python
from rivia.model import Model

# Open a HEC-RAS project
model = Model("path/to/project.prj")
print(model.version)       # e.g. "6.30"

# Switch plans
model.change_plan(title="Base Condition")
model.change_plan(short_id="BC")
model.change_plan(index=0)

# Read HDF results
area = model.hdf.flow_areas["Perimeter 1"]
wse_max = area.max_water_surface

# Export a WSE raster
vrt = model.export_wse(timestep=None, render_mode="sloping")
print(vrt.path)
```

## Package Structure

```
rivia/
├── com/       # COM interface to run/control HEC-RAS
├── model/     # Model - primary project interface; read/write text input files, read HDF results
├── hdf/       # Read HEC-RAS HDF5 geometry and result files
├── geo/       # Geospatial operations: raster export (geopandas/rasterio)
└── utils/     # Shared helpers
```

## Development

```bash
git clone https://github.com/gyanz/rivia.git
cd rivia
pip install -e ".[dev,geo,docs]"

# Run tests
pytest tests/ -x --tb=short

# Lint
ruff check src/

# Type check
mypy src/rivia

# Build docs
sphinx-build -b html docs docs/_build/html
```

## License

Copyright 2025 Gyan Basyal and WEST Consultants, Inc.

Licensed under the [Apache License, Version 2.0](LICENSE).
