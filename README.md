# RIVIA

**RAS Interface for Visualization, Information, and Automation**

A modern, modular Python library for interacting with [HEC-RAS](https://www.hec.usace.army.mil/software/hec-ras/) hydraulic modeling software.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)](https://www.hec.usace.army.mil/software/hec-ras/)
[![Documentation](https://readthedocs.org/projects/rivia/badge/?version=latest)](https://rivia.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/rivia.svg)](https://pypi.org/project/rivia/)

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
import numpy as np
from rivia.model import Project

# Open a HEC-RAS project
model = Project("path/to/project.prj")
print(model.version)       # e.g. "6.30"

# Switch plans and run
model.set_plan(short_id="BC")
model.run(hide_window=False)

# Export a WSE raster (pixel-perfect RASMapper equivalent)
vrt = model.export_wse(timestep=None, render_mode="sloping")

hdf = model.results          # UnsteadyPlan

# 2D flow areas
area = hdf.flow_areas["Perimeter 1"]
area.max_water_surface       # pd.DataFrame  — max WSE per cell, columns [value, time]
area.get_depth(timestep=0)   # np.ndarray   — water depth snapshot

# 1D cross sections — mapping interval
xs = hdf.cross_sections("mapping")["Butte Cr Upper 7"]
xs.wse              # pd.Series  — WSE indexed by pd.DatetimeIndex
xs.flow             # pd.Series
xs.velocity_channel # pd.Series

# Storage areas — DSS hydrograph interval
sa = hdf.storage_areas("output")["Reservoir 1"]
sa.wse              # pd.Series  — WSE indexed by pd.DatetimeIndex
sa.max_wse          # pd.DataFrame  — columns [value, time]

# Structures — DSS Profile interval
structs = hdf.structures("profile")
inl = structs.inlines["Butte Cr Upper 100"]
inl.stage_hw        # pd.Series  — headwater stage
inl.flow_total      # pd.Series  — total flow

conn = structs.connections["Dam"]
conn.stage_hw       # pd.Series
conn.stage_tw       # pd.Series
```

## Package Structure

```
rivia/
├── controller/  # COM interface to run/control HEC-RAS
├── model/       # Project — primary project interface; read/write text input files, read HDF results
├── hdf/         # Read HEC-RAS HDF5 geometry and result files
├── geo/         # Geospatial operations: raster export (geopandas/rasterio)
└── utils/       # Shared helpers
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
