# Introduction

**raspy** is a modern Python library for interacting with
[HEC-RAS](https://www.hec.usace.army.mil/software/hec-ras/), the U.S. Army Corps of
Engineers hydraulic modeling software.  It provides a clean, Pythonic API for the
full HEC-RAS workflow: launching the HEC-RAS model, running the compute, editing input files, reading HDF5
results, and exporting geospatial rasters — all from a single Python environment.

---

## What raspy can do

- **Running HEC-RAS** — open projects, switch plans, trigger computations, and close the application via the COM interface, without touching the GUI.
- **Reading and manipulating input files** — parse and edit plan, geometry, steady flow, and unsteady flow text files; write changes back to disk.
- **Reading HDF5 results** — access time-series and summary output (WSE, velocity, storage areas, SA/2D connections) from plan HDF files.
- **Exporting result rasters** — produce water-surface elevation, depth, velocity, and other hydraulic variable rasters using a pixel-perfect reimplementation of RASMapper's rendering pipeline.
- **Geospatial operations** — coordinate reference system handling, raster I/O, and mesh-to-raster interpolation via `geopandas` and `rasterio` (isolated to `raspy.geo`).
- **Display results** — visualise interpolated velocity vector fields around mesh cells as quiver plots.
- **Mesh validation** — check 2D mesh cells against HEC-RAS geometric validity rules (face count, convexity, collinearity, duplicates, boundary containment).
- **Canal automation** — drive gate operations and inflow hydrographs from sensor data by scripting the full run-read-act control loop.

---

## Licensing

raspy is released under the **Apache License 2.0**.

**Authors:** Gyan Basyal and WEST Consultants, Inc.

The Apache 2.0 license is a permissive open-source license.  In summary:

- You may use, copy, modify, and distribute raspy, including in commercial
  products, free of charge.
- You must retain the original copyright notice and the `NOTICE` file (if
  present) in any distribution.
- You must state any significant changes you made to the source code.
- No trademark rights are granted — you may not use the authors' or
  organisation's names to endorse or promote products derived from raspy without
  permission.
- The software is provided **as-is**, without warranties or conditions of any
  kind.  The authors are not liable for damages arising from its use.

The full license text is available in the `LICENSE` file at the root of the
repository and at <https://www.apache.org/licenses/LICENSE-2.0>.
