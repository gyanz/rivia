# Reading Model Data

raspy provides three distinct ways to access HEC-RAS results, each suited to
different data sources and use cases.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Data source          │  Entry point              │  What you get       │
├─────────────────────────────────────────────────────────────────────────┤
│  Text files (.prj/.p*/.g*/.u*) │ model.project/plan/geom/flow │ dataclasses │
│                       │                           │                     │
│  Plan HDF (.px.hdf)   │  model.hdf                │  NumPy / DataFrame  │
│    └─ 2D mesh results │    .flow_areas[name]       │  arrays + DataFrames│
│    └─ 1D XS results   │    .cross_sections["R Rc RS"|i|(r,rc,rs)] │      │
│    └─ structures      │    .structures[name|i|(r,rc,rs)]  │            │
│    └─ storage areas   │    .storage_areas[name]   │                     │
│                       │                           │                     │
│  DSS file (.dss)      │  model.dss                │  pd.Series          │
│    └─ XS time series  │    .flow(river, reach, rs)│  (full run period)  │
│    └─ inline struct.  │    .stage_hw(...)         │                     │
│                       │                           │                     │
│  Geometry HDF (*.gx.hdf) │ GeometryHdf(path)     │  NumPy arrays       │
│    └─ mesh geometry   │    .flow_areas[name]      │  (no results)       │
│    └─ structures      │    .structures[name]      │                     │
│    └─ cross sections  │    .cross_sections[key]   │                     │
└─────────────────────────────────────────────────────────────────────────┘
```

## When to use which

| Situation | Use |
|---|---|
| List all plans, units, file paths in the project | `model.project` |
| Read/edit plan settings (intervals, window, flow file) | `model.plan` |
| Read/edit cross-section geometry, Manning's n | `model.geom` |
| Read/edit boundary conditions and flow hydrographs | `model.flow` |
| 2D mesh results (WSE, velocity) at specific timesteps | `model.hdf.flow_areas` |
| Maximum/minimum summary over all timesteps | `model.hdf.flow_areas` |
| 1D cross-section results from a single run | `model.hdf.cross_sections` |
| 1D results spanning multiple restart-file runs | `model.dss` |
| Inline/lateral structure results | `model.hdf.structures` or `model.dss` |
| Mesh geometry only (no results) | `GeometryHdf` directly |


## Text Input Files

raspy exposes the four HEC-RAS text input files as lazily loaded objects on `Model`.

```
model.project  → ProjectFile    .prj  — project index: plan list, unit system, file paths
model.plan     → PlanFile       .p**  — plan settings: intervals, simulation window, file refs
model.geom     → GeometryFile   .g**  — geometry: cross sections, Manning's n, structures
model.flow     → SteadyFlowFile       — steady flow: profiles, boundary conditions
              → UnsteadyFlowEditor    — unsteady flow: hydrographs, gate openings, initial conditions
```

### `model.project` — project file

```python
proj = model.project             # ProjectFile

proj.title                       # project title string
proj.units                       # "English" or "SI"
proj.plan_files                  # list[Path] — all .p** files
proj.geom_files                  # list[Path] — all .g** files
proj.plans()                     # list of dicts with title, short_id, path per plan
proj.plan_titles                 # list of plan title strings
proj.plan_short_ids              # list of plan short IDs
```

### `model.plan` — plan file

```python
plan = model.plan                # PlanFile for the current plan

plan.plan_title                  # plan title string
plan.short_id                    # plan short identifier (e.g. "BC")
plan.is_unsteady                 # True for unsteady plans
plan.simulation_window           # ((start_date, start_time), (end_date, end_time))
plan.computation_interval        # e.g. "1MIN"
plan.output_interval             # mapping output interval
plan.dss_interval                # DSS output interval
plan.instantaneous_interval      # instantaneous output interval
plan.mapping_interval            # RASMapper output interval

plan.plan_title = "New Title"    # properties are settable
plan.save()                      # write back to disk
```

### `model.geom` — geometry file

```python
geom = model.geom                # GeometryFile

geom.reaches                     # list[(river, reach)]
geom.get_cross_section("Butte Cr", "Upper", "7")  # → CrossSection dataclass
geom.cross_sections("Butte Cr", "Upper")           # → list[CrossSection]

# Edit and save
geom.set_mannings("Butte Cr", "Upper", "7", entries)
geom.set_stations("Butte Cr", "Upper", "7", stations, elevations)
geom.save()

# Typed structure access (geometry only, no results)
geom.structures.inlines          # StructureIndex[Inline]
geom.structures.bridges          # StructureIndex[Bridge]
geom.structures.laterals         # StructureIndex[Lateral]
```

### `model.flow` — flow file

`model.flow` returns a `SteadyFlowFile` or `UnsteadyFlowEditor` depending on
the active plan.

```python
flow = model.flow

# Unsteady
flow.flow_hydrographs            # list[FlowHydrograph]
flow.lateral_inflows             # list[LateralInflow]
flow.gate_boundaries             # list[GateBoundary]

flow.get_flow_hydrograph("Butte Cr", "Upper", "7")       # → list[float]
flow.set_flow_hydrograph("Butte Cr", "Upper", "7", vals) # edit in place
flow.save()
```


## Plan HDF — `model.hdf`

`model.hdf` returns a lazily opened `PlanHdf` backed by the current plan's
HDF file (`*.px.hdf`).  It gives access to both geometry and results.

```
PlanHdf
├── .flow_areas          → FlowAreaResultsCollection
│     └── ["name"]       → FlowAreaResults
│           ├── .water_surface         h5py.Dataset  shape (n_t, n_cells)
│           ├── .face_velocity         h5py.Dataset  shape (n_t, n_faces)
│           ├── .max_water_surface     pd.DataFrame  columns [value, time]
│           ├── .max_face_velocity     pd.DataFrame  columns [value, time]
│           ├── .wse(timestep)         np.ndarray    shape (n_cells,)
│           └── .depth(timestep)       np.ndarray    shape (n_cells,)
│
├── .storage_areas       → StorageAreaResultsCollection
│     └── ["name"]       → StorageAreaResults
│           ├── .water_surface         np.ndarray    shape (n_t,)
│           └── .max_water_surface     pd.DataFrame  columns [value, time]
│
├── .structures          → PlanStructureCollection
│     │   full mixed collection  → [name | "River Reach RS" | index | (river, reach, rs)]
│     ├── .connections   → StructureIndex[SA2DConnectionResults]  keyed by connection name
│     ├── .bridges       → StructureIndex[BridgeResults]          keyed by "River Reach RS"
│     ├── .inlines       → StructureIndex[InlineResults]          keyed by "River Reach RS"
│     └── .laterals      → StructureIndex[LateralResults]         keyed by "River Reach RS"
│           each StructureIndex supports  [name | index]
│
├── .cross_sections      → CrossSectionResultsCollection  (mapping output interval)
│     └── ["River Reach RS" | index | (river, reach, rs)]  → CrossSectionResults
│
├── .cross_sections_dss  → CrossSectionResultsCollection  (DSS output interval)
│     └── ["River Reach RS" | index | (river, reach, rs)]  → CrossSectionResultsDss
│
└── .cross_sections_inst → CrossSectionResultsCollection  (instantaneous output interval)
      └── ["River Reach RS" | index | (river, reach, rs)]  → CrossSectionResultsInst
```

### 2D flow areas

```python
hdf = model.hdf
area = hdf.flow_areas["Perimeter 1"]   # FlowAreaResults

# Time-series datasets (lazy — slice to load into memory)
wse_t5 = area.water_surface[5]         # WSE at timestep 5, shape (n_cells,)
vel_t5 = area.face_velocity[5]         # face velocity at timestep 5

# Convenience wrappers
wse = area.wse(timestep=5)             # same as above, np.ndarray
depth = area.depth(timestep=5)         # WSE minus terrain

# Summary results
area.max_water_surface                 # pd.DataFrame, max WSE per cell
area.max_face_velocity                 # pd.DataFrame, max velocity per face
```

### Cross-section results (1D)

`PlanHdf` exposes three collections depending on which output interval you need:

| Property | Class | Interval |
|---|---|---|
| `.cross_sections` | `CrossSectionResults` | Mapping output interval |
| `.cross_sections_dss` | `CrossSectionResultsDss` | DSS output interval |
| `.cross_sections_inst` | `CrossSectionResultsInst` | Instantaneous output interval |

All three collections (`cross_sections`, `cross_sections_dss`, `cross_sections_inst`)
accept any of three key types:

```python
xs = hdf.cross_sections[0]                               # integer index (insertion order)
xs = hdf.cross_sections["Butte Cr Upper 7"]              # "River Reach RS" joined by spaces
xs = hdf.cross_sections[("Butte Cr", "Upper", "7")]      # (river, reach, rs) tuple

xs.water_surface_total    # np.ndarray shape (n_t,)
xs.flow_total             # np.ndarray shape (n_t,)
xs.velocity_total         # np.ndarray shape (n_t,)
```

The string key is the river, reach, and RS joined by spaces — the same
convention used by `StructureCollection` and the DSS Hydrograph Output group
names in the HDF.  The tuple form is more readable and avoids ambiguity when
any component contains spaces.

RS values are stored exactly as they appear in the HDF — no stripping or
normalisation is applied.

### Structures

```python
structs = hdf.structures                          # PlanStructureCollection

# Typed sub-collections — each is a StructureIndex supporting [name | index]
structs.connections                               # StructureIndex[SA2DConnectionResults]
structs.bridges                                   # StructureIndex[BridgeResults]
structs.inlines                                   # StructureIndex[InlineResults]
structs.laterals                                  # StructureIndex[LateralResults]

# Access via typed sub-collection
inl = structs.inlines["Butte Cr Upper 100"]       # by "River Reach RS"
inl = structs.inlines[0]                          # by integer index
conn = structs.connections["Reservoir Dam"]       # SA2DConnection by name
conn = structs.connections[0]

# Access via the full mixed collection (all types together)
# SA2DConnection → by connection name
conn = structs["Reservoir Dam"]
# Bridge / Inline / Lateral → by "River Reach RS", tuple, or index
inl = structs["Butte Cr Upper 100"]
inl = structs[("Butte Cr", "Upper", "100")]
inl = structs[2]

inl.variable_names                                # list of available output variables
inl.stage_hw                                      # headwater stage, np.ndarray shape (n_t,)
inl.stage_tw                                      # tailwater stage
inl.total_flow                                    # total flow
```


## DSS File — `model.dss`

`model.dss` returns a `DssReader` that reads directly from the DSS file
written alongside the unsteady simulation (`.dss`).

The key difference from `model.hdf.cross_sections`: the DSS file **accumulates
across restart runs**, so it covers the full simulation period even when the
model was run as a sequence of short durations.  The plan HDF is overwritten on
each run and only contains the most recent duration.

Requires the optional `pydsstools` dependency.

```
DssReader  (all methods accept an optional window=(start, end) keyword arg)
├── .flow(river, reach, rs)              → pd.Series   XS flow or inline structure total flow
├── .stage(river, reach, rs)             → pd.Series   stage at a cross section
├── .flow_cum(river, reach, rs)          → pd.Series   cumulative flow volume
├── .stage_hw(river, reach, rs)          → pd.Series   inline structure headwater stage
├── .stage_tw(river, reach, rs)          → pd.Series   inline structure tailwater stage
├── .total_gate_flow(river, reach, rs, gate=N) → pd.Series
└── .weir_flow(river, reach, rs)         → pd.Series
```

```python
dss = model.dss

# Flow at cross-section RS "7" — no window → current plan simulation window
q = dss.flow("Butte Cr", "Upper", "7")

# Stage at an inline structure
hw = dss.stage_hw("Butte Cr", "Upper", "100")

# Restrict to a time window
q = dss.flow("Butte Cr", "Upper", "7",
             window=(("01Jan2020", "0000"), ("31Jan2020","2400")))
```


## Geometry HDF — `GeometryHdf`

`GeometryHdf` reads a standalone geometry HDF (`*.gx.hdf`) for mesh geometry
without simulation results.

```{note}
`PlanHdf` inherits from `GeometryHdf`, so all geometry properties described
below are equally accessible via `model.hdf` on a plan file — no need to open
the geometry HDF separately when you already have a plan open.
```

```
GeometryHdf
├── .flow_areas        → FlowAreaCollection
│     └── ["name"]    → FlowArea
│           ├── .cell_centers           np.ndarray  shape (n_cells, 2)
│           ├── .facepoint_coordinates  np.ndarray  shape (n_fp, 2)
│           ├── .face_cell_indexes      np.ndarray  shape (n_faces, 2)
│           └── .cell_polygons          list[np.ndarray]
│
├── .structures        → StructureCollection
│     │   full mixed collection  → [name | "River Reach RS" | index | (river, reach, rs)]
│     ├── .connections  → StructureIndex[SA2DConnection]  keyed by connection name
│     ├── .bridges      → StructureIndex[Bridge]          keyed by "River Reach RS"
│     ├── .inlines      → StructureIndex[Inline]          keyed by "River Reach RS"
│     └── .laterals     → StructureIndex[Lateral]         keyed by "River Reach RS"
│
└── .cross_sections    → CrossSectionCollection
      └── ["River Reach RS" | index | (river, reach, rs)]  → CrossSection (geometry only, no time-series)
```

```python
from raspy.hdf import GeometryHdf

geom = GeometryHdf("model.g01.hdf")
area = geom.flow_areas["Perimeter 1"]   # FlowArea (geometry only)

area.cell_centers                        # np.ndarray shape (n_cells, 2)
area.facepoint_coordinates               # np.ndarray shape (n_fp, 2)
area.cell_polygons                       # list of polygon vertex arrays

# Cross sections support the same three key types as the results collections
xs = geom.cross_sections[0]                           # integer index
xs = geom.cross_sections["Butte Cr Upper 7"]          # "River Reach RS" joined by spaces
xs = geom.cross_sections[("Butte Cr", "Upper", "7")]  # (river, reach, rs) tuple
xs.station_elevation                                  # np.ndarray shape (n_pts, 2)
```
