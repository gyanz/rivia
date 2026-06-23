# Data Access

rivia exposes HEC-RAS data through four entry points depending on the source:

```{list-table}
:header-rows: 1
:widths: 30 35 35

* - Data source
  - Entry point
  - What you get
* - Text files (`.prj` / `.p*` / `.g*` / `.u*`)
  - `model.project` / `model.plan` / `model.geometry` / `model.flow`
  - Python dataclasses, readable and writable
* - Plan HDF (`.px.hdf`) — 2D mesh results, 1D XS, structures, storage areas
  - `model.results` → `UnsteadyPlan` / `SteadyPlan`
  - NumPy arrays, pandas DataFrames
* - DSS file (`.dss`) — XS and inline structure time series
  - `model.dss` → `DssReader`
  - `pd.Series` (spans full period across restart runs)
* - Geometry HDF (`.gx.hdf`) — mesh geometry only, no results
  - `hdf.Geometry(path)`
  - NumPy arrays
```

## When to use which

| Situation | Use |
|---|---|
| List all plans, units, file paths in the project | `model.project` |
| Read/edit plan settings (intervals, window, flow file) | `model.plan` |
| Read/edit cross-section geometry, Manning's n | `model.geometry` |
| Read/edit boundary conditions and flow hydrographs | `model.flow` |
| 2D mesh results (WSE, velocity) at specific timesteps | `model.results.flow_areas` |
| Maximum/minimum summary over all timesteps | `model.results.flow_areas` |
| 1D cross-section results from the latest run | `model.results.cross_sections` |
| 1D results spanning multiple restart-file runs | `model.dss` |
| Inline/lateral structure results | `model.results.structures` or `model.dss` |
| Mesh geometry only (no results) | `hdf.Geometry` directly |


## Text Input Files

rivia exposes the four HEC-RAS text input files as lazily loaded objects on `Project`.

```
model.project   → Proj         .prj  — project index: plan list, unit system, file paths
model.plan      → Plan         .p**  — plan settings: intervals, simulation window, file refs
model.geometry  → Geometry     .g**  — geometry: cross sections, Manning's n, structures
model.flow      → SteadyFlow         — steady flow: profiles, boundary conditions
               → UnsteadyFlow        — unsteady flow: hydrographs, gate openings, initial conditions
```

### `model.project` — project file

```python
proj = model.project             # Proj

proj.title                       # project title string
proj.units                       # "English" or "SI"
proj.plan_files                  # list[Path] — all .p** files
proj.geom_files                  # list[Path] — all .g** files
proj.plans                       # list of dicts with title, short_id, path per plan
proj.plan_titles                 # list of plan title strings
proj.plan_short_ids              # list of plan short IDs
```

### `model.plan` — plan file

```python
plan = model.plan                # Plan for the current plan

plan.title                       # plan title string
plan.short_id                    # plan short identifier (e.g. "BC")
plan.is_unsteady                 # True for unsteady plans
plan.simulation_window           # ((start_date, start_time), (end_date, end_time))
plan.computation_interval        # e.g. "1MIN"
plan.output_interval             # DSS hydrograph output interval
plan.instantaneous_interval      # instantaneous output interval
plan.mapping_interval            # RASMapper output interval

plan.title = "New Title"         # properties are settable
plan.save()                      # write back to disk
```

### `model.geometry` — geometry file

```python
geom = model.geometry            # Geometry

geom.reaches                     # list[(river, reach)]
geom.get_cross_section("Butte Cr", "Upper", "7")  # → CrossSection dataclass
geom.cross_sections("Butte Cr", "Upper")           # → list[CrossSection]

# Edit and save
geom.set_mannings("Butte Cr", "Upper", "7", entries)
geom.set_stations("Butte Cr", "Upper", "7", stations, elevations)
geom.save()

# Typed structure access (geometry only, no results)
geom.structures.inlines          # StructureIndex[InlineStructure]
geom.structures.bridges          # StructureIndex[Bridge]
geom.structures.laterals         # StructureIndex[LateralStructure]
```

### `model.flow` — flow file

`model.flow` returns an `UnsteadyFlow`.

```python
flow = model.flow   # UnsteadyFlow

flow.flow_hydrographs            # list[FlowHydrograph]
flow.lateral_inflows             # list[LateralInflow]
flow.gate_boundaries             # list[GateBoundary]

flow.set_flow_hydrograph(0, vals)                             # edit by index
flow.set_flow_hydrograph_at("Butte Cr", "Upper", "7", vals)  # edit by location

flow.set_lateral_inflow(0, vals)                              # edit by index
flow.set_lateral_inflow_at("Butte Cr", "Upper", "50", vals)  # edit by location
flow.set_all_lateral_inflows([vals0, vals1])                  # edit all at once
flow.save()
```


## Plan HDF — `model.results`

`model.results` returns a lazily opened `UnsteadyPlan` or `SteadyPlan`
backed by the current plan's HDF file (`*.px.hdf`).  It gives access to
both geometry and results.

```
UnsteadyPlan
├── .flow_areas          → FlowAreaResultsCollection
│     └── ["name"]       → FlowAreaResults
│           ├── .water_surface                          h5py.Dataset  shape (n_t, n_cells)
│           ├── .face_velocity                          h5py.Dataset  shape (n_t, n_faces)
│           ├── .max_water_surface                      pd.DataFrame  columns [value, time]
│           ├── .max_face_velocity                      pd.DataFrame  columns [value, time]
│           ├── .get_water_surface(timestep=t)          np.ndarray    shape (n_cells,)
│           ├── .get_water_surface(cell=c)              pd.Series     WSE over time
│           ├── .get_depth(timestep=t)                  np.ndarray    shape (n_cells,)
│           └── .get_max_depth()                        pd.DataFrame  columns [value, time]
│
├── .storage_areas       → StorageAreaResultsCollection
│     └── ["name"]       → StorageAreaResults
│           ├── .water_surface         np.ndarray    shape (n_t,)
│           └── .max_water_surface     pd.DataFrame  columns [value, time]
│
├── .structures          → StructureResultsCollection
│     │   full mixed collection  → [name | "River Reach RS" | index | (river, reach, rs)]
│     ├── .connections   → StructureIndex[SA2DConnectionResults]  keyed by connection name
│     ├── .bridges       → StructureIndex[BridgeResults]          keyed by "River Reach RS"
│     ├── .inlines       → StructureIndex[InlineResults]          keyed by "River Reach RS"
│     └── .laterals      → StructureIndex[LateralResults]         keyed by "River Reach RS"
│           each StructureIndex supports  [name | index]
│
├── .cross_sections()              → CrossSectionResultsCollection  (mapping interval, default)
│     └── ["River Reach RS" | index | (river, reach, rs)]  → CrossSectionMappingResults
│           ├── .timestamps    pd.DatetimeIndex
│           ├── .wse           pd.Series
│           ├── .flow          pd.Series
│           ├── .flow_lateral  pd.Series
│           ├── .velocity_channel / .velocity_total / .flow_cumulative  pd.Series
│
├── .cross_sections("output")      → CrossSectionResultsCollection  (DSS hydrograph interval)
│     └── ["River Reach RS" | index | (river, reach, rs)]  → CrossSectionOutputResults
│           ├── .timestamps  pd.DatetimeIndex
│           ├── .wse / .flow / .flow_cumulative  pd.Series
│
└── .cross_sections("instantaneous") → InstantaneousResultsCollection  (Post Process Profiles)
      ├── .timestamps     pd.DatetimeIndex  (length n; maps integer profile columns 0…n-1)
      ├── .profile_table("Water Surface")   pd.DataFrame  location × ["max_wse", 0, 1, …]
      ├── .wse() / .flow() / .energy_grade()  (named delegates to profile_table)
      ├── .velocity_channel() / .velocity_total() / .shear() / …  (Additional Variables)
      └── ["River Reach RS" | index | (river, reach, rs)]  → CrossSectionInstantaneousResults
            (geometry carrier only — no data accessors; use collection methods above)
```

### 2D flow areas

```python
hdf = model.results
area = hdf.flow_areas["Perimeter 1"]   # FlowAreaResults

# Raw time-series datasets (lazy h5py.Dataset — slice to control memory)
wse_t5 = area.water_surface[5]         # WSE at timestep 5, shape (n_cells + n_ghost,)
vel_t5 = area.face_velocity[5]         # face-normal velocity at timestep 5

# Derived snapshot (one timestep, all locations)
wse   = area.get_water_surface(timestep=5)   # np.ndarray (n_cells,)
depth = area.get_depth(timestep=5)           # np.ndarray (n_cells,), max(0, WSE - bed)

# Derived time-history (one location, all timesteps)
wse_series = area.get_water_surface(cell=0)  # pd.Series indexed by timestamps

# Scalar (one timestep, one location)
wse_val = area.get_water_surface(timestep=5, cell=0)  # float

# Velocity (snapshot)
vecs  = area.get_cell_velocity(5)                          # (n_cells, 2) [Vx, Vy]
speed = area.get_cell_velocity(5, component="speed")       # (n_cells,)
fvec  = area.get_face_velocity(timestep=5, component="vector")  # (n_faces, 2)

# Summary results (raw HDF reads)
area.max_water_surface                 # pd.DataFrame, max WSE per cell
area.max_face_velocity                 # pd.DataFrame, max velocity per face
area.get_max_depth()                   # pd.DataFrame, max depth per cell (computed)
```

### Cross-section results (1D)

`plan.cross_sections(output)` is a single method that selects which output
block to read via its `output` argument:

| `output=` | Collection type | Per-XS data |
|---|---|---|
| `"mapping"` (default) | `CrossSectionResultsCollection` | `pd.Series` per variable |
| `"output"` | `CrossSectionResultsCollection` | `pd.Series` per variable |
| `"instantaneous"` | `InstantaneousResultsCollection` | collection methods only |

#### Mapping and output blocks

```python
hdf = model.results

# mapping block (default)
coll = hdf.cross_sections()
xs = coll[0]                               # integer index
xs = coll["Butte Cr Upper 7"]             # "River Reach RS" joined by spaces
xs = coll[("Butte Cr", "Upper", "7")]     # (river, reach, rs) tuple

xs.timestamps       # pd.DatetimeIndex — mapping interval
xs.wse              # pd.Series indexed by timestamps
xs.flow             # pd.Series
xs.flow_lateral     # pd.Series
xs.velocity_channel # pd.Series
xs.velocity_total   # pd.Series
xs.flow_cumulative  # pd.Series

# DSS hydrograph block
coll_out = hdf.cross_sections("output")
xs_out = coll_out["Butte Cr Upper 7"]
xs_out.timestamps   # pd.DatetimeIndex — output interval
xs_out.wse          # pd.Series
xs_out.flow         # pd.Series
xs_out.flow_cumulative  # pd.Series
```

#### Instantaneous block (Post Process Profiles)

The instantaneous block exposes all data through the *collection*, not per-XS
objects. The profile axis is labeled `["max_wse", 0, 1, …, n-1]` where
`"max_wse"` is the Max WS envelope and integers map into `coll.timestamps`.

```python
coll = hdf.cross_sections("instantaneous")

# Timestamps for profiles 0…n-1 (excludes Max WS envelope)
coll.timestamps     # pd.DatetimeIndex length n

# Generic engine — returns location × profiles DataFrame
wse = coll.profile_table("Water Surface")
#                            max_wse     0      1      2
# (Butte Cr, Upper, 7)        102.5   98.1   99.3  100.7
# (Butte Cr, Upper, 6)        101.2   97.0   98.1   99.4

wse["max_wse"]                        # max-WS longitudinal profile (pd.Series)
wse[0]                                # WSE along reach at first instantaneous profile
wse.loc[("Butte Cr", "Upper", "7")]   # all profiles for one XS (pd.Series)

coll.timestamps[0]                    # datetime of profile column 0

# Named convenience methods (delegate to profile_table)
coll.wse()              # == coll.profile_table("Water Surface")
coll.flow()             # == coll.profile_table("Flow")
coll.energy_grade()     # == coll.profile_table("Energy Grade")
coll.velocity_channel() # == coll.profile_table("Velocity Channel")
coll.shear()            # == coll.profile_table("Shear")
# … one method per variable; see InstantaneousResultsCollection for the full list
```

The string key is the river, reach, and RS joined by spaces.  The tuple form
avoids ambiguity when any component contains spaces.  RS values are stored
exactly as they appear in the HDF.

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

```{note}
Only a limited set of output variables is currently supported (cross-section
flow/stage and inline structure flow/stage).  More variables will be added in
future updates.
```

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


## Geometry HDF — `hdf.Geometry`

`hdf.Geometry` reads a standalone geometry HDF (`*.gx.hdf`) for mesh geometry
without simulation results.

```{note}
`UnsteadyPlan` and `SteadyPlan` both inherit from `hdf.Geometry`, so all
geometry properties described below are equally accessible via `model.results`
on a plan file — no need to open the geometry HDF separately when you already
have a plan open.
```

```
Geometry
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
│     ├── .inlines      → StructureIndex[InlineStructure]  keyed by "River Reach RS"
│     └── .laterals     → StructureIndex[LateralStructure] keyed by "River Reach RS"
│
└── .cross_sections    → CrossSectionCollection
      └── ["River Reach RS" | index | (river, reach, rs)]  → CrossSection (geometry only, no time-series)
```

```python
from rivia.hdf import Geometry

geom = Geometry("model.g01.hdf")
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
