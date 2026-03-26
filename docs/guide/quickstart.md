# Quickstart

`Model` is the primary entry point for working with a HEC-RAS project.
It opens HEC-RAS via COM, binds to a project file, and provides access to
all associated files and results through a single object.

## 1. Opening a project

```python
from raspy.model import Model

model = Model("path/to/project.prj", ras_version=None, backup=False)
print(model.version)          # e.g. "6.30"
print(model.plan_file)        # Path to current plan file
print(model.geom_file)        # Path to current geometry file
```

`ras_version` is auto-detected from the project file's current plan; pass an explicit
version string (e.g. `"6.30"`) or integer (e.g. `630`) to override it.

Pass `backup=True` to snapshot all input files on open and restore them
automatically on exit — useful when running batch modifications:

```{important}
`Model` requires exactly one HEC-RAS process running for the version it
targets.  At construction time, all existing HEC-RAS instances of that
version are closed automatically before the new session is opened.

Different versions do not collide — you can hold a `Model` instance
targeting HEC-RAS 6.6 while independently working with a separate instance
targeting HEC-RAS 6.7.
```

```{warning}
When `backup=True` is used and the Python process terminates abnormally
(crash, forced kill, IDE restart), raspy backup files are left on disk.
The next time a `Model` instance is created for that project, those backup
files are automatically restored — **overwriting the current HEC-RAS input
files**.  Any edits made to the project in HEC-RAS or a text editor after
the abnormal exit will be silently lost.

If you suspect stale backup files are present, you may either delete them
manually or open the project with raspy so that the backup files are
ingested and the session closes normally.
```

## 2. Switching plans

```python
model.change_plan(title="Base Condition")
model.change_plan(short_id="BC")
model.change_plan(index=0)
```

After switching, `model.plan`, `model.hdf`, and all path properties
reflect the newly active plan.

## 3. Reloading after changes

`model.reload()` must be called after editing any input file to make
HEC-RAS and raspy pick up the changes.  It closes and discards the cached
`plan`, `hdf`, and project handles, then re-opens the project via COM.

The input files are accessed as:

- `model.plan` — plan file (`.p**`), a `PlanFile` instance
- `model.flow` — flow file, a `SteadyFlowFile` or `UnsteadyFlowEditor`
  depending on the active plan (`.f**` or `.u**`)

```python
model.plan.save()
model.reload()              # invalidates cached plan/hdf, re-opens project
```

For HEC-RAS 5.03 and above, `reload()` uses `Project_Close` +
`Project_Open` (no COM restart).  For older versions it restarts the COM
process entirely.

## 4. Controlling the HEC-RAS GUI

```python
model.show()    # make HEC-RAS window visible
model.hide()    # hide it
```

## 5. Running the model

`model.run()` computes the current plan and returns a `(success, messages)` tuple.
It raises `HecRasComputeError` if HEC-RAS reports a failure.

```python
success, messages = model.run(blocking=True, hide_window=False)
```

In most cases `blocking=True` (the default) is the right choice — the call
returns only once HEC-RAS finishes.  Pass `blocking=False` to return
immediately and poll for completion via the controller:

```python
model.run(blocking=False)
while not model.controller.Compute_Complete():
    time.sleep(1)
```

Pass `hide_window=True` to suppress the computation window during the run:

```python
success, messages = model.run(hide_window=True)
```

## 6. Resetting to original files

If opened with `backup=True`, call `reset()` to restore all input files
to their state at construction time:

```python
model.reset()
```

## 7. Exporting rasters

`Model` inherits all raster export methods from `MapperExtension`.
The rasterization pipeline is a pixel-perfect reimplementation of RASMapper's
rendering engine — output rasters are identical to those produced by RASMapper.
Each hydraulic variable has a dedicated `export_*` method (returns a
persistent `VrtMap`) and a matching `open_*` method (context manager that
yields a `rasterio.DatasetReader` and cleans up on exit).
`store_map()` is the generic underlying function — prefer the named methods
over calling it directly.

The `timestep` argument is either an integer index into the output timestep
series (zero-based) or `None`, which exports the maximum profile across all
timesteps.

**Water-surface elevation:**

```python
# Maximum WSE raster written to disk
vrt = model.export_wse(timestep=None, render_mode="sloping")
print(vrt.path)

# WSE at timestep 10, read directly via rasterio
with model.open_wse(timestep=10, render_mode="hybrid") as ds:
    data = ds.read(1)
```

**Depth:**

```python
vrt = model.export_depth(timestep=None)

with model.open_depth(timestep=5) as ds:
    data = ds.read(1)
```

**Velocity:**

```python
vrt = model.export_velocity(timestep=None, render_mode="sloping")

with model.open_velocity(timestep=0) as ds:
    data = ds.read(1)
```

**Other variables** (same `export_*` / `open_*` pattern):

```python
vrt = model.export_froude(timestep=None)
vrt = model.export_shear_stress(timestep=None)
vrt = model.export_dv(timestep=None)       # depth × velocity
vrt = model.export_dv2(timestep=None)      # depth × velocity²
```

**Terrain:**

```python
# GeoTIFF — single merged file, all source TIFFs mosaicked
terrain_path = model.export_plan_terrain("terrain.tif")

# VRT — lightweight wrapper over the original source TIFFs (no copy)
terrain_path = model.export_plan_terrain("terrain.vrt")

# VRT with sources copied — self-contained, safe to move to another machine
terrain_path = model.export_plan_terrain("terrain.vrt", copy=True)
```


## 8. Reading plan results (HDF)

`model.hdf` returns a lazily opened `PlanHdf` instance bound to the current
plan's HDF file.

```python
hdf = model.hdf
print(hdf.time_stamps)     # pd.DatetimeIndex of output timesteps
print(hdf.n_timesteps)     # number of timesteps
```

### 2D flow areas

`hdf.flow_areas` is a `FlowAreaResultsCollection` — a dict-like container
of `FlowAreaResults` objects, one per 2D flow area in the plan.

```python
print(hdf.flow_areas.names)            # list of area names
area = hdf.flow_areas["Perimeter 1"]   # FlowAreaResults

# Time-series datasets (h5py.Dataset — slice to read into memory)
wse = area.water_surface[5]            # WSE at timestep 5, shape (n_cells,)
vel = area.face_velocity[5]            # face-normal velocity at timestep 5

# Summary results (pd.DataFrame with columns [value, time])
area.max_water_surface                 # maximum WSE per cell across all timesteps
area.max_face_velocity                 # maximum face velocity per face
```

```{rubric} Exporting rasters
```

`area.export_raster()` rasterizes a result variable to a GeoTIFF using the
interpolation pipeline derived from `RasMapperLib.dll`.  The algorithm closely
follows what RASMapper implements internally, but nuanced differences exist and
edge cases may produce results that diverge from RASMapper's output.

`timestep` is a 0-based index into the output timestep series.  Pass `None`
to use the time of maximum WSE — valid for `"wse"` and `"depth"` only;
velocity requires an explicit timestep.

`render_mode` controls how values are interpolated across each cell:

- `"horizontal"` — flat per-cell value; no spatial interpolation
- `"sloping"` (default) — interpolates across cell-corner facepoints
- `"hybrid"` — like `"sloping"` but also incorporates face-centre contributions;
  accepts `use_depth_weights` (weight faces by water depth) and
  `shallow_to_flat` (render poorly-connected cells as flat)

```python
# Water-surface elevation — maximum profile (timestep=None), sloping (default)
wse_path = area.export_raster("wse", timestep=None, output_path="wse_max.tif",
                               reference_raster="topo.tif")

# Water-surface elevation — flat per-cell value
wse_path = area.export_raster("wse", timestep=None, output_path="wse_horizontal.tif",
                               reference_raster="topo.tif", render_mode="horizontal")

# Depth — WSE minus terrain; hybrid with depth weighting
depth_path = area.export_raster("depth", timestep=10, output_path="depth.tif",
                                 reference_raster="topo.tif", render_mode="hybrid",
                                 use_depth_weights=True, shallow_to_flat=True)

# Velocity magnitude — explicit timestep required
vel_path = area.export_raster("velocity", timestep=10, output_path="velocity.tif",
                               reference_raster="topo.tif", render_mode="sloping")
```

Use `export_hydraulic_rasters()` to export WSE, depth, and velocity in one call:

```python
results = area.export_hydraulic_rasters(
    timestep=10,
    reference_raster="topo.tif",
    wse_path="wse.tif",
    depth_path="depth.tif",
    velocity_path="velocity.tif",
    render_mode="sloping",
)
```

```{rubric} Velocity plot
```

`area.velocity_plot()` renders a quiver plot of interpolated velocity vectors
around a target cell — useful for inspecting local flow patterns.

```python
ax = area.velocity_plot(
    timestep=10,
    cell_index=42,          # target cell; neighbourhood expands from here
    buffer=2,               # rings of face-adjacent cells to include
    reference_raster="topo.tif",
    render_mode="sloping",
    n_arrows=400,           # approximate number of quiver arrows
)
ax.set_title("Velocity — timestep 10, cell 42")
```

### Storage areas

`hdf.storage_areas` is a `StorageAreaResultsCollection` — a dict-like
container of `StorageAreaResults` objects.

```python
print(hdf.storage_areas.names)
sa = hdf.storage_areas["Reservoir 1"]  # StorageAreaResults

sa.water_surface                       # WSE time series, shape (n_t,)
sa.max_water_surface                   # pd.DataFrame with columns [value, time]
```

### SA/2D connections

`hdf.storage_area_connections` is an `SA2DConnectionCollection` — a
dict-like container of `SA2DConnectionResults` objects for structures
connecting storage areas and 2D flow areas (dams, levees, gates, weirs).

```python
print(hdf.storage_area_connections.names)
conn = hdf.storage_area_connections["Dam"]  # SA2DConnectionResults

conn.total_flow                             # total flow time series, shape (n_t,)
conn.stage_hw                               # headwater stage time series
conn.stage_tw                               # tailwater stage time series
```
