# Quickstart

`Model` is the primary entry point for working with a HEC-RAS project.
It opens HEC-RAS via COM, binds to a project file, and provides access to
all associated files and results through a single object.

## Opening a project

```python
from raspy.model import Model

model = Model("path/to/project.prj")
print(model.version)          # e.g. "6.30"
print(model.plan_file)        # Path to current plan file
print(model.geom_file)        # Path to current geometry file
```

Pass `backup=True` to snapshot all input files on open and restore them
automatically on exit — useful when running batch modifications:

```python
model = Model("path/to/project.prj", backup=True)
```

## Switching plans

```python
model.change_plan(title="Base Condition")
model.change_plan(short_id="BC")
model.change_plan(index=0)
```

After switching, `model.plan`, `model.hdf`, and all path properties
reflect the newly active plan.

## Reloading after changes

`model.reload()` must be called after any modification to input files to
make HEC-RAS and raspy pick up the changes.  It closes and discards the
cached `plan`, `hdf`, and project handles, then re-opens the project via COM:

```python
model.plan.save()
model.reload()              # invalidates cached plan/hdf, re-opens project
```

For HEC-RAS 5.03 and above, `reload()` uses `Project_Close` +
`Project_Open` (no COM restart).  For older versions it restarts the COM
process entirely.

## Controlling the HEC-RAS window

```python
model.show()                        # make HEC-RAS window visible
model.hide()                        # hide it
model.show_compute(True)            # show the computation window during runs
model.compute_blocking = True       # wait for compute to finish (default)
```

## Resetting to original files

If opened with `backup=True`, call `reset()` to restore all input files
to their state at construction time:

```python
model.reset()
```

## Reading plan results (HDF)

`model.hdf` returns a lazily opened
[`PlanHdf`](../api/generated/raspy.hdf._plan.PlanHdf.rst) instance bound
to the current plan's HDF file:

```python
area = model.hdf.flow_area_results["Perimeter 1"]

# Maximum water-surface elevation across all timesteps
wse_max = area.max_water_surface()

# Water surface at a specific timestep
wse_t5 = area.water_surface[5]

# Maximum face velocity
vel_max = area.max_face_velocity()
```


## Exporting rasters

`Model` inherits all raster export methods from `MapperExtension`.
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

## Reading and editing the plan file

`model.plan` gives a lazily parsed [`PlanFile`](../api/generated/raspy.model.plan.PlanFile.rst):

```python
print(model.plan.simulation_date)
model.plan.save()
model.reload()
```

## Reading and editing the geometry file

```python
from raspy.model import GeometryFile

gf = GeometryFile(model.geom_file)
xs = gf.get_cross_section("Butte Cr", "Upper Reach", "1234.56")
print(xs.stations, xs.elevations)

gf.set_mannings("Butte Cr", "Upper Reach", "1234.56", n_left=0.04, n_channel=0.035, n_right=0.04)
gf.save()
model.reload()
```
