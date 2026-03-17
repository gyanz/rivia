# Terrain Export — HDF Structure and Algorithm

This document describes how `raspy` reads a RasMapper terrain HDF5 file and
exports it to a GeoTIFF, including the rasterisation of **Levee-type
ground-line modifications**.

---

## 1  Terrain HDF file structure

RasMapper stores terrain data in `*.hdf` files (root attribute
`File Type = "HEC Terrain"`).  A single HDF may contain:

| HDF group | Contents |
|---|---|
| `Terrain/` | One sub-group per source TIFF, holding pyramid-level Mask, Perimeter, and Min-Max datasets |
| `Modifications/` | Optional — vector ground-line modifications (Levee, Channel, …) |
| `Terrain/Stitch TIN Points` / `Stitches` | Tile-boundary stitching data (not used for export) |

### 1.1  Terrain source groups (`Terrain/{name}/`)

Each sub-group has:

| Attribute / Dataset | Type | Description |
|---|---|---|
| `@File` | string | Relative path to the source GeoTIFF |
| `@Priority` | int32 | Compositing priority — **lower number = higher priority** (rendered on top) |
| `{level}/Mask` | uint8 `(N_tiles, tile_px²)` | Per-pixel validity bitmask (bit 0 = has data) |
| `{level}/Perimeter` | float32 `(N_tiles, 2·tile+1)` | Edge elevation values used for tile-boundary stitching during rendering |
| `{level}/Min-Max` | float32 `(N_tiles, 3)` | Per-tile min, max, and all-NoData flag |

**Level 0** is full resolution; each successive level doubles the cell size
(overview pyramid).

> **Key point:** pixel elevation values are stored in the source TIFF files,
> not in the HDF.  The HDF stores only the pyramid metadata, mask, and
> perimeter used by the RasMapper rendering engine.

### 1.2  Modification groups (`Modifications/{name}/`)

Present when the user has drawn ground-line modifications (levees, channels)
in RasMapper.  Each sub-group has:

| Attribute / Dataset | Type | Description |
|---|---|---|
| `@Subtype` | string | `"Levee"` or `"Channel"` (other types planned) |
| `@Priority` | int32 | Modification rendering priority (lower = applied last / wins) |
| `Attributes` | structured array | Cross-section parameters — see table below |
| `Polyline Points` | float64 `(N, 2)` | Centerline XY vertices in the terrain CRS |
| `Polyline Info` | int32 `(1, 4)` | `[start_vertex, vertex_count, start_part, part_count]` |
| `Polyline Parts` | int32 `(P, 2)` | Per-part `[start_vertex, vertex_count]` |
| `Profile Values` | float32 `(M, 2)` | `[station_ft, elevation]` pairs along the centerline |
| `Profile Info` | int32 `(1, 2)` | `[start_idx, count]` into Profile Values |

**Attributes structured array fields:**

| Field | Meaning |
|---|---|
| `Name` | User label |
| `Elevation Type` | `"SetIfHigher"`, `"SetIfLower"`, `"SetValue"`, `"AddValue"` |
| `Top Elevation` | Fallback crest elevation (overridden by Profile Values) |
| `Top Width` | Width of the flat crest (ft or m, same units as CRS) |
| `Left Slope` | Left-side H:V slope ratio; `0` → vertical wall |
| `Right Slope` | Right-side H:V slope ratio; `0` → vertical wall |
| `Max Reach` | Maximum perpendicular distance from centerline to outer edge |
| `Transition Percent` | Edge-blending fraction (not yet used in raspy export) |
| `Elev Pt Tolerance` | Control-point snapping tolerance (pre-computed in Profile Values) |

---

## 2  Export algorithm

### 2.1  Source TIFF mosaic

1. Open the terrain HDF, collect all sub-groups of `Terrain/` that have a
   `@File` attribute.
2. Resolve each `@File` value relative to the HDF directory → absolute path.
3. Sort entries by `@Priority` ascending (0 = highest priority).
4. Mosaic with `rasterio.merge` — the **first dataset wins** for overlapping
   pixels, so the highest-priority source dominates.
5. If there are no modifications, write the mosaic directly to the output path.

### 2.2  Applying Levee modifications

Modifications are applied **after** mosaicking, in **reverse priority order**
(lower-priority modifications applied first so higher-priority ones overwrite
them).

For each `Modifications/{name}/` group with `Subtype = "Levee"`:

#### Step 1 — Read parameters
- Polyline centerline vertices (XY, terrain CRS)
- Profile Values `(station, elevation)` pairs
- `TopWidth`, `LeftSlope`, `RightSlope`, `MaxReach`
- `Elevation Type`

Slope normalisation: `NaN → 3.0` (archive default); `0.0 → ∞` (vertical
wall — no elevation drop outside the flat crest).

#### Step 2 — Compute footprint
Buffer the polyline by `MaxReach` to get a polygon bounding box in map
coordinates.  Convert to raster pixel coordinates and clamp to the raster
extent.

#### Step 3 — Project pixels onto centerline
For each pixel in the bounding box, find the closest point on the polyline:

```
For each polyline segment [A, B]:
    t = clamp(((P - A) · (B - A)) / |B - A|², 0, 1)
    closest = A + t * (B - A)
    dist = |P - closest|            # perpendicular distance
    station = cum_length[i] + t * segment_length
    signed_dist = (ux * (Py - Cy) - uy * (Px - Cx))   # + = left, - = right
```

Keep the segment that gives the smallest distance.

#### Step 4 — Interpolate crest elevation

```python
crest_elev = np.interp(station, profile[:, 0], profile[:, 1])
```

#### Step 5 — Compute cross-section elevation

```
half_w = TopWidth / 2

|perp| ≤ half_w               → mod_elev = crest_elev          (flat top)
half_w < |perp| ≤ MaxReach   → mod_elev = crest_elev           (left or right side)
                                           - (|perp| - half_w) / slope
|perp| > MaxReach              → no modification
```

When `slope = ∞` (original value 0): the slope term is 0, so the full
`MaxReach` corridor receives the flat crest elevation.

#### Step 6 — Apply elevation type

| HDF value | Operation |
|---|---|
| `"SetIfHigher"` / `"TakeHigher"` | `np.maximum(terrain, mod_elev)` |
| `"SetIfLower"` / `"TakeLower"` | `np.minimum(terrain, mod_elev)` |
| `"SetValue"` / `"FixedElevation"` | replace unconditionally |
| `"AddValue"` | `terrain + mod_elev` |

NoData pixels in the original terrain are replaced by the modification value
regardless of elevation type.

---

## 3  Supported and planned modification subtypes

| Subtype | Status |
|---|---|
| `Levee` | Implemented |
| `Channel` | Planned (future version) |
| Others | Warned and skipped |

---

## 4  API

```python
from raspy.hdf._terrain import export_terrain

# Standalone — given a terrain HDF path
out = export_terrain("terrain/drone survey 2020.hdf", "output/terrain.tif")

# Via the model — uses the plan's associated terrain automatically
out = model.plan_terrain_export("output/terrain.tif")
```

`plan_terrain_export` calls `plan_terrain()` to resolve the terrain HDF from
the geometry HDF's `Terrain Layername` attribute and the project `.rasmap`
file, then delegates to `export_terrain`.

---

## 5  Archive reference

Implementation derived from analysis of:

- `archive/DLLs/RasMapperLib/RasMapperLib/TerrainLayer.cs` — tile rendering
  and `ObstructRaster` hook
- `archive/DLLs/RasMapperLib/RasMapperLib.Terrain/RasterFileInfo.cs` — HDF
  pyramid metadata
- `archive/DLLs/RasMapperLib/RasMapperLib/GroundLineModificationLayer.cs` —
  TIN construction and rasterisation
- `archive/DLLs/RasMapperLib/RasMapperLib/ElevationModificationGroup.cs` —
  compositing and `GetReplacementValue`
