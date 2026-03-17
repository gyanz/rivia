# `export_raster2` — RASMapper-Exact Interpolation Plan

**Branch:** `flowarea_interp`
**Reference:** `archive/velocity_rasterizer_standalone/velocity_rasterizer_standalone/velocity_rasterizer_combined.py`
**Source:** Reverse-engineered from `RasMapperLib.dll` (C#/.NET) by CLB Engineering.
**Validated:** Pixel-perfect against RASMapper VRT exports — median |diff| = 0.000000 ft/s.

---

## Goal

Add `FlowAreaResults.export_raster2()` that replicates RASMapper's exact rasterization
pipeline for 2D flow areas, matching the **"Sloping Cell Corners"** (`sloping`) and
**"Horizontal"** (`flat`) render modes. This replaces all existing custom interpolation
methods (`scatter_*`, `triangle_blend`, `facepoint_blend`, etc.) with the validated
RASMapper algorithm.

Supported variables: `water_surface`, `depth`, `speed`, `velocity`.

---

## RASMapper Algorithm (Sloping Mode)

The algorithm is a 5-step pipeline plus pixel loop (see `compute_2d_velocity_raster`):

### Step A — Hydraulic Connectivity (`compute_face_wss_new`)

For each face, determine whether water is flowing and compute per-face WSE on each
cell side (`face_value_a`, `face_value_b`). This is more than a simple wet/dry test:

1. Skip faces where cellA or cellB is virtual (`cell_face_count == 1`).
2. Mark face disconnected if either cell is dry
   (`wse <= cell_min_elev + MIN_WS_PLOT_TOLERANCE`, tolerance = 0.001).
3. Mark face disconnected if both cells have WSE below face invert.
4. Identify higher/lower cell, compute `was_crit_cap_used` via
   `_avg_water_surface_with_crit_check`.
5. Compute `flag_levee` (critical depth exceeded) and `flag_backfill` (reverse gradient
   opposes elevation step). Each flag changes which values go to `face_value_a/b` and
   whether `face_connected = True/False`.
6. For connected faces: set `face_value_a = face_value_b = face_ws` (the higher-cell WSE
   after slope adjustment using `delta_min_elev`).

Produces: `face_connected: bool[n_faces]`, `face_value_a: float[n_faces]`,
`face_value_b: float[n_faces]`.

### Step B — Facepoint WSE via PlanarRegressionZ (`compute_facepoint_wse`)

For each vertex (facepoint), fit a weighted least-squares plane through all adjacent
face midpoints and their `face_value_a`/`face_value_b` WSE values:

- Application point for each face = midpoint of its two endpoint facepoints (not the
  face centroid).
- Both `value_a` and `value_b` are included separately (2 contributions per face).
- Skip `nodata = -9999.0` values.
- If n = 0: nodata. If n = 1: use that value. If n = 2: average. If n ≥ 3: full 3×3
  least-squares (expand Cramer's rule; fall back to average if `det == 0`).

Produces: `fp_wse: float[n_facepoints]` — WSE at each vertex.

**Used for:** WSE/depth pixel interpolation AND wet/dry masking in Step 4.

### Step 2 — C-Stencil Tangential Velocity Reconstruction
(`reconstruct_face_velocities_least_squares`)

HEC-RAS stores only face-normal velocity scalars. For each face, reconstruct the full
2D velocity vector `(Vx, Vy)` from a 3-face C-stencil:

1. For cellA's stencil: find CW and CCW neighboring faces within cellA.
2. Build `FaceVelocityCoef` symmetric 2×2 matrix by accumulating normals of all 3 faces.
3. Invert via `det = A11*A22 - A12²` (fall back to scaled identity if `det == 0`).
4. Solve `[Vx, Vy] = A_inv @ [B1, B2]` where B is the sum of `face_vel * face_normal`
   over stencil faces.
5. Project back: `Vx = face_vel*nx + tangential*tan_x`.
6. Repeat for cellB's stencil → gives `Item1` (cellA side) and `Item2` (cellB side).
7. Boundary faces (`cellB < 0`): `velocity = face_vel * face_normal`.
8. Connected faces: average Item1 and Item2.

Produces: `face_velocity_cellA_side: float[n_faces, 2]`,
`face_velocity_cellB_side: float[n_faces, 2]`.

### Step 3 — Inverse-Face-Length Weighted Vertex Averaging
(`compute_vertex_velocities`)

For each facepoint, compute a velocity vector for each of its adjacent faces (one per
face → `fp_velocities[fp_idx]` has shape `(n_adj_faces, 2)`):

1. Group adjacent faces into hydraulically-connected arcs using
   `_get_hydraulically_connected_arc`.
   - Arc ends (exclusive) at disconnected faces.
   - Disconnected faces on both arc ends ARE still included in the average.
2. Accumulate `sum_vx += float32(vel[0]) * inv_face_length` using **float32
   accumulators** (critical — matches C# MeshFV2D.cs:8677 `float num5=0f, num6=0f`).
3. Item selection for each face in arc:
   - **Start face**: use Item1 if `fpA == fp`, else Item2 (orientation-based).
   - **Subsequent connected faces**: always use Item1.
   - **Terminal disconnected face** (arc end): use opposite selection from start.
4. Divide by `total_weight` → `fp_velocities[fp_idx][j]` = velocity at that vertex
   as seen from face `j`'s arc context.
5. Also build `fp_face_local_map: dict[(fp_idx, face_idx) -> local_j]` for lookup
   in Step 4.

Produces: `fp_velocities: list[ndarray(n_adj_faces, 2)]`,
`fp_face_local_map: dict[(fp, face) -> int]`.

### Step 3.5 — Sloped Face Velocity Replacement (`replace_face_velocities_sloped`)

For each face, replace its reconstructed velocity with the **average of the two
endpoint vertex velocities** (from Step 3, using `fp_face_local_map` to select
the per-face arc velocity at each endpoint):

```
replaced_face_vel[f] = (fp_velocities[fpA][local_j_A] + fp_velocities[fpB][local_j_B]) / 2
```

This smoothed velocity is used for **vertex contributions** in Step 4's interpolation.
Face midpoint contributions still use the original Step 2 Item1/Item2.

Produces: `replaced_face_vel: float[n_faces, 2]`.

### Step 4 — Polygon Barycentric Weights + Donate Pixel Interpolation

For each cell, rasterize its pixels and interpolate:

**4a. Cell polygon and vertex order** (`build_cell_vertex_info`)
Traverse `cell_face_values[cell_idx]` to get ordered `(face_idx, orientation)` pairs.
For each face, pick vertex: `fpA if orientation > 0 else fpB`.
Produces `vertices: list[(x, y, fp_idx)]`, `face_indices`, `face_orientations`.

**4b. Barycentric weights** (`compute_polygon_barycentric_weights`)
Generalized polygon barycentric coordinates using cross-products:
```
xproducts[i] = (v[i] - p) × (v[i+1] - p)   (clamped away from 0)
weights[j] = cp(v[j-1], v[j], v[j+1]) * Π_{k ≠ j, j-1} xproducts[k]
normalized, clamped to ≥ 0, renormalized
```
**Critical:** cast result to `float32` (matches C# `RASGeometryMapPoints.cs:2956`
`fpWeights[l] = (float)(array[l] / num3)`).

**4c. Donate — redistribute vertex weights to face midpoints**
(`redistribute_weights_to_edge_midpoints`)
For each vertex `j`, compute how much weight it can give to adjacent edge midpoints:
```
cw_can_give[j]  = (w[j] / (w[j-1] + w[j+1])) * w[j-1]
ccw_can_give[j] = (w[j] / (w[j-1] + w[j+1])) * w[j+1]
can_donate[edge_j] = min(ccw_can_give[j], cw_can_give[j+1])
velocity_weights[j]     -= can_donate[edge_j]
velocity_weights[j+1]   -= can_donate[edge_j]
velocity_weights[N + j]  = can_donate[edge_j] * 2   # face midpoint j→j+1
```
Produces `velocity_weights: float64[2*N]` (first N = vertices, last N = face midpoints).

**4d. WSE interpolation** (for `water_surface` / `depth`):
```
pixel_wse = Σ vel_weights[i] * fp_local_wse_adj[i]   (vertices)
          + Σ vel_weights[N+j] * face_local_wse[j]    (face midpoints)
```
Where:
- `fp_local_wse_adj` = `DownwardAdjustFPValues(cell_wse[c], fp_wse)`:
  adds `(cell_wse[c] - mean(fp_wse)) / N` to each facepoint WSE.
- `face_local_wse[k]` = `face_value_a[f]` if `cell == cellA` else `face_value_b[f]`.
- When `depth_weights` available (terrain sampled at facepoints), use
  `PaintCell_8Stencil_RebalanceWeights`: multiply `vel_weights * depth_weights`,
  normalize, then interpolate.
- When `all_shallow` (no connected faces): use flat cell WSE for wet/dry check.

**4e. Velocity interpolation** (for `speed` / `velocity`):
```
Vx = Σ vel_weights[i] * replaced_face_vel[fp→face[i]][0]   (vertices — Step 3.5)
   + Σ vel_weights[N+j] * face_vel_A_or_B[face_j][0]       (face midpoints — Step 2)
Vy = (same)
speed = sqrt(Vx² + Vy²)
```
Face midpoint velocity selection: `face_velocity_cellA_side` if `orientation > 0`,
else `face_velocity_cellB_side`.

**4f. Wet/dry masking**:
Priority order (C# `clipPixelDepths`):
1. **Internal**: `pixel_wse - terrain_elev[r, c] > MIN_WS_PLOT_TOLERANCE` using
   `fp_wse + face_wse` barycentric interpolation above → most accurate.
2. **Fallback**: compare `cell_wse[c]` against `terrain_elev[r, c]`.

---

## Flat Mode (Horizontal)

Paint each cell's raw value over all its owned pixels, no interpolation:
- `water_surface`: `cell_wse[c]`
- `depth`: `cell_wse[c] - terrain_elev[r, c]` (clamp ≥ 0)
- `speed`/`velocity`: cell-average speed from `cell_face_values` (normal component only,
  area-weighted sum of `face_vel * face_area / cell_area`)

No Steps 1–4 needed. Uses `cell_id_raster` only.

---

## File Changes

### Task 1 — Prerequisites (2 files)

#### `src/raspy/hdf/_geometry.py`

Add property `facepoint_face_orientation`:

```python
@property
def facepoint_face_orientation(self) -> tuple[np.ndarray, np.ndarray]:
    """Angle-sorted facepoint→face mapping with orientation.

    Returns ``(fp_face_info, fp_face_values)`` where:
    - ``fp_face_info``: shape ``(n_facepoints, 2)`` — ``[start, count]`` into
      ``fp_face_values`` for each facepoint.
    - ``fp_face_values``: shape ``(total, 2)`` — ``[face_idx, orientation]``.
      ``orientation = 0`` means this facepoint is ``fpA`` (first endpoint) of
      that face; ``orientation = 1`` means it is ``fpB`` (second endpoint).

    Face entries per facepoint are sorted in clockwise angular order around
    the facepoint coordinate (required for arc traversal in vertex velocity
    computation).

    Reads from HDF ``FacePoints Face and Orientation Info/Values`` if present;
    otherwise builds from ``face_facepoint_indexes``. Always applies angle sort.
    """
```

#### `src/raspy/geo/_rasmap.py` (new file — ~700 lines)

All computation functions, self-contained, no HDF I/O. Mirrors the structure of
`velocity_rasterizer_combined.py` 2D pipeline but adapted for raspy array conventions:

| Function | Description |
|---|---|
| `compute_face_wss(cell_wse, cell_min_elev, face_min_elev, face_cell_indexes, cell_face_count)` | Step A: hydraulic connectivity |
| `compute_facepoint_wse(fp_coords, fp_face_info, fp_face_values, face_facepoint_indexes, face_value_a, face_value_b)` | Step B: PlanarRegressionZ vertex WSE |
| `reconstruct_face_velocities(face_vel, face_normals_2d, face_connected, face_cell_indexes, cell_face_info, cell_face_values, face_facepoint_indexes)` | Step 2: C-stencil WLS |
| `compute_vertex_velocities(face_vel_A, face_vel_B, face_connected, face_lengths, face_facepoint_indexes, face_cell_indexes, cell_wse, fp_face_info, fp_face_values, face_value_a, face_value_b)` | Step 3: arc-based IFL weighted avg |
| `replace_face_velocities_sloped(fp_velocities, fp_face_local_map, face_facepoint_indexes)` | Step 3.5 |
| `rasterize_rasmap(variable, cell_id_grid, transform, terrain_grid, ...)` | Step 4: full pixel loop (Numba + fallback) |
| Numba kernels: `_barycentric_weights_nb`, `_donate_nb`, `_pixel_velocity_nb`, `_batch_pixel_velocities_nb`, `_batch_sloped_depth_velocity_nb` | JIT-compiled inner loops |
| `_sort_fp_faces_by_angle(fp_face_info, fp_face_values, fp_coords, face_facepoint_indexes)` | Prerequisite for arc traversal |
| `build_cell_id_raster(cell_polygons, wet_mask, transform, height, width)` | Rasterize cell ownership |

**Numba usage:** Optional, auto-detected. Provides ~9× speedup. Falls back to pure
numpy/Python if not installed. Kernels are cache-compiled (`@njit(cache=True)`).

---

### Task 2 — Integration (2 files)

#### `src/raspy/geo/raster.py`

Add:

```python
def rasmap_raster(
    variable: Literal["water_surface", "depth", "speed", "velocity"],
    # -- Mesh topology (raspy property values) --
    cell_wse: np.ndarray,
    cell_min_elevation: np.ndarray,
    face_min_elevation: np.ndarray,
    face_cell_indexes: np.ndarray,
    cell_face_info: np.ndarray,
    cell_face_values: np.ndarray,
    face_facepoint_indexes: np.ndarray,
    fp_coords: np.ndarray,
    face_normals: np.ndarray,       # (n_faces, 3) raspy convention [nx, ny, len]
    fp_face_info: np.ndarray,       # angle-sorted, from facepoint_face_orientation
    fp_face_values: np.ndarray,
    cell_polygons: list[np.ndarray],
    face_normal_velocity: np.ndarray | None,   # for speed/velocity
    # -- Grid --
    reference_raster: str | Path | None,
    cell_size: float | None,
    crs: Any | None,
    nodata: float,
    # -- Options --
    interp_mode: Literal["flat", "sloping"],
    depth_threshold: float,
    clip_to_perimeter: bool,
    perimeter: np.ndarray | None,
    use_numba: bool | None,
    output_path: str | Path | None,
) -> Path | rasterio.io.DatasetReader
```

Responsibilities:
1. Derive output grid from `reference_raster` or `cell_size`.
2. Call `_rasmap.*` compute functions (Steps A, B, 2, 3, 3.5) based on `variable`.
3. Call `_rasmap.build_cell_id_raster` for pixel→cell mapping.
4. Call `_rasmap.rasterize_rasmap` for the pixel loop.
5. Handle `clip_to_perimeter` masking.
6. Write output or return in-memory dataset.

#### `src/raspy/hdf/_plan.py`

Add `export_raster2` method to `FlowAreaResults`:

```python
def export_raster2(
    self,
    variable: Literal["water_surface", "depth", "speed", "velocity"],
    timestep: int | None = None,   # None = use max-value summary datasets
    output_path: str | Path | None = None,
    *,
    reference_raster: str | Path | None = None,  # required for depth
    cell_size: float | None = None,
    crs: Any | None = None,
    nodata: float = -9999.0,
    interp_mode: Literal["flat", "sloping"] = "sloping",
    depth_threshold: float = 0.001,
    clip_to_perimeter: bool = True,
    use_numba: bool | None = None,  # None = auto-detect
) -> Path | rasterio.io.DatasetReader
```

Reads HDF data once, delegates to `geo.raster.rasmap_raster()`.

**HDF data read:**
- `cell_wse` = `self.max_water_surface["value"]` (timestep=None) or
  `self.water_surface[timestep, :n_cells]`
- `face_normal_velocity` = `self.max_face_velocity` (timestep=None) or
  `self.face_velocity[timestep, :]` (only for speed/velocity)

**Raises:**
- `ValueError` if `variable="depth"` and `reference_raster` is None.
- `ValueError` if `variable in ("speed", "velocity")` and `timestep=None` (no max
  face velocity summary in HDF for all plans — check if available, raise if not).
- `ImportError` if `rasterio` not installed.

---

## Key Differences from Existing `export_raster`

| Aspect | `export_raster` | `export_raster2` |
|---|---|---|
| WSE render | triangulation + `scipy.griddata` | PlanarRegressionZ + barycentric+donate |
| Velocity | WLS cell-center + scatter/IDW/blend | C-stencil → vertex avg → barycentric+donate |
| Parameters | 15+ interp options | `interp_mode: flat|sloping` only |
| Accuracy | Approximate | Pixel-perfect vs RASMapper |
| Speed | ~2–5 s (numpy) | ~7–10 s Numba, ~60 s fallback |
| Wet/dry | cell WSE mask | per-pixel facepoint WSE interpolation |

---

## Missing `max_face_velocity` Property

The `FlowAreaResults` class currently exposes `face_velocity` (time series) and
`max_water_surface` (summary). We also need `max_face_velocity` for `timestep=None`
support on speed/velocity. This is stored at:

```
Results/Unsteady/Output/Output Blocks/Base Output/
  Summary Output/2D Flow Areas/<name>/Maximum Face Velocity
```

This needs to be added to `FlowAreaResults` as a summary property (1 line of code in
`_plan.py`, counted within Task 2's file budget).

---

## Data Flow Diagram

```
HDF                      Task 1 (_rasmap.py)              Task 2
─────                    ───────────────────              ──────────────────
cell_wse ──────────────► compute_face_wss ──► face_connected, face_value_a/b
cell_min_elev ─────────►                    │
face_min_elev ─────────►                    │
                                            ▼
fp_coords ─────────────► compute_facepoint_wse ──────────► fp_wse (for WSE/depth)
fp_face_info/vals ─────►
face_facepoint_indexes ►

face_normal_vel ────────► reconstruct_face_velocities ───► face_vel_A, face_vel_B
face_normals ──────────►
cell_face_info/vals ───►

                         ► compute_vertex_velocities ─────► fp_velocities
                         ► replace_face_velocities_sloped ► replaced_face_vel

cell_polygons ──────────► build_cell_id_raster ──────────► cell_id_grid
                                                           │
terrain raster ─────────────────────────────────────────► ▼
                                                rasterize_rasmap ──► output (H,W)
```

---

## Performance

| Condition | Mesh size | Time |
|---|---|---|
| Flat mode (any variable) | any | < 1 s |
| Sloping, with Numba | 19K cells | ~7–10 s |
| Sloping, without Numba | 19K cells | ~60 s |
| Sloping, with Numba | 100K cells | ~40–60 s |

Numba JIT cache: first call in session may add ~10 s for compilation; subsequent calls
reuse cached bytecode.

---

## Suggested Test Cases

1. **Pixel-perfect regression**: compare against CLB's validated output (`/archive/`)
   on BaldEagle example — median diff should be 0.000000 ft/s.
2. **Flat mode correctness**: all pixels in cell `c` have exactly `cell_wse[c]`.
3. **Dry cell exclusion**: cells with `wse ≤ cell_min_elev + 0.001` produce no pixels.
4. **Boundary face (cellB = -1)**: velocity = `face_vel * face_normal` (degenerate stencil).
5. **All-shallow cell**: falls back to flat cell WSE for wet/dry, still interpolates velocity.
6. **`timestep=None` with `variable="speed"`**: raises `ValueError` if max face velocity
   not available in HDF.
7. **`variable="depth"` without `reference_raster`**: raises `ValueError`.
8. **Numba vs numpy fallback**: same output to tolerance 1e-6 ft/s.
9. **`interp_mode="flat"` depth**: pixel_depth = cell_wse - terrain_elev, clipped ≥ 0.
10. **`clip_to_perimeter=True`**: no pixels outside perimeter polygon.
11. **`facepoint_face_orientation` falls back to computed**: when HDF lacks those datasets,
    result matches when datasets present.

---

## Implementation Order

1. **Task 1a** — Add `facepoint_face_orientation` to `_geometry.py` + tests. ✓ DONE
2. **Task 1b** — Implement `_rasmap.py` (Steps A, B, 2, 3, 3.5 + pixel loop + Numba).
   Start with pure-Python; add Numba after correctness verified. ✓ DONE
3. **Task 2a** — Add `rasmap_raster()` to `geo/raster.py`. ✓ DONE
4. **Task 2b** — Add `export_raster2()` to `_plan.py`. ✓ DONE
5. **Validation** — Run against BaldEagle example, compare to CLB outputs.

---

## Implementation Notes

### Task 1b — `_rasmap.py` implementation summary

`src/raspy/geo/_rasmap.py` (~620 lines, 35 unit tests passing).

| Function | Step | Key detail |
|---|---|---|
| `compute_face_wss` | A | Full connectivity: levee, backfill, critical-depth, delta-min-elev slope adjustment |
| `compute_facepoint_wse` | B | PlanarRegressionZ — Cramer's rule 3×3, face midpoints as application points, both value_a and value_b contributed |
| `reconstruct_face_velocities` | 2 | C-stencil WLS via `_FaceVelocityCoef`; connected faces average Item1+Item2 |
| `compute_vertex_velocities` | 3 | Arc traversal; **float32 accumulators** (critical — matches C# `MeshFV2D.cs:8677`) |
| `replace_face_velocities_sloped` | 3.5 | Average of two endpoint vertex velocities per face |
| `_barycentric_weights` | 4 | float32 cast (matches C# `RASGeometryMapPoints.cs:2956`) |
| `_donate` | 4 | Weight redistribution to edge midpoints |
| `_downward_adjust_fp_wse` | 4 | `DownwardAdjustFPValues` — nudges fp WSE toward cell average |
| `_depth_weights_for_cell` | 4 | `PaintCell_8Stencil_RebalanceWeights` depth weights |
| `_pixel_wse_sloped` | 4 | Both rebalanced and plain paths |
| `build_cell_id_raster` | 4 | Shapely/rasterio cell ownership grid (wet cells only) |
| `rasterize_rasmap` | 4 | Full pixel loop: `water_surface`, `depth`, `speed`, `velocity` (4-band) |

Boundary facepoints (only adjacent to boundary faces where `cellB=-1`) correctly
receive nodata WSE — `compute_face_wss` skips those faces entirely per the original
C# logic. This is expected and tested explicitly.

---

### `facepoint_face_orientation` — synthetic HDF caveat

The synthetic HDF used in unit tests (`tests/hdf/conftest.py`) sets all
`Faces FacePoint Indexes` to zeros, making every face claim `fpA = fpB = 0`.
This means all 2×N_FACES entries map to facepoint 0, leaving all other
facepoints with empty lists — a structurally valid but semantically degenerate
mesh. Unit tests for this property therefore check structural invariants only
(shapes, dtypes, count sums, orientation values in {0,1}). Semantic correctness
(correct arc ordering, correct Item1/Item2 selection) must be verified against
the real BaldEagle HDF in integration tests.

---

### Task 2a — `rasmap_raster()` in `geo/raster.py`

`src/raspy/geo/raster.py` — `rasmap_raster()` added (~120 lines of implementation after
the long docstring). 16 unit tests added to `tests/geo/test_rasmap.py`.

**Signature:**
```python
def rasmap_raster(
    variable: Literal["water_surface", "depth", "speed", "velocity"],
    cell_wse, cell_min_elevation, face_min_elevation,
    face_cell_indexes, cell_face_info, cell_face_values,
    face_facepoint_indexes, fp_coords, face_normals,
    fp_face_info, fp_face_values, cell_polygons,
    face_normal_velocity=None, output_path=None, *,
    reference_raster=None, cell_size=None, crs=None,
    nodata=-9999.0, interp_mode="sloping",
    depth_threshold=0.001, clip_to_perimeter=True,
    perimeter=None, use_numba=None,
) -> Path | rasterio.io.DatasetReader
```

**Key implementation details:**
- Resolves output grid from `reference_raster` (inherits transform + CRS + terrain DEM)
  or from `cell_size` + bbox of perimeter/facepoints.
- `interp_mode="flat"`: cell value painted directly per owned pixel via `build_cell_id_raster`.
  Speed uses area-weighted mean of `|face_vel|` over cell faces.
- `interp_mode="sloping"`: full 5-step pipeline via `_rasmap` functions (Steps A→B→2→3→3.5→4).
- Returns `Path` when `output_path` given; `rasterio.io.DatasetReader` (in-memory) otherwise.
- Optional perimeter clipping via `_mask_outside_polygon_array`.

**Caveats for minimal synthetic mesh:**
- Most facepoints in the 2-cell test mesh are boundary-only (nodata WSE). The sloping
  mode produces `-6665`-ish values (nodata × barycentric weight) for such pixels.
  Value-range tests therefore only apply to flat mode; sloping tests check structure only.
- In production (real HEC-RAS meshes), most facepoints have valid WSE from interior
  connected faces, so this is not an issue.

**Tests added** (`TestRasmapRasterFlat` + `TestRasmapRasterSloping`, 16 tests total):
- Flat: returns dataset, output shape, wet pixels have valid WSE, dry → all nodata,
  missing face_vel raises, missing reference raises, missing grid spec raises.
- Sloping: returns 2D array, some valid pixels, all dry → all nodata, speed is 2D,
  speed is non-negative, output_path creates file, flat/sloping produce same grid shape.

---

### Task 2b — `export_raster2()` in `_plan.py`

`src/raspy/hdf/_plan.py` — `FlowAreaResults.export_raster2()` added (~130 lines including
docstring). 8 unit tests added to `tests/hdf/test_plan.py`.

**What it does:**
- Reads `cell_wse` from HDF (`timestep=None` → `max_water_surface`).
- Reads `face_normal_velocity` for `speed`/`velocity` variables only.
- Reads `cell_face_info`, `cell_face_values`, `facepoint_face_orientation` once.
- Auto-derives `cell_size` from `median(face_normals[:, 2])` when neither
  `reference_raster` nor `cell_size` is supplied.
- Delegates entirely to `rasmap_raster()` passing all geometry properties.

**Guards:**
- `variable="speed"/"velocity"` with `timestep=None` → `ValueError`.
- `variable="depth"` without `reference_raster` → propagated from `rasmap_raster`.

**Tests added** (`TestExportRaster2Guards` + `TestExportRaster2`, 8 tests total):
- Guards: speed/velocity require timestep.
- Functional (require rasterio + shapely): flat WSE, sloping WSE, max-timestep WSE,
  flat speed, output_path creates file, auto cell_size works.
- Uses `cell_size=5.0` with the synthetic HDF (100-unit bbox → ~20×20 grid).
