# Channel Ground-Line Modification

This document describes the `Channel` subtype of RasMapper ground-line
modifications and how it is implemented in `raspy`.

---

## 1  Overview

Ground-line modifications are stored in the `Modifications/` HDF group and
identified by a `Subtype` attribute.  `raspy` supports two subtypes:

| Subtype | Shape | Default elevation op | Physical use |
|---|---|---|---|
| `Levee` | ⋀ mound — flat crest, sides slope **down** | `SetIfHigher` | Embankments, dikes |
| `Channel` | ∨ trough — flat bottom, sides slope **up** | `SetIfLower` | Stream channels, ditches |

Both subtypes share the same HDF structure and attribute fields; only the
cross-section geometry and default elevation operation differ.

---

## 2  HDF structure

Path: `Modifications/{name}/`

| Item | Kind | Description |
|---|---|---|
| `@Type` | attribute `str` | Always `"Levee"` (the HDF type for all ground-line mods) |
| `@Subtype` | attribute `str` | `"Levee"` or `"Channel"` |
| `@Priority` | attribute `int` | Lower number = applied last = higher precedence |
| `Attributes` | structured array (1 row) | Per-feature parameters (see below) |
| `Polyline Points` | float64 `(N, 2+)` | Centerline vertices — only columns 0 (X) and 1 (Y) are used |
| `Profile Values` | float64 `(M, 2)` | `[cumulative_station, elevation]` pairs along the centerline |

### 2.1  `Attributes` fields

| Field name | Type | Description |
|---|---|---|
| `Elevation Type` | string | `SetIfHigher`, `SetIfLower`, `SetValue`, or `Add` |
| `Top Width` | float32 | Full width of the flat bottom/crest (metres or feet) |
| `Left Slope` | float32 | H:V slope of the left side (positive; 0 = vertical wall) |
| `Right Slope` | float32 | H:V slope of the right side (positive; 0 = vertical wall) |
| `Max Reach` | float32 | Maximum lateral extent from the centerline |

---

## 3  Cross-section geometry

For each pixel within the bounding box of a modification the algorithm:

1. Projects the pixel onto the centerline polyline to obtain a signed
   perpendicular distance `d` (positive = left of travel direction) and a
   cumulative station `s`.
2. Interpolates the **crest/bottom elevation** `z` from `Profile Values` at
   station `s`.
3. Computes the modification elevation from the trapezoidal cross-section:

```
half_w = Top Width / 2

|d| <= half_w          →  mod_elev = z                          (flat zone)
half_w < |d| <= Max Reach →  drop = (|d| − half_w) / slope
                              Levee:   mod_elev = z − drop       (slopes down)
                              Channel: mod_elev = z + drop       (slopes up)
|d| > Max Reach        →  pixel not affected (mod_elev = NaN)
```

The only algorithmic difference between Levee and Channel is the **sign of
`drop`**: negative for Levee (sides fall away from the crest), positive for
Channel (sides rise away from the trough).

4. Applies the `Elevation Type` rule:

| `Elevation Type` | Result |
|---|---|
| `SetIfHigher` | `max(original, mod_elev)` |
| `SetIfLower` | `min(original, mod_elev)` |
| `SetValue` | `mod_elev` unconditionally |
| `Add` | `original + mod_elev` |

Pixels whose original value equals NoData always receive `mod_elev`
regardless of the rule.

---

## 4  Slope normalisation

Slopes stored in HDF are the user-entered positive H:V values.  Before use:

- `NaN` → `3.0` (RasMapper default)
- `0.0` → `inf` (vertical wall — drop term becomes zero)

---

## 5  Suggested test cases

### 5.1  Basic Channel lowers terrain

Create a synthetic terrain (flat at elevation 10.0), a Channel modification
with bottom elevation 5.0, `Top Width = 4`, `Left Slope = Right Slope = 2`,
`Max Reach = 8`, `Elevation Type = SetIfLower`.

Expected results:
- Pixels within `|d| <= 2`: elevation = 5.0
- Pixels at `|d| = 4` (2 m into the slope): elevation = 5.0 + (4−2)/2 = 6.0
- Pixels at `|d| = 8` (outer edge): elevation = 5.0 + (8−2)/2 = 8.0
- Pixels beyond `Max Reach`: elevation = 10.0 (unchanged)

### 5.2  Channel does not raise terrain

Same setup, but original terrain is flat at **4.0** (already below the
channel bottom of 5.0).  With `SetIfLower` the original must not be raised:
all pixels should remain at 4.0.

### 5.3  Channel vs. Levee symmetry

Given the same centerline, profile, width, and slopes:

- Levee with `SetIfHigher` should produce `original + drop` on the sloping
  sides relative to the crest.
- Channel with `SetIfLower` should produce the mirror-image depression.

The absolute difference `|levee_pixel − original|` should equal
`|channel_pixel − original|` for the same `|d|` on flat terrain.

### 5.4  Vertical walls (`slope = 0`)

`Top Width = 4`, `Left Slope = Right Slope = 0`, `Max Reach = 8`.

- Within `|d| <= 2`: modified elevation.
- At `|d| = 2 + ε` (just outside the flat zone): `drop = ε / inf = 0`, so
  elevation = crest/bottom with zero slope effect — step change at the wall
  edge, then falls outside `Max Reach` at `|d| > 8`.

### 5.5  `SetValue` elevation type

Channel with `Elevation Type = SetValue` on terrain with varying elevation.
All pixels within `Max Reach` should be forced to `mod_elev` regardless of
whether `mod_elev` is above or below the original terrain.

### 5.6  Multiple overlapping modifications (priority ordering)

A Channel (priority 1) and a Levee (priority 2) both covering the same
pixels.  The Channel (lower priority number = applied last) should win at
overlapping pixels.

### 5.7  NoData pixels receive modification value

Create a mosaic with NoData (`-9999`) in the channel footprint.  After
applying a Channel modification, NoData pixels inside the footprint should be
filled with `mod_elev` (not remain as NoData).

### 5.8  VRT sidecar contains only modified pixels

When `export_terrain` writes a `.vrt`, the `_mods.tif` sidecar should be
NoData at all pixels that were unaffected and should match the GeoTIFF output
at pixels that were modified.
