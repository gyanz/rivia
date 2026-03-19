# RasProcess.exe StoreAllMaps — Render Mode Limitation

## Summary

`RasProcess.exe -Command=StoreAllMaps` (and `-Command=StoreMap`) **always produces rasters using
basic "sloping" (cell-corners only) interpolation**, regardless of the `<RenderMode>`,
`<UseDepthWeightedFaces>`, or `<ReduceShallowToHorizontal>` settings in the `.rasmap` file.
These settings are silently ignored by the scripting commands.

This means `raspy.model.Model.store_map()`, which delegates to `RasProcess.exe`, always
generates maps equivalent to `rasterize_rasmap(..., with_faces=False, use_depth_weights=False,
shallow_to_flat=True)`.

---

## How RenderMode Works in the Interactive GUI

The `.rasmap` file stores three top-level render-mode elements:

```xml
<RenderMode>slopingPretty</RenderMode>
<UseDepthWeightedFaces>true</UseDepthWeightedFaces>
<ReduceShallowToHorizontal>true</ReduceShallowToHorizontal>
```

Valid values for `<RenderMode>`:

| Value | Description |
|---|---|
| `horizontal` | Flat per-cell water surface. No interpolation across cell edges. |
| `sloping` | Sloping interpolation using cell-corner facepoints only. |
| `slopingPretty` | Sloping using cell corners **and** face centroids (`CellStencilMethod.WithFaces`). Supports depth-weighted faces and shallow reduction. |

`<UseDepthWeightedFaces>` and `<ReduceShallowToHorizontal>` are only meaningful when
`<RenderMode>` is `slopingPretty`.

These settings are read exclusively by the interactive RasMapper GUI in
`RASMapper.XMLLoad()` (source: `RasMapperLib/RasMapperLib/RASMapper.cs`, ~line 12876):

```csharp
switch (XML.StringTextElement(node, "RenderMode"))
{
    case "horizontal":
        SharedData.SetHorizontalRenderingMode();
        break;
    case "sloping":
    case "hybrid":
        SharedData.SetSlopingRenderingMode();
        break;
    default:  // "slopingPretty" or absent → always falls here
        bool useDepthWeightedFaces =
            XML.StringTextElement(node, "UseDepthWeightedFaces").RoughlyEquals("true");
        SharedData.SetSlopingPrettyRenderingMode(
            XML.StringTextElement(node, "ReduceShallowToHorizontal").RoughlyEquals("true"),
            useDepthWeightedFaces);
        break;
}
```

---

## What StoreAllMapsCommand Actually Does

Source: `RasMapperLib/RasMapperLib.Scripting/StoreAllMapsCommand.cs`

```csharp
public override void Execute(ProgressReporter prog = null)
{
    using (new SetSRSHelper(RasMapFilename))  // only sets coordinate reference system
    {
        SharedData.RasMapFilename = RasMapFilename;
        XElement xElement = XElement.Load(RasMapFilename);  // reads <Terrains> and <Results> only
        // ... iterates result map layers and calls rASResultsMap.StoreMap() ...
    }
}
```

`SetSRSHelper` only sets the spatial reference system from the rasmap. It does not read render
mode settings.

`StoreAllMapsCommand` never calls `RASMapper.XMLLoad()` — it reads the rasmap file only to find
terrain filenames and result layer definitions. The `<RenderMode>` element is never accessed.

`ExecuteWith()` (the entry point for command-line arguments) accepts only two parameters:
- `RasMapFilename` — path to the `.rasmap` file
- `ResultFilename` — path to the plan HDF file (optional filter)

There is no argument for render mode.

---

## SharedData Defaults (Used by StoreAllMaps)

When `RasProcess.exe` starts, `SharedData` static fields are initialized to
(source: `SharedData.cs`, module-level initialization, ~line 1926):

```csharp
CellRenderMode        = CellRenderMethod.Sloping;
CellStencilMode       = CellStencilMethod.JustFacepoints;  // NOT WithFaces
FaceWSMode            = FaceWSMethod.Adjusted;
FacepointAdjustmentMode = FacepointAdjustmentMethod.None;
ShallowBehaviorMode   = ShallowBehavior.ReduceToHorizontal;
UseDepthWeightedFaces = false;
```

`CellStencilMethod.JustFacepoints` means only cell-corner facepoints are used in the
interpolation stencil — face centroids are excluded. This is the "sloping (cell corners)"
mode in the RasMapper options dialog.

---

## Render Mode Internal State Machine

`SharedData` exposes three render-mode setters
(source: `SharedData.cs`, ~line 1765):

| Setter | CellRenderMode | CellStencilMode | UseDepthWeightedFaces | ShallowBehavior |
|---|---|---|---|---|
| `SetHorizontalRenderingMode()` | `Horizontal` | `JustFacepoints` | false | None |
| `SetSlopingRenderingMode()` | `Sloping` | `JustFacepoints` | false | None |
| `SetSlopingPrettyRenderingMode(reduceShallow, depthWeights)` | `Sloping` | **`WithFaces`** | per argument | per argument |

Only `slopingPretty` mode uses `CellStencilMethod.WithFaces` and can enable depth-weighted faces
or shallow-to-horizontal reduction.

`WaterSurfaceRenderer` branches on these values at render time:

```csharp
// in WaterSurfaceRenderer.cs
if (SharedData.CellRenderMode == SharedData.CellRenderMethod.Horizontal)
    // ... flat rendering
if (SharedData.CellRenderMode == SharedData.CellRenderMethod.Sloping)
    // ... sloping rendering (corners or corners+faces depending on CellStencilMode)
if (SharedData.CellStencilMode == SharedData.CellStencilMethod.WithFaces
    && SharedData.UseDepthWeightedFaces)
    // ... compute depth-weighted face weights
```

---

## Mapping to `rasterize_rasmap` Parameters

The Python `rasterize_rasmap()` function in `src/raspy/geo/_rasmap.py` implements the same
interpolation logic and directly mirrors the RasMapper modes:

| RasMapper option | `<RenderMode>` XML | `rasterize_rasmap` equivalent |
|---|---|---|
| Horizontal | `horizontal` | `with_faces=False` + override to flat (not yet implemented) |
| Sloping (Cell Corners) | `sloping` | `with_faces=False, shallow_to_flat=True, use_depth_weights=False` |
| Sloping (Cell Corners + Face Centers) | `slopingPretty` | `with_faces=True, shallow_to_flat=False, use_depth_weights=False` |
| Sloping + Shallow Reduces to Horizontal | `slopingPretty` + `<ReduceShallowToHorizontal>true</ReduceShallowToHorizontal>` | `with_faces=True, shallow_to_flat=True, use_depth_weights=False` |
| Sloping + Depth-Weighted Faces | `slopingPretty` + `<UseDepthWeightedFaces>true</UseDepthWeightedFaces>` | `with_faces=True, shallow_to_flat=False, use_depth_weights=True` |

`store_map()` output is always equivalent to **Sloping (Cell Corners)**:
`with_faces=False, shallow_to_flat=True, use_depth_weights=False`.

---

## GUI RasMapper Call Chain

The interactive RasMapper GUI uses the **same rendering engine** as `RasProcess.exe` — the
difference is solely in how `SharedData` render mode is initialized before rendering begins.

**GUI call chain (triggered from the "Store Map" menu):**

```
ManageResultsMaps.StoreMapPopup()          // RasMapperLib/ManageResultsMaps.cs
  → RASResultsMap.StoreMap(reporter)       // RasMapperLib/RASResultsMap.cs (~line 5530)
      → MapProcessingEngine.StoreMap(...)  // RasMapperLib/MapProcessingEngine.cs (~line 1320)
          → Renderer.GetSimplifiedComputer(...)   // RasMapperLib.Render/Renderer.cs (~line 1674)
              → WaterSurfaceRenderer.GetComputer(...)   // for WSE maps
              → DepthRenderer.GetComputer(...)          // for depth maps
              → VelocityRenderer.GetComputer(...)       // for velocity maps
```

The GUI sets `SharedData` render mode **before** reaching this chain:

```
RASMapper.XMLLoad()  // called at application startup when a rasmap is opened
  → reads <RenderMode>, <UseDepthWeightedFaces>, <ReduceShallowToHorizontal>
  → calls SharedData.SetHorizontalRenderingMode()
       or SharedData.SetSlopingRenderingMode()
       or SharedData.SetSlopingPrettyRenderingMode(reduceShallow, depthWeights)
```

**Why `RasProcess.exe` produces different output:**

Both paths call identical `MapProcessingEngine.StoreMap()` code. The only difference is
`SharedData` state at the time of the call:

| Caller | SharedData state when StoreMap() runs |
|---|---|
| GUI (`ManageResultsMaps`) | Set by `RASMapper.XMLLoad()` from rasmap XML |
| `RasProcess.exe` (`StoreAllMapsCommand`) | Module-level defaults (basic `Sloping`, `JustFacepoints`) |

There is no per-call render mode argument to `MapProcessingEngine.StoreMap()` — it reads
`SharedData` global state implicitly. Since `RasProcess.exe` never calls `RASMapper.XMLLoad()`,
`SharedData` is never updated from the rasmap, and the defaults apply regardless of what
the `.rasmap` file contains.

---

## Recommendation

When full render-mode fidelity is needed (especially `slopingPretty` with depth-weighted faces
to match RasMapper's default display), use `rasterize_rasmap()` from `raspy.geo` directly
instead of `store_map()` / `export_wse()`.

`rasterize_rasmap()` is also faster for single-variable exports because it avoids launching a
subprocess and does not require a terrain HDF — it rasterizes directly from the HDF mesh
geometry and result arrays.

`store_map()` remains useful when:
- You need a persistent VRT + GeoTIFF tile set on disk (native RasProcess format)
- You need variables not yet supported by `rasterize_rasmap`
- You need exact byte-for-byte parity with RasMapper's stored-map output files

---

## Source Files Referenced

All source files are decompiled VB.NET/C# from `RasMapperLib.dll` v2.0.0.0 (HEC-RAS 6.6),
located in `archive/DLLs/RasMapperLib/`.

| File | Topic |
|---|---|
| `RasMapperLib.Scripting/StoreAllMapsCommand.cs` | `StoreAllMaps` entry point — does not read render mode |
| `RasMapperLib.Scripting/StoreMapCommand.cs` | `StoreMap` entry point — does not read render mode |
| `RasMapperLib.Scripting/SetSRSHelper.cs` | Only sets coordinate reference system from rasmap |
| `RasMapperLib/RASMapper.cs` (~line 12876) | GUI `XMLLoad` — only place `<RenderMode>` is read |
| `RasMapperLib/SharedData.cs` (~line 1765) | `SetHorizontalRenderingMode`, `SetSlopingRenderingMode`, `SetSlopingPrettyRenderingMode` |
| `RasMapperLib/MapperOptionWSRenderMode.cs` | GUI dialog — `LoadCellRenderMode` / `StoreCellRenderMode` |
| `RasMapperLib.Render/WaterSurfaceRenderer.cs` | Branches on `SharedData.CellRenderMode` / `UseDepthWeightedFaces` |
| `RasMapperLib.Render/SlopingFactors.cs` | Computes face-point water surface factors for sloping modes |
| `RasMapperLib/ManageResultsMaps.cs` | GUI entry point — `StoreMapPopup()` → `RASResultsMap.StoreMap()` |
| `RasMapperLib/RASResultsMap.cs` (~line 5530) | `StoreMap()` — shared by GUI and scripting layer |
| `RasMapperLib/MapProcessingEngine.cs` (~line 1320) | Core raster generation engine — called by both GUI and RasProcess |
| `RasMapperLib.Render/Renderer.cs` (~line 1674) | `GetSimplifiedComputer()` — dispatches to variable-specific renderer |
