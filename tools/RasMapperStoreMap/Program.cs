/*
 * RasMapperStoreMap.exe
 *
 * Headless stored-map generator for HEC-RAS that respects the <RenderMode>,
 * <UseDepthWeightedFaces>, <ReduceShallowToHorizontal>, and <TightExtent>
 * settings.  RasProcess.exe ignores render-mode state (SharedData defaults to
 * basic Sloping/JustFacepoints); this stub reads them and calls the appropriate
 * SharedData setter before rendering.
 *
 * Two execution paths:
 *
 *   UseDepthWeightedFaces=false, TightExtent=false:
 *     Delegates to StoreAllMapsCommand.Execute() — the simplest path.
 *
 *   UseDepthWeightedFaces=true OR TightExtent=true (default: both true):
 *     Bypasses StoreAllMapsCommand; runs a custom loop over rasmap XML layers:
 *     1. Creates RASResults and ensures terrain is set.
 *     2. If UseDepthWeightedFaces=true: writes FacePoint Elevation data to
 *        PostProcessing.hdf via PostProcessor.EnsureFacepointElevations(), then
 *        populates the in-memory RASD2FlowArea cache via
 *        EnsureGetPerMeshFacepointElevations() on the SAME RASResults instance.
 *        This is necessary because WaterSurfaceRenderer.GetComputer() reads
 *        PerMeshFacepointElevations (a per-instance field) before the normal
 *        code path that populates it runs.
 *        See docs/rasprocess_render_mode_limitation.md.
 *     3. Constructs RASResultsMap with that SAME RASResults (shares the cache).
 *     4. If TightExtent=true: calls StoreMapTerrainResample with the model's
 *        geometry extent (D2FlowArea + XS + StorageArea bounding box) so output
 *        tiles are pixel-aligned and clipped to the model footprint — matching
 *        RasMapper's "Original Extent" behaviour.
 *        If TightExtent=false: calls StoreMap, which produces terrain-extent
 *        tiles (NoData outside the model, but tiles as large as the terrain).
 *     The two flags compose: UseDepthWeightedFaces and TightExtent each address
 *     independent concerns and work correctly together or separately.
 *
 * RasMapperLib.dll is loaded dynamically at runtime from the HEC-RAS installation
 * directory — this project has zero compile-time references to HEC-RAS assemblies.
 *
 * Usage:
 *   RasMapperStoreMap.exe
 *       -RasMapFilename=<path>
 *       [-ResultFilename=<path>]
 *       [-RenderMode=sloping|slopingPretty|horizontal]   (default: sloping)
 *       [-UseDepthWeightedFaces=true|false]              (default: false)
 *       [-ReduceShallowToHorizontal=true|false]          (default: true)
 *       [-TightExtent=true|false]                        (default: true)
 *       [-RasMapperLibDir=<HEC-RAS install dir>]
 *
 * If -RasMapperLibDir is omitted the exe searches:
 *   1. Environment variable HEC_RAS_LIB_DIR
 *   2. Common HEC-RAS installation paths (6.6 down to 6.1)
 *
 * Source references:
 *   archive/DLLs/RasMapperLib/RasMapperLib.Scripting/StoreAllMapsCommand.cs
 *   archive/DLLs/RasMapperLib/RasMapperLib/SharedData.cs  (~line 1765, ~line 1926)
 *   archive/DLLs/RasMapperLib/RasMapperLib/RASMapper.cs   (~line 12876)
 *   archive/DLLs/RasMapperLib/RasMapperLib/PostProcessor.cs (~line 532)
 *   archive/DLLs/RasMapperLib/RasMapperLib/RASD2FlowArea.cs (~line 3834)
 *   archive/DLLs/RasMapperLib/RasMapperLib.Render/WaterSurfaceRenderer.cs (~line 2055)
 *   archive/DLLs/RasMapperLib/RasMapperLib/MapProcessingEngine.cs (~line 1346)
 *   archive/DLLs/RasMapperLib/RasMapperLib/InterpolatedLayer.cs (~line 2843)
 *   docs/rasprocess_render_mode_limitation.md
 */

using System.Linq.Expressions;
using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;

// ── Argument parsing ──────────────────────────────────────────────────────────

string? rasMapFilename = null;
string? resultFilename = null;
string  renderMode               = "sloping";
bool    useDepthWeightedFaces    = false;
bool    reduceShallowToHorizontal = true;
bool    tightExtent               = true;
string? libDir = null;

foreach (var arg in args)
{
    if      (Starts(arg, "-RasMapFilename="))           rasMapFilename            = Val(arg);
    else if (Starts(arg, "-ResultFilename="))            resultFilename            = Val(arg);
    else if (Starts(arg, "-RenderMode="))                renderMode                = Val(arg);
    else if (Starts(arg, "-UseDepthWeightedFaces="))     useDepthWeightedFaces     = Bool(arg);
    else if (Starts(arg, "-ReduceShallowToHorizontal=")) reduceShallowToHorizontal = Bool(arg);
    else if (Starts(arg, "-TightExtent="))               tightExtent               = Bool(arg);
    else if (Starts(arg, "-RasMapperLibDir="))           libDir                    = Val(arg);
}

if (rasMapFilename is null)
{
    Console.Error.WriteLine(
        "RasMapperStoreMap: stores HEC-RAS result maps with full render-mode fidelity.\n" +
        "\n" +
        "Usage:\n" +
        "  RasMapperStoreMap.exe -RasMapFilename=<path>\n" +
        "                        [-ResultFilename=<path>]\n" +
        "                        [-RenderMode=sloping|slopingPretty|horizontal]\n" +
        "                        [-UseDepthWeightedFaces=true|false]\n" +
        "                        [-ReduceShallowToHorizontal=true|false]\n" +
        "                        [-TightExtent=true|false]\n" +
        "                        [-RasMapperLibDir=<HEC-RAS install dir>]");
    return 1;
}

// ── Locate RasMapperLib.dll ───────────────────────────────────────────────────

libDir ??= ResolveLibDir();

if (libDir is null)
{
    Console.Error.WriteLine(
        "RasMapperStoreMap: cannot locate RasMapperLib.dll.\n" +
        "Pass -RasMapperLibDir=<dir>, or set the HEC_RAS_LIB_DIR environment variable.");
    return 1;
}

string rasMapperLibPath = Path.Combine(libDir, "RasMapperLib.dll");
if (!File.Exists(rasMapperLibPath))
{
    Console.Error.WriteLine($"RasMapperStoreMap: RasMapperLib.dll not found in '{libDir}'.");
    return 1;
}

// ── Load RasMapperLib + resolve transitive dependencies ──────────────────────

// All HEC-RAS assemblies (Utility.dll, etc.) live in the same directory.
// Use AppDomain.AssemblyResolve — the .NET Framework equivalent of
// AssemblyLoadContext.Default.Resolving — since RasMapperLib targets net4x.
AppDomain.CurrentDomain.AssemblyResolve += (sender, e) =>
{
    var assemblyName = new AssemblyName(e.Name);
    var candidate = Path.Combine(libDir, assemblyName.Name + ".dll");
    return File.Exists(candidate) ? Assembly.LoadFrom(candidate) : null;
};

// ── Initialise GDAL before loading RasMapperLib ───────────────────────────────
//
// RasMapperLib's static initialiser (and several command classes) calls
// GDALSetup.InitializeMultiplatform() with a path derived from
// Assembly.GetExecutingAssembly().Location.  When our stub runs, that location
// is src/raspy/bin/ — nowhere near the HEC-RAS GDAL folder — so the library
// emits "The 'GDAL' is not a sub folder to …" on stderr.
//
// Fix: load Geospatial.GDALAssist.dll ourselves and call
// GDALSetup.InitializeMultiplatform(libDir\GDAL) before loading RasMapperLib,
// mirroring exactly what GetProjection.cs and CreateTerrainCommand.cs do.
// Reference: archive/DLLs/RasProcess/RasProcess/GetProjection.cs ~line 28

var gdalDir = Path.Combine(libDir, "GDAL");
if (Directory.Exists(gdalDir))
{
    var gdalAssistPath = Path.Combine(libDir, "Geospatial.GDALAssist.dll");
    if (File.Exists(gdalAssistPath))
    {
        try
        {
            var gdalAsm = Assembly.LoadFrom(gdalAssistPath);
            var gdalSetupType = gdalAsm.GetType("Geospatial.GDALAssist.GDALSetup");
            if (gdalSetupType is not null)
            {
                var initMethod = gdalSetupType.GetMethod(
                    "InitializeMultiplatform",
                    BindingFlags.Public | BindingFlags.Static,
                    null,
                    [typeof(string)],
                    null);
                initMethod?.Invoke(null, [gdalDir]);
            }
        }
        catch (Exception ex)
        {
            // Non-fatal: raster maps may still work without full GDAL init.
            Console.Error.WriteLine(
                $"RasMapperStoreMap: GDAL initialisation warning — {ex.Message}");
        }
    }
}

Assembly asm;
try
{
    asm = Assembly.LoadFrom(rasMapperLibPath);
}
catch (Exception ex)
{
    Console.Error.WriteLine($"RasMapperStoreMap: failed to load RasMapperLib.dll — {ex.Message}");
    return 1;
}

// ── Apply render mode to SharedData ──────────────────────────────────────────
//
// SharedData module-level defaults (used by RasProcess.exe) are:
//   CellRenderMode  = Sloping
//   CellStencilMode = JustFacepoints   ← excludes face centroids
//   UseDepthWeightedFaces = false
//   ShallowBehaviorMode   = ReduceToHorizontal
//
// Only SetSlopingPrettyRenderingMode() sets CellStencilMode = WithFaces, which
// enables face-centroid interpolation and depth-weighted faces.

var sharedDataType = asm.GetType("RasMapperLib.SharedData")
    ?? throw new InvalidOperationException("RasMapperLib.SharedData not found in assembly.");

// Install the classifying writer now so the header line below and all
// subsequent Console.WriteLine calls (including from RasMapperLib) are
// prefixed with "DEBUG: " or "INFO: " for the Python caller.
Console.SetOut(new ClassifyingWriter(Console.Out));

Console.WriteLine($"RasMapperStoreMap: RenderMode={renderMode} " +
                  $"UseDepthWeightedFaces={useDepthWeightedFaces} " +
                  $"ReduceShallowToHorizontal={reduceShallowToHorizontal}");

switch (renderMode.ToLowerInvariant())
{
    case "horizontal":
        InvokeStatic(sharedDataType, "SetHorizontalRenderingMode");
        break;

    case "sloping":
        InvokeStatic(sharedDataType, "SetSlopingRenderingMode");
        break;

    case "slopingpretty":
        // Signature: SetSlopingPrettyRenderingMode(bool reduceShallow, bool depthWeights)
        // Matches RASMapper.XMLLoad() call order (SharedData.cs ~line 1765).
        InvokeStatic(sharedDataType, "SetSlopingPrettyRenderingMode",
            reduceShallowToHorizontal, useDepthWeightedFaces);
        break;

    default:
        Console.Error.WriteLine(
            $"RasMapperStoreMap: unknown RenderMode '{renderMode}'. " +
            "Use horizontal, sloping, or slopingPretty.");
        return 1;
}

// ── Depth-weighted rendering: ensure FacePoint Elevations then run StoreMap ────
//
// UseDepthWeightedFaces=true requires a per-facepoint terrain elevation array
// stored in <model_dir>/<PlanShortID>/PostProcessing.hdf at:
//   Results/Unsteady/.../2D Flow Areas/<mesh>/Processed Data/
//     Profile (Horizontal)/FacePoint Elevation
//
// Root cause of "Error loading facepoint elevations for precip rendering method":
//   WaterSurfaceRenderer.GetComputer() (line ~2055) reads the in-memory field
//   RASD2FlowArea.PerMeshFacepointElevations directly, before
//   ComputeWaterSurfaceInternal2D (the only other code path that calls
//   EnsureGetPerMeshFacepointElevations at line ~2644) has a chance to run.
//   PerMeshFacepointElevations is a per-instance field — populated by
//   EnsureGetPerMeshFacepointElevations — so it is always null on a freshly
//   created RASResults object.
//
// Fix: for the depth-weighted case we bypass StoreAllMapsCommand and instead:
//   1. Create RASResults ourselves and ensure terrain is set.
//   2. Call PostProcessor.EnsureFacepointElevations() → writes PostProcessing.hdf.
//   3. Call RASD2FlowArea.EnsureGetPerMeshFacepointElevations() → populates the
//      in-memory PerMeshFacepointElevations field on THIS rasResults instance.
//   4. Construct RASResultsMap(rasResults) — SAME rasResults — and call StoreMap.
//      WaterSurfaceRenderer._geometry is rasResults.Geometry, so it sees the
//      already-populated PerMeshFacepointElevations and no longer throws.
//
// Reference:
//   RasMapperLib.PostProcessor.EnsureFacepointElevations()       (PostProcessor.cs ~532)
//   RasMapperLib.RASD2FlowArea.EnsureGetPerMeshFacepointElevations() (~3834)
//   RasMapperLib.Render.WaterSurfaceRenderer.GetComputer()        (~1979, ~2055)
//   RasMapperLib.Scripting.StoreAllMapsCommand.Execute()           (~60)

if (useDepthWeightedFaces || tightExtent)
{
    if (resultFilename is null)
    {
        Console.Error.WriteLine(
            "RasMapperStoreMap: -ResultFilename=<path> is required when " +
            "-UseDepthWeightedFaces=true or -TightExtent=true.");
        return 1;
    }

    // ── Build ConsoleProgressReporter (same as non-depth-weighted path) ───────
    object? progressReporterDW = null;
    var utilityCoreAsmDW = AppDomain.CurrentDomain.GetAssemblies()
        .FirstOrDefault(a => a.GetName().Name == "Utility.Core");
    if (utilityCoreAsmDW is null)
    {
        var utilityCoreLibPathDW = Path.Combine(libDir, "Utility.Core.dll");
        if (File.Exists(utilityCoreLibPathDW))
            utilityCoreAsmDW = Assembly.LoadFrom(utilityCoreLibPathDW);
    }
    if (utilityCoreAsmDW is not null)
    {
        var cprTypeDW = utilityCoreAsmDW.GetType("Utility.Progress.ConsoleProgressReporter");
        if (cprTypeDW is not null)
        {
            var cprCtorDW = cprTypeDW.GetConstructor([typeof(bool)]);
            if (cprCtorDW is not null)
                progressReporterDW = cprCtorDW.Invoke([false]);
        }
    }

    try
    {
        // ── 1. Set SharedData.RasMapFilename (mirrors StoreAllMapsCommand line 62) ──
        var sharedDataRasMapFilenameProp = sharedDataType.GetProperty("RasMapFilename",
            BindingFlags.Public | BindingFlags.Static);
        sharedDataRasMapFilenameProp?.SetValue(null, rasMapFilename);

        // ── 2. Parse rasmap XML for terrain dict + result map layers ──────────
        string rasMapDir = Path.GetDirectoryName(rasMapFilename) ?? ".";
        var xRoot = System.Xml.Linq.XElement.Load(rasMapFilename);

        // Build terrain name → filename dict (mirrors StoreAllMapsCommand lines 77-95)
        var terrainDict = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var terrainsEl = xRoot.Element("Terrains");
        if (terrainsEl is not null)
        {
            foreach (var tLayer in terrainsEl.Elements("Layer"))
            {
                var tName = tLayer.Attribute("Name")?.Value;
                var tFile = tLayer.Attribute("Filename")?.Value;
                if (tName is null || tFile is null) continue;
                if (!Path.IsPathRooted(tFile))
                    tFile = Path.GetFullPath(Path.Combine(rasMapDir, tFile));
                if (File.Exists(tFile) && !terrainDict.ContainsKey(tName))
                    terrainDict.Add(tName, tFile);
            }
        }

        // ── 3. Create RASResults ──────────────────────────────────────────────
        var rasResultsType = asm.GetType("RasMapperLib.RASResults")
            ?? throw new InvalidOperationException("RasMapperLib.RASResults not found.");
        var rasResultsCtor = rasResultsType.GetConstructor([typeof(string)])
            ?? throw new InvalidOperationException("RASResults(string) constructor not found.");
        var rasResults = rasResultsCtor.Invoke([resultFilename]);

        // ── 4. Terrain resolution ─────────────────────────────────────────────
        // EnsureFacepointElevations silently no-ops when Geometry.Terrain is
        // null.  If automatic resolution fails (common in headless), parse the
        // rasmap XML and set Geometry.Terrain directly.
        var geometryProp = rasResultsType.GetProperty("Geometry",
            BindingFlags.Public | BindingFlags.Instance);
        var geometry = geometryProp?.GetValue(rasResults)
            ?? throw new InvalidOperationException("RASResults.Geometry is null.");

        var terrainProp = geometry.GetType().GetProperty("Terrain",
            BindingFlags.Public | BindingFlags.Instance);
        object? terrain = null;
        try { terrain = terrainProp?.GetValue(geometry); }
        catch { /* getter can throw if file missing */ }

        if (terrain is null)
        {
            Console.WriteLine(
                "RasMapperStoreMap: Terrain not auto-resolved; setting from rasmap XML...");

            string? terrainFile = terrainDict.Count > 0
                ? terrainDict.Values.First() : null;
            string  terrainName = terrainDict.Count > 0
                ? terrainDict.Keys.First() : "Terrain";

            if (terrainFile is not null)
            {
                var terrainLayerType = asm.GetType("RasMapperLib.TerrainLayer");
                var tlCtor = terrainLayerType?.GetConstructor(
                    [typeof(string), typeof(string), typeof(bool)]);
                if (tlCtor is not null)
                {
                    var terrainLayer = tlCtor.Invoke([terrainName, terrainFile, false]);
                    terrainProp!.SetValue(geometry, terrainLayer);
                    Console.WriteLine(
                        $"RasMapperStoreMap: Terrain set to '{terrainName}' ({terrainFile})");
                }
                else
                {
                    Console.Error.WriteLine(
                        "RasMapperStoreMap: warning — TerrainLayer ctor not found; " +
                        "FacePoint Elevation computation may fail.");
                }
            }
            else
            {
                Console.Error.WriteLine(
                    "RasMapperStoreMap: warning — no terrain file found in rasmap XML; " +
                    "FacePoint Elevation computation will fail.");
            }
        }

        // ── 5 & 6. FacePoint Elevation pre-computation (depth-weighted only) ──
        // Not needed for tight_extent-only; only required when
        // UseDepthWeightedFaces=true to pre-populate PerMeshFacepointElevations
        // before WaterSurfaceRenderer.GetComputer() reads it.
        if (useDepthWeightedFaces)
        {
        Console.WriteLine(
            "RasMapperStoreMap: Ensuring FacePoint Elevations for depth-weighted rendering...");

        var getPostProcessorMethod = rasResultsType.GetMethod("GetPostProcessor")
            ?? throw new InvalidOperationException("RASResults.GetPostProcessor() not found.");
        var postProcessor = getPostProcessorMethod.Invoke(rasResults, null);

        var ppFilenameProp = postProcessor!.GetType().GetProperty("Filename",
            BindingFlags.Public | BindingFlags.Instance);
        string ppFilename = ppFilenameProp?.GetValue(postProcessor) as string ?? "(unknown)";
        Console.WriteLine($"RasMapperStoreMap: PostProcessing.hdf path: {ppFilename}");

        var ensureFPEMethod = postProcessor.GetType()
            .GetMethod("EnsureFacepointElevations", BindingFlags.Public | BindingFlags.Instance)
            ?? throw new InvalidOperationException(
                   "PostProcessor.EnsureFacepointElevations() not found.");
        ensureFPEMethod.Invoke(postProcessor, [null]);
        Console.WriteLine("RasMapperStoreMap: FacePoint Elevation data written to PostProcessing.hdf.");

        // KEY FIX: WaterSurfaceRenderer.GetComputer reads PerMeshFacepointElevations
        // (an instance field on RASD2FlowArea) BEFORE ComputeWaterSurfaceInternal2D
        // ever calls EnsureGetPerMeshFacepointElevations.  By calling it here on
        // this rasResults, the field is populated BEFORE StoreMap/GetComputer runs.
        // We pass this same rasResults to RASResultsMap below so that
        // WaterSurfaceRenderer._geometry is THIS geometry with the populated cache.
        var d2FlowAreaProp = geometry.GetType().GetProperty("D2FlowArea",
            BindingFlags.Public | BindingFlags.Instance);
        var d2FlowArea = d2FlowAreaProp?.GetValue(geometry);
        if (d2FlowArea is not null)
        {
            var ensureGetMethod = d2FlowArea.GetType()
                .GetMethod("EnsureGetPerMeshFacepointElevations",
                           BindingFlags.Public | BindingFlags.Instance);
            ensureGetMethod?.Invoke(d2FlowArea, null);
            Console.WriteLine(
                "RasMapperStoreMap: FacePoint Elevation in-memory cache populated.");
        }
        } // end if (useDepthWeightedFaces)

        // ── 7. Run StoreMap / StoreMapTerrainResample per layer ──────────────
        // Mirrors StoreAllMapsCommand.Execute() but reuses the rasResults instance
        // created above (so PerMeshFacepointElevations is shared with the renderer
        // for depth-weighted, and StoreMapTerrainResample can be used for tight extent).
        var resultsEl = xRoot.Element("Results");
        if (resultsEl is null)
            throw new InvalidOperationException(
                "rasmap XML has no <Results> element.");

        var rasResultsMapType = asm.GetType("RasMapperLib.RASResultsMap")
            ?? throw new InvalidOperationException("RasMapperLib.RASResultsMap not found.");
        var rasResultsMapCtor = rasResultsMapType.GetConstructor([rasResultsType])
            ?? throw new InvalidOperationException(
                   "RASResultsMap(RASResults) constructor not found.");

        // XMLLoad may be on a base class — search through hierarchy.
        var xmlLoadMethod = FindMethodInHierarchy(rasResultsMapType, "XMLLoad",
            [typeof(System.Xml.XmlElement)]);

        var outputModeIsStoredTypeProp = rasResultsMapType.GetProperty(
            "OutputModeIsStoredType", BindingFlags.Public | BindingFlags.Instance);

        var setOverrideDictMethod = rasResultsMapType.GetMethod(
            "SetOverrideTerrainFilenamesDictionary",
            BindingFlags.Public | BindingFlags.Instance);

        // StoreMap(ProgressReporter reporter, bool showFinishedMessage) — 2 params
        var storeMapMethod = rasResultsMapType.GetMethods(
                BindingFlags.Public | BindingFlags.Instance)
            .FirstOrDefault(m => m.Name == "StoreMap" &&
                                 m.GetParameters().Length == 2 &&
                                 m.GetParameters()[1].ParameterType == typeof(bool));

        // ── Tight-extent support: resolve types for StoreMapTerrainResample ──
        //
        // MapProcessingEngine.StoreMapTerrainResample(map, terrain, baseFilename,
        //   extent, resampleCellSize, addItemsToSpIdx, reporter)
        //
        // addItemsToSpIdx is Action<SpatialIndex<int>> — populated in the per-layer
        // loop below using Expression.Lambda so we avoid a compile-time dependency
        // on the generic SpatialIndex<T> type.
        //
        // We mirror RasMapper's ViewExtent clip (InterpolatedLayer.cs ~2919):
        //   addItemsToSpIdx = spIdx => spIdx.Add(modelExtent, 0)
        // The spatial index then returns > 0 elements for every terrain tile cell
        // overlapping the model's bounding box, giving a pixel-aligned tight output.
        //
        // Reference: RasMapperLib.MapProcessingEngine.StoreMapTerrainResample (~1346)
        //            RasMapperLib.InterpolatedLayer.ExentToClipFromExportRasterOptions (~2843)
        var extentType     = asm.GetType("RasMapperLib.Extent");
        var spIdxOpenType  = asm.GetType("RasMapperLib.SpatialIndex`1");
        var spIdxIntType   = spIdxOpenType?.MakeGenericType(typeof(int));
        var mapProcEngType = asm.GetType("RasMapperLib.MapProcessingEngine");

        // Add(Extent bounds, int key) on SpatialIndex<int>
        var addSpIdxMethod = spIdxIntType is not null && extentType is not null
            ? spIdxIntType.GetMethod("Add", [extentType, typeof(int)])
            : null;

        // StoreMapTerrainResample — static, 7 parameters
        var storeMapTRMethod = mapProcEngType?.GetMethods(
                BindingFlags.Public | BindingFlags.Static)
            .FirstOrDefault(m => m.Name == "StoreMapTerrainResample" &&
                                 m.GetParameters().Length == 7);

        // StoredMapBaseFilename — the output tile base path (no extension)
        var storedMapBaseFilenameProp = rasResultsMapType.GetProperty(
            "StoredMapBaseFilename",
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

        // Extent property on RASResultsMap: union of D2FlowArea + XS + StorageArea extents
        var extentPropRM = rasResultsMapType.GetProperty(
            "Extent", BindingFlags.Public | BindingFlags.Instance);

        bool tightExtentReady = tightExtent &&
                                storeMapTRMethod is not null &&
                                extentType      is not null &&
                                spIdxIntType    is not null &&
                                addSpIdxMethod  is not null &&
                                storedMapBaseFilenameProp is not null &&
                                extentPropRM    is not null;

        if (tightExtent && !tightExtentReady)
            Console.Error.WriteLine(
                "RasMapperStoreMap: warning — tight extent setup incomplete " +
                "(missing type or method); falling back to StoreMap.");

        string resultFileFullPath = Path.GetFullPath(resultFilename);
        int mapsGenerated = 0;

        foreach (var resultLayer in resultsEl.Elements("Layer"))
        {
            var fileAttr = resultLayer.Attribute("Filename");
            if (fileAttr is null) continue;

            string rPath = fileAttr.Value;
            if (!Path.IsPathRooted(rPath))
                rPath = Path.GetFullPath(Path.Combine(rasMapDir, rPath));

            // Only process the result file that matches our -ResultFilename arg
            if (!string.Equals(Path.GetFullPath(rPath), resultFileFullPath,
                    StringComparison.OrdinalIgnoreCase))
                continue;

            // Re-check terrain (should be non-null since we set it above)
            try { terrain = terrainProp?.GetValue(geometry); } catch { }
            if (terrain is null)
            {
                Console.Error.WriteLine(
                    $"RasMapperStoreMap: No terrain for '{Path.GetFileName(rPath)}' — skipping.");
                continue;
            }

            foreach (var mapLayerEl in resultLayer.Elements("Layer")
                         .Where(l => (string?)l.Attribute("Type") == "RASResultsMap"))
            {
                try
                {
                    // Construct RASResultsMap with the SAME rasResults (shares cache)
                    var rasResultsMap = rasResultsMapCtor.Invoke([rasResults]);

                    // Load layer settings from XML
                    if (xmlLoadMethod is not null)
                    {
                        var xmlDoc = new System.Xml.XmlDocument();
                        using var xmlReader = mapLayerEl.CreateReader();
                        var xmlNode = xmlDoc.ReadNode(xmlReader);
                        xmlLoadMethod.Invoke(rasResultsMap, [xmlNode as System.Xml.XmlElement]);
                    }

                    bool isStored = (bool)(outputModeIsStoredTypeProp?.GetValue(rasResultsMap)
                                          ?? false);
                    if (!isStored) continue;

                    setOverrideDictMethod?.Invoke(rasResultsMap, [terrainDict]);

                    if (tightExtentReady)
                    {
                        // Build Action<SpatialIndex<int>>(spIdx => spIdx.Add(modelExtent, 0))
                        // Mirrors RasMapper's ViewExtent clip: a single bounding-box item
                        // causes ComputeRFIMetaDataAndIntersectingTiles to include all terrain
                        // pixels overlapping the model extent (pixel-aligned tight output).
                        var modelExtent    = extentPropRM!.GetValue(rasResultsMap);
                        var smBaseFilename = storedMapBaseFilenameProp!.GetValue(rasResultsMap) as string;

                        if (modelExtent is not null && smBaseFilename is not null)
                        {
                            var spIdxParam = Expression.Parameter(spIdxIntType!, "spIdx");
                            var addCall    = Expression.Call(
                                spIdxParam, addSpIdxMethod!,
                                Expression.Constant(modelExtent, extentType!),
                                Expression.Constant(0, typeof(int)));
                            var addDelegate = Expression.Lambda(
                                typeof(Action<>).MakeGenericType(spIdxIntType!),
                                addCall, spIdxParam).Compile();

                            storeMapTRMethod!.Invoke(null,
                                [rasResultsMap, terrain, smBaseFilename,
                                 modelExtent, -1.0, addDelegate, progressReporterDW]);
                        }
                        else
                        {
                            Console.Error.WriteLine(
                                "RasMapperStoreMap: tight extent — could not resolve model " +
                                "extent or output filename; falling back to StoreMap.");
                            storeMapMethod?.Invoke(rasResultsMap, [progressReporterDW, false]);
                        }
                    }
                    else
                    {
                        storeMapMethod?.Invoke(rasResultsMap, [progressReporterDW, false]);
                    }

                    mapsGenerated++;
                }
                catch (TargetInvocationException tie) when (tie.InnerException is not null)
                {
                    Console.Error.WriteLine(
                        $"RasMapperStoreMap: StoreMap layer failed — {tie.InnerException.Message}");
                    // Mirror StoreAllMapsCommand behaviour: log and continue to next layer
                }
            }
        }

        Console.WriteLine(
            $"RasMapperStoreMap: {mapsGenerated} Maps generated " +
            $"for '{Path.GetFileName(resultFilename)}'.");
        return 0;   // ← early exit — skip the StoreAllMapsCommand block below
    }
    catch (TargetInvocationException tie) when (tie.InnerException is not null)
    {
        Console.Error.WriteLine(
            $"RasMapperStoreMap: depth-weighted setup failed — {tie.InnerException.Message}");
        return 1;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine(
            $"RasMapperStoreMap: depth-weighted setup error — {ex.Message}");
        return 1;
    }
}

// ── Instantiate and execute StoreAllMapsCommand ───────────────────────────────
//
// Constructor: StoreAllMapsCommand(string rmFilename, string resFilename = "")
// Execute(ProgressReporter prog) — pass a real ConsoleProgressReporter so that
//   progress messages reach stdout; see the ConsoleProgressReporter block below.

var cmdType = asm.GetType("RasMapperLib.Scripting.StoreAllMapsCommand")
    ?? throw new InvalidOperationException(
           "RasMapperLib.Scripting.StoreAllMapsCommand not found in assembly.");

var ctor = cmdType.GetConstructor([typeof(string), typeof(string)])
    ?? throw new InvalidOperationException(
           "StoreAllMapsCommand(string, string) constructor not found.");

var cmd = ctor.Invoke([rasMapFilename, resultFilename ?? ""]);

var executeMethod = cmdType.GetMethod("Execute")
    ?? throw new InvalidOperationException("StoreAllMapsCommand.Execute() not found.");

// ── Build a ConsoleProgressReporter so progress messages reach stdout ─────────
//
// ConsoleProgressReporter lives in Utility.Core.dll (Utility.Progress namespace).
// The single constructor parameter:
//   progressOverwritesLine=false → each update on its own line (correct for
//   line-by-line stdout streaming; true would emit bare \r carriage returns).
//
// Reference: Utility.Core.dll — Utility.Progress.ConsoleProgressReporter

object? progressReporter = null;
var utilityCoreAsm = AppDomain.CurrentDomain.GetAssemblies()
    .FirstOrDefault(a => a.GetName().Name == "Utility.Core");
if (utilityCoreAsm is null)
{
    var utilityCoreLibPath = Path.Combine(libDir, "Utility.Core.dll");
    if (File.Exists(utilityCoreLibPath))
        utilityCoreAsm = Assembly.LoadFrom(utilityCoreLibPath);
}
if (utilityCoreAsm is not null)
{
    var cprType = utilityCoreAsm.GetType("Utility.Progress.ConsoleProgressReporter");
    if (cprType is not null)
    {
        var cprCtor = cprType.GetConstructor([typeof(bool)]);
        if (cprCtor is not null)
            progressReporter = cprCtor.Invoke([false]);
    }
}


try
{
    executeMethod.Invoke(cmd, [progressReporter]);
}
catch (TargetInvocationException tie) when (tie.InnerException is not null)
{
    Console.Error.WriteLine($"RasMapperStoreMap: StoreAllMapsCommand failed — {tie.InnerException.Message}");
    return 1;
}

return 0;

// ── Helpers ───────────────────────────────────────────────────────────────────

static bool   Starts(string arg, string prefix) =>
    arg.StartsWith(prefix, StringComparison.OrdinalIgnoreCase);

static string Val(string arg) =>
    arg.Substring(arg.IndexOf('=') + 1);

static bool Bool(string arg) =>
    Val(arg).Equals("true", StringComparison.OrdinalIgnoreCase);

static void InvokeStatic(Type type, string methodName, params object[] methodArgs)
{
    var method = type.GetMethod(methodName, BindingFlags.Public | BindingFlags.Static)
        ?? throw new InvalidOperationException($"{type.Name}.{methodName} not found.");
    method.Invoke(null, methodArgs.Length > 0 ? methodArgs : null);
}

// Walk the inheritance chain to find a method with the given parameter types.
// GetMethod() with an explicit Type[] only searches the declared type, not base
// classes, when BindingFlags are omitted on certain .NET versions.
static MethodInfo? FindMethodInHierarchy(Type type, string name, Type[] paramTypes)
{
    for (var t = type; t is not null; t = t.BaseType)
    {
        var m = t.GetMethod(name,
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance,
            null, paramTypes, null);
        if (m is not null) return m;
    }
    return null;
}

// ResolveLibDir() is a fallback for ad-hoc/manual use of the exe.
// When called from raspy's store_map(), -RasMapperLibDir is always passed
// explicitly (resolved via installed_ras_directory()), so this is never invoked.
static string? ResolveLibDir()
{
    // 1. Environment variable
    var envDir = Environment.GetEnvironmentVariable("HEC_RAS_LIB_DIR");
    if (envDir is not null && File.Exists(Path.Combine(envDir, "RasMapperLib.dll")))
        return envDir;

    // 2. Common HEC-RAS installation paths, newest first
    string[] candidates =
    [
        @"C:\Program Files (x86)\HEC\HEC-RAS\6.6",
        @"C:\Program Files (x86)\HEC\HEC-RAS\6.5",
        @"C:\Program Files (x86)\HEC\HEC-RAS\6.4.1",
        @"C:\Program Files (x86)\HEC\HEC-RAS\6.3.1",
        @"C:\Program Files (x86)\HEC\HEC-RAS\6.1",
        @"C:\Program Files\HEC\HEC-RAS\6.6",
    ];

    foreach (var dir in candidates)
    {
        if (File.Exists(Path.Combine(dir, "RasMapperLib.dll")))
            return dir;
    }

    return null;
}

// ── ClassifyingWriter ─────────────────────────────────────────────────────────
//
// Wraps Console.Out and prefixes every output line with "DEBUG: " or "INFO: "
// so the Python caller (_mapper.py) can log fine-grained GDAL progress at DEBUG
// and meaningful milestones at INFO.
//
// DEBUG patterns:
//   "File N of M: N% processed (...)"  — per-terrain-file GDAL percentage
//   "0...10...20...30...100 - done."   — GDAL GDALTermProgress bar
//   "RasMapperStoreMap: RenderMode=..."— our own header line
// INFO: everything else (Progress: N%, milestones, summaries)

sealed class ClassifyingWriter : TextWriter
{
    private readonly TextWriter _inner;
    private readonly StringBuilder _buf = new StringBuilder();

    public ClassifyingWriter(TextWriter inner) { _inner = inner; }

    public override Encoding Encoding => _inner.Encoding;

    public override void Write(char value)
    {
        if (value == '\n') Flush();
        else _buf.Append(value);
    }

    public override void WriteLine(string? value)
    {
        _buf.Append(value);
        Flush();
    }

    public override void Flush()
    {
        if (_buf.Length == 0) return;
        string line = _buf.ToString().TrimEnd('\r');
        _buf.Clear();
        string prefix = IsDebugLine(line) ? "DEBUG: " : "INFO: ";
        _inner.WriteLine(prefix + line);
        _inner.Flush();
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing) Flush();
        base.Dispose(disposing);
    }

    private static readonly Regex _reFilePct =
        new Regex(@"^File \d+ of \d+: \d+% processed", RegexOptions.Compiled);
    private static readonly Regex _reGdal =
        new Regex(@"^\d+\.\.\.\d+.*done\.", RegexOptions.Compiled);

    private static bool IsDebugLine(string line) =>
        _reFilePct.IsMatch(line) ||
        _reGdal.IsMatch(line) ||
        line.StartsWith("RasMapperStoreMap: RenderMode=", StringComparison.Ordinal);
}
