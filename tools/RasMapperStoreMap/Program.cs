/*
 * RasMapperStoreMap.exe
 *
 * Thin wrapper around RasMapperLib.Scripting.StoreAllMapsCommand that initialises
 * SharedData render-mode state before executing the command.  RasProcess.exe never
 * sets render mode (SharedData defaults to basic Sloping/JustFacepoints), so rasters
 * produced by RasProcess.exe always ignore the <RenderMode> settings in the .rasmap
 * file.  This stub fixes that by reading the render-mode arguments and calling the
 * appropriate SharedData setter before delegating to StoreAllMapsCommand.
 *
 * RasMapperLib.dll is loaded dynamically at runtime from the HEC-RAS installation
 * directory so this project has zero compile-time references to HEC-RAS assemblies.
 *
 * Usage:
 *   RasMapperStoreMap.exe
 *       -RasMapFilename=<path>
 *       [-ResultFilename=<path>]
 *       [-RenderMode=sloping|slopingPretty|horizontal]   (default: sloping)
 *       [-UseDepthWeightedFaces=true|false]              (default: false)
 *       [-ReduceShallowToHorizontal=true|false]          (default: true)
 *       [-RasMapperLibDir=<HEC-RAS install dir>]
 *
 * If -RasMapperLibDir is omitted the exe tries:
 *   1. Environment variable HEC_RAS_LIB_DIR
 *   2. Common HEC-RAS installation paths (6.6 down to 6.1)
 *
 * Source references:
 *   archive/DLLs/RasMapperLib/RasMapperLib.Scripting/StoreAllMapsCommand.cs
 *   archive/DLLs/RasMapperLib/RasMapperLib/SharedData.cs  (~line 1765, ~line 1926)
 *   archive/DLLs/RasMapperLib/RasMapperLib/RASMapper.cs   (~line 12876)
 *   docs/rasprocess_render_mode_limitation.md
 */

using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;

// ── Argument parsing ──────────────────────────────────────────────────────────

string? rasMapFilename = null;
string? resultFilename = null;
string  renderMode               = "sloping";
bool    useDepthWeightedFaces    = false;
bool    reduceShallowToHorizontal = true;
string? libDir = null;

foreach (var arg in args)
{
    if      (Starts(arg, "-RasMapFilename="))           rasMapFilename            = Val(arg);
    else if (Starts(arg, "-ResultFilename="))            resultFilename            = Val(arg);
    else if (Starts(arg, "-RenderMode="))                renderMode                = Val(arg);
    else if (Starts(arg, "-UseDepthWeightedFaces="))     useDepthWeightedFaces     = Bool(arg);
    else if (Starts(arg, "-ReduceShallowToHorizontal=")) reduceShallowToHorizontal = Bool(arg);
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

// ── Ensure FacePoint Elevations (required for depth-weighted rendering) ────────
//
// UseDepthWeightedFaces=true requires a per-facepoint terrain elevation array
// stored in <model_dir>/<PlanShortID>/PostProcessing.hdf at:
//   Results/Unsteady/.../2D Flow Areas/<mesh>/Processed Data/
//     Profile (Horizontal)/FacePoint Elevation
//
// RasMapper GUI triggers this computation via a UI popup on first use.  In
// headless mode the popup is skipped, so WaterSurfaceRenderer finds the array
// null and throws "Error loading facepoint elevations for precip rendering".
//
// Fix: call PostProcessor.EnsureFacepointElevations() directly before
// StoreAllMapsCommand, mirroring RasMapper's own staleness logic:
//   • PostProcessing.hdf missing / corrupt / older than result HDF → full
//     rebuild via PostProcessor.NeedsBaseFileUpdate() + CreatePostProcessBase()
//   • PostProcessing.hdf current but FacePoint Elevation dataset absent →
//     compute and write just that dataset (fast)
//   • PostProcessing.hdf current and dataset present → no-op (fastest)
//
// Critical: SharedData.RasMapFilename must be set before constructing
// RASResults so that RASResults.Geometry.Terrain resolves via the rasmap.
// StoreAllMapsCommand does this at line 62; we mirror that here.
//
// Reference:
//   RasMapperLib.PostProcessor.EnsureFacepointElevations()       (PostProcessor.cs)
//   RasMapperLib.PostProcessor.NeedsBaseFileUpdate()              (PostProcessor.cs)
//   RasMapperLib.Scripting.StoreAllMapsCommand.Execute() line 62  (StoreAllMapsCommand.cs)

if (useDepthWeightedFaces)
{
    if (resultFilename is null)
    {
        Console.Error.WriteLine(
            "RasMapperStoreMap: -ResultFilename=<path> is required when " +
            "-UseDepthWeightedFaces=true (needed to locate PostProcessing.hdf).");
        return 1;
    }

    Console.WriteLine(
        "RasMapperStoreMap: Ensuring FacePoint Elevations for depth-weighted rendering...");
    try
    {
        // Set SharedData.RasMapFilename before constructing RASResults so the
        // geometry can resolve the terrain layer — exactly what StoreAllMapsCommand
        // does before its own new RASResults(value3) call.
        var sharedDataRasMapFilenameProp = sharedDataType.GetProperty("RasMapFilename",
            BindingFlags.Public | BindingFlags.Static);
        sharedDataRasMapFilenameProp?.SetValue(null, rasMapFilename);

        var rasResultsType = asm.GetType("RasMapperLib.RASResults")
            ?? throw new InvalidOperationException("RasMapperLib.RASResults not found.");
        var rasResultsCtor = rasResultsType.GetConstructor([typeof(string)])
            ?? throw new InvalidOperationException("RASResults(string) constructor not found.");
        var rasResults = rasResultsCtor.Invoke([resultFilename]);

        var getPostProcessorMethod = rasResultsType.GetMethod("GetPostProcessor")
            ?? throw new InvalidOperationException("RASResults.GetPostProcessor() not found.");
        var postProcessor = getPostProcessorMethod.Invoke(rasResults, null);

        var ensureMethod = postProcessor.GetType()
            .GetMethod("EnsureFacepointElevations",
                       BindingFlags.Public | BindingFlags.Instance)
            ?? throw new InvalidOperationException(
                   "PostProcessor.EnsureFacepointElevations() not found.");

        // Pass null → the method defaults to ProgressReporter.None() internally.
        ensureMethod.Invoke(postProcessor, [null]);
    }
    catch (TargetInvocationException tie) when (tie.InnerException is not null)
    {
        Console.Error.WriteLine(
            "RasMapperStoreMap: EnsureFacepointElevations failed — " +
            tie.InnerException.Message);
        return 1;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine(
            $"RasMapperStoreMap: EnsureFacepointElevations error — {ex.Message}");
        return 1;
    }
}

// ── Instantiate and execute StoreAllMapsCommand ───────────────────────────────
//
// Constructor: StoreAllMapsCommand(string rmFilename, string resFilename = "")
// Execute(ProgressReporter prog = null) — null is handled internally via
//   ProgressReporter.None(), so no Utility.dll reference is needed here.

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
// StoreAllMapsCommand.Execute(null) uses ProgressReporter.None() — a no-op that
// silently drops all ReportMessage / ReportProgress calls.  Passing a
// ConsoleProgressReporter instead causes every message and percentage update to
// be written to stdout, matching the behaviour callers expect from RasProcess.exe.
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
