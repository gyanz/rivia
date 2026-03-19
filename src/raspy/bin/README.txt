RasMapperStoreMap.exe belongs here.

Build steps (one-time, from the repo root):

    cd tools/RasMapperStoreMap
    dotnet build -c Release
    copy bin\Release\net9.0-windows\RasMapperStoreMap.exe ..\..\src\raspy\bin\

The exe is loaded by store_map() when render_mode != "sloping" or when
use_depth_weights / shallow_to_flat differ from their defaults.  It dynamically
loads RasMapperLib.dll from the HEC-RAS installation directory at runtime, so no
HEC-RAS DLLs need to be copied here.

This file is a placeholder so that setuptools includes the bin/ directory in the
source tree.  The .exe itself is intentionally excluded from version control.
