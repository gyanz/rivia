"""COM interface to run and control HEC-RAS."""

from typing import Any

from .registry import installed_ras_progids, ras_registry_xxx

_installed_ras: dict[str, Any] = {}
_installed_ras_versions: list = []


def _ensure_loaded() -> None:
    global _installed_ras
    global _installed_ras_versions
    if not _installed_ras:
        for entry in installed_ras_progids():
            xxx = entry["registry_xxx"]
            xxxx = entry["version_xxxx"]
            if xxx:
                _installed_ras[xxx] = entry
            
    if not _installed_ras_versions:
        for entry in installed_ras_progids():
            xxxx = entry["version_xxxx"]
            if xxxx:
                _installed_ras_versions.append(xxxx)


def installed_ras_versions(descriptive:bool=False) -> list[str]:
    _ensure_loaded()
    if descriptive:
        display_names = []
        for entry in installed_ras_progids():
            display_names.append(entry["display_name"])
        return display_names

    return list(_installed_ras_versions)


def installed_ras_progid(version: str | int) -> dict[str, str | None]:
    _ensure_loaded()
    xxx = ras_registry_xxx(version)

    if xxx not in _installed_ras:
        raise RuntimeError(f"HEC-RAS {version} is not installed.")
    
    entry = _installed_ras[xxx]
    xxxx = int(entry["version_xxxx"])

    entry = _installed_ras[xxx]
    progids: dict[str, str | None] = {
        "controller": None, "geometry": None, "flow": None
    }
    for key in progids:
        com = entry.get(key)
        if com and com["exists"]:
            progids[key] = com["progid"]

    return xxxx,progids
