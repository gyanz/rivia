"""Helpers for sampling hydraulic results along a polyline profile."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _to_xy(line: "np.ndarray | Any") -> np.ndarray:
    """Normalise a line-like input to an ``(n, 2)`` float64 ndarray.

    Parameters
    ----------
    line : ndarray or object with ``__geo_interface__``
        Either an array-like of shape ``(n, 2)`` containing ``(x, y)``
        pairs, or any object that exposes a ``__geo_interface__`` property
        returning a GeoJSON-like mapping (e.g. a :class:`shapely.LineString`).
        Only ``"LineString"`` geometry type is accepted; all others raise
        :exc:`TypeError`.  Z coordinates are silently dropped.

    Returns
    -------
    ndarray, shape (n, 2), dtype float64

    Raises
    ------
    TypeError
        If a ``__geo_interface__`` object is not a ``LineString``.
    ValueError
        If the resulting array is not shape ``(n, 2)`` with ``n >= 2``.
    """
    if hasattr(line, "__geo_interface__"):
        geom = line.__geo_interface__
        if geom["type"] != "LineString":
            raise TypeError(
                f"Expected a LineString geometry, got {geom['type']!r}."
            )
        xy = np.array(geom["coordinates"], dtype=np.float64)[:, :2]
    else:
        xy = np.asarray(line, dtype=np.float64)

    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
        raise ValueError(
            f"Line must resolve to shape (n, 2) with n >= 2; got shape {xy.shape}."
        )
    return xy


def _stations_to_xy(stations: np.ndarray, polyline_xy: np.ndarray) -> np.ndarray:
    """Convert along-line station distances to ``(x, y)`` coordinates.

    Parameters
    ----------
    stations : ndarray, shape (n,)
        Sorted along-line distances in model units (from polyline start).
    polyline_xy : ndarray, shape (m, 2)
        Polyline vertices as ``(x, y)`` pairs.

    Returns
    -------
    ndarray, shape (n, 2)
        ``(x, y)`` coordinates for each station.
    """
    stations = np.asarray(stations, dtype=np.float64)
    polyline_xy = np.asarray(polyline_xy, dtype=np.float64)

    seg_vecs = np.diff(polyline_xy, axis=0)                    # (m-1, 2)
    seg_lens = np.hypot(seg_vecs[:, 0], seg_vecs[:, 1])        # (m-1,)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])          # (m,)

    # For each station, find the segment it falls in (searchsorted right-biased
    # so that the very start of a segment is in that segment, not the previous).
    idx = np.searchsorted(cum, stations, side="right") - 1
    idx = np.clip(idx, 0, len(seg_lens) - 1)

    t = np.where(
        seg_lens[idx] > 0.0,
        (stations - cum[idx]) / seg_lens[idx],
        0.0,
    )
    t = np.clip(t, 0.0, 1.0)

    xy = polyline_xy[idx] + t[:, np.newaxis] * seg_vecs[idx]
    return xy


def _sample_raster_at_points(raster_path: Path, xy_pts: np.ndarray) -> np.ndarray:
    """Sample raster elevation at ``(x, y)`` points.

    Parameters
    ----------
    raster_path : Path
        Path to a GDAL-readable raster (VRT, GeoTIFF, etc.).
    xy_pts : ndarray, shape (n, 2)
        ``(x, y)`` sample points in the raster's CRS.

    Returns
    -------
    ndarray, shape (n,), dtype float64
        Elevation at each point.  ``NaN`` for points outside the raster
        extent or on nodata pixels.
    """
    import rasterio  # lazy — keeps geo/ the sole rasterio importer

    with rasterio.open(raster_path) as ds:
        nodata = ds.nodata
        raw = list(ds.sample([(float(x), float(y)) for x, y in xy_pts], indexes=1))

    values = np.array(raw, dtype=np.float64).ravel()
    if nodata is not None:
        values[values == nodata] = np.nan
    return values
