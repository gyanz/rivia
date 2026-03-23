# Installation

raspy requires Python 3.10+ and runs on **Windows only** (HEC-RAS is Windows-only).

## From PyPI

```bash
pip install raspy
```

## With geospatial extras

The `geo` subpackage requires `geopandas` and `rasterio`:

```bash
pip install raspy[geo]
```

## Development install

```bash
git clone https://github.com/gbasyal/raspy.git
cd raspy
pip install -e ".[dev,geo,docs]"
```
