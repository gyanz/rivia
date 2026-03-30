# Installation

rivia requires Python 3.10+ and runs on **Windows only** (HEC-RAS is Windows-only).

## From PyPI

```bash
pip install rivia
```

## With geospatial extras

The `geo` subpackage requires `geopandas` and `rasterio`:

```bash
pip install rivia[geo]
```

## Development install

```bash
git clone https://github.com/gyanz/rivia.git
cd rivia
pip install -e ".[dev,geo,docs]"
```
