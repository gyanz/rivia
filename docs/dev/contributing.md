# Contributing

## Environment setup

```bash
conda activate raspy-dev
pip install -e ".[dev,geo,docs]"
```

## Running tests

```bash
pytest tests/ -x --tb=short
```

## Linting and type checking

```bash
ruff check src/
mypy src/raspy
```

## Building docs

```bash
sphinx-build -b html docs docs/_build/html
```

Or using the Makefile:

```bash
cd docs && make html
```
