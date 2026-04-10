"""COM interface to run and control HEC-RAS."""

from .controller import HecRasComputeError, connect  # noqa: F401

__all__ = ["connect", "HecRasComputeError"]
