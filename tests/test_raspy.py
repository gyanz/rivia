"""Basic smoke tests for raspy package."""

import raspy


def test_version():
    assert raspy.__version__ == "0.1.0"


def test_subpackage_imports():
    import raspy.controller
    import raspy.io
    import raspy.hdf
    import raspy.geometry
    import raspy.utils
