"""Basic smoke tests for raspy package."""

import raspy


def test_version():
    assert raspy.__version__ == "0.1.0"


def test_subpackage_imports():
    import raspy.com
    import raspy.model
    import raspy.hdf
    import raspy.geo
    import raspy.utils
