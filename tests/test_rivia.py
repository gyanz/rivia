"""Basic smoke tests for rivia package."""

import rivia


def test_version():
    assert rivia.__version__ == "0.1.0"


def test_subpackage_imports():
    import rivia.com
    import rivia.model
    import rivia.hdf
    import rivia.geo
    import rivia.utils
